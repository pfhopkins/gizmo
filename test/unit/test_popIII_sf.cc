// Unit test: Pop III switch, star-formation criterion and IMF sampler (src/galaxy_sf/single_star_popIII.h).
// Criterion: every condition alone vetoes; an ionised cell never qualifies.
// IMF: mass-weighted draws through the LYRA 1/M acceptance reproduce the target number IMF (KS test, 1% level).
#include "test_harness.h"
#include <cmath>
#include <cstdio>
#include "src/galaxy_sf/single_star_popIII.h"
#include <algorithm>
#include <random>
#include <vector>

static const double ZSUN = 0.014;

TEST_CASE("pristine switch at Z_crit = 1e-4 Zsun") {
    double Zc = 1e-4 * ZSUN;
    CHECK(popIII_gas_is_pristine(0.0, Zc));
    CHECK(popIII_gas_is_pristine(0.99 * Zc, Zc));
    CHECK(!popIII_gas_is_pristine(Zc, Zc));
    CHECK(!popIII_gas_is_pristine(1e-3 * ZSUN, Zc));
}

TEST_CASE("criterion: each condition vetoes, ionised cells never form") {
    popIII_sf_params p = {1e-4 * ZSUN, 1e6, 1e-3, 1000.};
    double nH = 1.44e6, divv = -1e-13, xH2 = 1.05e-3, MJ = 468.; // Z=0 one-zone state where the criterion first passes
    CHECK(popIII_sf_allowed(nH, divv, xH2, MJ, 0, &p));
    CHECK(!popIII_sf_allowed(nH, divv, xH2, MJ, 1, &p));
    CHECK(!popIII_sf_allowed(nH, divv, xH2, MJ, 2, &p));
    CHECK(!popIII_sf_allowed(0.9e6, divv, xH2, MJ, 0, &p));
    CHECK(!popIII_sf_allowed(nH, 0.0, xH2, MJ, 0, &p));
    CHECK(!popIII_sf_allowed(nH, 1e-13, xH2, MJ, 0, &p));
    CHECK(!popIII_sf_allowed(nH, divv, 0.9e-3, MJ, 0, &p));
    CHECK(!popIII_sf_allowed(nH, divv, xH2, 1000., 0, &p));
    CHECK(!popIII_sf_allowed(1e9, divv, 0.5, 10., 1, &p)); // far above every threshold, still ionised
}

// KS distance between accepted masses and the number IMF dN/dlnM on [Mlo, Mhi]
static double ks_number_imf(int type, double Mchar, double Mlo, double Mhi, int n_accept, double *mmin, double *mmax)
{
    std::mt19937_64 gen(12345 + type);
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    auto uniform = [&]() {return u01(gen);};
    std::vector<double> m;
    *mmin = 1e30; *mmax = 0;
    while((int)m.size() < n_accept) {
        double M = popIII_draw_mass_weighted(uniform, type, Mchar, Mlo, Mhi);
        *mmin = std::min(*mmin, M); *mmax = std::max(*mmax, M);
        if(uniform() < Mlo / M) {m.push_back(M);} // LYRA: P proportional to M_cell / M_star
    }
    std::sort(m.begin(), m.end());
    const int NG = 20000; std::vector<double> lnM(NG + 1), cdf(NG + 1, 0.0); // reference CDF in ln M (trapezoid)
    for(int k = 0; k <= NG; k++) {lnM[k] = log(Mlo) + (log(Mhi) - log(Mlo)) * k / NG;}
    for(int k = 1; k <= NG; k++) {cdf[k] = cdf[k-1] + 0.5 * (popIII_imf_dNdlnM(exp(lnM[k-1]), type, Mchar) + popIII_imf_dNdlnM(exp(lnM[k]), type, Mchar)) * (lnM[k] - lnM[k-1]);}
    double D = 0;
    for(size_t i = 0; i < m.size(); i++) {
        double x = (log(m[i]) - log(Mlo)) / (log(Mhi) - log(Mlo)) * NG; int k = std::min((int)x, NG - 1); double f = x - k;
        double F = (cdf[k] + f * (cdf[k+1] - cdf[k])) / cdf[NG];
        D = std::max(D, std::max(fabs(F - (double)i / m.size()), fabs(F - (double)(i + 1) / m.size())));
    }
    return D;
}

TEST_CASE("IMF: characteristic mass 20 Msun on 1-300 Msun") {
    int N = 100000; double mmin, mmax;
    double D = ks_number_imf(POPIII_IMF_CHARACTERISTIC_MASS, 20., 1., 300., N, &mmin, &mmax);
    std::printf("  char-mass IMF: KS D = %.4g (1%% critical %.4g), draws in [%.3f, %.2f]\n", D, 1.63 / sqrt(N), mmin, mmax);
    CHECK(D < 1.63 / sqrt(N));
    CHECK(mmin >= 1. && mmax <= 300.);
}

TEST_CASE("IMF: log-flat on 5-210 Msun") {
    int N = 100000; double mmin, mmax;
    double D = ks_number_imf(POPIII_IMF_LOG_FLAT, 20., 5., 210., N, &mmin, &mmax);
    std::printf("  log-flat IMF: KS D = %.4g (1%% critical %.4g), draws in [%.3f, %.2f]\n", D, 1.63 / sqrt(N), mmin, mmax);
    CHECK(D < 1.63 / sqrt(N));
    CHECK(mmin >= 5. && mmax <= 210.);
}

TEST_MAIN()
