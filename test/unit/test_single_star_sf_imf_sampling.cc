// Unit test: single-star SF event (src/galaxy_sf/single_star_sf_event.h) with the IMF draw of src/galaxy_sf/single_star_imf.h must
// form stars from the Kroupa number IMF and the expected mass X for any X = M_cell * pfac. Bars per X (~5e4 stars): KS D < 1.63/sqrt(N); chi^2 (10 log bins, 9 dof) < 27.88;
// f(<0.5) and f(>8) within 4 sigma of Kroupa; mean formed mass per event within 4 standard errors of X.
#include "test_harness.h"
#include "src/galaxy_sf/single_star_imf.h"
#include "src/galaxy_sf/single_star_sf_event.h"
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <random>
#include <vector>

static const double MLO = 0.08, MHI = 350.;

// Kroupa number IMF on [MLO, MHI]: dN/dM ~ 2 M^-1.3 below 0.5, M^-2.3 above
static double kroupa_cum(double M) // unnormalised N(<M)
{
    double a = std::min(M, 0.5), c = 2. * (pow(MLO, -0.3) - pow(a, -0.3)) / 0.3;
    if(M > 0.5) {c += (pow(0.5, -1.3) - pow(M, -1.3)) / 1.3;}
    return c;
}
static double kroupa_cdf(double M) {return kroupa_cum(M) / kroupa_cum(MHI);}
static double kroupa_mean()
{
    double m1 = 2. * (pow(0.5, 0.7) - pow(MLO, 0.7)) / 0.7 + (pow(0.5, -0.3) - pow(MHI, -0.3)) / 0.3;
    return m1 / kroupa_cum(MHI);
}

static void check_event_sampling(double X)
{
    std::mt19937_64 gen(1000 + (unsigned long long)(X * 1000));
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    auto uniform = [&]() {return u01(gen);};
    const double G0 = single_star_imf_cumulative(MLO), G1 = single_star_imf_cumulative(MHI);
    auto draw = [&]() {return single_star_imf_inverse_cumulative(G0 + uniform() * (G1 - G0));}; // the mass-weighted draw of single_star_imf.h
    long E = (long)(5.0e4 * kroupa_mean() / X) + 1;
    std::vector<double> m, out;
    double s1 = 0, s2 = 0;
    for(long e = 0; e < E; e++) {
        double M0 = draw();
        single_star_sf_event(X, MLO, M0, [&](long) {return draw();}, [&](long) {return uniform();}, out);
        double me = 0;
        for(double x : out) {m.push_back(x); me += x;}
        s1 += me; s2 += me * me;
    }
    double N = (double)m.size();
    std::sort(m.begin(), m.end());
    double D = 0;
    for(size_t i = 0; i < m.size(); i++) {
        double F = kroupa_cdf(m[i]);
        D = std::max(D, std::max(fabs(F - i / N), fabs(F - (i + 1) / N)));
    }
    double chi2 = 0; int NB = 10;
    for(int b = 0; b < NB; b++) {
        double lo = MLO * pow(MHI / MLO, (double)b / NB), hi = MLO * pow(MHI / MLO, (double)(b + 1) / NB);
        double expct = N * (kroupa_cdf(hi) - kroupa_cdf(lo));
        double obs = (double)(std::lower_bound(m.begin(), m.end(), hi) - std::lower_bound(m.begin(), m.end(), lo));
        chi2 += (obs - expct) * (obs - expct) / expct;
    }
    double fl = (std::lower_bound(m.begin(), m.end(), 0.5) - m.begin()) / N, fh = (m.end() - std::upper_bound(m.begin(), m.end(), 8.0)) / N;
    double fl0 = kroupa_cdf(0.5), fh0 = 1 - kroupa_cdf(8.0);
    double mean_ev = s1 / E, se_ev = sqrt((s2 / E - mean_ev * mean_ev) / E);
    double mean_m = 0; for(double x : m) {mean_m += x;} mean_m /= N;
    std::printf("  X=%5.2f: N=%.0f KS D=%.4f (bar %.4f) chi2=%.1f (bar 27.88) f<0.5=%.4f (Kroupa %.4f) f>8=%.5f (Kroupa %.5f) <m>=%.3f (%.3f) mass/event/X=%.4f +- %.4f\n",
                X, N, D, 1.63 / sqrt(N), chi2, fl, fl0, fh, fh0, mean_m, kroupa_mean(), mean_ev / X, se_ev / X);
    CHECK(D < 1.63 / sqrt(N));
    CHECK(chi2 < 27.88);
    CHECK(fabs(fl - fl0) < 4 * sqrt(fl0 * (1 - fl0) / N));
    CHECK(fabs(fh - fh0) < 4 * sqrt(fh0 * (1 - fh0) / N));
    CHECK(fabs(mean_ev - X) < 4 * se_ev);
}

TEST_CASE("SF event X = 0.01 Msun") {check_event_sampling(0.01);}
TEST_CASE("SF event X = 0.1 Msun") {check_event_sampling(0.1);}
TEST_CASE("SF event X = 0.5 Msun") {check_event_sampling(0.5);}
TEST_CASE("SF event X = 1 Msun") {check_event_sampling(1.0);}
TEST_CASE("SF event X = 2 Msun") {check_event_sampling(2.0);}
TEST_CASE("SF event X = 5 Msun") {check_event_sampling(5.0);}

TEST_MAIN()
