// Unit test: H- photodetachment from band energy densities (cooling/sgchem/sgchem_hminus_bands.h) against the direct
// integral k = c int u_nu sigma/(h nu) dnu over blackbody fields. Exact for the 3e4 K in-band shape, within 25% for
// 1e4-1e5 K stars. A hot-star field normalised to G0 = 1 (6-13.6 eV) detaches ~1e-2 of the ISRF-calibrated 4.2e-7 s^-1.
#include "test_harness.h"
#include <cmath>
#include <cstdio>
#include "src/cooling/sgchem/sgchem_hminus_bands.h"

static const double H = 6.62607015e-27, KB = 1.380649e-16, EV = 1.602176634e-12, C = 2.99792458e10;

// energy density of a blackbody shape u_nu = A nu^3/(e^{h nu/kT}-1) between E_lo and E_hi [eV], and its direct rate
static void bb_band(double T, double A, double lo, double hi, double *u, double *k)
{
    const int n = 20000; double du = 0, dk = 0, dnu = (hi - lo) * EV / H / n;
    for(int i = 0; i < n; i++) {
        double nu = lo * EV / H + (i + 0.5) * dnu, unu = A * nu * nu * nu / expm1(H * nu / (KB * T));
        du += unu * dnu; dk += C * unu * sgchem_hminus_sigma(nu) / (H * nu) * dnu;
    }
    *u = du; *k = dk;
}

static double rate_ratio(double T, double *k_per_G0)
{
    double uo, ko, un, kn, uf, kf;
    bb_band(T, 1.0, 0.4133, 3.444, &uo, &ko); bb_band(T, 1.0, 3.444, 6.0, &un, &kn); bb_band(T, 1.0, 6.0, 13.6, &uf, &kf);
    double k_direct = ko + kn + kf, k_band = sgchem_hminus_rate_from_bands(uo, un, uf);
    *k_per_G0 = k_band / (uf / 5.257e-14); // per Habing G0 in 6-13.6 eV
    return k_band / k_direct;
}

TEST_CASE("cross-section peaks near 4e-17 cm^2 at ~1.5 eV") {
    double smax = 0, Emax = 0;
    for(double E = 0.76; E < 13.6; E += 0.001) {double s = sgchem_hminus_sigma(E * EV / H); if(s > smax) {smax = s; Emax = E;}}
    std::printf("  sigma_max = %.3e cm^2 at %.3f eV\n", smax, Emax);
    CHECK(smax > 3.5e-17 && smax < 4.5e-17 && Emax > 1.3 && Emax < 1.7);
    CHECK(sgchem_hminus_sigma(0.75 * EV / H) == 0);
}

TEST_CASE("band rate against the direct blackbody integral") {
    for(double T : {1e4, 2e4, 3e4, 5e4, 1e5}) {
        double kG0, r = rate_ratio(T, &kG0);
        std::printf("  T = %.0e K: k_band/k_direct = %.4f, k(G0 = 1) = %.3e s^-1 (%.4f x 4.2e-7)\n", T, r, kG0, kG0 / 4.2e-7);
        if(T == 3e4) {CHECK_CLOSE(r, 1.0, 1e-3);} else {CHECK(fabs(r - 1) < 0.25);}
        if(T >= 2e4) {CHECK(kG0 < 0.05 * 4.2e-7);}
    }
}

TEST_CASE("linear in the band energies, zero without light") {
    CHECK(sgchem_hminus_rate_from_bands(0, 0, 0) == 0);
    CHECK_CLOSE(sgchem_hminus_rate_from_bands(2e-14, 4e-15, 6e-15), 2 * sgchem_hminus_rate_from_bands(1e-14, 2e-15, 3e-15), 1e-20);
}

TEST_MAIN()
