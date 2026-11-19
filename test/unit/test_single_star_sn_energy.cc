// Unit test: single-star supernova energies by death type (src/galaxy_sf/single_star_sn_energy.h; stella_dev 61f0f5902, F6).
// PISN: Heger & Woosley 2002 (ApJ 567, 532) via the Wise et al. 2012a fit, E = [5.0 + 1.304 (M_He/Msun - 64)] 1e51 erg with
// M_He = (13/24)(M_ZAMS - 20) Msun, He core held to HW02's PISN range 64-133 Msun; the default (no flag).
// ECSN: 1e50 erg (Kitaura, Janka & Hillebrandt 2006, A&A 450, 345: 1.2e50 erg for the O-Ne-Mg core of an 8.8 Msun star).
// Other explosive deaths (CCSN, PPISN) 1e51 erg; failed SNe and direct collapse 0.
#include <cmath>
#include <cstdio>
#include "test_harness.h"
#include "src/galaxy_sf/single_star_sn_energy.h"

TEST_CASE("energies by death type") {
    CHECK(single_star_sn_energy_erg(0, 0, 1, 15.) == 1e51);   // CCSN, PPISN
    CHECK(single_star_sn_energy_erg(0, 1, 1, 8.5) == 1e50);   // ECSN
    CHECK(single_star_sn_energy_erg(0, 0, 0, 40.) == 0);      // failed SN, direct collapse
    CHECK(single_star_sn_energy_erg(0, 1, 0, 8.5) == 0);      // no explosion wins
}

TEST_CASE("PISN from the He core, held to 64-133 Msun") {
    for(double M : {100., 160., 200., 260., 320., 500.}) {
        double M_He = (13. / 24.) * (M - 20.); M_He = (M_He < 64.) ? 64. : ((M_He > 133.) ? 133. : M_He);
        double want = (5.0 + 1.304 * (M_He - 64.)) * 1e51, got = single_star_sn_energy_erg(1, 0, 1, M);
        std::printf("  M = %g Msun: M_He = %.3f, E = %.4e erg\n", M, M_He, got);
        CHECK(std::fabs(got / want - 1) < 1e-15);
    }
    CHECK(single_star_sn_energy_erg(1, 0, 1, 320.) > 9.4e52 && single_star_sn_energy_erg(1, 0, 1, 320.) < 9.6e52);
}

TEST_MAIN()
