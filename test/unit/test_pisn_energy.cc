// Unit test: supernova explosion energies (galaxy_sf/resolvedism_sn_energy.h).
// PISN: Heger & Woosley 2002 via the Wise+12a fit, E = [5.0 + 1.304 (M_He - 64)] 1e51 erg, M_He = (13/24)(M - 20),
// He core held to HW02's PISN range 64-133 Msun (opt-in GALSF_RESOLVEDISM_PISN_ENERGY_FROM_HELIUM_CORE; default PISN 1e51);
// other explosive deaths 1e51 erg; failed SNe and direct BHs 0.
#include "test_harness.h"
#include "galaxy_sf/resolvedism_sn_energy.h"

TEST_CASE("PISN energy from the helium core") {
    CHECK_CLOSE(resolvedism_pisn_energy_erg(200.) / 1e51, 5.0 + 1.304 * (97.5 - 64.), 1e-9); // design report: 4.9e52
    CHECK_CLOSE(resolvedism_pisn_energy_erg(140.) / 1e51, 5.0 + 1.304 * (65. - 64.), 1e-9);
    CHECK_CLOSE(resolvedism_pisn_energy_erg(20. + 64. * 24. / 13.) / 1e51, 5.0, 1e-9);     // lower edge of the HW02 range
    CHECK_CLOSE(resolvedism_pisn_energy_erg(130.) / 1e51, 5.0, 1e-9);                      // table PISN below the range: held at M_He = 64
    CHECK_CLOSE(resolvedism_pisn_energy_erg(20. + 133. * 24. / 13.) / 9.35e52, 1.0, 0.02); // HW02 maximum, 9.35e52 for the 133 Msun core
    CHECK_CLOSE(resolvedism_pisn_energy_erg(400.), resolvedism_pisn_energy_erg(20. + 133. * 24. / 13.), 1e-9 * 1e52);
    double prev = 0;
    for(double M = 100; M <= 500; M += 0.5) {double E = resolvedism_pisn_energy_erg(M); CHECK(E >= prev && E >= 5e51 && E <= 9.5e52); prev = E;}
}

TEST_CASE("energy by death type") {
    CHECK(resolvedism_sn_energy_erg(0, 1, 15.) == 1e51);  // CCSN, ECSN, PPISN
    CHECK(resolvedism_sn_energy_erg(0, 0, 40.) == 0);     // failed SN, direct BH
#ifdef GALSF_RESOLVEDISM_PISN_ENERGY_FROM_HELIUM_CORE
    CHECK(resolvedism_sn_energy_erg(1, 1, 200.) == resolvedism_pisn_energy_erg(200.));
#else
    CHECK(resolvedism_sn_energy_erg(1, 1, 200.) == 1e51); // default: flat 1e51 for PISN
#endif
}

TEST_MAIN()
