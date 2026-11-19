// Unit test: supernova explosion energies by death type (galaxy_sf/resolvedism_sn_energy.h).
// Default: every explosive death (CCSN, ECSN, PPISN, PISN) 1e51 erg; failed SNe and direct collapse 0.
// With RESOLVED_ISM_VARIABLE_SN_ENERGY (test_sn_energy_variable.cc):
//   PISN: Heger & Woosley 2002 (ApJ 567, 532) via the Wise et al. 2012a fit, E = [5.0 + 1.304 (M_He/Msun - 64)] 1e51 erg with
//   M_He = (13/24)(M_ZAMS - 20) Msun (HW02 eq. 1), He core held to HW02's PISN range 64-133 Msun;
//   ECSN: 5e50 erg, Uli's choice 2026-11-19 (Kitaura, Janka & Hillebrandt 2006, A&A 450, 345, give ~1e50 erg for the 8.8 Msun O-Ne-Mg core).
#include "test_harness.h"
#include "galaxy_sf/resolvedism_sn_energy.h"

TEST_CASE("PISN energy from the helium core (the fit itself)") {
    CHECK_CLOSE(resolvedism_pisn_energy_erg(200.) / 1e51, 5.0 + 1.304 * (97.5 - 64.), 1e-9);
    CHECK_CLOSE(resolvedism_pisn_energy_erg(140.) / 1e51, 5.0 + 1.304 * (65. - 64.), 1e-9);
    CHECK_CLOSE(resolvedism_pisn_energy_erg(20. + 64. * 24. / 13.) / 1e51, 5.0, 1e-9);     // lower edge of the HW02 range
    CHECK_CLOSE(resolvedism_pisn_energy_erg(130.) / 1e51, 5.0, 1e-9);                      // table PISN below the range: held at M_He = 64
    CHECK_CLOSE(resolvedism_pisn_energy_erg(20. + 133. * 24. / 13.) / 9.35e52, 1.0, 0.02); // HW02 maximum, 9.35e52 for the 133 Msun core
    CHECK_CLOSE(resolvedism_pisn_energy_erg(400.), resolvedism_pisn_energy_erg(20. + 133. * 24. / 13.), 1e-9 * 1e52);
    double prev = 0;
    for(double M = 100; M <= 500; M += 0.5) {double E = resolvedism_pisn_energy_erg(M); CHECK(E >= prev && E >= 5e51 && E <= 9.5e52); prev = E;}
}

TEST_CASE("energy by death type") {
    CHECK(resolvedism_sn_energy_erg(0, 0, 1, 15.) == 1e51);  // CCSN, PPISN
    CHECK(resolvedism_sn_energy_erg(0, 0, 0, 40.) == 0);     // failed SN, direct collapse
#ifdef RESOLVED_ISM_VARIABLE_SN_ENERGY
    CHECK(resolvedism_sn_energy_erg(0, 1, 1, 8.5) == 5e50);  // ECSN
    CHECK(resolvedism_sn_energy_erg(1, 0, 1, 200.) == resolvedism_pisn_energy_erg(200.)); // PISN: He core
#else
    CHECK(resolvedism_sn_energy_erg(0, 1, 1, 8.5) == 1e51);  // ECSN: fixed
    CHECK(resolvedism_sn_energy_erg(1, 0, 1, 200.) == 1e51); // PISN: fixed
#endif
}

TEST_MAIN()
