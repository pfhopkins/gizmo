// Unit test: supernova explosion energies by death type (galaxy_sf/resolvedism_sn_energy.h).
// Default: every explosive death (CCSN, ECSN, PPISN, PISN) 1e51 erg; failed SNe and direct collapse 0.
// With RESOLVED_ISM_VARIABLE_SN_ENERGY (test_sn_energy_variable.cc), Uli's choice 2026-11-19: PISN 1e52 erg, ECSN 5e50 erg,
// CCSN and PPISN 1e51 erg. For reference: the He-core fit of Heger & Woosley 2002 via Wise+12a gives 4.87e52 erg at 200 Msun,
// Kitaura, Janka & Hillebrandt 2006 (A&A 450, 345) ~1e50 erg for the 8.8 Msun O-Ne-Mg core.
#include "test_harness.h"
#include "galaxy_sf/resolvedism_sn_energy.h"

TEST_CASE("energy by death type") {
    CHECK(resolvedism_sn_energy_erg(0, 0, 1) == 1e51);  // CCSN, PPISN
    CHECK(resolvedism_sn_energy_erg(0, 0, 0) == 0);     // failed SN, direct collapse
#ifdef RESOLVED_ISM_VARIABLE_SN_ENERGY
    CHECK(resolvedism_sn_energy_erg(0, 1, 1) == 5e50);  // ECSN
    CHECK(resolvedism_sn_energy_erg(1, 0, 1) == 1e52);  // PISN
#else
    CHECK(resolvedism_sn_energy_erg(0, 1, 1) == 1e51);  // ECSN: fixed
    CHECK(resolvedism_sn_energy_erg(1, 0, 1) == 1e51);  // PISN: fixed
#endif
}

TEST_MAIN()
