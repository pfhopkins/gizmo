// Unit test: SGCHEM_PRIMORDIAL_HELIUM sets He/H = 0.079 by number in the chemistry (ABHE) and the matching hydrogen
// mass fraction for the gas (HYDROGEN_MASSFRAC = 0.76, Y = 0.24 in pristine cells), as the primordial network 1 does.
#define SGCHEM
#define CHEMISTRYNETWORK 17
#define SGCHEM_PRIMORDIAL_HELIUM
#include <cmath>
#include <cstdio>
#include "test_harness.h"
#include "src/declarations/constants.h"
#include "src/cooling/sgchem/sgchem_consts.h"

TEST_CASE("net 17 with primordial helium") {
    CHECK_CLOSE((double)ABHE, 0.079, 1e-12);
    CHECK_CLOSE((double)HYDROGEN_MASSFRAC, 0.76, 1e-12);
    double X_chem = 1.0 / (1.0 + 4.0 * ABHE); // the chemistry's own X_H
    std::printf("  X_H: gas %.5f, chemistry %.5f; Y = %.4f\n", (double)HYDROGEN_MASSFRAC, X_chem, 1.0 - HYDROGEN_MASSFRAC);
    CHECK(std::fabs(X_chem / HYDROGEN_MASSFRAC - 1) < 1e-3);
}

TEST_MAIN()
