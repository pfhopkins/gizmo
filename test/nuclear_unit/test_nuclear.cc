/* Standalone unit test for the nuclear reaction network solver.
   Tests the aprox13 alpha-chain solver, NSE composition, Ye/Abar computation,
   and energy conservation without running a full GIZMO simulation.

   Compiled by the pytest harness (test_nuclear_unit.py) which creates
   GIZMO_config.h with the right flags and compiles against the header-only
   nuclear_physics_functions.h using the full GIZMO header infrastructure.
*/

#include "../../src/nuclear/nuclear_physics_functions.h"
#include <cstdio>
#include <cmath>
#include <cstring>

/* provide the global All struct and ThisTask that nuclear_physics.cc references via allvars.h */
struct global_data_all_processes All;
int ThisTask = 0;

static int npass = 0, nfail = 0;

static void check(const char *name, int condition) {
    if (condition) { npass++; }
    else { nfail++; printf("  FAIL: %s\n", name); }
}

int main()
{
    /* set up minimal global state: CGS unit system (all unit conversions = 1) */
    memset(&All, 0, sizeof(All));
    All.UnitLength_in_cm = 1.0;
    All.UnitMass_in_g = 1.0;
    All.UnitVelocity_in_cm_per_s = 1.0;
    All.HubbleParam = 1.0;
    All.cf_a3inv = 1.0;
    All.cf_atime = 1.0;
    All.MinEgySpec = 0;
    All.NuclearBurningFloor_T = 1.0e8;
    All.NuclearNSE_T_threshold = 6.0e9;

    printf("=== Nuclear Network Unit Test ===\n\n");

    /* ---- Test 1: Ye/Abar computation ---- */
    printf("--- Test 1: Ye/Abar from composition ---\n");
    {
        double X[NUM_NUCLEAR_SPECIES] = {};
        double Ye, Abar;

        /* pure He4: Ye=0.5, Abar=4 */
        memset(X, 0, sizeof(X));
        X[NUCLEAR_HE4] = 1.0;
        nuclear_compute_ye_abar(X, &Ye, &Abar);
        printf("  Pure He4: Ye=%.6f Abar=%.6f\n", Ye, Abar);
        check("He4 Ye", fabs(Ye - 0.5) < 1e-10);
        check("He4 Abar", fabs(Abar - 4.0) < 1e-10);

        /* pure Ni56: Ye=0.5, Abar=56 */
        memset(X, 0, sizeof(X));
        X[NUCLEAR_NI56] = 1.0;
        nuclear_compute_ye_abar(X, &Ye, &Abar);
        printf("  Pure Ni56: Ye=%.6f Abar=%.6f\n", Ye, Abar);
        check("Ni56 Ye", fabs(Ye - 0.5) < 1e-10);
        check("Ni56 Abar", fabs(Abar - 56.0) < 1e-10);

        /* 50/50 C12/O16 */
        memset(X, 0, sizeof(X));
        X[NUCLEAR_C12] = 0.5; X[NUCLEAR_O16] = 0.5;
        nuclear_compute_ye_abar(X, &Ye, &Abar);
        double Abar_expected = 1.0 / (0.5/12.0 + 0.5/16.0);
        printf("  50/50 CO: Ye=%.6f Abar=%.6f (expect %.6f)\n", Ye, Abar, Abar_expected);
        check("CO Ye", fabs(Ye - 0.5) < 1e-10);
        check("CO Abar", fabs(Abar - Abar_expected) < 1e-10);
    }

    /* ---- Test 2: Triple-alpha burning (He4 -> C12) ---- */
    printf("\n--- Test 2: Triple-alpha burning ---\n");
    {
        struct nuclear_input in = {};
        in.rho = 1.0e7;  /* g/cc */
        in.T   = 2.0e9;  /* 2 GK */
        in.Ye  = 0.5;
        in.dt  = 1.0;    /* 1 second */
        in.X[NUCLEAR_HE4] = 1.0;

        struct nuclear_output out;
        int ierr = nuclear_aprox13_solve(&in, &out);
        printf("  X(He4)=%.6f  X(C12)=%.6f  X(O16)=%.6f\n",
               out.X[NUCLEAR_HE4], out.X[NUCLEAR_C12], out.X[NUCLEAR_O16]);
        printf("  de=%.4e erg/g  Ye=%.6f  Abar=%.6f\n", out.de, out.Ye, out.Abar);

        check("converged", ierr == 0);
        check("He4 decreased", out.X[NUCLEAR_HE4] < 1.0);
        check("C12 produced", out.X[NUCLEAR_C12] > 0.0);
        check("energy released > 0", out.de > 0.0);
        check("Ye = 0.5", fabs(out.Ye - 0.5) < 1e-6);
        double sum = 0; for (int k = 0; k < NUM_NUCLEAR_SPECIES; k++) sum += out.X[k];
        check("sum(X) = 1", fabs(sum - 1.0) < 1e-10);
    }

    /* ---- Test 3: Alpha-chain burning at detonation conditions ---- */
    /* Note: aprox13 is an alpha-chain — it burns via alpha captures, not C+C or C+O
       fusion. So we need He4 present as fuel. A realistic WD composition after
       some initial C+C burning would have He4 available. */
    printf("\n--- Test 3: Alpha-chain burning at WD conditions ---\n");
    {
        struct nuclear_input in = {};
        in.rho = 2.0e9;  in.T = 4.0e9;  in.Ye = 0.5;  in.dt = 0.01;
        in.X[NUCLEAR_HE4] = 0.2; in.X[NUCLEAR_C12] = 0.4; in.X[NUCLEAR_O16] = 0.4;

        struct nuclear_output out;
        int ierr = nuclear_aprox13_solve(&in, &out);
        printf("  X(C12)=%.6f  X(O16)=%.6f  X(Si28)=%.6f  X(Ni56)=%.6f\n",
               out.X[NUCLEAR_C12], out.X[NUCLEAR_O16], out.X[NUCLEAR_SI28], out.X[NUCLEAR_NI56]);
        printf("  de=%.4e erg/g\n", out.de);

        check("converged", ierr == 0);
        check("He4 consumed", out.X[NUCLEAR_HE4] < 0.2);
        check("energy released > 0", out.de > 0.0);
        check("composition changed", fabs(out.X[NUCLEAR_C12] - 0.4) > 0.01 ||
              fabs(out.X[NUCLEAR_O16] - 0.4) > 0.01);
    }

    /* ---- Test 4: NSE composition ---- */
    printf("\n--- Test 4: NSE composition ---\n");
    {
        double X[NUM_NUCLEAR_SPECIES];
        nuclear_nse_composition(1.0e9, 8.0, 0.5, X);
        double iron_group = X[NUCLEAR_FE52] + X[NUCLEAR_NI56];
        printf("  T9=8, rho=1e9: X(He4)=%.4f X(Fe52)=%.4f X(Ni56)=%.4f\n",
               X[NUCLEAR_HE4], X[NUCLEAR_FE52], X[NUCLEAR_NI56]);
        check("NSE at 8GK: iron-group > 80%%", iron_group > 0.8);

        nuclear_nse_composition(1.0e9, 10.0, 0.5, X);
        printf("  T9=10, rho=1e9: X(Ni56)=%.4f\n", X[NUCLEAR_NI56]);
        check("NSE at 10GK: Ni56 > 90%%", X[NUCLEAR_NI56] > 0.9);

        double sum = 0; for (int k = 0; k < NUM_NUCLEAR_SPECIES; k++) sum += X[k];
        check("NSE sum(X) = 1", fabs(sum - 1.0) < 1e-10);
    }

    /* ---- Test 5: Cold gas — no burning ---- */
    printf("\n--- Test 5: Cold gas (no burning) ---\n");
    {
        struct nuclear_input in = {};
        in.rho = 1.0e6; in.T = 1.0e7; in.Ye = 0.5; in.dt = 1.0;
        in.X[NUCLEAR_C12] = 0.5; in.X[NUCLEAR_O16] = 0.5;

        struct nuclear_output out;
        nuclear_aprox13_solve(&in, &out);
        check("cold: composition unchanged", fabs(out.X[NUCLEAR_C12] - 0.5) < 1e-6);
        check("cold: no energy", fabs(out.de) < 1e-10);
    }

    /* ---- Test 6: Energy sign and magnitude ---- */
    printf("\n--- Test 6: Energy check ---\n");
    {
        /* He4 -> heavier: should release energy (positive de, since heavier nuclei are more bound) */
        struct nuclear_input in = {};
        in.rho = 1.0e8; in.T = 3.0e9; in.Ye = 0.5; in.dt = 0.1;
        in.X[NUCLEAR_HE4] = 1.0;

        struct nuclear_output out;
        nuclear_aprox13_solve(&in, &out);
        printf("  He4 burning at 3GK: de=%.4e erg/g (should be > 0)\n", out.de);
        check("exothermic (de > 0)", out.de > 0.0);
        /* Q-value for He4->Ni56 is ~8.6 MeV/nucleon - 7.07 MeV/nucleon ~ 1.6 MeV/nucleon
           = 1.6e-6 * 6e23 / 4 ~ 2.4e17 erg/g maximum. de should be less than this. */
        /* He4->C12: Q ~ 0.61 MeV/nucleon * N_A ~ 5.8e17 erg/g; He4->Ni56 ~ 1.5e18 erg/g max */
        check("energy magnitude reasonable", out.de < 2.0e18);
    }

    printf("\n========================================\n");
    printf("  PASSED: %d    FAILED: %d\n", npass, nfail);
    printf("========================================\n");

    return (nfail > 0) ? 1 : 0;
}
