// Unit test: PPISN enrichment for a 150 Msun star at Z=0.001
//
// Reproduces the crash at t=0.0369 in the Stella test_SN run (job 150573).
// A 150 Msun star classified as PPISN (rem_type=4) with remnant mass 4.2 Msun
// ejects 145.8 Msun. The table yields produce ejecta with Z ~ 50%, which when
// injected into ~4 Msun gas cells gives abundc ~ 0.01, abundo ~ 0.05 —
// metallicities far above what is physical for a 0.1-solar progenitor.
//
// The test checks:
//   1. What ejecta metallicity the yields produce
//   2. What post-enrichment abundc/abundo a neighbor cell gets
//   3. Whether the remnant mass (4.2 Msun) is sensible for a PPISN

#include "test_harness.h"
#include <cmath>
#include <cstdio>

static const int NELEM = 27;

// Element indices (matching resolvedism_stellar_tables.h)
enum { ELEM_H=0, ELEM_He=1, ELEM_C=2, ELEM_N=3, ELEM_O=4, ELEM_F=5,
       ELEM_Ne=6, ELEM_Na=7, ELEM_Mg=8, ELEM_Al=9, ELEM_Si=10 };

static const char *elem_names[NELEM] = {
    "H","He","C","N","O","F","Ne","Na","Mg","Al","Si","P","S","Cl",
    "Ar","K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn"
};

// Solar mass fractions (Asplund+ 2009)
static const double solar_X[NELEM] = {
    0.7381, 0.2485, 2.36e-3, 6.91e-4, 5.72e-3, 3.26e-7, 1.25e-3,
    2.98e-5, 5.91e-4, 5.57e-5, 6.65e-4, 5.16e-6, 3.10e-4, 3.15e-6,
    7.37e-5, 2.93e-6, 6.44e-5, 3.48e-8, 3.59e-6, 2.30e-7, 1.37e-5,
    9.17e-6, 1.17e-3, 3.30e-6, 6.99e-5, 7.20e-7, 1.67e-6
};

struct GasParticle {
    double Mass;
    double Metallicity;
    double ElementAbundance[NELEM];
};

static GasParticle make_gas(double M_solar, double Z_frac_solar)
{
    GasParticle p;
    p.Mass = M_solar;
    p.Metallicity = 0;
    double Z_solar = 0.014;
    for(int k = 0; k < NELEM; k++) {
        if(k < 2)
            p.ElementAbundance[k] = solar_X[k] * (1.0 - Z_solar * Z_frac_solar) / (1.0 - Z_solar);
        else
            p.ElementAbundance[k] = solar_X[k] * Z_frac_solar;
        if(k >= 2) p.Metallicity += p.ElementAbundance[k];
    }
    double s = 0;
    for(int k = 0; k < NELEM; k++) s += p.ElementAbundance[k];
    for(int k = 0; k < NELEM; k++) p.ElementAbundance[k] /= s;
    p.Metallicity /= s;
    return p;
}

static double sum_X(const GasParticle &p) {
    double s = 0;
    for(int k = 0; k < NELEM; k++) s += p.ElementAbundance[k];
    return s;
}

// Yield computation from resolvedism_fb_thermal.cc lines 109-135
static void compute_yields(const double net_yields[NELEM],
                           const double X_birth[NELEM],
                           double Mej,
                           double ElemYields[NELEM],
                           double *MetalMass_out)
{
    double metal_mass = 0;
    for(int k = 0; k < NELEM; k++) {
        double M_elem_ej = net_yields[k] + X_birth[k] * Mej;
        if(M_elem_ej < 0) M_elem_ej = 0;
        ElemYields[k] = M_elem_ej;
        if(k >= 2) metal_mass += M_elem_ej;
    }
    *MetalMass_out = metal_mass;
}

// Enrichment update from resolvedism_fb_thermal.cc
static void enrich(GasParticle &p, double wk, double Mej,
                   double MetalMass, const double ElemYields[NELEM])
{
    double Mass_j = p.Mass;
    double dM = wk * Mej;

    double dMZ = wk * MetalMass;
    double dZ = (dMZ - p.Metallicity * dM) / (Mass_j + dM);
    p.Metallicity += dZ;

    for(int k = 0; k < NELEM; k++) {
        double X_old = p.ElementAbundance[k];
        double dMX = wk * ElemYields[k];
        double dX = (dMX - X_old * dM) / (Mass_j + dM);
        p.ElementAbundance[k] += dX;
    }

    p.Mass += dM;
}


// ---- Actual net yields from stellar_tables_unified.hdf5 ----
// M=150 Msun, Z=0.001, rem_type=4 (PPISN), rem_mass=4.217 Msun
static const double ppisn_net_yields[NELEM] = {
    -7.9508342795e+01,  // H
    +6.5449645632e+00,  // He
    +8.6547896724e+00,  // C
    +3.7743500186e+00,  // N
    +5.5392020613e+01,  // O
    -2.4985461619e-06,  // F
    +3.7731021576e+00,  // Ne
    +1.3914156559e-01,  // Na
    +7.2608586963e-01,  // Mg
    +3.2210319251e-03,  // Al
    +5.7965648772e-03,  // Si
    +0.0000000000e+00,  // P
    +8.4400853400e-06,  // S
    +0.0000000000e+00,  // Cl
    +7.0075590000e-04,  // Ar
    +0.0000000000e+00,  // K
    -1.7010633700e-07,  // Ca
    +0.0000000000e+00,  // Sc
    -8.6911414000e-09,  // Ti
    +0.0000000000e+00,  // V
    +1.6988240000e-04,  // Cr
    +0.0000000000e+00,  // Mn
    +7.2549666767e-04,  // Fe
    +0.0000000000e+00,  // Co
    +7.2902530000e-04,  // Ni
    +0.0000000000e+00,  // Cu
    +1.8181200000e-05   // Zn
};

static const double PPISN_MSTAR    = 150.0;
static const double PPISN_REM_MASS = 4.21663;
static const double PPISN_MEJ      = PPISN_MSTAR - PPISN_REM_MASS; // 145.78 Msun


TEST_CASE("PPISN 150Msun: ejecta composition") {
    // Birth abundances at 0.1 solar
    double X_birth[NELEM];
    for(int k = 0; k < NELEM; k++)
        X_birth[k] = (k < 2) ? solar_X[k] : solar_X[k] * 0.1;

    double ElemYields[NELEM];
    double MetalMass;
    compute_yields(ppisn_net_yields, X_birth, PPISN_MEJ, ElemYields, &MetalMass);

    double yield_sum = 0;
    for(int k = 0; k < NELEM; k++) yield_sum += ElemYields[k];

    // Ejecta composition (mass fractions in the ejecta)
    double Z_ej    = MetalMass / yield_sum;

    std::printf("\n");
    std::printf("    === PPISN 150 Msun at Z=0.001 (0.07 solar) ===\n");
    std::printf("    Remnant mass:  %.2f Msun (rem_type=4 PPISN)\n", PPISN_REM_MASS);
    std::printf("    Ejecta mass:   %.2f Msun\n", PPISN_MEJ);
    std::printf("    sum(ElemYields)=%.2f vs Mej=%.2f (deficit=%.2f Msun)\n",
                yield_sum, PPISN_MEJ, PPISN_MEJ - yield_sum);
    std::printf("\n");
    std::printf("    Ejecta composition:\n");
    for(int k = 0; k < NELEM; k++) {
        if(ElemYields[k] > 1e-6)
            std::printf("      %-4s: %10.4f Msun  (X_ej = %.4e)\n",
                        elem_names[k], ElemYields[k], ElemYields[k] / yield_sum);
    }
    std::printf("    Ejecta Z = %.4f (%.1f x solar)\n", Z_ej, Z_ej / 0.014);
    std::printf("\n");

    // --- DIAGNOSTIC: is this sensible for a PPISN? ---
    // PPISN should leave a massive BH (40-90 Msun), not 4.2 Msun
    std::printf("    SANITY CHECKS:\n");
    std::printf("    Remnant fraction: %.1f%% of initial mass\n",
                PPISN_REM_MASS / PPISN_MSTAR * 100);
    std::printf("    Ejecta Z = %.1f x solar — physical for 0.1 solar progenitor? %s\n",
                Z_ej / 0.014, Z_ej > 0.1 ? "NO" : "yes");

    // A PPISN remnant should be >> 4 Msun. Typical: 40-90 Msun BH.
    CHECK(PPISN_REM_MASS > 30.0); // EXPECTED TO FAIL — demonstrates the bug

    // Ejecta metallicity should not be > 10x solar for a 0.1 solar progenitor
    CHECK(Z_ej < 0.14); // 10x solar — EXPECTED TO FAIL
}


TEST_CASE("PPISN 150Msun: post-enrichment neighbor abundances") {
    // Simulate what happens to a 4 Msun gas cell (1 of 32 neighbors)
    // when the PPISN ejecta arrives

    double X_birth[NELEM];
    for(int k = 0; k < NELEM; k++)
        X_birth[k] = (k < 2) ? solar_X[k] : solar_X[k] * 0.1;

    double ElemYields[NELEM];
    double MetalMass;
    compute_yields(ppisn_net_yields, X_birth, PPISN_MEJ, ElemYields, &MetalMass);

    GasParticle p = make_gas(4.0, 0.1);  // 4 Msun cell at 0.1 solar
    double wk = 1.0 / 32.0;  // 1 of 32 neighbors

    std::printf("\n");
    std::printf("    === Post-enrichment: 4 Msun cell, wk=1/32 ===\n");
    std::printf("    Before: X_H=%.4f X_C=%.4e X_O=%.4e Z=%.4e\n",
                p.ElementAbundance[ELEM_H], p.ElementAbundance[ELEM_C],
                p.ElementAbundance[ELEM_O], p.Metallicity);

    enrich(p, wk, PPISN_MEJ, MetalMass, ElemYields);

    double X_H = p.ElementAbundance[ELEM_H];
    double X_C = p.ElementAbundance[ELEM_C];
    double X_O = p.ElementAbundance[ELEM_O];
    double abundc = (X_C / 12.0) / X_H;
    double abundo = (X_O / 16.0) / X_H;

    std::printf("    After:  X_H=%.4f X_C=%.4e X_O=%.4e Z=%.4e\n",
                X_H, X_C, X_O, p.Metallicity);
    std::printf("    abundc = %.4e  (crash value was 1.25e-2)\n", abundc);
    std::printf("    abundo = %.4e  (crash value was 5.41e-2)\n", abundo);
    std::printf("    Z = %.4f = %.1f x solar\n", p.Metallicity, p.Metallicity / 0.014);
    std::printf("    sum(X) = %.10f\n", sum_X(p));
    std::printf("    dM received = %.4f Msun\n", PPISN_MEJ * wk);
    std::printf("    New cell mass = %.4f Msun\n", p.Mass);
    std::printf("\n");

    // Compare with crash values from stella.out.150573
    // abundc = 1.2524709458929627E-002
    // abundo = 5.4085914719215165E-002
    std::printf("    Match with crash particle 3095686:\n");
    std::printf("      abundc: test=%.4e  crash=1.2525e-2  ratio=%.3f\n",
                abundc, abundc / 1.2525e-2);
    std::printf("      abundo: test=%.4e  crash=5.4086e-2  ratio=%.3f\n",
                abundo, abundo / 5.4086e-2);

    // Post-enrichment metallicity must be physical
    CHECK(p.Metallicity < 0.14);  // < 10x solar — EXPECTED TO FAIL
    CHECK(X_H > 0.5);  // H should still dominate
    for(int k = 0; k < NELEM; k++) CHECK(p.ElementAbundance[k] >= 0);
}


TEST_CASE("PPISN 150Msun: remnant mass comparison across types") {
    // Print what a sensible PPISN remnant should look like
    std::printf("\n");
    std::printf("    === Remnant mass sanity ===\n");
    std::printf("    Current table:  M_rem = %.2f Msun for 150 Msun PPISN\n", PPISN_REM_MASS);
    std::printf("    Expected PPISN: M_rem ~ 40-90 Msun (massive BH after pulsations)\n");
    std::printf("    This looks like: PISN (complete disruption) or wrong table entry\n");
    std::printf("\n");

    // With a proper PPISN remnant of e.g. 60 Msun:
    double proper_rem = 60.0;
    double proper_Mej = PPISN_MSTAR - proper_rem;  // 90 Msun

    double X_birth[NELEM];
    for(int k = 0; k < NELEM; k++)
        X_birth[k] = (k < 2) ? solar_X[k] : solar_X[k] * 0.1;

    double ElemYields[NELEM];
    double MetalMass;
    // Use same net yields but with reduced ejecta
    compute_yields(ppisn_net_yields, X_birth, proper_Mej, ElemYields, &MetalMass);

    double yield_sum = 0;
    for(int k = 0; k < NELEM; k++) yield_sum += ElemYields[k];
    double Z_ej = MetalMass / yield_sum;

    GasParticle p = make_gas(4.0, 0.1);
    enrich(p, 1.0/32.0, proper_Mej, MetalMass, ElemYields);

    double abundc = (p.ElementAbundance[ELEM_C] / 12.0) / p.ElementAbundance[ELEM_H];
    double abundo = (p.ElementAbundance[ELEM_O] / 16.0) / p.ElementAbundance[ELEM_H];

    std::printf("    With M_rem=60 Msun (proper PPISN):\n");
    std::printf("      Mej = %.1f Msun\n", proper_Mej);
    std::printf("      Ejecta Z = %.4f (%.1f x solar)\n", Z_ej, Z_ej / 0.014);
    std::printf("      Post-enrichment: abundc=%.4e abundo=%.4e Z=%.4f\n",
                abundc, abundo, p.Metallicity);

    // Even with proper remnant, net yields are still huge — but less extreme
    CHECK(Z_ej < 1.0);  // ejecta Z < 100%
}

TEST_MAIN()
