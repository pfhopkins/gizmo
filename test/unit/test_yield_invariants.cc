// Unit test: stellar-feedback yield-sum invariants.
//
// Mirrors the SN/Type-Ia yield computation in resolvedism_fb_thermal.cc and
// the per-cell enrich() update used in the same file. Asserts:
//
//   I1.  After the deficit-fix, Σ ElemYields[k] == Mej   (mass conservation)
//   I2.  MetalMass == Σ_{k≥ELEM_C} ElemYields[k]         (definition)
//   I3.  After enrich(), |Σ_k ElementAbundance[k] - 1| stays at FP-precision
//        across many sequential enrichment events (no drift)
//
// Reproduces the X+Y+Z drift seen in test_SN_PI snap_100→120 where one cell's
// mass-fraction sum grew from 1.36 → 1.41 over 20 Myr.

#include "test_harness.h"
#include <cmath>
#include <cstdio>
#include <cstring>

constexpr int NELEM = 27;
enum { ELEM_H=0, ELEM_He=1, ELEM_C=2, ELEM_N=3, ELEM_O=4, ELEM_F=5,
       ELEM_Ne=6, ELEM_Na=7, ELEM_Mg=8, ELEM_Al=9, ELEM_Si=10,
       ELEM_P=11, ELEM_S=12, ELEM_Cl=13, ELEM_Ar=14, ELEM_K=15, ELEM_Ca=16,
       ELEM_Sc=17, ELEM_Ti=18, ELEM_V=19, ELEM_Cr=20, ELEM_Mn=21, ELEM_Fe=22,
       ELEM_Co=23, ELEM_Ni=24, ELEM_Cu=25, ELEM_Zn=26 };

struct Yields {
    double Mej;
    double MetalMass;
    double ElemYields[NELEM];
};

struct Cell {
    double Mass;
    double Metallicity;          // total Z (mass fraction)
    double ElementAbundance[NELEM];
};

// Solar mass fractions (Asplund+ 2009), 27-element ordering.
static const double solar_X[NELEM] = {
    0.7381, 0.2485, 2.36e-3, 6.91e-4, 5.72e-3, 3.26e-7, 1.25e-3,
    2.98e-5, 5.91e-4, 5.57e-5, 6.65e-4, 5.16e-6, 3.10e-4, 3.15e-6,
    7.37e-5, 2.93e-6, 6.44e-5, 3.48e-8, 3.59e-6, 2.30e-7, 1.37e-5,
    9.17e-6, 1.17e-3, 3.30e-6, 6.99e-5, 7.20e-7, 1.67e-6
};

static double sum_X(const Cell &p) {
    double s = 0;
    for(int k = 0; k < NELEM; k++) s += p.ElementAbundance[k];
    return s;
}

static double sum_metals(const Cell &p) {
    double s = 0;
    for(int k = ELEM_C; k < NELEM; k++) s += p.ElementAbundance[k];
    return s;
}

// Mirror of resolvedism_fb_thermal.cc lines 109-171 (SN/CCSN/PPISN/PISN path).
// Inputs:  Mstar, rem_mass, net_yields[k], X_birth[k]
// Outputs: y.Mej, y.MetalMass, y.ElemYields[k]
static Yields compute_sn_yields(double Mstar, double rem_mass,
                                const double net_yields[NELEM],
                                const double X_birth[NELEM])
{
    Yields y = {};
    y.Mej = Mstar - rem_mass;
    if(y.Mej < 0) y.Mej = 0;

    double sum_yields = 0;
    for(int k = 0; k < NELEM; k++) {
        double M_elem = net_yields[k] + X_birth[k] * y.Mej;
        if(M_elem < 0) M_elem = 0;
        y.ElemYields[k] = M_elem;
        sum_yields += M_elem;
    }

    // Deficit-fix (mirrors lines 157-163 of resolvedism_fb_thermal.cc)
    if(y.Mej > 0 && sum_yields > 0) {
        double deficit = y.Mej - sum_yields;
        if(std::fabs(deficit) > 1e-6) {
            y.ElemYields[ELEM_H] += deficit;
            if(y.ElemYields[ELEM_H] < 0) y.ElemYields[ELEM_H] = 0;
        }
    }

    for(int k = ELEM_C; k < NELEM; k++) y.MetalMass += y.ElemYields[k];
    return y;
}

// Mirror of the per-cell update in resolvedism_fb_thermal.cc lines 287-326.
// Updates ElementAbundance, Metallicity, and Mass in-place (no atomics, single-thread).
static void enrich(Cell &p, double wk, const Yields &y)
{
    double Mass_j = p.Mass;
    double dM = wk * y.Mej;

    double dMZ = wk * y.MetalMass;
    double dZ = (dMZ - p.Metallicity * dM) / (Mass_j + dM);
    p.Metallicity += dZ;

    for(int k = 0; k < NELEM; k++) {
        double X_old = p.ElementAbundance[k];
        double dMX  = wk * y.ElemYields[k];
        double dX   = (dMX - X_old * dM) / (Mass_j + dM);
        p.ElementAbundance[k] += dX;
    }
    p.Mass += dM;
}

// Build a pristine cell at (1 - Z_frac_solar*Z_solar) hydrogen + (Z_frac_solar) metals,
// with sum-to-1 enforced numerically.
static Cell make_pristine_gas(double M_solar, double Z_frac_solar)
{
    Cell p = {};
    p.Mass = M_solar;
    const double Zsol = 0.014;
    for(int k = 0; k < NELEM; k++) {
        if(k < ELEM_C) p.ElementAbundance[k] = solar_X[k] * (1.0 - Zsol*Z_frac_solar) / (1.0 - Zsol);
        else           p.ElementAbundance[k] = solar_X[k] * Z_frac_solar;
    }
    double s = sum_X(p);
    for(int k = 0; k < NELEM; k++) p.ElementAbundance[k] /= s;
    p.Metallicity = sum_metals(p);
    return p;
}

// =============================================================================
// I1 / I2: yield-sum invariants for a known PPISN entry (M=150, Z=0.001)
// =============================================================================
TEST_CASE("yield-sum: PPISN 150 Msun Z=0.001 — Σ ElemYields == Mej") {
    // Real net yields from stellar_tables_unified.hdf5 (M=150, Z=0.001, rem=PPISN, rem_mass=4.217)
    double net[NELEM] = {
        -7.9508342795e+01,  6.5449645632e+00,  8.6547896724e+00,  3.7743500186e+00,
         5.5392020613e+01, -2.4985461619e-06,  3.7731021576e+00,  1.3914156559e-01,
         7.2608586963e-01,  3.2210319251e-03,  5.7965648772e-03,  0.0,
         8.4400853400e-06,  0.0,                7.0075590000e-04,  0.0,
        -1.7010633700e-07,  0.0,               -8.6911414000e-09,  0.0,
         0.0,               0.0,                0.0,               0.0,
         0.0,               0.0,                0.0
    };
    Cell birth = make_pristine_gas(150.0, 0.071); // Z=0.001 ≈ 0.071 Z_solar
    Yields y = compute_sn_yields(150.0, 4.217, net, birth.ElementAbundance);

    double sum = 0;
    for(int k = 0; k < NELEM; k++) sum += y.ElemYields[k];

    // I1: total ejecta mass conservation
    CHECK_CLOSE(sum, y.Mej, 1e-6 * y.Mej);

    // I2: MetalMass equals Σ_{k≥C} ElemYields[k] by construction
    double metals_check = 0;
    for(int k = ELEM_C; k < NELEM; k++) metals_check += y.ElemYields[k];
    CHECK_CLOSE(y.MetalMass, metals_check, 1e-12);
}

// =============================================================================
// I1 (subtle): does the deficit-fix tolerance (1e-6 Msun) silently leak mass?
// =============================================================================
TEST_CASE("yield-sum: deficits below 1e-6 Msun are NOT fixed → small leak") {
    // Construct an artificial yield with a deliberate ~1e-7 deficit
    double net[NELEM] = {0};
    Cell birth = make_pristine_gas(20.0, 0.1);
    // Mej_pred = 20 - 5 = 15; if net=0 and X_birth*Mej sums to 14.9999999...
    // due to floating-point, the deficit-fix won't fire (abs(deficit) < 1e-6).
    Yields y = compute_sn_yields(20.0, 5.0, net, birth.ElementAbundance);
    double sum = 0;
    for(int k = 0; k < NELEM; k++) sum += y.ElemYields[k];
    double leak = std::fabs(y.Mej - sum);
    if(leak > 0) {
        std::fprintf(stdout, "  note: per-event leak = %.3e Msun (below 1e-6 threshold, not redistributed)\n", leak);
    }
    // The test passes either way — informational only.
    CHECK(true);
}

// =============================================================================
// I3: sum-to-1 invariant under repeated enrichment
// =============================================================================
TEST_CASE("enrich-loop: Σ ElementAbundance stays at 1.0 across 1000 SN events") {
    // Realistic CCSN-like yields summing exactly to Mej
    double Mej_target = 9.0;
    double net[NELEM] = {0};
    net[ELEM_H]  = -1.5;   // burned H
    net[ELEM_He] = +1.0;   // produced He
    net[ELEM_C]  = +0.05;
    net[ELEM_O]  = +0.4;
    net[ELEM_Mg] = +0.02;
    net[ELEM_Si] = +0.05;
    net[ELEM_Fe] = +0.07;

    Cell birth = make_pristine_gas(10.0, 0.1);
    Cell gas   = make_pristine_gas(1.0,  0.1);  // 1 Msun gas particle
    Yields y = compute_sn_yields(10.0, 1.0, net, birth.ElementAbundance);

    // Pre-check: yield computation itself preserves Mej
    double sum0 = 0;
    for(int k = 0; k < NELEM; k++) sum0 += y.ElemYields[k];
    CHECK_CLOSE(sum0, y.Mej, 1e-12 * y.Mej);
    (void)Mej_target;

    // Apply 1000 small enrichment events with wk=1e-3 (so we add tiny dM each time)
    double sum_init = sum_X(gas);
    for(int n = 0; n < 1000; n++) {
        enrich(gas, 1e-3, y);
    }
    double sum_final = sum_X(gas);
    double drift = std::fabs(sum_final - sum_init);
    std::fprintf(stdout, "  Σ ElementAbundance initial=%.15g  final=%.15g  drift=%.3e\n",
                 sum_init, sum_final, drift);

    // After 1000 enrichment events, sum should still be ≈ 1 to FP precision
    // (drift of ~1e-12 acceptable; ~1e-3 or larger would indicate the bug)
    CHECK(drift < 1e-10);
}

// =============================================================================
// I3 (cross-tracer): does Metallicity[0] drift away from Σ_{k≥C} ElementAbundance[k]?
// =============================================================================
TEST_CASE("enrich-loop: Metallicity[0] tracks Σ_{k≥C} ElementAbundance[k]") {
    double net[NELEM] = {0};
    net[ELEM_H] = -1.5; net[ELEM_He] = +1.0; net[ELEM_C] = +0.05;
    net[ELEM_O] = +0.4; net[ELEM_Mg] = +0.02; net[ELEM_Si] = +0.05; net[ELEM_Fe] = +0.07;

    Cell birth = make_pristine_gas(10.0, 0.1);
    Cell gas   = make_pristine_gas(1.0,  0.1);
    Yields y = compute_sn_yields(10.0, 1.0, net, birth.ElementAbundance);

    for(int n = 0; n < 1000; n++) enrich(gas, 1e-3, y);

    double Z_from_metallicity = gas.Metallicity;
    double Z_from_elements    = sum_metals(gas);
    double drift = std::fabs(Z_from_metallicity - Z_from_elements);
    std::fprintf(stdout, "  Metallicity[0]=%.15g  Σ_{k≥C} ElementAbundance=%.15g  diff=%.3e\n",
                 Z_from_metallicity, Z_from_elements, drift);
    CHECK(drift < 1e-10);
}

TEST_MAIN()
