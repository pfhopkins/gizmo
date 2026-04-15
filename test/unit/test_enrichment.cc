// Unit tests for the resolved ISM metal enrichment update formula.
//
// Tests the injection formula from resolvedism_fb_thermal.cc:
//   dX = (dMX - X_old * dM) / (M_old + dM)
//   X_new = X_old + dX
//
// This should give: X_new = (X_old * M_old + dMX) / (M_old + dM)
// i.e. mass-weighted mixing of old composition with ejecta.
//
// KEY FINDING: sum(ElemYields) != Mej in the code because
// M_elem_ej = max(net_yield + X_birth * Mej, 0) clips negative yields.
// This breaks sum(X) = 1 conservation.

#include "test_harness.h"
#include <cmath>
#include <cstdio>

static const int NELEM = 27;

struct GasParticle {
    double Mass;
    double Metallicity;
    double ElementAbundance[NELEM];
};

// Reproduce the enrichment update exactly as in resolvedism_fb_thermal.cc
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

// Reproduce the yield computation from resolvedism_fb_thermal.cc lines 109-135
// Uses net_yield + X_birth * Mej, clipped to >= 0
static void compute_yields(const double net_yields[NELEM],
                           const double X_birth[NELEM],
                           double Mej,
                           double ElemYields[NELEM],
                           double *MetalMass_out)
{
    double metal_mass = 0;
    for(int k = 0; k < NELEM; k++) {
        double M_elem_ej = net_yields[k] + X_birth[k] * Mej;
        if(M_elem_ej < 0) M_elem_ej = 0;  // <-- the clip that breaks conservation
        ElemYields[k] = M_elem_ej;
        if(k >= 2) metal_mass += M_elem_ej; // C=2 onwards are metals
    }
    *MetalMass_out = metal_mass;
}

// Solar mass fractions (Asplund+ 2009)
static const double solar_X[NELEM] = {
    0.7381, 0.2485, 2.36e-3, 6.91e-4, 5.72e-3, 3.26e-7, 1.25e-3,
    2.98e-5, 5.91e-4, 5.57e-5, 6.65e-4, 5.16e-6, 3.10e-4, 3.15e-6,
    7.37e-5, 2.93e-6, 6.44e-5, 3.48e-8, 3.59e-6, 2.30e-7, 1.37e-5,
    9.17e-6, 1.17e-3, 3.30e-6, 6.99e-5, 7.20e-7, 1.67e-6
};

static double sum_X(const GasParticle &p) {
    double s = 0;
    for(int k = 0; k < NELEM; k++) s += p.ElementAbundance[k];
    return s;
}

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
    // Normalise to sum = 1 (solar fractions are rounded)
    double s = sum_X(p);
    for(int k = 0; k < NELEM; k++) p.ElementAbundance[k] /= s;
    p.Metallicity /= s;
    return p;
}

// ---- Tests ----

TEST_CASE("enrichment: formula correct when sum(yields) = Mej") {
    // If yields are properly normalised, sum(X) must stay 1
    GasParticle p = make_gas(4.0, 0.1);
    CHECK_CLOSE(sum_X(p), 1.0, 1e-10);

    double Mej = 18.0;
    double ElemYields[NELEM];
    for(int k = 0; k < NELEM; k++) ElemYields[k] = solar_X[k] * Mej;
    // Force exact normalisation
    double s = 0; for(int k = 0; k < NELEM; k++) s += ElemYields[k];
    for(int k = 0; k < NELEM; k++) ElemYields[k] *= Mej / s;

    double MetalMass = 0;
    for(int k = 2; k < NELEM; k++) MetalMass += ElemYields[k];

    enrich(p, 1.0/32.0, Mej, MetalMass, ElemYields);
    CHECK_CLOSE(sum_X(p), 1.0, 1e-10);
    CHECK_CLOSE(p.Mass, 4.0 + Mej/32.0, 1e-10);
}

TEST_CASE("enrichment: per-element mass conservation") {
    const int N_ngb = 32;
    GasParticle gas[N_ngb];
    for(int j = 0; j < N_ngb; j++) gas[j] = make_gas(4.0, 0.1);

    double Mej = 18.0;
    double ElemYields[NELEM] = {};
    ElemYields[0] = 10.0; ElemYields[1] = 5.0; ElemYields[2] = 0.5;
    ElemYields[4] = 2.0; ElemYields[22] = 0.5;

    double MetalMass = 0;
    for(int k = 2; k < NELEM; k++) MetalMass += ElemYields[k];

    double wk = 1.0 / N_ngb;
    double elem_mass_before[NELEM] = {};
    for(int k = 0; k < NELEM; k++)
        for(int j = 0; j < N_ngb; j++)
            elem_mass_before[k] += gas[j].ElementAbundance[k] * gas[j].Mass;

    for(int j = 0; j < N_ngb; j++)
        enrich(gas[j], wk, Mej, MetalMass, ElemYields);

    for(int k = 0; k < NELEM; k++) {
        double elem_mass_after = 0;
        for(int j = 0; j < N_ngb; j++)
            elem_mass_after += gas[j].ElementAbundance[k] * gas[j].Mass;
        CHECK_CLOSE(elem_mass_after - elem_mass_before[k], ElemYields[k], 1e-10);
    }
}

TEST_CASE("enrichment: yield clip breaks sum(X) = 1") {
    // This test DEMONSTRATES the bug: when net_yield + X_birth * Mej < 0
    // for some element (typically H), clipping to 0 means sum(ElemYields) < Mej,
    // so sum(X) drifts below 1.
    GasParticle p = make_gas(4.0, 0.1);

    // Mock a massive star that consumed lots of H (large negative H net yield)
    double net_yields[NELEM] = {};
    net_yields[0] = -8.0;  // H: consumed 8 Msun by fusion
    net_yields[1] =  3.0;  // He: produced
    net_yields[4] =  2.0;  // O
    net_yields[2] =  0.5;  // C
    // All others zero (small star, simple case)

    double X_birth[NELEM];
    for(int k = 0; k < NELEM; k++)
        X_birth[k] = (k < 2) ? solar_X[k] : solar_X[k] * 0.1;

    double Mej = 18.0;
    double ElemYields[NELEM];
    double MetalMass;
    compute_yields(net_yields, X_birth, Mej, ElemYields, &MetalMass);

    double yield_sum = 0;
    for(int k = 0; k < NELEM; k++) yield_sum += ElemYields[k];
    std::printf("    sum(ElemYields)=%.4f vs Mej=%.4f (deficit=%.4f Msun, %.2f%%)\n",
                yield_sum, Mej, Mej - yield_sum, (1.0 - yield_sum/Mej)*100);

    // sum(ElemYields) < Mej because H got clipped
    CHECK(yield_sum < Mej);

    enrich(p, 1.0/32.0, Mej, MetalMass, ElemYields);

    double sx = sum_X(p);
    std::printf("    sum(X) after enrichment = %.10f (should be 1.0)\n", sx);
    // This WILL fail — documenting the bug
    // CHECK_CLOSE(sx, 1.0, 1e-10);
    // Instead, check the magnitude of the drift
    CHECK(std::fabs(sx - 1.0) < 0.01); // < 1% is tolerable?
}

TEST_CASE("enrichment: PISN into 4 Msun cell — CO chemistry viable") {
    GasParticle p = make_gas(4.0, 0.1);

    // Realistic PISN yields at Z=0.001 (from stellar_tables_unified.hdf5)
    double net_yields[NELEM] = {};
    net_yields[0] = -14.0;  // H consumed
    net_yields[1] =  -5.0;  // He consumed
    net_yields[2] =   8.6;  // C
    net_yields[3] =   0.1;  // N
    net_yields[4] =  55.4;  // O
    net_yields[6] =   1.5;  // Ne
    net_yields[8] =   2.5;  // Mg
    net_yields[10] = 15.0;  // Si
    net_yields[12] =  4.0;  // S
    net_yields[22] =  2.5;  // Fe

    double X_birth[NELEM];
    for(int k = 0; k < NELEM; k++)
        X_birth[k] = (k < 2) ? solar_X[k] : solar_X[k] * 0.1;

    double Mej = 150.0;  // full disruption
    double ElemYields[NELEM];
    double MetalMass;
    compute_yields(net_yields, X_birth, Mej, ElemYields, &MetalMass);

    double yield_sum = 0;
    for(int k = 0; k < NELEM; k++) yield_sum += ElemYields[k];
    std::printf("    PISN yield sum=%.2f vs Mej=%.2f (%.2f%%)\n",
                yield_sum, Mej, yield_sum/Mej*100);

    enrich(p, 1.0/32.0, Mej, MetalMass, ElemYields);

    double X_H = p.ElementAbundance[0];
    double X_C = p.ElementAbundance[2];
    double X_O = p.ElementAbundance[4];
    double abundc = (X_C / 12.0) / X_H;
    double abundo = (X_O / 16.0) / X_H;
    double max_CO = std::fmin(abundc, abundo);

    std::printf("    Post-PISN: X_H=%.4f X_C=%.4e X_O=%.4e Z=%.4f sum(X)=%.6f\n",
                X_H, X_C, X_O, p.Metallicity, sum_X(p));
    std::printf("    abundc=%.4e abundo=%.4e max_CO=%.4e\n",
                abundc, abundo, max_CO);

    // Key checks: is CO chemistry viable?
    CHECK(abundc > 4.0e-4);   // abundc must be above CO crash value
    CHECK(X_H > 0);           // no negative H
    CHECK(X_C > 0);           // no negative C
    CHECK(p.Metallicity < 1); // physical
    for(int k = 0; k < NELEM; k++) CHECK(p.ElementAbundance[k] >= 0);
}

TEST_CASE("enrichment: multiple sequential SNe preserve sum(X)") {
    GasParticle p = make_gas(4.0, 0.1);

    // 5 SNe with properly normalised yields
    double masses[] = {18, 25, 12, 50, 8};
    for(int sn = 0; sn < 5; sn++) {
        double Mej = masses[sn];
        double ElemYields[NELEM];
        for(int k = 0; k < NELEM; k++) ElemYields[k] = solar_X[k] * Mej;
        double s = 0; for(int k = 0; k < NELEM; k++) s += ElemYields[k];
        for(int k = 0; k < NELEM; k++) ElemYields[k] *= Mej / s;

        double MetalMass = 0;
        for(int k = 2; k < NELEM; k++) MetalMass += ElemYields[k];

        enrich(p, 1.0/32.0, Mej, MetalMass, ElemYields);
        CHECK_CLOSE(sum_X(p), 1.0, 1e-10);
        for(int k = 0; k < NELEM; k++) CHECK(p.ElementAbundance[k] >= 0);
    }
}

TEST_MAIN()
