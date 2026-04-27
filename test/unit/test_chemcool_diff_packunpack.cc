// Unit test: CHEMCOOL ↔ TURB_DIFF pack/unpack mass conservation.
//
// In hydro_toplevel.cc:980-1018, when CHEMCOOL+TURB_DIFF_METALS are both on,
// elements with chemistry-locked atoms (H, C, O for net 5) go through a
// special "free + locked" path:
//
//   1. PACK:    locked_X = TracAbund[k] * molwt[k] * X_H_OLD
//               free_X   = ElementAbundance[X] - locked_X
//   2. DIFFUSE: free_X += Dyield_free[X]/Mass
//               trac_mf[k] = TracAbund[k]*molwt[k]*X_H_OLD + Dyield_trac[k]/Mass
//   3. UNPACK:  ElementAbundance[X] = free_X + Σ trac_mf[k] × (X-fraction in molecule)
//               TracAbund[k] = trac_mf[k] / (molwt[k] * X_H_NEW)
//
// He has no species in net 5, so it just diffuses normally — and the snapshot
// data shows He is EXACTLY conserved while X_H drifts upward in chemistry-
// active (compressing) cells. This test reproduces that pathway with zero
// diffusion deltas — drift should be at FP precision; anything larger is the
// bug.

#include "test_harness.h"
#include <cmath>
#include <cstdio>

constexpr int NELEM = 27;
enum { ELEM_H=0, ELEM_He=1, ELEM_C=2, ELEM_O=4 };

constexpr int TRAC_H2 = 0, TRAC_HP = 1, TRAC_CO = 2;
constexpr int TRAC_NUM = 3;
static const double trac_molwt[TRAC_NUM] = {2.0, 1.0, 28.0};
// fraction of the molecule mass that is H (for H2/H+) and C/O (for CO)
static const double mol_H_frac[TRAC_NUM]  = {1.0, 1.0, 0.0};         // H2: 2/2, H+: 1/1, CO: 0/28
static const double mol_C_frac           = 12.0/28.0;
static const double mol_O_frac           = 16.0/28.0;

struct Cell {
    double Mass;
    double X[NELEM];
    double TracAbund[TRAC_NUM];   // n_X / n_H
};

static double DMAX(double a, double b) { return a > b ? a : b; }

// Verbatim mirror of hydro_toplevel.cc:984-1019 (network-5 path)
// dYield_free[k]: diffusion delta for free element k (ELEM_H,C,O only)
// dYield_trac[k]: diffusion delta for tracer mass-fraction k (H2/H+/CO)
static void chemcool_pack_diffuse_unpack(Cell &c,
    const double dYield_free[NELEM], const double dYield_trac[TRAC_NUM])
{
    double X_H = DMAX(c.X[ELEM_H], 1e-10);

    double old_CO_locked_C = c.TracAbund[TRAC_CO] * 12.0 * X_H;
    double old_CO_locked_O = c.TracAbund[TRAC_CO] * 16.0 * X_H;
    double old_H2_locked_H = c.TracAbund[TRAC_H2] * 2.0  * X_H;
    double old_HP_locked_H = c.TracAbund[TRAC_HP] * 1.0  * X_H;

    double free_C = DMAX(c.X[ELEM_C] - old_CO_locked_C, 0);
    double free_O = DMAX(c.X[ELEM_O] - old_CO_locked_O, 0);
    double free_H = DMAX(c.X[ELEM_H] - old_H2_locked_H - old_HP_locked_H, 0);

    free_C = DMAX(free_C + dYield_free[ELEM_C] / c.Mass, 0.01*free_C);
    free_O = DMAX(free_O + dYield_free[ELEM_O] / c.Mass, 0.01*free_O);
    free_H = DMAX(free_H + dYield_free[ELEM_H] / c.Mass, 0.01*free_H);

    double trac_mf[TRAC_NUM];
    for(int k = 0; k < TRAC_NUM; k++) {
        double mf_old = c.TracAbund[k] * trac_molwt[k] * X_H;
        trac_mf[k] = DMAX(mf_old + dYield_trac[k] / c.Mass, 0.01*mf_old);
    }

    c.X[ELEM_H] = free_H + trac_mf[TRAC_H2] + trac_mf[TRAC_HP];
    c.X[ELEM_C] = free_C + trac_mf[TRAC_CO] * mol_C_frac;
    c.X[ELEM_O] = free_O + trac_mf[TRAC_CO] * mol_O_frac;

    double X_H_new = DMAX(c.X[ELEM_H], 1e-10);
    for(int k = 0; k < TRAC_NUM; k++) {
        c.TracAbund[k] = trac_mf[k] / (trac_molwt[k] * X_H_new);
    }
}

// Build an ISM cell at the test_SN_PI initial state (Z = 0.1 solar)
static Cell make_ism_cell(double T_H2_n, double T_HP_n, double T_CO_n)
{
    Cell c = {};
    c.Mass = 1.0;
    c.X[ELEM_H]  = 0.7579;
    c.X[ELEM_He] = 0.2409;
    c.X[ELEM_C]  = 5.0e-4;
    c.X[ELEM_O]  = 4.0e-3;
    // (other elements remain zero for this isolated test — sum < 1, but only H/C/O matter here)
    c.TracAbund[TRAC_H2] = T_H2_n;   // n(H2)/n(H)
    c.TracAbund[TRAC_HP] = T_HP_n;   // n(H+)/n(H)
    c.TracAbund[TRAC_CO] = T_CO_n;   // n(CO)/n(H)
    return c;
}

// =============================================================================
// Test 1: ZERO diffusion → round-trip is identity
// =============================================================================
TEST_CASE("ZERO Dyield: pack/unpack preserves X_H exactly") {
    Cell c = make_ism_cell(0.1, 1e-4, 1e-7);  // mostly molecular
    Cell c0 = c;
    double zero_free[NELEM] = {0};
    double zero_trac[TRAC_NUM] = {0};

    chemcool_pack_diffuse_unpack(c, zero_free, zero_trac);

    std::fprintf(stdout, "  X_H  before=%.15g  after=%.15g  drift=%.3e\n",
                 c0.X[ELEM_H], c.X[ELEM_H], c.X[ELEM_H] - c0.X[ELEM_H]);
    std::fprintf(stdout, "  X_C  before=%.15g  after=%.15g  drift=%.3e\n",
                 c0.X[ELEM_C], c.X[ELEM_C], c.X[ELEM_C] - c0.X[ELEM_C]);
    std::fprintf(stdout, "  X_O  before=%.15g  after=%.15g  drift=%.3e\n",
                 c0.X[ELEM_O], c.X[ELEM_O], c.X[ELEM_O] - c0.X[ELEM_O]);

    CHECK_CLOSE(c.X[ELEM_H], c0.X[ELEM_H], 1e-14);
    CHECK_CLOSE(c.X[ELEM_C], c0.X[ELEM_C], 1e-14);
    CHECK_CLOSE(c.X[ELEM_O], c0.X[ELEM_O], 1e-14);
}

// =============================================================================
// Test 2: Many ZERO-diffusion round-trips (chemistry sets new TracAbund each step)
// — does X_H drift even when no diffusion happens, just because the chemistry
// keeps re-shuffling H ↔ H2 ↔ H+ between calls?
// =============================================================================
TEST_CASE("loop: 1000 zero-diffusion round-trips, with chemistry shifting TracAbund") {
    Cell c = make_ism_cell(0.1, 1e-4, 1e-7);
    Cell c0 = c;
    double zero_free[NELEM] = {0};
    double zero_trac[TRAC_NUM] = {0};

    // Each step: simulate chemistry shifting H → H2 (or back) by tweaking TracAbund
    // before the next pack/unpack. This mimics how compression activates H2 formation
    // between TURB_DIFF substeps.
    double H2_drift = 0.001;  // each step, n(H2)/n(H) grows by 0.1% of current
    for(int n = 0; n < 1000; n++) {
        c.TracAbund[TRAC_H2] *= (1.0 + H2_drift);   // chemistry forms more H2
        chemcool_pack_diffuse_unpack(c, zero_free, zero_trac);
    }

    double drift = c.X[ELEM_H] - c0.X[ELEM_H];
    std::fprintf(stdout, "  after 1000 round-trips with H2 forming: X_H drift = %.3e\n", drift);
    std::fprintf(stdout, "    final X_H = %.6f (initial %.6f)\n", c.X[ELEM_H], c0.X[ELEM_H]);
    std::fprintf(stdout, "    final TracAbund = [H2=%.4e, H+=%.4e, CO=%.4e]\n",
                 c.TracAbund[TRAC_H2], c.TracAbund[TRAC_HP], c.TracAbund[TRAC_CO]);

    // Mass of H must be conserved exactly (no actual diffusion happened)
    CHECK(std::fabs(drift) < 1e-12);
}

// =============================================================================
// Test 3: Conservative diffusion of free H (Dyield in ↔ out cancel across pair).
// =============================================================================
TEST_CASE("conservative Dyield_free[H] preserves total H across two cells") {
    Cell A = make_ism_cell(0.1, 1e-4, 1e-7);
    Cell B = make_ism_cell(0.05, 1e-4, 1e-7);
    A.X[ELEM_H] = 0.65;     // A is enriched (low free_H)
    B.X[ELEM_H] = 0.78;     // B is pristine

    // Conservative pair flux: A receives, B gives
    double dM = 0.001;
    double freeA[NELEM] = {0}; freeA[ELEM_H] = +dM;
    double freeB[NELEM] = {0}; freeB[ELEM_H] = -dM;
    double zero_trac[TRAC_NUM] = {0};

    double H_before = A.X[ELEM_H]*A.Mass + B.X[ELEM_H]*B.Mass;
    chemcool_pack_diffuse_unpack(A, freeA, zero_trac);
    chemcool_pack_diffuse_unpack(B, freeB, zero_trac);
    double H_after  = A.X[ELEM_H]*A.Mass + B.X[ELEM_H]*B.Mass;

    std::fprintf(stdout, "  total H before=%.15g  after=%.15g  drift=%.3e\n",
                 H_before, H_after, H_after - H_before);
    CHECK(std::fabs(H_after - H_before) < 1e-12);
}

TEST_MAIN()
