// Unit test: reproduce the pre-SF X_H inflation seen in test_SN_PI snap 16-28.
//
// Observed in the live run: a single gas cell, no stars in the system yet,
// no feedback. Cell starts at X_H = 0.7579 (IC) with constant Mass. Once it
// begins compressing (nH 0.79 → 9 in one snap), X_H grows monotonically.
// After 12 Myr (12 snaps) X_H reaches 0.95 with H_mass × Mass having
// inflated by 25% — i.e. mass is genuinely created.
//
// Hypothesis: the unpack clamp DMAX(free_H + Dyield/Mass, 0.01*free_H)
// fires when chemistry locks most H into H2 (so free_H is small relative
// to the total cell H). In dense / chemistry-active cells, free_H ≪ X_H,
// and the 99% loss threshold becomes much easier to hit even for modest
// diffusion deltas.
//
// Setup:
//   Cell A: dense, "compressed". TracAbund[H2] grown to ~0.4 → free_H is small.
//   Cell B: pristine, low TracAbund.
//   Both same X_H_total = 0.76 initially.
//
// Apply CONSERVATIVE pair-flux diffusion (any flux out of A appears as
// equal-magnitude flux into B). Run multiple substeps. Check whether total
// H mass across A+B is conserved.

#include "test_harness.h"
#include <cmath>
#include <cstdio>

constexpr int NELEM = 27;
enum { ELEM_H=0, ELEM_He=1, ELEM_C=2, ELEM_O=4 };
constexpr int TRAC_H2 = 0, TRAC_HP = 1, TRAC_CO = 2;
constexpr int TRAC_NUM = 3;
static const double trac_molwt[TRAC_NUM] = {2.0, 1.0, 28.0};
static const double mol_C_frac = 12.0/28.0, mol_O_frac = 16.0/28.0;

struct Cell {
    double Mass;
    double X[NELEM];
    double TracAbund[TRAC_NUM];
};

static double DMAX(double a, double b) { return a > b ? a : b; }

// Verbatim mirror of hydro_toplevel.cc:984-1019 (network-5 path)
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

// Cell-mass total for element k (k=H or whatever)
static double total_mass_X(const Cell *c, int n, int elem) {
    double s = 0;
    for(int i = 0; i < n; i++) s += c[i].X[elem] * c[i].Mass;
    return s;
}

// =============================================================================
// Test: dense chemistry-active cell vs pristine, conservative pair-flux diffusion.
// Reproduces the live-run scenario:
//   A:  high H2 → small free_H
//   B:  no H2 → large free_H
//   diffusion exchanges small amounts; check if total H_mass is conserved
// =============================================================================
TEST_CASE("dense + pristine pair-flux: total H conservation under repeated unpack") {
    Cell A = {}, B = {};
    A.Mass = 1.0;  B.Mass = 1.0;
    A.X[ELEM_H] = 0.76; A.X[ELEM_He] = 0.24; A.X[ELEM_C] = 5e-4; A.X[ELEM_O] = 4e-3;
    B.X[ELEM_H] = 0.76; B.X[ELEM_He] = 0.24; B.X[ELEM_C] = 5e-4; B.X[ELEM_O] = 4e-3;
    // A is dense + chemistry-active: 95% of cell H locked in H2
    // TracAbund[H2] = (locked_H_mass_fraction) / (2 * X_H) = (0.95*0.76)/(2*0.76) = 0.475
    A.TracAbund[TRAC_H2] = 0.475;       // n(H2)/n(H), mass-locked fraction = 0.95*X_H
    A.TracAbund[TRAC_HP] = 1e-4;
    A.TracAbund[TRAC_CO] = 1e-7;
    // B is pristine: almost no H2
    B.TracAbund[TRAC_H2] = 1e-6;
    B.TracAbund[TRAC_HP] = 1e-4;
    B.TracAbund[TRAC_CO] = 1e-7;

    Cell cells[2] = {A, B};
    double H_init = total_mass_X(cells, 2, ELEM_H);

    // Apply N substeps of small conservative pair-flux diffusion
    // Per step, transfer 1% of (free_H_B - free_H_A) plus 1% of (mf_H2_A - mf_H2_B)
    // (both directions — each tracer diffuses on its own gradient)
    for(int n = 0; n < 1000; n++) {
        double X_H_A = DMAX(cells[0].X[ELEM_H], 1e-10);
        double X_H_B = DMAX(cells[1].X[ELEM_H], 1e-10);
        double mf_H2_A = cells[0].TracAbund[TRAC_H2] * 2.0 * X_H_A;
        double mf_H2_B = cells[1].TracAbund[TRAC_H2] * 2.0 * X_H_B;
        double free_H_A = cells[0].X[ELEM_H] - mf_H2_A - cells[0].TracAbund[TRAC_HP]*X_H_A;
        double free_H_B = cells[1].X[ELEM_H] - mf_H2_B - cells[1].TracAbund[TRAC_HP]*X_H_B;

        double dY_free_H = 0.01 * (free_H_B - free_H_A);   // B → A (B is higher)
        double dY_trac_H2 = 0.01 * (mf_H2_B - mf_H2_A);     // A → B (A has more H2)

        double freeA[NELEM] = {0}; freeA[ELEM_H] = +dY_free_H;
        double freeB[NELEM] = {0}; freeB[ELEM_H] = -dY_free_H;
        double tracA[TRAC_NUM] = {0}; tracA[TRAC_H2] = +dY_trac_H2;
        double tracB[TRAC_NUM] = {0}; tracB[TRAC_H2] = -dY_trac_H2;

        chemcool_pack_diffuse_unpack(cells[0], freeA, tracA);
        chemcool_pack_diffuse_unpack(cells[1], freeB, tracB);
    }

    double H_after = total_mass_X(cells, 2, ELEM_H);
    double drift = H_after - H_init;
    std::fprintf(stdout, "  after 1000 substeps:\n");
    std::fprintf(stdout, "    cell A: X_H=%.6f  TracAbund[H2]=%.6f\n", cells[0].X[ELEM_H], cells[0].TracAbund[TRAC_H2]);
    std::fprintf(stdout, "    cell B: X_H=%.6f  TracAbund[H2]=%.6f\n", cells[1].X[ELEM_H], cells[1].TracAbund[TRAC_H2]);
    std::fprintf(stdout, "    total H_mass: before=%.10g  after=%.10g  drift=%+.3e\n",
                 H_init, H_after, drift);

    CHECK(std::fabs(drift) < 1e-10);
}

// =============================================================================
// Edge case: A's free_H is forced very small by extreme chemistry, then the
// SAME small conservative-flux diffusion. Does the asymmetric clamp fire and
// create H mass?
// =============================================================================
TEST_CASE("extreme: A's free_H ≈ 0 (chemistry locks 99.9% of H), conservative flux") {
    Cell A = {}, B = {};
    A.Mass = 1.0; B.Mass = 1.0;
    A.X[ELEM_H] = 0.76; A.X[ELEM_He] = 0.24;
    B.X[ELEM_H] = 0.76; B.X[ELEM_He] = 0.24;
    // Push 99.9% of A's H into H2 (so locked = 0.999*0.76 = 0.759, free_H ≈ 7.6e-4)
    A.TracAbund[TRAC_H2] = 0.4995;       // mass-locked = 0.999*X_H
    A.TracAbund[TRAC_HP] = 1e-6;
    A.TracAbund[TRAC_CO] = 1e-7;
    B.TracAbund[TRAC_H2] = 1e-6;
    B.TracAbund[TRAC_HP] = 1e-6;
    B.TracAbund[TRAC_CO] = 1e-7;

    Cell cells[2] = {A, B};
    double H_init = total_mass_X(cells, 2, ELEM_H);

    // Now apply a flux that REMOVES some free_H from A (e.g., diffusion sweeps it out)
    // — this should trigger the 0.01*free_H clamp because free_H is tiny.
    for(int n = 0; n < 1000; n++) {
        double X_H_A = DMAX(cells[0].X[ELEM_H], 1e-10);
        double X_H_B = DMAX(cells[1].X[ELEM_H], 1e-10);
        double mf_H2_A = cells[0].TracAbund[TRAC_H2] * 2.0 * X_H_A;
        double free_H_A = cells[0].X[ELEM_H] - mf_H2_A - cells[0].TracAbund[TRAC_HP]*X_H_A;

        // Very small absolute flux but it removes 50% of A's free_H per step
        // (A's free_H is tiny so this is small in absolute terms)
        double dY_free_H = -0.5 * free_H_A;   // A loses 50% of its small free_H

        double freeA[NELEM] = {0}; freeA[ELEM_H] = +dY_free_H;     // negative
        double freeB[NELEM] = {0}; freeB[ELEM_H] = -dY_free_H;     // positive
        double zero_trac[TRAC_NUM] = {0};

        chemcool_pack_diffuse_unpack(cells[0], freeA, zero_trac);
        chemcool_pack_diffuse_unpack(cells[1], freeB, zero_trac);
    }

    double H_after = total_mass_X(cells, 2, ELEM_H);
    double drift = H_after - H_init;
    std::fprintf(stdout, "  edge: after 1000 substeps:\n");
    std::fprintf(stdout, "    cell A: X_H=%.6f\n", cells[0].X[ELEM_H]);
    std::fprintf(stdout, "    cell B: X_H=%.6f\n", cells[1].X[ELEM_H]);
    std::fprintf(stdout, "    total H_mass: before=%.10g  after=%.10g  drift=%+.3e (rel %+.3f%%)\n",
                 H_init, H_after, drift, 100*drift/H_init);

    // Expectation: the asymmetric clamp creates H mass — drift should be POSITIVE
    if(drift > 1e-8) {
        std::fprintf(stdout, "    → CLAMP FIRES, H mass is created (this is the bug)\n");
    }
    CHECK(true); // informational test — CHECK is just to count
}

// =============================================================================
// Test the proposed fix: same scenario but with floor at 0.0 instead of 0.01*old.
// =============================================================================
static void chemcool_pack_diffuse_unpack_FIXED(Cell &c,
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

    free_C = DMAX(free_C + dYield_free[ELEM_C] / c.Mass, 0.0);    // FIX: floor at 0, not 0.01*old
    free_O = DMAX(free_O + dYield_free[ELEM_O] / c.Mass, 0.0);
    free_H = DMAX(free_H + dYield_free[ELEM_H] / c.Mass, 0.0);

    double trac_mf[TRAC_NUM];
    for(int k = 0; k < TRAC_NUM; k++) {
        double mf_old = c.TracAbund[k] * trac_molwt[k] * X_H;
        trac_mf[k] = DMAX(mf_old + dYield_trac[k] / c.Mass, 0.0);   // FIX
    }

    c.X[ELEM_H] = free_H + trac_mf[TRAC_H2] + trac_mf[TRAC_HP];
    c.X[ELEM_C] = free_C + trac_mf[TRAC_CO] * mol_C_frac;
    c.X[ELEM_O] = free_O + trac_mf[TRAC_CO] * mol_O_frac;

    double X_H_new = DMAX(c.X[ELEM_H], 1e-10);
    for(int k = 0; k < TRAC_NUM; k++) {
        c.TracAbund[k] = trac_mf[k] / (trac_molwt[k] * X_H_new);
    }
}

TEST_CASE("FIX: floor=0 should preserve H mass even in extreme regime") {
    Cell A = {}, B = {};
    A.Mass = 1.0; B.Mass = 1.0;
    A.X[ELEM_H] = 0.76; A.X[ELEM_He] = 0.24;
    B.X[ELEM_H] = 0.76; B.X[ELEM_He] = 0.24;
    A.TracAbund[TRAC_H2] = 0.4995;
    A.TracAbund[TRAC_HP] = 1e-6; A.TracAbund[TRAC_CO] = 1e-7;
    B.TracAbund[TRAC_H2] = 1e-6;
    B.TracAbund[TRAC_HP] = 1e-6; B.TracAbund[TRAC_CO] = 1e-7;

    Cell cells[2] = {A, B};
    double H_init = total_mass_X(cells, 2, ELEM_H);

    for(int n = 0; n < 1000; n++) {
        double X_H_A = DMAX(cells[0].X[ELEM_H], 1e-10);
        double mf_H2_A = cells[0].TracAbund[TRAC_H2] * 2.0 * X_H_A;
        double free_H_A = cells[0].X[ELEM_H] - mf_H2_A - cells[0].TracAbund[TRAC_HP]*X_H_A;
        double dY_free_H = -0.5 * free_H_A;

        double freeA[NELEM] = {0}; freeA[ELEM_H] = +dY_free_H;
        double freeB[NELEM] = {0}; freeB[ELEM_H] = -dY_free_H;
        double zero_trac[TRAC_NUM] = {0};

        chemcool_pack_diffuse_unpack_FIXED(cells[0], freeA, zero_trac);
        chemcool_pack_diffuse_unpack_FIXED(cells[1], freeB, zero_trac);
    }

    double H_after = total_mass_X(cells, 2, ELEM_H);
    double drift = H_after - H_init;
    std::fprintf(stdout, "  FIXED: after 1000 substeps with 0.0 floor:\n");
    std::fprintf(stdout, "    total H_mass drift=%+.3e (rel %+.3f%%)\n", drift, 100*drift/H_init);

    CHECK(std::fabs(drift) < 1e-10);
}

TEST_MAIN()
