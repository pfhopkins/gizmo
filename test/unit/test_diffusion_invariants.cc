// Unit test: TURB_DIFF mass-fraction conservation under the post-diffusion
// unpack code in hydro_toplevel.cc.
//
// The unpack does (e.g.):
//   free_H = DMAX(free_H + Dyield[ELEM_H] / Mass, 0.01 * free_H);
//
// This clamp prevents free_H from dropping below 1% of its previous value
// in a single step — a "safety floor" against negative abundances. The clamp
// is **asymmetric**: it floors losses but never caps gains. For a properly
// mass-conservative diffusion solver, Σ Dyield_k over neighbors is zero;
// the per-cell update should preserve total mass of each element across
// the system.
//
// We test two regimes:
//
//   Test 1 (realistic):  small per-step gradient on normal ISM gas
//                        — clamp should NOT trigger; mass should be conserved.
//   Test 2 (edge case):  super-enriched cell with tiny free_H against a
//                        normal cell, with a diffusion delta large enough
//                        to push free_H near zero — clamp WILL trigger;
//                        prediction: total mass increases (bug confirmed).

#include "test_harness.h"
#include <cmath>
#include <cstdio>

constexpr int NELEM = 27;
enum { ELEM_H=0, ELEM_He=1, ELEM_C=2, ELEM_O=4 };

struct Cell {
    double Mass;
    double X[NELEM];          // mass fractions
    double Dyield[NELEM];     // mass of each element transferred IN this step
};

// Mirror of the post-diffusion unpack in hydro_toplevel.cc:949-957
static void apply_diffusion_with_clamp(Cell &c) {
    for(int k = 0; k < NELEM; k++) {
        double new_X = c.X[k] + c.Dyield[k] / c.Mass;
        c.X[k] = std::fmax(new_X, 0.01 * c.X[k]);
    }
}

// Same update WITHOUT the clamp (truthful conservative arithmetic)
static void apply_diffusion_no_clamp(Cell &c) {
    for(int k = 0; k < NELEM; k++) {
        c.X[k] = c.X[k] + c.Dyield[k] / c.Mass;
    }
}

// Total mass of element k across all cells
static double total_mass_k(const Cell *cells, int n, int k) {
    double s = 0;
    for(int i = 0; i < n; i++) s += cells[i].X[k] * cells[i].Mass;
    return s;
}

// =============================================================================
// Test 1: realistic ISM gradient — clamp should NOT trigger; mass conserved
// =============================================================================
TEST_CASE("TURB_DIFF: realistic ISM gradient preserves total H mass") {
    Cell A = {}, B = {};
    A.Mass = 1.0; B.Mass = 1.0;
    A.X[ELEM_H] = 0.71; B.X[ELEM_H] = 0.74;          // normal ISM, mild gradient
    A.X[ELEM_He] = 0.27; B.X[ELEM_He] = 0.25;
    A.X[ELEM_C] = 0.005; B.X[ELEM_C] = 0.003;
    // (other elements zero for simplicity)

    // Conservative flux: ΔM = κ × (X_B - X_A) × (small step)
    double dM_H = 0.001;   // 0.001 Msun of H transferred B → A
    A.Dyield[ELEM_H] = +dM_H;
    B.Dyield[ELEM_H] = -dM_H;

    Cell cells[2] = {A, B};
    double H_before = total_mass_k(cells, 2, ELEM_H);
    apply_diffusion_with_clamp(cells[0]);
    apply_diffusion_with_clamp(cells[1]);
    double H_after = total_mass_k(cells, 2, ELEM_H);

    double drift = std::fabs(H_after - H_before);
    std::fprintf(stdout, "  realistic: total H mass before=%.10g after=%.10g drift=%.3e\n",
                 H_before, H_after, drift);
    CHECK(drift < 1e-12);
}

// =============================================================================
// Test 2: edge case — extreme gradient that triggers the clamp
// =============================================================================
TEST_CASE("TURB_DIFF: extreme gradient + clamp creates H mass") {
    Cell A = {}, B = {};
    A.Mass = 1.0; B.Mass = 1.0;
    // A is super-enriched (low free_H), B is pristine
    A.X[ELEM_H] = 0.001; A.X[ELEM_He] = 0.0001; A.X[ELEM_C] = 0.5;
    B.X[ELEM_H] = 0.74;  B.X[ELEM_He] = 0.25;   B.X[ELEM_C] = 0.003;

    // Diffusion will push H from B → A. If the per-step delta is large enough
    // to drive B's free_H below 1% of its original, the clamp fires.
    // We construct an artificially large delta (not unrealistic for very-strong
    // gradients × long substeps): transfer most of B's H to A.
    double dM_H = 0.74 * 0.999;  // 99.9% of B's H tries to migrate
    A.Dyield[ELEM_H] = +dM_H;
    B.Dyield[ELEM_H] = -dM_H;

    Cell cells[2] = {A, B};
    double H_before = total_mass_k(cells, 2, ELEM_H);
    apply_diffusion_with_clamp(cells[0]);
    apply_diffusion_with_clamp(cells[1]);
    double H_after = total_mass_k(cells, 2, ELEM_H);

    double drift = H_after - H_before;
    std::fprintf(stdout, "  edge:      total H mass before=%.10g after=%.10g drift=%+0.3e\n",
                 H_before, H_after, drift);
    std::fprintf(stdout, "             cell A H_old=%.4g H_new=%.4g  cell B H_old=%.4g H_new=%.4g\n",
                 0.001, cells[0].X[ELEM_H], 0.74, cells[1].X[ELEM_H]);

    // Expectation: the clamp on B prevented losing all of B's H
    // → some H was "created" at the system level.
    CHECK(drift > 1e-6);   // confirms mass is created (positive drift)
}

// =============================================================================
// Test 3: many small steps with a moderate gradient — does drift accumulate
// even when individual clamps don't fire each step?
// =============================================================================
TEST_CASE("TURB_DIFF: 1000 small steps, moderate gradient — accumulating drift?") {
    Cell A = {}, B = {};
    A.Mass = 1.0; B.Mass = 1.0;
    // Moderate gradient: enriched cell at X_H=0.05, normal at X_H=0.70
    A.X[ELEM_H] = 0.05; B.X[ELEM_H] = 0.70;

    double H_before = total_mass_k((Cell[]){A, B}, 2, ELEM_H);

    // 1000 small diffusion sub-steps, each transferring 1% of (B-A) gradient
    Cell cells[2] = {A, B};
    for(int n = 0; n < 1000; n++) {
        double dM_H = 0.01 * (cells[1].X[ELEM_H] - cells[0].X[ELEM_H]);
        cells[0].Dyield[ELEM_H] = +dM_H;
        cells[1].Dyield[ELEM_H] = -dM_H;
        apply_diffusion_with_clamp(cells[0]);
        apply_diffusion_with_clamp(cells[1]);
    }
    double H_after = total_mass_k(cells, 2, ELEM_H);
    double drift = H_after - H_before;
    std::fprintf(stdout, "  loop:      total H mass before=%.10g after=%.10g drift=%+0.3e\n",
                 H_before, H_after, drift);
    std::fprintf(stdout, "             final A.X[H]=%.6f  B.X[H]=%.6f\n",
                 cells[0].X[ELEM_H], cells[1].X[ELEM_H]);
    CHECK(std::fabs(drift) < 1e-10);   // expect no drift if clamp never fires
}

TEST_MAIN()
