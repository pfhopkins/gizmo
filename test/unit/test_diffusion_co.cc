// Unit test for the TracAbund / ElementAbundance diffusion cycle.
//
// Reproduces the exact pack → apply Dyield → unpack logic from
// hydro_toplevel.cc and checks whether CO can ever exceed
// min(abundc, abundo) after unpacking.  By construction that must
// never happen — CO cannot exceed the available carbon or oxygen.
//
// The test sweeps many parameter combinations:
//   - varying CO fractions relative to abundc/abundo
//   - C-rich (abundc > abundo) and O-rich (abundo > abundc) regimes
//   - positive and negative Dyield (CO influx and outflux)
//   - large and small Dyield magnitudes
//   - multiple consecutive diffusion cycles (accumulation test)
//
// If the pack/unpack logic has a units mismatch, double-counting, or
// missing clamp, this test will catch it.

#include "test_harness.h"
#include <cmath>
#include <cstdio>
#include <algorithm>

// ── helpers matching the code ──────────────────────────────────────

static const int TRAC_NUM = 3;   // H2, H+, CO
static const double trac_molwt[TRAC_NUM] = {2.0, 1.0, 28.0};

static inline double DMAX(double a, double b) { return a > b ? a : b; }

// Mimics a single particle's state
struct Particle {
    double Mass;
    double ElementAbundance_H;   // hydrogen mass fraction
    double ElementAbundance_C;   // total carbon mass fraction (free + CO-locked)
    double ElementAbundance_O;   // total oxygen mass fraction (free + CO-locked)
    double TracAbund[TRAC_NUM];  // abundance ratios: H2, H+, CO  (n_X / n_H)
};

// ── packing (hydro_toplevel.cc ~line 497-520) ──────────────────────

struct PackedMetallicity {
    double H;           // free-H mass fraction (neutral atomic H only)
    double C;           // free-C mass fraction
    double O;           // free-O mass fraction
    double trac[TRAC_NUM]; // TracAbund converted to mass fractions
};

static PackedMetallicity pack(const Particle &p)
{
    PackedMetallicity m;
    double X_H = DMAX(p.ElementAbundance_H, 1e-10);

    // free C and O (subtract CO-locked fraction)
    double CO_locked_C = p.TracAbund[2] * 12.0 * X_H;
    double CO_locked_O = p.TracAbund[2] * 16.0 * X_H;
    m.C = DMAX(p.ElementAbundance_C - CO_locked_C, 0);
    m.O = DMAX(p.ElementAbundance_O - CO_locked_O, 0);

    // free H (subtract H2-locked and H+-locked fractions)
    double H2_locked_H = p.TracAbund[0] * 2.0 * X_H;
    double HP_locked_H = p.TracAbund[1] * 1.0 * X_H;
    m.H = DMAX(p.ElementAbundance_H - H2_locked_H - HP_locked_H, 0);

    // TracAbund → mass fractions
    for(int k = 0; k < TRAC_NUM; k++)
        m.trac[k] = p.TracAbund[k] * trac_molwt[k] * X_H;

    return m;
}

// ── unpacking (hydro_toplevel.cc ~line 878-900) ────────────────────
// Dyield_C, Dyield_O are mass fluxes for free-C, free-O.
// Dyield_trac[k] are mass fluxes for TracAbund (in mass-fraction units).
//
// This reproduces the CURRENT code exactly, including any bugs.

static void unpack(Particle &p,
                   double Dyield_H, double Dyield_C, double Dyield_O,
                   const double Dyield_trac[TRAC_NUM])
{
    double X_H = DMAX(p.ElementAbundance_H, 1e-10);

    // Step 1: strip locked fractions to get free components
    double old_CO_locked_C = p.TracAbund[2] * 12.0 * X_H;
    double old_CO_locked_O = p.TracAbund[2] * 16.0 * X_H;
    double old_H2_locked_H = p.TracAbund[0] * 2.0 * X_H;
    double old_HP_locked_H = p.TracAbund[1] * 1.0 * X_H;
    double free_C = DMAX(p.ElementAbundance_C - old_CO_locked_C, 0);
    double free_O = DMAX(p.ElementAbundance_O - old_CO_locked_O, 0);
    double free_H = DMAX(p.ElementAbundance_H - old_H2_locked_H - old_HP_locked_H, 0);

    // Step 2: apply free-fraction diffusion deltas
    free_C = DMAX(free_C + Dyield_C / p.Mass, 0.01 * free_C);
    free_O = DMAX(free_O + Dyield_O / p.Mass, 0.01 * free_O);
    free_H = DMAX(free_H + Dyield_H / p.Mass, 0.01 * free_H);

    // Step 3: update species masses (in mass-fraction units)
    double trac_mf[TRAC_NUM];
    for(int k = 0; k < TRAC_NUM; k++) {
        double mf_old = p.TracAbund[k] * trac_molwt[k] * X_H;
        trac_mf[k] = DMAX(mf_old + Dyield_trac[k] / p.Mass, 0.01 * mf_old);
    }

    // Step 4: reconstruct element totals from free + locked
    p.ElementAbundance_H = free_H + trac_mf[0] + trac_mf[1];       // free_H + H2_mf + HP_mf
    p.ElementAbundance_C = free_C + trac_mf[2] * (12.0/28.0);      // free_C + CO_locked_C
    p.ElementAbundance_O = free_O + trac_mf[2] * (16.0/28.0);      // free_O + CO_locked_O

    // Step 5: convert species masses to abundance ratios using NEW X_H
    double X_H_new = DMAX(p.ElementAbundance_H, 1e-10);
    for(int k = 0; k < TRAC_NUM; k++) {
        p.TracAbund[k] = trac_mf[k] / (trac_molwt[k] * X_H_new);
    }
}

// ── invariant check ────────────────────────────────────────────────

static bool check_co_ceiling(const Particle &p, const char* label, int iter = -1)
{
    double X_H = DMAX(p.ElementAbundance_H, 1e-10);
    double abundc = p.ElementAbundance_C / (12.0 * X_H);
    double abundo = p.ElementAbundance_O / (16.0 * X_H);
    double CO = p.TracAbund[2];
    double ceiling = std::min(abundc, abundo);

    // Allow tiny floating-point overshoot (1e-12 relative)
    if(CO > ceiling * (1.0 + 1e-12)) {
        std::fprintf(stderr,
            "  VIOLATION [%s, iter=%d]: CO=%.6e > min(abundc,abundo)=%.6e "
            "(abundc=%.6e, abundo=%.6e)\n",
            label, iter, CO, ceiling, abundc, abundo);
        return false;
    }
    return true;
}

// ── tests ──────────────────────────────────────────────────────────

TEST_CASE("single diffusion step: CO stays below ceiling (O-rich)") {
    // O-rich regime: abundc < abundo, ceiling = abundc
    // Carbon: 2.8e-4 mass fraction → abundc ~ 3.1e-4
    // Oxygen: 5.6e-3 mass fraction → abundo ~ 4.6e-4
    // CO = 2.0e-4 (< abundc, below ceiling)
    double Dyield_fracs[] = {-0.1, 0.0, 0.1, 0.5, 1.0, -0.5};
    for(double frac : Dyield_fracs) {
        Particle p;
        p.Mass = 1.0;
        p.ElementAbundance_H = 0.76;
        p.ElementAbundance_C = 2.8e-4;
        p.ElementAbundance_O = 5.6e-3;
        p.TracAbund[0] = 0.3;   // H2
        p.TracAbund[1] = 1e-6;  // H+
        p.TracAbund[2] = 2.0e-4; // CO

        PackedMetallicity m = pack(p);
        // Dyield = frac * packed value * Mass  (simulates diffusion flux)
        double Dyield_C = frac * m.C * p.Mass;
        double Dyield_O = frac * m.O * p.Mass;
        double Dyield_trac[TRAC_NUM] = {0, 0, frac * m.trac[2] * p.Mass};

        unpack(p, 0, Dyield_C, Dyield_O, Dyield_trac);

        char label[64];
        std::snprintf(label, sizeof(label), "O-rich frac=%.1f", frac);
        CHECK(check_co_ceiling(p, label));
    }
}

TEST_CASE("single diffusion step: CO stays below ceiling (C-rich)") {
    // C-rich regime: abundo < abundc, ceiling = abundo
    // Carbon: 5.6e-3 mass fraction → abundc ~ 6.1e-4
    // Oxygen: 2.8e-4 mass fraction → abundo ~ 2.3e-5
    // CO = 1.0e-5 (< abundo, below ceiling)
    double Dyield_fracs[] = {-0.1, 0.0, 0.1, 0.5, 1.0, -0.5};
    for(double frac : Dyield_fracs) {
        Particle p;
        p.Mass = 1.0;
        p.ElementAbundance_H = 0.76;
        p.ElementAbundance_C = 5.6e-3;
        p.ElementAbundance_O = 2.8e-4;
        p.TracAbund[0] = 0.3;
        p.TracAbund[1] = 1e-6;
        p.TracAbund[2] = 1.0e-5;

        PackedMetallicity m = pack(p);
        double Dyield_C = frac * m.C * p.Mass;
        double Dyield_O = frac * m.O * p.Mass;
        double Dyield_trac[TRAC_NUM] = {0, 0, frac * m.trac[2] * p.Mass};

        unpack(p, 0, Dyield_C, Dyield_O, Dyield_trac);

        char label[64];
        std::snprintf(label, sizeof(label), "C-rich frac=%.1f", frac);
        CHECK(check_co_ceiling(p, label));
    }
}

TEST_CASE("CO near ceiling: diffusion must not push over") {
    // CO = 0.99 * min(abundc, abundo) — nearly saturated
    double X_H = 0.76;
    double elem_C = 3.0e-4;
    double elem_O = 4.0e-4;
    double abundc = elem_C / (12.0 * X_H);
    double abundo = elem_O / (16.0 * X_H);
    double ceiling = std::min(abundc, abundo);
    double CO_init = 0.99 * ceiling;

    double Dyield_fracs[] = {0.01, 0.05, 0.1, 0.5, 1.0};
    for(double frac : Dyield_fracs) {
        Particle p;
        p.Mass = 1.0;
        p.ElementAbundance_H = X_H;
        p.ElementAbundance_C = elem_C;
        p.ElementAbundance_O = elem_O;
        p.TracAbund[0] = 0.3;
        p.TracAbund[1] = 1e-6;
        p.TracAbund[2] = CO_init;

        PackedMetallicity m = pack(p);
        // Only CO gets positive Dyield (influx from neighbor with more CO)
        double Dyield_trac[TRAC_NUM] = {0, 0, frac * m.trac[2] * p.Mass};

        unpack(p, 0, 0, 0, Dyield_trac);

        char label[64];
        std::snprintf(label, sizeof(label), "near-ceiling frac=%.2f", frac);
        CHECK(check_co_ceiling(p, label));
    }
}

TEST_CASE("repeated diffusion cycles: no CO drift above ceiling") {
    // Run 1000 pack→unpack cycles with small random-ish Dyields.
    // This tests whether double-counting or leaks accumulate over time.
    // abundc = 3.0e-3/(12*0.76) = 3.29e-4, abundo = 5.0e-3/(16*0.76) = 4.11e-4
    // CO = 1.0e-4 < min(abundc,abundo) ✓
    Particle p;
    p.Mass = 100.0;  // 100 Msun particle
    p.ElementAbundance_H = 0.76;
    p.ElementAbundance_C = 3.0e-3;
    p.ElementAbundance_O = 5.0e-3;
    p.TracAbund[0] = 0.25;
    p.TracAbund[1] = 1e-8;
    p.TracAbund[2] = 1.0e-4;

    int violations = 0;
    for(int iter = 0; iter < 1000; iter++) {
        PackedMetallicity m = pack(p);

        // Small diffusion flux: ±1% of packed value, alternating sign
        double sign = (iter % 3 == 0) ? 1.0 : ((iter % 3 == 1) ? -1.0 : 0.5);
        double frac = 0.01 * sign;

        double Dyield_C     = frac * m.C * p.Mass;
        double Dyield_O     = frac * m.O * p.Mass;
        double Dyield_trac[TRAC_NUM] = {
            0,                          // H2: don't diffuse in single-particle CO test
            0,                          // H+: don't diffuse in single-particle CO test
            frac * m.trac[2] * p.Mass   // CO: the quantity under test
        };

        unpack(p, 0, Dyield_C, Dyield_O, Dyield_trac);

        if(!check_co_ceiling(p, "1000-cycle", iter))
            violations++;

        // Bail early if clearly broken
        if(violations > 5) break;
    }
    CHECK(violations == 0);
}

TEST_CASE("repeated cycles: ElementAbundance[C] must not grow unbounded") {
    // Checks for the double-counting leak: if CO-locked C is added
    // without subtracting the old copy, ElementAbundance[C] grows
    // each cycle even with zero net Dyield.
    // abundc = 3.0e-3/(12*0.76) = 3.29e-4, abundo = 5.0e-3/(16*0.76) = 4.11e-4
    // CO = 1.0e-4 < min(abundc,abundo) ✓
    Particle p;
    p.Mass = 100.0;
    p.ElementAbundance_H = 0.76;
    p.ElementAbundance_C = 3.0e-3;
    p.ElementAbundance_O = 5.0e-3;
    p.TracAbund[0] = 0.25;
    p.TracAbund[1] = 1e-8;
    p.TracAbund[2] = 1.0e-4;

    double C_init = p.ElementAbundance_C;

    for(int iter = 0; iter < 100; iter++) {
        (void)pack(p);
        // Zero Dyield — no actual diffusion.  C should stay constant.
        double Dyield_trac[TRAC_NUM] = {0, 0, 0};
        unpack(p, 0, 0, 0, Dyield_trac);
    }

    // After 100 cycles with zero Dyield, C should not have changed
    // Allow 1e-10 relative tolerance for floating-point
    double drift = std::fabs(p.ElementAbundance_C - C_init) / C_init;
    if(drift > 1e-10) {
        std::fprintf(stderr,
            "  C drifted by %.6e relative after 100 zero-Dyield cycles "
            "(C_init=%.6e, C_final=%.6e)\n",
            drift, C_init, p.ElementAbundance_C);
    }
    CHECK(drift < 1e-10);
}

TEST_CASE("high metallicity C>O: CO must not exceed abundo") {
    // The regime that crashed in the DVODE test suite.
    // Solar C ~ 2.4e-3, O ~ 5.7e-3 mass fraction.
    // At 2x solar C, 0.5x solar O: C-dominated, abundo is the ceiling.
    Particle p;
    p.Mass = 50.0;
    p.ElementAbundance_H = 0.70;
    p.ElementAbundance_C = 4.8e-3;   // 2x solar
    p.ElementAbundance_O = 2.85e-3;  // 0.5x solar
    double abundo = p.ElementAbundance_O / (16.0 * p.ElementAbundance_H);
    p.TracAbund[0] = 0.4;
    p.TracAbund[1] = 1e-10;
    p.TracAbund[2] = 0.95 * abundo;  // CO near oxygen ceiling

    int violations = 0;
    for(int iter = 0; iter < 500; iter++) {
        PackedMetallicity m = pack(p);

        // Moderate CO influx (neighbor has 2x more CO)
        double frac_CO = 0.02;
        // Small C/O fluxes
        double frac_CO2 = 0.005;
        double Dyield_C = frac_CO2 * m.C * p.Mass;
        double Dyield_O = frac_CO2 * m.O * p.Mass;
        double Dyield_trac[TRAC_NUM] = {0, 0, frac_CO * m.trac[2] * p.Mass};

        unpack(p, 0, Dyield_C, Dyield_O, Dyield_trac);

        if(!check_co_ceiling(p, "highZ-C>O", iter))
            violations++;
        if(violations > 5) break;
    }
    CHECK(violations == 0);
}

TEST_CASE("dense gas yn=550000: reproduces crash conditions") {
    // Mimics the crash particle: yn ~ 550000 cm^-3, high H2,
    // CO near ceiling, moderate diffusion flux.
    Particle p;
    p.Mass = 0.5;  // small dense particle
    p.ElementAbundance_H = 0.76;
    p.ElementAbundance_C = 1.4e-4;
    p.ElementAbundance_O = 3.2e-4;
    double abundc = p.ElementAbundance_C / (12.0 * p.ElementAbundance_H);
    double abundo = p.ElementAbundance_O / (16.0 * p.ElementAbundance_H);
    double ceiling = std::min(abundc, abundo);
    p.TracAbund[0] = 0.27;
    p.TracAbund[1] = 1e-14;
    p.TracAbund[2] = 0.9 * ceiling;

    int violations = 0;
    for(int iter = 0; iter < 2000; iter++) {
        PackedMetallicity m = pack(p);

        // Alternating sign to mimic turbulent mixing
        double sign = (iter % 2 == 0) ? 0.03 : -0.01;
        double Dyield_C = sign * m.C * p.Mass;
        double Dyield_O = sign * m.O * p.Mass;
        double Dyield_trac[TRAC_NUM] = {
            0,                          // H2: don't diffuse in single-particle CO test
            0,                          // H+: don't diffuse in single-particle CO test
            sign * m.trac[2] * p.Mass   // CO
        };

        unpack(p, 0, Dyield_C, Dyield_O, Dyield_trac);

        if(!check_co_ceiling(p, "dense-gas", iter))
            violations++;
        if(violations > 5) break;
    }
    CHECK(violations == 0);
}

// ── two-particle diffusion step ────────────────────────────────────
// Mimics the actual diffusion solver: pack both particles, compute
// Dyield from the gradient (neighbor - self) scaled by a diffusion
// coefficient, apply to both particles (conservatively).

static void diffuse_pair(Particle &a, Particle &b, double kappa)
{
    PackedMetallicity ma = pack(a);
    PackedMetallicity mb = pack(b);

    // Dyield = kappa * (neighbor - self) * Mass  (simplified SPH flux)
    double Dy_H_a = kappa * (mb.H - ma.H) * a.Mass;
    double Dy_C_a = kappa * (mb.C - ma.C) * a.Mass;
    double Dy_O_a = kappa * (mb.O - ma.O) * a.Mass;
    double Dy_H_b = kappa * (ma.H - mb.H) * b.Mass;
    double Dy_C_b = kappa * (ma.C - mb.C) * b.Mass;
    double Dy_O_b = kappa * (ma.O - mb.O) * b.Mass;

    double Dy_trac_a[TRAC_NUM], Dy_trac_b[TRAC_NUM];
    for(int k = 0; k < TRAC_NUM; k++) {
        Dy_trac_a[k] = kappa * (mb.trac[k] - ma.trac[k]) * a.Mass;
        Dy_trac_b[k] = kappa * (ma.trac[k] - mb.trac[k]) * b.Mass;
    }

    unpack(a, Dy_H_a, Dy_C_a, Dy_O_a, Dy_trac_a);
    unpack(b, Dy_H_b, Dy_C_b, Dy_O_b, Dy_trac_b);
}

// Helper: build a consistent particle
static Particle make_particle(double Mass, double X_H,
    double elem_C, double elem_O, double CO_abund, double H2, double Hp)
{
    Particle p;
    p.Mass = Mass;
    p.ElementAbundance_H = X_H;
    p.ElementAbundance_C = elem_C;
    p.ElementAbundance_O = elem_O;
    p.TracAbund[0] = H2;
    p.TracAbund[1] = Hp;
    p.TracAbund[2] = CO_abund;
    return p;
}

// ── two-particle tests: extreme neighbor contrasts ─────────────────

TEST_CASE("two-particle: 100x solar vs primordial, strong diffusion") {
    // High-Z particle next to near-primordial gas.  CO saturated on high-Z side.
    // abundc(hi) = 1.41e-2/(12*0.76) = 1.55e-3, abundo = 3.16e-2/(16*0.76) = 2.6e-3
    Particle hi = make_particle(10.0, 0.76, 1.41e-2, 3.16e-2,
        0.99 * 1.41e-2/(12.0*0.76), 0.4, 1e-8);
    Particle lo = make_particle(10.0, 0.76, 1.41e-6, 3.16e-6,
        1e-10, 0.01, 1e-4);

    int violations = 0;
    for(int iter = 0; iter < 2000; iter++) {
        diffuse_pair(hi, lo, 0.1);  // aggressive diffusion
        if(!check_co_ceiling(hi, "100x-vs-prim HI", iter)) violations++;
        if(!check_co_ceiling(lo, "100x-vs-prim LO", iter)) violations++;
        if(violations > 5) break;
    }
    CHECK(violations == 0);
}

TEST_CASE("two-particle: 1000x solar C-rich vs 0.01x solar, kappa=0.5") {
    // Extreme C-rich (CCSN pocket) next to very metal-poor gas.
    // abundc = 0.141/(12*0.70) = 0.0168, abundo = 3.16e-3/(16*0.70) = 2.82e-4
    // ceiling = abundo.  CO near abundo.
    double X_H = 0.70;
    double elem_C = 1.41e-1;   // 1000x solar
    double elem_O = 3.16e-3;   // 10x solar (O-limited)
    double abundo = elem_O / (16.0 * X_H);
    Particle hi = make_particle(5.0, X_H, elem_C, elem_O,
        0.99 * abundo, 0.3, 1e-10);
    Particle lo = make_particle(5.0, X_H, 1.41e-6, 3.16e-6,
        1e-12, 0.001, 1e-4);

    int violations = 0;
    for(int iter = 0; iter < 2000; iter++) {
        diffuse_pair(hi, lo, 0.5);  // very aggressive
        if(!check_co_ceiling(hi, "1000xC-rich HI", iter)) violations++;
        if(!check_co_ceiling(lo, "1000xC-rich LO", iter)) violations++;
        if(violations > 5) break;
    }
    CHECK(violations == 0);
}

TEST_CASE("two-particle: 500x solar symmetric, CO saturated both sides") {
    // Both particles very metal-rich, CO near ceiling, slightly different.
    // Tests whether diffusion between two high-Z particles preserves ceiling.
    double X_H = 0.72;
    double elem_C_a = 7.0e-2, elem_O_a = 1.5e-1;
    double elem_C_b = 8.0e-2, elem_O_b = 1.2e-1;
    double ceil_a = std::min(elem_C_a/(12*X_H), elem_O_a/(16*X_H));
    double ceil_b = std::min(elem_C_b/(12*X_H), elem_O_b/(16*X_H));
    Particle a = make_particle(20.0, X_H, elem_C_a, elem_O_a,
        0.999 * ceil_a, 0.45, 1e-12);
    Particle b = make_particle(20.0, X_H, elem_C_b, elem_O_b,
        0.999 * ceil_b, 0.40, 1e-10);

    int violations = 0;
    for(int iter = 0; iter < 2000; iter++) {
        diffuse_pair(a, b, 0.2);
        if(!check_co_ceiling(a, "500x-symm A", iter)) violations++;
        if(!check_co_ceiling(b, "500x-symm B", iter)) violations++;
        if(violations > 5) break;
    }
    CHECK(violations == 0);
}

TEST_CASE("two-particle: SN ejecta blob into pristine ISM") {
    // A single SN dumps metals into one particle; neighbor is pristine.
    // Ejecta: ~3 Msun metals into a 100 Msun particle.
    // C yield ~ 0.15 Msun, O yield ~ 0.7 Msun for 15 Msun CCSN
    double X_H = 0.76;
    double M_gas = 100.0;
    double C_frac = 0.15 / M_gas;  // 1.5e-3
    double O_frac = 0.70 / M_gas;  // 7.0e-3
    double abundc = C_frac / (12.0 * X_H);  // 1.64e-4
    double abundo = O_frac / (16.0 * X_H);  // 5.76e-4
    // CO saturated to ceiling (cold dense enriched gas forms CO fast)
    Particle ejecta = make_particle(M_gas, X_H, C_frac, O_frac,
        0.999 * std::min(abundc, abundo), 0.35, 1e-10);
    Particle pristine = make_particle(M_gas, X_H, 1e-7, 3e-7,
        1e-14, 1e-3, 1e-4);

    int violations = 0;
    for(int iter = 0; iter < 5000; iter++) {
        diffuse_pair(ejecta, pristine, 0.05);
        if(!check_co_ceiling(ejecta, "SN-ejecta", iter)) violations++;
        if(!check_co_ceiling(pristine, "SN-pristine", iter)) violations++;
        if(violations > 5) break;
    }
    CHECK(violations == 0);
}

TEST_CASE("two-particle: asymmetric mass, small enriched into large pristine") {
    // Small dense enriched clump diffusing into large diffuse pristine cloud.
    // Mass ratio 1:100.  Strong gradient, asymmetric Dyield.
    double X_H = 0.76;
    Particle small_p = make_particle(1.0, X_H, 5e-2, 1e-1,
        0.95 * 5e-2/(12*X_H), 0.45, 1e-12);
    Particle big_p = make_particle(100.0, X_H, 1e-6, 3e-6,
        1e-12, 0.01, 1e-4);

    int violations = 0;
    for(int iter = 0; iter < 3000; iter++) {
        diffuse_pair(small_p, big_p, 0.3);
        if(!check_co_ceiling(small_p, "asym-small", iter)) violations++;
        if(!check_co_ceiling(big_p, "asym-big", iter)) violations++;
        if(violations > 5) break;
    }
    CHECK(violations == 0);
}

TEST_CASE("two-particle: CO-rich meets CO-poor, same total metals") {
    // Same total C and O, but one has all CO, other has none.
    // Tests whether CO diffusion alone (no free-C gradient) is safe.
    double X_H = 0.76;
    double elem_C = 3e-3, elem_O = 6e-3;
    double abundc = elem_C / (12*X_H);
    Particle co_rich = make_particle(10.0, X_H, elem_C, elem_O,
        0.99 * abundc, 0.4, 1e-10);
    Particle co_poor = make_particle(10.0, X_H, elem_C, elem_O,
        1e-10, 0.4, 1e-10);

    int violations = 0;
    for(int iter = 0; iter < 2000; iter++) {
        diffuse_pair(co_rich, co_poor, 0.2);
        if(!check_co_ceiling(co_rich, "CO-rich", iter)) violations++;
        if(!check_co_ceiling(co_poor, "CO-poor", iter)) violations++;
        if(violations > 5) break;
    }
    CHECK(violations == 0);
}

// ── H+ overshoot tests ────────────────────────────────────────────
// Can metal diffusion push H+ past 1.0 after SN enrichment?
//
// After SN enrichment, X_H drops (0.75 -> 0.45). The diffusion pack
// converts TracAbund to mass fractions using local X_H, and unpack
// converts back using local X_H. If neighbors have different X_H,
// the mass-fraction flux divided by local X_H could overshoot.

static bool check_hp_bound(const Particle &p, const char* label, int iter = -1)
{
    if(p.TracAbund[1] > 1.0 + 1e-12) {
        std::fprintf(stderr,
            "  H+ VIOLATION [%s, iter=%d]: H+=%.10f > 1.0 (X_H=%.4f)\n",
            label, iter, p.TracAbund[1], p.ElementAbundance_H);
        return false;
    }
    return true;
}

TEST_CASE("H+ overshoot: crash particle 3095686 conditions") {
    // Reproduce the exact crash scenario with a single diffusion step.
    //
    // Post-SN enrichment state of particle 3095686:
    //   Mass ~ 8.56 Msun (4 + 4.56 from ejecta)
    //   X_H ~ 0.45 (dropped from 0.75 by metal-rich ejecta)
    //   H+ = 2.1e-6 (cold neutral, not yet ionized)
    //   H2 = 0.074
    //   abundc = 0.0125, abundo = 0.054
    //
    // Neighbor: un-enriched, same SN heated it but no mass dumped
    //   X_H = 0.75 (pristine)
    //   H+ = 1e-6 (also cold neutral — first SN in this run)
    //
    // Question: can diffusion between these two push H+ past 1.0?

    Particle crash_p = make_particle(8.56, 0.45, 3.2e-2, 2.0e-1,
        2e-7,    // CO
        0.074,   // H2
        2.1e-6); // H+ (pre-SN value, from snapshot_036)

    // Neighbor: pristine composition, also cold neutral
    Particle pristine_nbr = make_particle(4.0, 0.75, 2.4e-4, 5.7e-4,
        2e-9,    // CO
        0.074,   // H2
        2.1e-6); // H+ (same cold neutral)

    std::printf("\n    === Single diffusion step: crash conditions ===\n");
    std::printf("    Before: crash_p H+=%.6e (X_H=%.4f), nbr H+=%.6e (X_H=%.4f)\n",
        crash_p.TracAbund[1], crash_p.ElementAbundance_H,
        pristine_nbr.TracAbund[1], pristine_nbr.ElementAbundance_H);

    // Show what happens with pack/unpack
    PackedMetallicity m_crash = pack(crash_p);
    PackedMetallicity m_nbr = pack(pristine_nbr);
    std::printf("    Packed H+ mass frac: crash=%.6e, nbr=%.6e\n",
        m_crash.trac[1], m_nbr.trac[1]);
    std::printf("    Ratio nbr/crash packed = %.4f (X_H ratio = %.4f)\n",
        m_nbr.trac[1] / m_crash.trac[1],
        pristine_nbr.ElementAbundance_H / crash_p.ElementAbundance_H);

    // Run diffusion
    diffuse_pair(crash_p, pristine_nbr, 0.1);

    std::printf("    After 1 step (kappa=0.1): crash_p H+=%.6e, nbr H+=%.6e\n",
        crash_p.TracAbund[1], pristine_nbr.TracAbund[1]);

    // Now try with neighbor already fully ionized (e.g. heated by previous SN)
    Particle crash_p2 = make_particle(8.56, 0.45, 3.2e-2, 2.0e-1,
        2e-7, 0.074, 2.1e-6);
    Particle hot_nbr = make_particle(4.0, 0.75, 2.4e-4, 5.7e-4,
        1e-10, 1e-8, 1.0);  // fully ionized

    std::printf("\n    === With fully ionized neighbor ===\n");
    PackedMetallicity m_crash2 = pack(crash_p2);
    PackedMetallicity m_hot = pack(hot_nbr);
    std::printf("    Packed H+ mass frac: crash=%.6e, hot_nbr=%.6e\n",
        m_crash2.trac[1], m_hot.trac[1]);

    // Single step
    diffuse_pair(crash_p2, hot_nbr, 0.1);
    std::printf("    After 1 step: crash_p H+=%.6e, hot_nbr H+=%.6e\n",
        crash_p2.TracAbund[1], hot_nbr.TracAbund[1]);

    // Many steps — does it converge past 1.0?
    int violations = 0;
    double max_hp = 0;
    for(int iter = 0; iter < 500; iter++) {
        diffuse_pair(crash_p2, hot_nbr, 0.1);
        if(crash_p2.TracAbund[1] > max_hp) max_hp = crash_p2.TracAbund[1];
        if(!check_hp_bound(crash_p2, "crash-ionized", iter)) violations++;
        if(violations > 3) break;
    }
    std::printf("    After 500 steps: crash_p H+=%.10f, max_hp=%.10f\n",
        crash_p2.TracAbund[1], max_hp);
    std::printf("    Violations: %d\n", violations);

    CHECK(violations == 0);
}

TEST_CASE("H+ overshoot: enriched particle, sweep kappa and neighbor H+") {
    // Systematic sweep: enriched particle (X_H=0.45) diffusing with
    // un-enriched neighbor (X_H=0.75) at various H+ levels and kappa.
    double kappas[] = {0.01, 0.05, 0.1, 0.3, 0.5};
    double nbr_hp[] = {0.5, 0.9, 0.99, 0.999, 1.0};

    int total_violations = 0;
    std::printf("\n    === Sweep: kappa x neighbor_H+ ===\n");
    std::printf("    %8s %8s %12s %12s %8s\n",
        "kappa", "nbr_H+", "max_H+", "final_H+", "status");

    for(double kappa : kappas) {
        for(double hp_n : nbr_hp) {
            Particle p = make_particle(8.56, 0.45, 3.2e-2, 2.0e-1,
                2e-7, 0.074, 2.1e-6);
            Particle nbr = make_particle(4.0, 0.75, 2.4e-4, 5.7e-4,
                1e-10, 1e-8, hp_n);

            int v = 0;
            double max_hp = 0;
            for(int iter = 0; iter < 200; iter++) {
                diffuse_pair(p, nbr, kappa);
                if(p.TracAbund[1] > max_hp) max_hp = p.TracAbund[1];
                if(p.TracAbund[1] > 1.0 + 1e-12) v++;
            }
            const char *status = (v > 0) ? "FAIL" : "ok";
            std::printf("    %8.3f %8.4f %12.8f %12.8f %8s\n",
                kappa, hp_n, max_hp, p.TracAbund[1], status);
            total_violations += v;
        }
    }
    CHECK(total_violations == 0);
}

// ── Comprehensive sweep: all constraints (H+, H2, CO) ──────────────
// Sweep a wide range of physical conditions and verify that diffusion
// never violates: H+ <= 1, H2 <= 0.5, CO <= min(abundc, abundo),
// and total H = free_H + H2_locked + HP_locked >= 0.

static bool check_all_bounds(const Particle &p, const char* label, int iter)
{
    bool ok = true;
    double X_H = DMAX(p.ElementAbundance_H, 1e-10);
    double abundc = p.ElementAbundance_C / (12.0 * X_H);
    double abundo = p.ElementAbundance_O / (16.0 * X_H);
    double co_ceil = std::min(abundc, abundo);

    if(p.TracAbund[1] > 1.0 + 1e-10) {
        if(iter <= 3) std::fprintf(stderr, "  H+ > 1: %.8e [%s iter=%d]\n", p.TracAbund[1], label, iter);
        ok = false;
    }
    if(p.TracAbund[0] > 0.5 + 1e-10) {
        if(iter <= 3) std::fprintf(stderr, "  H2 > 0.5: %.8e [%s iter=%d]\n", p.TracAbund[0], label, iter);
        ok = false;
    }
    if(p.TracAbund[2] > co_ceil * (1.0 + 1e-10) && co_ceil > 1e-20) {
        if(iter <= 3) std::fprintf(stderr, "  CO > ceil: CO=%.6e ceil=%.6e [%s iter=%d]\n", p.TracAbund[2], co_ceil, label, iter);
        ok = false;
    }
    if(p.ElementAbundance_H < -1e-15) {
        if(iter <= 3) std::fprintf(stderr, "  X_H < 0: %.8e [%s iter=%d]\n", p.ElementAbundance_H, label, iter);
        ok = false;
    }
    if(p.TracAbund[0] < -1e-15 || p.TracAbund[1] < -1e-15 || p.TracAbund[2] < -1e-15) {
        if(iter <= 3) std::fprintf(stderr, "  Negative TracAbund [%s iter=%d]\n", label, iter);
        ok = false;
    }
    return ok;
}

TEST_CASE("comprehensive sweep: H+, H2, CO constraints across conditions") {
    // Sweep: X_H contrast, H+ levels, H2 levels, metallicities, kappa, mass ratios
    // All pair-wise diffusion, 200 iterations each

    struct TestCase {
        const char *name;
        double M_a, XH_a, XC_a, XO_a, CO_a, H2_a, HP_a;
        double M_b, XH_b, XC_b, XO_b, CO_b, H2_b, HP_b;
        double kappa;
    };

    // Helper to set CO near ceiling
    #define NEAR_CEIL(XC, XO, XH, f) (f * std::min((XC)/(12.0*(XH)), (XO)/(16.0*(XH))))

    TestCase cases[] = {
        // SN-enriched vs pristine (the crash scenario)
        {"SN-enrich/pristine neutral",
         8.56, 0.45, 3.2e-2, 2.0e-1, 2e-7, 0.074, 2.1e-6,
         4.0, 0.75, 2.4e-4, 5.7e-4, 2e-9, 0.074, 2.1e-6, 0.1},

        // SN-enriched vs fully ionized
        {"SN-enrich/ionized",
         8.56, 0.45, 3.2e-2, 2.0e-1, 2e-7, 0.074, 2.1e-6,
         4.0, 0.75, 2.4e-4, 5.7e-4, 1e-10, 1e-8, 1.0, 0.1},

        // Extreme X_H contrast, strong diffusion
        {"extreme XH contrast",
         4.0, 0.20, 1e-1, 3e-1, 1e-6, 0.01, 1e-4,
         4.0, 0.75, 2.4e-4, 5.7e-4, 1e-10, 0.3, 0.5, 0.3},

        // Both enriched, different ionization
        {"both enriched, diff ionization",
         8.0, 0.40, 5e-2, 1.5e-1, 1e-5, 0.2, 0.001,
         6.0, 0.50, 3e-2, 8e-2, 5e-6, 0.01, 0.9, 0.2},

        // High H2 vs low H2
        {"high H2 vs low H2",
         4.0, 0.76, 2.4e-4, 5.7e-4, NEAR_CEIL(2.4e-4,5.7e-4,0.76,0.9), 0.45, 1e-6,
         4.0, 0.76, 2.4e-4, 5.7e-4, 1e-10, 0.001, 1e-4, 0.1},

        // 100x solar vs 0.01x solar
        {"100x vs 0.01x solar",
         10.0, 0.60, 2.4e-2, 5.7e-2, NEAR_CEIL(2.4e-2,5.7e-2,0.60,0.5), 0.1, 0.01,
         10.0, 0.76, 2.4e-6, 5.7e-6, 1e-12, 0.3, 1e-6, 0.2},

        // Mass asymmetry 1:20
        {"mass asymmetry 1:20",
         1.0, 0.45, 5e-2, 1e-1, 1e-5, 0.05, 0.5,
         20.0, 0.75, 2.4e-4, 5.7e-4, 1e-10, 0.3, 1e-6, 0.1},

        // Both fully ionized, different metals
        {"both ionized, diff Z",
         4.0, 0.45, 5e-2, 1.5e-1, 1e-8, 1e-8, 0.999,
         4.0, 0.75, 2.4e-4, 5.7e-4, 1e-10, 1e-8, 0.999, 0.3},

        // CO saturated on both sides, different X_H
        {"CO saturated, diff XH",
         4.0, 0.50, 3e-3, 6e-3, NEAR_CEIL(3e-3,6e-3,0.50,0.99), 0.2, 1e-4,
         4.0, 0.76, 3e-3, 6e-3, NEAR_CEIL(3e-3,6e-3,0.76,0.99), 0.2, 1e-4, 0.2},

        // Very aggressive diffusion
        {"aggressive kappa=0.5",
         4.0, 0.30, 8e-2, 2e-1, 1e-6, 0.1, 0.01,
         4.0, 0.76, 2.4e-4, 5.7e-4, 1e-10, 0.001, 0.99, 0.5},

        // Nearly all H in H2 (molecular cloud)
        {"molecular cloud",
         4.0, 0.76, 2.4e-4, 5.7e-4, NEAR_CEIL(2.4e-4,5.7e-4,0.76,0.8), 0.499, 1e-8,
         4.0, 0.76, 2.4e-4, 5.7e-4, 1e-10, 0.001, 0.5, 0.1},

        // PISN ejecta: extreme metal enrichment
        {"PISN ejecta",
         4.0+4.69, 0.20, 6e-2, 3.9e-1, 1e-8, 0.01, 1e-5,
         4.0, 0.76, 2.4e-4, 5.7e-4, 2e-9, 0.1, 1e-4, 0.1},
    };

    int n_cases = sizeof(cases) / sizeof(cases[0]);
    int total_violations = 0;

    for(int c = 0; c < n_cases; c++) {
        TestCase &tc = cases[c];
        Particle a = make_particle(tc.M_a, tc.XH_a, tc.XC_a, tc.XO_a, tc.CO_a, tc.H2_a, tc.HP_a);
        Particle b = make_particle(tc.M_b, tc.XH_b, tc.XC_b, tc.XO_b, tc.CO_b, tc.H2_b, tc.HP_b);

        int v = 0;
        for(int iter = 0; iter < 200; iter++) {
            diffuse_pair(a, b, tc.kappa);
            if(!check_all_bounds(a, tc.name, iter)) v++;
            if(!check_all_bounds(b, tc.name, iter)) v++;
            if(v > 5) break;
        }
        if(v > 0)
            std::printf("    FAIL: %s (%d violations)\n", tc.name, v);
        else
            std::printf("    ok:   %s\n", tc.name);
        total_violations += v;
    }

    std::printf("    Total: %d cases, %d violations\n", n_cases, total_violations);
    CHECK(total_violations == 0);
}

TEST_MAIN()
