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
                   double Dyield_C, double Dyield_O,
                   const double Dyield_trac[TRAC_NUM])
{
    double X_H = DMAX(p.ElementAbundance_H, 1e-10);

    // Step 1: strip old CO-locked C/O so we work on free fractions only
    double old_CO_locked_C = p.TracAbund[2] * 12.0 * X_H;
    double old_CO_locked_O = p.TracAbund[2] * 16.0 * X_H;
    double free_C = DMAX(p.ElementAbundance_C - old_CO_locked_C, 0);
    double free_O = DMAX(p.ElementAbundance_O - old_CO_locked_O, 0);

    // Step 2: apply free-C/O diffusion delta
    free_C = DMAX(free_C + Dyield_C / p.Mass, 0.01 * free_C);
    free_O = DMAX(free_O + Dyield_O / p.Mass, 0.01 * free_O);

    // Step 3: update TracAbund with diffusion delta
    for(int k = 0; k < TRAC_NUM; k++) {
        double da = Dyield_trac[k] / (p.Mass * trac_molwt[k] * X_H);
        p.TracAbund[k] = DMAX(p.TracAbund[k] + da, 0.01 * p.TracAbund[k]);
    }

    // Step 4: reconstruct total C/O = free + new CO-locked
    double new_CO_locked_C = p.TracAbund[2] * 12.0 * X_H;
    double new_CO_locked_O = p.TracAbund[2] * 16.0 * X_H;
    p.ElementAbundance_C = free_C + new_CO_locked_C;
    p.ElementAbundance_O = free_O + new_CO_locked_O;
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

        unpack(p, Dyield_C, Dyield_O, Dyield_trac);

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

        unpack(p, Dyield_C, Dyield_O, Dyield_trac);

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

        unpack(p, 0, 0, Dyield_trac);

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
            frac * m.trac[0] * p.Mass,
            frac * m.trac[1] * p.Mass,
            frac * m.trac[2] * p.Mass
        };

        unpack(p, Dyield_C, Dyield_O, Dyield_trac);

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
        PackedMetallicity m = pack(p);
        // Zero Dyield — no actual diffusion.  C should stay constant.
        double Dyield_trac[TRAC_NUM] = {0, 0, 0};
        unpack(p, 0, 0, Dyield_trac);
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

        unpack(p, Dyield_C, Dyield_O, Dyield_trac);

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
            sign * m.trac[0] * p.Mass,
            sign * m.trac[1] * p.Mass,
            sign * m.trac[2] * p.Mass
        };

        unpack(p, Dyield_C, Dyield_O, Dyield_trac);

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
    double Dy_C_a = kappa * (mb.C - ma.C) * a.Mass;
    double Dy_O_a = kappa * (mb.O - ma.O) * a.Mass;
    double Dy_C_b = kappa * (ma.C - mb.C) * b.Mass;
    double Dy_O_b = kappa * (ma.O - mb.O) * b.Mass;

    double Dy_trac_a[TRAC_NUM], Dy_trac_b[TRAC_NUM];
    for(int k = 0; k < TRAC_NUM; k++) {
        Dy_trac_a[k] = kappa * (mb.trac[k] - ma.trac[k]) * a.Mass;
        Dy_trac_b[k] = kappa * (ma.trac[k] - mb.trac[k]) * b.Mass;
    }

    unpack(a, Dy_C_a, Dy_O_a, Dy_trac_a);
    unpack(b, Dy_C_b, Dy_O_b, Dy_trac_b);
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

TEST_MAIN()
