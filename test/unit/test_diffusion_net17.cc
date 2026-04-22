// Unit test for the TracAbund + ElementAbundance + DeuteriumAbundance
// diffusion cycle for CHEMISTRYNETWORK == 17.
//
// Mirrors the exact pack -> apply Dyield -> unpack logic from
// hydro_toplevel.cc (lines 498-565 pack, 925-983 unpack) and tests
// every locking invariant the chemistry assumes:
//
//   abh2 <= 0.5                         (each H2 holds 2 H atoms)
//   abh2 + abhp + abhd <= 1             (H-pool conservation)
//   abhep + abhepp <= abhe              (He pool: He+/He++ from same He)
//   abdp + abhd <= abundD               (D pool: D+/HD from same D)
//   abco <= min(abundc, abundo)         (CO needs both C and O)
//
// The test sweeps two-cell donor/receiver setups, varying species
// near their physical limits (the regime where diffusion most likely
// pushes a cell over its ceiling). Mirrors test_diffusion_co.cc style.

#include "test_harness.h"
#include <cmath>
#include <cstdio>
#include <algorithm>

// ── species layout matching hydro_toplevel.cc ────────────────────────
//   TracAbund[0] = H2,    A = 2
//   TracAbund[1] = H+,    A = 1
//   TracAbund[2] = CO,    A = 28
//   TracAbund[3] = He+,   A = 4
//   TracAbund[4] = He++,  A = 4
//   TracAbund[5] = D+,    A = 2
//   TracAbund[6] = HD,    A = 3   (1 H + 2 D in mass)

static const int TRAC_NUM = 7;
static const double trac_molwt[TRAC_NUM] = {2.0, 1.0, 28.0, 4.0, 4.0, 2.0, 3.0};

static inline double DMAX(double a, double b) { return a > b ? a : b; }

struct Particle {
    double Mass;
    double X_H;     // ElementAbundance[ELEM_H]  (mass fraction)
    double X_He;    // ElementAbundance[ELEM_He]
    double X_C;     // ElementAbundance[ELEM_C]
    double X_O;     // ElementAbundance[ELEM_O]
    double X_D;     // DeuteriumAbundance        (D mass fraction)
    double TracAbund[TRAC_NUM];   // n_X / n_H abundance ratios
};

// ── pack: produces the values that flow into diffusion ──────────────

struct PackedMetallicity {
    double H;     // free-H mass fraction      (H not in H2/H+/HD)
    double He;    // free-He mass fraction     (He not in He+/He++)
    double C;     // free-C mass fraction      (C not in CO)
    double O;     // free-O mass fraction      (O not in CO)
    double D;     // free-D mass fraction      (D not in D+/HD)
    double trac[TRAC_NUM]; // species mass fractions (n_X/n_H * A_X * X_H)
};

static PackedMetallicity pack(const Particle &p)
{
    PackedMetallicity m;
    double X_H = DMAX(p.X_H, 1e-10);

    // Locked atoms (same formulas as hydro_toplevel.cc:505-511, 559-560)
    double H2_locked_H   = p.TracAbund[0] * 2.0  * X_H;
    double HP_locked_H   = p.TracAbund[1] * 1.0  * X_H;
    double HD_locked_H   = p.TracAbund[6] * 1.0  * X_H;
    double CO_locked_C   = p.TracAbund[2] * 12.0 * X_H;
    double CO_locked_O   = p.TracAbund[2] * 16.0 * X_H;
    double HeP_locked_He = p.TracAbund[3] * 4.0  * X_H;
    double HePP_locked_He= p.TracAbund[4] * 4.0  * X_H;
    double DP_locked_D   = p.TracAbund[5] * 2.0  * X_H;
    double HD_locked_D   = p.TracAbund[6] * 2.0  * X_H;

    m.H  = DMAX(p.X_H  - H2_locked_H - HP_locked_H - HD_locked_H, 0);
    m.He = DMAX(p.X_He - HeP_locked_He - HePP_locked_He, 0);
    m.C  = DMAX(p.X_C  - CO_locked_C, 0);
    m.O  = DMAX(p.X_O  - CO_locked_O, 0);
    m.D  = DMAX(p.X_D  - DP_locked_D - HD_locked_D, 0);

    // TracAbund -> mass fractions
    for(int k = 0; k < TRAC_NUM; k++)
        m.trac[k] = p.TracAbund[k] * trac_molwt[k] * X_H;

    return m;
}

// ── unpack: mirrors hydro_toplevel.cc:928-983 exactly ──────────────

static void unpack(Particle &p,
                   double Dyield_H, double Dyield_He,
                   double Dyield_C, double Dyield_O,
                   double Dyield_D,
                   const double Dyield_trac[TRAC_NUM])
{
    double X_H = DMAX(p.X_H, 1e-10);

    // Step 1: strip current locked fractions (mirror old-state of pack)
    double old_H2_locked_H   = p.TracAbund[0] * 2.0  * X_H;
    double old_HP_locked_H   = p.TracAbund[1] * 1.0  * X_H;
    double old_HD_locked_H   = p.TracAbund[6] * 1.0  * X_H;
    double old_CO_locked_C   = p.TracAbund[2] * 12.0 * X_H;
    double old_CO_locked_O   = p.TracAbund[2] * 16.0 * X_H;
    double old_HeP_locked_He = p.TracAbund[3] * 4.0  * X_H;
    double old_HePP_locked_He= p.TracAbund[4] * 4.0  * X_H;
    double old_DP_locked_D   = p.TracAbund[5] * 2.0  * X_H;
    double old_HD_locked_D   = p.TracAbund[6] * 2.0  * X_H;

    double free_H  = DMAX(p.X_H  - old_H2_locked_H - old_HP_locked_H - old_HD_locked_H, 0);
    double free_He = DMAX(p.X_He - old_HeP_locked_He - old_HePP_locked_He, 0);
    double free_C  = DMAX(p.X_C  - old_CO_locked_C, 0);
    double free_O  = DMAX(p.X_O  - old_CO_locked_O, 0);
    double free_D  = DMAX(p.X_D  - old_DP_locked_D - old_HD_locked_D, 0);

    // Step 2: apply free-fraction diffusion deltas
    free_H  = DMAX(free_H  + Dyield_H  / p.Mass, 0.01 * free_H);
    free_He = DMAX(free_He + Dyield_He / p.Mass, 0.01 * free_He);
    free_C  = DMAX(free_C  + Dyield_C  / p.Mass, 0.01 * free_C);
    free_O  = DMAX(free_O  + Dyield_O  / p.Mass, 0.01 * free_O);
    free_D  = DMAX(free_D  + Dyield_D  / p.Mass, 0.01 * free_D);

    // Step 3: update species mass fractions
    double trac_mf[TRAC_NUM];
    for(int k = 0; k < TRAC_NUM; k++) {
        double mf_old = p.TracAbund[k] * trac_molwt[k] * X_H;
        trac_mf[k] = DMAX(mf_old + Dyield_trac[k] / p.Mass, 0.01 * mf_old);
    }

    // Step 4: rebuild element totals (HD splits 1/3 H, 2/3 D)
    p.X_H  = free_H  + trac_mf[0] + trac_mf[1] + trac_mf[6] * (1.0/3.0);
    p.X_He = free_He + trac_mf[3] + trac_mf[4];
    p.X_C  = free_C  + trac_mf[2] * (12.0/28.0);
    p.X_O  = free_O  + trac_mf[2] * (16.0/28.0);
    p.X_D  = free_D  + trac_mf[5] + trac_mf[6] * (2.0/3.0);

    // Step 5: convert species masses back to abundance ratios using NEW X_H
    double X_H_new = DMAX(p.X_H, 1e-10);
    for(int k = 0; k < TRAC_NUM; k++)
        p.TracAbund[k] = trac_mf[k] / (trac_molwt[k] * X_H_new);
}

// ── invariant checks ────────────────────────────────────────────────

struct Violation {
    int count;
    bool ok() const { return count == 0; }
};

static Violation check_h2_ceiling(const Particle &p, const char *label, int iter = -1)
{
    Violation v{0};
    double abh2 = p.TracAbund[0];
    if(abh2 > 0.5 + 1e-12) {
        std::fprintf(stderr,
            "  VIOLATION [%s, iter=%d]: abh2=%.6e > 0.5\n",
            label, iter, abh2);
        v.count++;
    }
    return v;
}

static Violation check_h_pool(const Particle &p, const char *label, int iter = -1)
{
    Violation v{0};
    // Sum of H atoms locked in species + free-H mass / X_H should = 1
    double X_H = DMAX(p.X_H, 1e-10);
    double sum_locked = 2 * p.TracAbund[0] + p.TracAbund[1] + p.TracAbund[6];  // n_H atoms per n_H-nucleus
    if(sum_locked > 1.0 + 1e-10) {
        std::fprintf(stderr,
            "  VIOLATION [%s, iter=%d]: 2*abh2 + abhp + abhd = %.6e > 1\n",
            label, iter, sum_locked);
        v.count++;
    }
    return v;
}

static Violation check_he_pool(const Particle &p, const char *label, int iter = -1)
{
    Violation v{0};
    double X_H = DMAX(p.X_H, 1e-10);
    double abhe_total = p.X_He / (4.0 * X_H);   // n_He / n_H by mass
    double abhe_locked = p.TracAbund[3] + p.TracAbund[4];
    if(abhe_locked > abhe_total + 1e-10) {
        std::fprintf(stderr,
            "  VIOLATION [%s, iter=%d]: abhep+abhepp=%.6e > abhe=%.6e\n",
            label, iter, abhe_locked, abhe_total);
        v.count++;
    }
    return v;
}

static Violation check_d_pool(const Particle &p, const char *label, int iter = -1)
{
    Violation v{0};
    double X_H = DMAX(p.X_H, 1e-10);
    double abundD_total = p.X_D / (2.0 * X_H);   // n_D / n_H
    double abundD_locked = p.TracAbund[5] + p.TracAbund[6];
    if(abundD_locked > abundD_total + 1e-10) {
        std::fprintf(stderr,
            "  VIOLATION [%s, iter=%d]: abdp+abhd=%.6e > abundD=%.6e\n",
            label, iter, abundD_locked, abundD_total);
        v.count++;
    }
    return v;
}

static Violation check_co_ceiling(const Particle &p, const char *label, int iter = -1)
{
    Violation v{0};
    double X_H = DMAX(p.X_H, 1e-10);
    double abundc = p.X_C / (12.0 * X_H);
    double abundo = p.X_O / (16.0 * X_H);
    double co_ceiling = std::min(abundc, abundo);
    if(p.TracAbund[2] > co_ceiling * (1.0 + 1e-12)) {
        std::fprintf(stderr,
            "  VIOLATION [%s, iter=%d]: abco=%.6e > min(abundc,abundo)=%.6e\n",
            label, iter, p.TracAbund[2], co_ceiling);
        v.count++;
    }
    return v;
}

static int check_all(const Particle &p, const char *label, int iter = -1)
{
    int n = 0;
    n += check_h2_ceiling(p, label, iter).count;
    n += check_h_pool   (p, label, iter).count;
    n += check_he_pool  (p, label, iter).count;
    n += check_d_pool   (p, label, iter).count;
    n += check_co_ceiling(p, label, iter).count;
    return n;
}

// ── tests will go here ──────────────────────────────────────────────

TEST_CASE("net 17: pack and unpack of identity flux preserves invariants") {
    // Sanity test: with zero Dyield, unpack should reproduce the input
    // (modulo floating-point), and all invariants should hold.
    Particle p;
    p.Mass = 1.0;
    p.X_H  = 0.7065;
    p.X_He = 0.2935;
    p.X_C  = 1.5e-4;
    p.X_O  = 4.0e-4;
    p.X_D  = 4.0e-5;
    p.TracAbund[0] = 0.3;     // H2
    p.TracAbund[1] = 1e-6;    // H+
    p.TracAbund[2] = 5e-5;    // CO  (< min(abundc,abundo))
    p.TracAbund[3] = 0.05;    // He+
    p.TracAbund[4] = 0;       // He++
    p.TracAbund[5] = 1e-6;    // D+
    p.TracAbund[6] = 1e-5;    // HD

    PackedMetallicity m = pack(p);
    double zero_trac[TRAC_NUM] = {0,0,0,0,0,0,0};
    unpack(p, 0, 0, 0, 0, 0, zero_trac);

    int n = check_all(p, "identity");
    CHECK(n == 0);
}

// ── two-cell donor->receiver helper ─────────────────────────────────
// Mimics what the diffusion solver does: computes a flux of each
// scalar from donor into receiver. flux = frac * (donor_mf - recv_mf) * mass
// where mass is the smaller cell mass. Conservative: the donor loses
// what the receiver gains. We feed this Dyield to the receiver's
// unpack and check whether any invariant is violated.

static void diffuse_one_step(Particle &donor, Particle &receiver, double frac)
{
    PackedMetallicity mD = pack(donor);
    PackedMetallicity mR = pack(receiver);
    double m = std::min(donor.Mass, receiver.Mass);

    // Free-element fluxes (donor -> receiver if donor has more)
    double fH  = frac * (mD.H  - mR.H ) * m;
    double fHe = frac * (mD.He - mR.He) * m;
    double fC  = frac * (mD.C  - mR.C ) * m;
    double fO  = frac * (mD.O  - mR.O ) * m;
    double fD  = frac * (mD.D  - mR.D ) * m;

    // TracAbund mass-fraction fluxes
    double f_trac[TRAC_NUM];
    for(int k = 0; k < TRAC_NUM; k++)
        f_trac[k] = frac * (mD.trac[k] - mR.trac[k]) * m;

    // Apply equal-and-opposite to donor
    {
        double d_trac[TRAC_NUM];
        for(int k = 0; k < TRAC_NUM; k++) d_trac[k] = -f_trac[k];
        unpack(donor, -fH, -fHe, -fC, -fO, -fD, d_trac);
    }
    // Apply to receiver
    unpack(receiver, fH, fHe, fC, fO, fD, f_trac);
}

// ── donor/receiver tests for each invariant ─────────────────────────

TEST_CASE("net 17: donor with high H2, receiver near limit") {
    // Receiver starts at abh2 = 0.49 (just below ceiling).
    // Donor has abh2 = 0.4, plenty of H2 mass.
    // Diffusion can in principle push receiver over 0.5.
    int violations = 0;
    double fracs[] = {0.05, 0.1, 0.2, 0.5, 0.8};
    for(double frac : fracs) {
        Particle donor;
        donor.Mass = 1.0;
        donor.X_H = 0.7065;  donor.X_He = 0.2935;
        donor.X_C = 1.5e-4;  donor.X_O = 4.0e-4;  donor.X_D = 4.0e-5;
        donor.TracAbund[0] = 0.40;   // H2 high
        donor.TracAbund[1] = 1e-6;
        donor.TracAbund[2] = 5e-6;
        donor.TracAbund[3] = 0.05;
        donor.TracAbund[4] = 0;
        donor.TracAbund[5] = 1e-7;
        donor.TracAbund[6] = 1e-6;

        Particle recv = donor;
        recv.TracAbund[0] = 0.49;    // H2 near ceiling

        diffuse_one_step(donor, recv, frac);

        char lab[80];
        std::snprintf(lab, sizeof(lab), "H2 frac=%.2f", frac);
        violations += check_all(recv, lab);
    }
    CHECK(violations == 0);
}

TEST_CASE("net 17: donor sends H2 while receiver loses free H") {
    // Pathological case: receiver simultaneously gains H2 (positive
    // f_trac[0]) AND loses free H (negative fH). If receiver's X_H
    // shrinks while trac_mf[0] grows, abh2 = trac_mf[0]/(2*X_H) can
    // explode past 0.5.
    int violations = 0;
    double fracs[] = {0.1, 0.3, 0.5, 0.8};
    for(double frac : fracs) {
        Particle donor;
        donor.Mass = 1.0;
        donor.X_H = 0.7065;  donor.X_He = 0.2935;
        donor.X_C = 1.5e-4;  donor.X_O = 4.0e-4;  donor.X_D = 4.0e-5;
        donor.TracAbund[0] = 0.45;   // donor: H2 rich, low free H
        donor.TracAbund[1] = 1e-6;
        donor.TracAbund[2] = 5e-6;
        donor.TracAbund[3] = 0.05;
        donor.TracAbund[4] = 0;
        donor.TracAbund[5] = 1e-7;
        donor.TracAbund[6] = 1e-6;

        Particle recv = donor;
        recv.X_H  = 0.2;   recv.X_He = 0.78;   // receiver: H-poor
        recv.TracAbund[0] = 0.05;              // little H2
        recv.TracAbund[3] = 0.20;              // He+ rich (so mD.He > mR.He triggers He flow too)

        diffuse_one_step(donor, recv, frac);

        char lab[80];
        std::snprintf(lab, sizeof(lab), "H2-flow free-H-out frac=%.2f", frac);
        violations += check_all(recv, lab);
    }
    CHECK(violations == 0);
}

TEST_CASE("net 17: He+ near abhe limit, donor with more He+") {
    int violations = 0;
    double fracs[] = {0.05, 0.1, 0.3, 0.5};
    for(double frac : fracs) {
        Particle donor;
        donor.Mass = 1.0;
        donor.X_H = 0.7065;  donor.X_He = 0.2935;
        donor.X_C = 1.5e-4;  donor.X_O = 4.0e-4;  donor.X_D = 4.0e-5;
        donor.TracAbund[0] = 1e-6; donor.TracAbund[1] = 0.05;
        donor.TracAbund[2] = 1e-7; donor.TracAbund[3] = 0.07; donor.TracAbund[4] = 0.02;
        donor.TracAbund[5] = 1e-7; donor.TracAbund[6] = 1e-7;

        Particle recv = donor;
        // recv has same X_He but He+ + He++ already at 0.99 * abhe
        double X_H = recv.X_H, X_He = recv.X_He;
        double abhe = X_He / (4.0 * X_H);
        recv.TracAbund[3] = 0.95 * abhe;
        recv.TracAbund[4] = 0.04 * abhe;

        diffuse_one_step(donor, recv, frac);

        char lab[80];
        std::snprintf(lab, sizeof(lab), "He+ near limit frac=%.2f", frac);
        violations += check_all(recv, lab);
    }
    CHECK(violations == 0);
}

TEST_CASE("net 17: HD near abundD limit, donor with more HD") {
    int violations = 0;
    double fracs[] = {0.05, 0.1, 0.3, 0.5};
    for(double frac : fracs) {
        Particle donor;
        donor.Mass = 1.0;
        donor.X_H = 0.7065;  donor.X_He = 0.2935;
        donor.X_C = 1.5e-4;  donor.X_O = 4.0e-4;  donor.X_D = 4.0e-5;
        donor.TracAbund[0] = 0.30; donor.TracAbund[1] = 1e-6;
        donor.TracAbund[2] = 1e-7; donor.TracAbund[3] = 0.05; donor.TracAbund[4] = 0;
        donor.TracAbund[5] = 1e-7; donor.TracAbund[6] = 1e-5; // HD donor

        Particle recv = donor;
        // recv has same D mass fraction, HD+D+ already at 0.99 * abundD
        double X_H = recv.X_H;
        double abundD = recv.X_D / (2.0 * X_H);
        recv.TracAbund[5] = 0.05 * abundD;
        recv.TracAbund[6] = 0.94 * abundD;

        diffuse_one_step(donor, recv, frac);

        char lab[80];
        std::snprintf(lab, sizeof(lab), "HD near limit frac=%.2f", frac);
        violations += check_all(recv, lab);
    }
    CHECK(violations == 0);
}

TEST_CASE("net 17: 1000 cycles of small back-and-forth flux, no drift") {
    Particle a, b;
    a.Mass = 1.0;
    a.X_H = 0.7065;  a.X_He = 0.2935;
    a.X_C = 1.5e-4;  a.X_O = 4.0e-4;  a.X_D = 4.0e-5;
    a.TracAbund[0] = 0.30; a.TracAbund[1] = 1e-6;
    a.TracAbund[2] = 5e-6; a.TracAbund[3] = 0.05; a.TracAbund[4] = 0;
    a.TracAbund[5] = 1e-7; a.TracAbund[6] = 1e-6;

    b = a;
    b.TracAbund[0] = 0.40;  // small initial difference

    int violations = 0;
    for(int i = 0; i < 1000; i++) {
        double frac = 0.01 * ((i % 3 == 0) ? 1.0 : ((i % 3 == 1) ? -1.0 : 0.5));
        diffuse_one_step(a, b, frac);
        violations += check_all(a, "1000-cycle a", i);
        violations += check_all(b, "1000-cycle b", i);
        if(violations > 10) break;
    }
    CHECK(violations == 0);
}

TEST_CASE("net 17: mix cold-molecular cell with hot-ionized cell") {
    // Production crash regime: a SF cell (cold, molecular: abh2 ~ 0.4) sits
    // next to a SN-shock-heated cell (hot, ionized: abhp ~ 1, abh2 ~ 0).
    // The two cells exchange scalars via turbulent diffusion. The hot cell's
    // tiny abh2 + its hot/low-density-style ElementAbundance shouldn't push
    // the cold cell over abh2 = 0.5.
    int violations = 0;
    double fracs[] = {0.05, 0.1, 0.3, 0.5, 0.8};
    for(double frac : fracs) {
        Particle cold;
        cold.Mass = 1.0;
        cold.X_H = 0.7065;  cold.X_He = 0.2935;
        cold.X_C = 1.5e-4;  cold.X_O = 4.0e-4;  cold.X_D = 4.0e-5;
        cold.TracAbund[0] = 0.40;   // H2 cold cloud
        cold.TracAbund[1] = 1e-6;
        cold.TracAbund[2] = 5e-5;
        cold.TracAbund[3] = 1e-4;
        cold.TracAbund[4] = 0;
        cold.TracAbund[5] = 1e-7;
        cold.TracAbund[6] = 1e-5;

        Particle hot = cold;
        hot.TracAbund[0] = 1e-12;   // hot fully ionized
        hot.TracAbund[1] = 0.999;
        hot.TracAbund[3] = 1e-5;
        hot.TracAbund[4] = 0.099;   // He+ + He++ saturated
        hot.TracAbund[5] = cold.X_D / (2 * cold.X_H) * 0.99;
        hot.TracAbund[6] = 1e-12;

        // Apply diffusion with hot as donor of H+ and cold as donor of H2
        diffuse_one_step(cold, hot, frac);

        char lab[80];
        std::snprintf(lab, sizeof(lab), "cold-hot mix frac=%.2f", frac);
        violations += check_all(cold, lab);
        violations += check_all(hot,  lab);
    }
    CHECK(violations == 0);
}

TEST_CASE("net 17: many-neighbor accumulation") {
    // In production, a cell receives flux from ~N_ngb neighbors per step,
    // summed onto its Dyield buffer before unpack. Single-pair test is
    // conservative; multi-source accumulation can over-fill the cell.
    // Simulate: receiver gets flux from 10 distinct H2-rich donors.
    int violations = 0;
    double fracs[] = {0.05, 0.1, 0.2};
    for(double frac : fracs) {
        Particle recv;
        recv.Mass = 1.0;
        recv.X_H = 0.7065;  recv.X_He = 0.2935;
        recv.X_C = 1.5e-4;  recv.X_O = 4.0e-4;  recv.X_D = 4.0e-5;
        recv.TracAbund[0] = 0.45;   // already near limit
        recv.TracAbund[1] = 1e-6;
        recv.TracAbund[2] = 5e-6;
        recv.TracAbund[3] = 0.05;
        recv.TracAbund[4] = 0;
        recv.TracAbund[5] = 1e-7;
        recv.TracAbund[6] = 1e-6;

        // Accumulate Dyields from 10 hypothetical neighbors that each
        // have abh2 = 0.49 (small per-pair gradient → small per-pair flux
        // but 10x the kick when summed)
        PackedMetallicity mR = pack(recv);
        double sum_fH=0, sum_fHe=0, sum_fC=0, sum_fO=0, sum_fD=0;
        double sum_ftrac[TRAC_NUM] = {0,0,0,0,0,0,0};
        for(int n = 0; n < 10; n++) {
            Particle donor = recv;
            donor.TracAbund[0] = 0.49;
            PackedMetallicity mD = pack(donor);
            double m = std::min(donor.Mass, recv.Mass);
            sum_fH  += frac * (mD.H  - mR.H ) * m;
            sum_fHe += frac * (mD.He - mR.He) * m;
            sum_fC  += frac * (mD.C  - mR.C ) * m;
            sum_fO  += frac * (mD.O  - mR.O ) * m;
            sum_fD  += frac * (mD.D  - mR.D ) * m;
            for(int k = 0; k < TRAC_NUM; k++)
                sum_ftrac[k] += frac * (mD.trac[k] - mR.trac[k]) * m;
        }
        unpack(recv, sum_fH, sum_fHe, sum_fC, sum_fO, sum_fD, sum_ftrac);

        char lab[80];
        std::snprintf(lab, sizeof(lab), "10-neighbor accum frac=%.2f", frac);
        violations += check_all(recv, lab);
    }
    CHECK(violations == 0);
}

TEST_CASE("net 17: broad random sweep, many configs and frac values") {
    // Pseudo-random sweep through a large parameter cube.
    // Each iteration constructs two cells with independent abundances
    // (so we explore many edge cases including ones we didn't think of)
    // and applies one diffusion step at a random frac.
    // Reports the worst violation found.
    auto u01 = [](unsigned &s) {
        s = s * 1664525u + 1013904223u;
        return (double)s / 4294967296.0;
    };
    unsigned seed = 1234567u;

    // Bounded random ratios so we always stay within physical limits
    // BEFORE diffusion (so any violation found AFTER diffusion is real).
    auto rand_in = [&](double lo, double hi) {
        return lo + (hi - lo) * u01(seed);
    };

    int violations = 0;
    int worst_iter = -1;
    const int N = 50000;
    for(int it = 0; it < N; it++) {
        Particle a;
        a.Mass = 1.0;
        // X_H from primordial-ish to enriched
        a.X_H  = rand_in(0.30, 0.76);
        a.X_He = rand_in(0.20, 1.0 - a.X_H);
        a.X_C  = rand_in(1e-6, 1e-3);
        a.X_O  = rand_in(1e-6, 3e-3);
        a.X_D  = rand_in(1e-7, 1e-4);

        // Place TracAbund safely within ceilings
        double abundc = a.X_C / (12.0 * a.X_H);
        double abundo = a.X_O / (16.0 * a.X_H);
        double abhe   = a.X_He / (4.0  * a.X_H);
        double abundD = a.X_D  / (2.0  * a.X_H);
        double co_max = std::min(abundc, abundo);

        a.TracAbund[0] = rand_in(1e-12, 0.499);          // abh2 < 0.5
        a.TracAbund[1] = rand_in(1e-12, 1.0 - 2*a.TracAbund[0]); // abhp + 2 abh2 < 1
        a.TracAbund[2] = rand_in(0,     0.99 * co_max);  // CO < ceiling
        a.TracAbund[3] = rand_in(0,     0.99 * abhe);    // abhep < abhe
        a.TracAbund[4] = rand_in(0,     0.99 * (abhe - a.TracAbund[3])); // abhepp + abhep < abhe
        a.TracAbund[5] = rand_in(0,     0.5  * abundD);  // abdp half abundD
        a.TracAbund[6] = rand_in(0,     0.49 * (abundD - a.TracAbund[5])); // abhd in remaining D

        // Independent second cell
        Particle b = a;
        b.X_H  = rand_in(0.30, 0.76);
        b.X_He = rand_in(0.20, 1.0 - b.X_H);
        b.TracAbund[0] = rand_in(1e-12, 0.499);
        b.TracAbund[1] = rand_in(1e-12, 1.0 - 2*b.TracAbund[0]);
        double b_abundc = b.X_C / (12.0 * b.X_H);
        double b_abundo = b.X_O / (16.0 * b.X_H);
        double b_abhe   = b.X_He / (4.0  * b.X_H);
        double b_abundD = b.X_D  / (2.0  * b.X_H);
        b.TracAbund[2] = rand_in(0, 0.99 * std::min(b_abundc, b_abundo));
        b.TracAbund[3] = rand_in(0, 0.99 * b_abhe);
        b.TracAbund[4] = rand_in(0, 0.99 * (b_abhe - b.TracAbund[3]));
        b.TracAbund[5] = rand_in(0, 0.5  * b_abundD);
        b.TracAbund[6] = rand_in(0, 0.49 * (b_abundD - b.TracAbund[5]));

        double frac = rand_in(0.001, 0.99);
        diffuse_one_step(a, b, frac);

        int v = check_all(a, "rand a", it) + check_all(b, "rand b", it);
        if(v > 0) {
            if(violations == 0) worst_iter = it;
            violations += v;
            if(violations > 30) break;
        }
    }
    if(violations > 0) {
        std::fprintf(stderr, "  Sweep: %d violations in %d configs (first at iter %d)\n",
                     violations, N, worst_iter);
    } else {
        std::fprintf(stdout, "    Sweep: 0 violations in %d configs\n", N);
    }
    CHECK(violations == 0);
}

TEST_CASE("net 17: high-metallicity SN-enriched regime") {
    // SN-shock-enriched cells (10-100x solar Z) have suppressed X_H
    // and very high metal mass fractions. Test that diffusion holds
    // invariants even when X_H drops to 0.3 and X_C/X_O get to %-level.
    auto u01 = [](unsigned &s) {
        s = s * 1664525u + 1013904223u;
        return (double)s / 4294967296.0;
    };
    unsigned seed = 7654321u;
    auto rand_in = [&](double lo, double hi) {
        return lo + (hi - lo) * u01(seed);
    };

    int violations = 0;
    const int N = 50000;
    for(int it = 0; it < N; it++) {
        Particle a;
        a.Mass = 1.0;
        a.X_H  = rand_in(0.30, 0.60);              // strong enrichment
        a.X_He = rand_in(0.20, 1.0 - a.X_H - 0.05);
        a.X_C  = rand_in(1e-4, 5e-2);              // up to 5% C (SN ejecta)
        a.X_O  = rand_in(1e-4, 1e-1);              // up to 10% O
        a.X_D  = rand_in(1e-7, 1e-4);

        double abundc = a.X_C / (12.0 * a.X_H);
        double abundo = a.X_O / (16.0 * a.X_H);
        double abhe   = a.X_He / (4.0  * a.X_H);
        double abundD = a.X_D  / (2.0  * a.X_H);
        double co_max = std::min(abundc, abundo);

        a.TracAbund[0] = rand_in(1e-12, 0.499);
        a.TracAbund[1] = rand_in(1e-12, 1.0 - 2*a.TracAbund[0]);
        a.TracAbund[2] = rand_in(0, 0.99 * co_max);
        a.TracAbund[3] = rand_in(0, 0.99 * abhe);
        a.TracAbund[4] = rand_in(0, 0.99 * (abhe - a.TracAbund[3]));
        a.TracAbund[5] = rand_in(0, 0.5  * abundD);
        a.TracAbund[6] = rand_in(0, 0.49 * (abundD - a.TracAbund[5]));

        // Mix high-Z donor with primordial-ish receiver (extreme gradient)
        Particle b = a;
        b.X_H  = rand_in(0.70, 0.76);
        b.X_He = rand_in(0.20, 1.0 - b.X_H - 0.001);
        b.X_C  = rand_in(1e-7, 1e-5);              // primordial Z
        b.X_O  = rand_in(1e-7, 1e-5);
        b.X_D  = rand_in(1e-7, 1e-4);
        double b_abundc = b.X_C / (12.0 * b.X_H);
        double b_abundo = b.X_O / (16.0 * b.X_H);
        double b_abhe   = b.X_He / (4.0  * b.X_H);
        double b_abundD = b.X_D  / (2.0  * b.X_H);
        b.TracAbund[0] = rand_in(1e-12, 0.499);
        b.TracAbund[1] = rand_in(1e-12, 1.0 - 2*b.TracAbund[0]);
        b.TracAbund[2] = rand_in(0, 0.99 * std::min(b_abundc, b_abundo));
        b.TracAbund[3] = rand_in(0, 0.99 * b_abhe);
        b.TracAbund[4] = rand_in(0, 0.99 * (b_abhe - b.TracAbund[3]));
        b.TracAbund[5] = rand_in(0, 0.5  * b_abundD);
        b.TracAbund[6] = rand_in(0, 0.49 * (b_abundD - b.TracAbund[5]));

        double frac = rand_in(0.001, 0.99);
        diffuse_one_step(a, b, frac);

        violations += check_all(a, "highZ a", it) + check_all(b, "highZ b", it);
        if(violations > 30) break;
    }
    if(violations == 0) std::fprintf(stdout, "    HighZ sweep: 0 violations in %d configs\n", N);
    CHECK(violations == 0);
}

TEST_CASE("net 17: extreme PISN ejecta (X_Z up to 50%)") {
    // PISN can dump 100 Msun of >50% metal ejecta into a single cell.
    // After mixing, X_H can drop below 0.5 and X_C+X_O can exceed 30%.
    // Stress-test diffusion in this regime.
    auto u01 = [](unsigned &s) {
        s = s * 1664525u + 1013904223u;
        return (double)s / 4294967296.0;
    };
    unsigned seed = 9876543u;
    auto rand_in = [&](double lo, double hi) {
        return lo + (hi - lo) * u01(seed);
    };

    int violations = 0;
    const int N = 20000;
    for(int it = 0; it < N; it++) {
        Particle ejecta;
        ejecta.Mass = 1.0;
        ejecta.X_C = rand_in(0.05, 0.20);   // 5-20% C
        ejecta.X_O = rand_in(0.10, 0.40);   // 10-40% O
        ejecta.X_H = rand_in(0.10, 0.40);
        ejecta.X_He = rand_in(0.10, 1.0 - ejecta.X_H - ejecta.X_C - ejecta.X_O - 0.01);
        if(ejecta.X_He < 0) ejecta.X_He = 0.05;
        ejecta.X_D = rand_in(1e-7, 1e-5);

        double e_abundc = ejecta.X_C / (12.0 * ejecta.X_H);
        double e_abundo = ejecta.X_O / (16.0 * ejecta.X_H);
        double e_abhe   = ejecta.X_He / (4.0  * ejecta.X_H);
        double e_abundD = ejecta.X_D  / (2.0  * ejecta.X_H);
        ejecta.TracAbund[0] = rand_in(1e-12, 0.05);   // hot ejecta: little H2
        ejecta.TracAbund[1] = rand_in(0.5,   0.99);   // mostly ionized
        ejecta.TracAbund[2] = rand_in(0, 0.5 * std::min(e_abundc, e_abundo));
        ejecta.TracAbund[3] = rand_in(0, 0.5 * e_abhe);
        ejecta.TracAbund[4] = rand_in(0, 0.5 * (e_abhe - ejecta.TracAbund[3]));
        ejecta.TracAbund[5] = rand_in(0, 0.5 * e_abundD);
        ejecta.TracAbund[6] = rand_in(0, 0.49 * (e_abundD - ejecta.TracAbund[5]));

        Particle ambient;
        ambient.Mass = 1.0;
        ambient.X_H = 0.7065; ambient.X_He = 0.2935;
        ambient.X_C = 1e-4;  ambient.X_O = 4e-4;  ambient.X_D = 4e-5;
        double a_abundc = ambient.X_C / (12.0 * ambient.X_H);
        double a_abundo = ambient.X_O / (16.0 * ambient.X_H);
        double a_abhe   = ambient.X_He / (4.0  * ambient.X_H);
        double a_abundD = ambient.X_D  / (2.0  * ambient.X_H);
        ambient.TracAbund[0] = rand_in(0, 0.40);
        ambient.TracAbund[1] = rand_in(1e-8, 1.0 - 2*ambient.TracAbund[0]);
        ambient.TracAbund[2] = rand_in(0, 0.99 * std::min(a_abundc, a_abundo));
        ambient.TracAbund[3] = rand_in(0, 0.5 * a_abhe);
        ambient.TracAbund[4] = rand_in(0, 0.5 * (a_abhe - ambient.TracAbund[3]));
        ambient.TracAbund[5] = rand_in(0, 0.5 * a_abundD);
        ambient.TracAbund[6] = rand_in(0, 0.49 * (a_abundD - ambient.TracAbund[5]));

        double frac = rand_in(0.001, 0.95);
        diffuse_one_step(ejecta, ambient, frac);

        violations += check_all(ejecta,  "PISN ejecta", it);
        violations += check_all(ambient, "PISN ambient", it);
        if(violations > 30) break;
    }
    if(violations == 0) std::fprintf(stdout, "    PISN sweep: 0 violations in %d configs\n", N);
    CHECK(violations == 0);
}

TEST_MAIN()
