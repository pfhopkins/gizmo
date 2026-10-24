// Ketju-integrator tests: GIZMO's additions to the library (contact collisions of non-PN particles, the external
// tide, the collision_radius initialisation).
// Copyright (C) 2026 Ulrich Steinwandel.
// Licensed under the GPLv3 license.
// See LICENSE.md or <https://www.gnu.org/licenses/> for details.

#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <mpi.h>
#include <vector>

#include "ketju_integrator/internal.h"
#include "ketju_integrator/ketju_integrator.h"

namespace {

// Newtonian energy, plus -1/2 sum m x.T.x of the external tide a = T x if T is given (xx, yy, zz, xy, yz, xz)
double energy(const struct ketju_system *R, const double *t = nullptr) {
    const auto *ps = R->physical_state;
    const double G = R->constants->G;
    double E = 0;
    for (int i = 0; i < R->num_particles; ++i) {
        E += 0.5 * ps->mass[i] * (ps->vel[i][0] * ps->vel[i][0] + ps->vel[i][1] * ps->vel[i][1] + ps->vel[i][2] * ps->vel[i][2]);
        for (int j = i + 1; j < R->num_particles; ++j) {
            double r2 = 0;
            for (int k = 0; k < 3; ++k) r2 += (ps->pos[i][k] - ps->pos[j][k]) * (ps->pos[i][k] - ps->pos[j][k]);
            E -= G * ps->mass[i] * ps->mass[j] / std::sqrt(r2);
        }
        if (t) {
            const double T[3][3] = {{t[0], t[3], t[5]}, {t[3], t[1], t[4]}, {t[5], t[4], t[2]}};
            for (int a = 0; a < 3; ++a)
                for (int b = 0; b < 3; ++b) E -= 0.5 * ps->mass[i] * ps->pos[i][a] * T[a][b] * ps->pos[i][b];
        }
    }
    return E;
}

void setup_stars(struct ketju_system *R, int n) {
    ketju_create_system(R, 0, n, MPI_COMM_WORLD);
    R->constants->G = 1;
    R->constants->c = 1;
    R->options->PN_flags = KETJU_PN_NONE;
    R->options->gbs_relative_tolerance = 1e-12;
    R->options->output_time_relative_tolerance = 1e-12;
}

// Relative two-body state with pericentre rp, at separation r0 approaching, total mass M (G = 1), speed v0 at r0
void flyby_state(double M, double rp, double r0, double v0, double x[3], double v[3]) {
    const double vp = std::sqrt(v0 * v0 + 2 * M * (1 / rp - 1 / r0)), h = rp * vp;
    const double vt = h / r0, vr = -std::sqrt(v0 * v0 - vt * vt);
    x[0] = r0, x[1] = 0, x[2] = 0;
    v[0] = vr, v[1] = vt, v[2] = 0;
}

} // namespace

// aa8b3bdc: ketju_create_system sets collision_radius, whatever the struct held before (the original unit tests
// declare it on the stack and segfaulted in merge_bhs without this)
TEST(Gizmo_collision_radius, null_after_create) {
    struct ketju_system R;
    std::memset(&R, 0xA5, sizeof(R));
    ketju_create_system(&R, 2, 1, MPI_COMM_WORLD);
    ASSERT_EQ(R.collision_radius, nullptr);
    ketju_free_system(&R);
}

// b1fa9895: two stars falling together from rest merge on contact (r < R1 + R2) into one at their CoM; mass and
// momentum are conserved and the energy change is booked in dE_collision
TEST(Gizmo_star_collisions, head_on_contact) {
    for (int with_radius = 1; with_radius >= 0; --with_radius) {
        struct ketju_system R;
        setup_stars(&R, 2);
        std::vector<double> radius = {0.5, 0.5};
        if (with_radius) R.collision_radius = radius.data();
        const double m[2] = {0.5, 1.0}, V = 0.1, r0 = 10;
        for (int i = 0; i < 2; ++i) {
            R.physical_state->mass[i] = m[i];
            R.physical_state->pos[i][0] = (i ? -m[0] : m[1]) / 1.5 * r0;
            R.physical_state->vel[i][1] = V;
        }
        const double E0 = energy(&R), t_end = 50; // free fall from 10 to 0: (pi/2) sqrt(r0^3 / (2 G M)) = 28.7
        ketju_run_integrator(&R, t_end);
        if (!with_radius) { // feature off: no merger, the stars pass through each other
            EXPECT_EQ(R.num_particles, 2);
            EXPECT_EQ(R.num_mergers, 0);
            ketju_free_system(&R);
            continue;
        }
        ASSERT_EQ(R.num_particles, 1);
        ASSERT_EQ(R.num_mergers, 1);
        const auto &mi = R.merger_infos[0];
        EXPECT_EQ(mi.is_collision, 1);
        EXPECT_EQ(mi.m1, 1.0); // the heavier star survives
        EXPECT_EQ(mi.m2, 0.5);
        EXPECT_EQ(R.physical_state->mass[0], 1.5);
        // contact time of the radial Kepler orbit, x = (R1 + R2) / r0
        const double x = 1 / r0, t_contact = std::sqrt(r0 * r0 * r0 / (2 * 1.5)) * (std::sqrt(x * (1 - x)) + std::acos(std::sqrt(x)));
        EXPECT_GE(mi.t_merger, t_contact * (1 - 1e-9));
        EXPECT_LE(mi.t_merger, 0.5 * M_PI * std::sqrt(r0 * r0 * r0 / (2 * 1.5)) * (1 + 1e-9)); // before r = 0
        // CoM motion: x = V t along y, at rest along x
        EXPECT_NEAR(R.physical_state->pos[0][0], 0, 1e-10);
        EXPECT_NEAR(R.physical_state->pos[0][1], V * t_end, 1e-10);
        EXPECT_NEAR(R.physical_state->vel[0][0], 0, 1e-12);
        EXPECT_NEAR(R.physical_state->vel[0][1], V, 1e-12);
        EXPECT_EQ(R.collision_radius[0], 0.5);
        // energy: E_end = E0 + dE_collision (= the product's kinetic energy 1/2 M V^2)
        EXPECT_NEAR(energy(&R), E0 + mi.dE_collision, 1e-10 * std::abs(E0));
        EXPECT_NEAR(energy(&R), 0.5 * 1.5 * V * V, 1e-12);
        ketju_free_system(&R);
    }
}

// b1fa9895: a flyby whose two-body pericentre is 0.8 (R1 + R2) collides even if no accepted step lands inside contact
// (pericentre passage test); 1.25 (R1 + R2) misses. A bound third star stays: E_end = E0 + dE_collision.
TEST(Gizmo_star_collisions, graze_and_miss) {
    for (double f : {0.8, 1.25}) {
        struct ketju_system R;
        setup_stars(&R, 3);
        std::vector<double> radius = {0.05, 0.05, 0.05};
        R.collision_radius = radius.data();
        double x[3], v[3];
        flyby_state(2, f * 0.1, 20, 0.6, x, v);
        for (int k = 0; k < 3; ++k) {
            R.physical_state->pos[0][k] = 0.5 * x[k];
            R.physical_state->pos[1][k] = -0.5 * x[k];
            R.physical_state->vel[0][k] = 0.5 * v[k];
            R.physical_state->vel[1][k] = -0.5 * v[k];
        }
        R.physical_state->mass[0] = R.physical_state->mass[1] = 1;
        R.physical_state->mass[2] = 0.1;
        R.physical_state->pos[2][1] = 200;               // third star, slowly bound to the pair
        R.physical_state->vel[2][0] = 0.05;
        const double E0 = energy(&R);
        ketju_run_integrator(&R, 80);                       // pericentre at t ~ 20 / 0.6 ... 30
        if (f > 1) {
            EXPECT_EQ(R.num_mergers, 0);
            EXPECT_EQ(R.num_particles, 3);
            EXPECT_NEAR(energy(&R), E0, 1e-9 * std::abs(E0));
        } else {
            ASSERT_EQ(R.num_mergers, 1);
            EXPECT_EQ(R.num_particles, 2);
            EXPECT_EQ(R.merger_infos[0].is_collision, 1);
            EXPECT_EQ(R.physical_state->mass[0], 2.0);
            EXPECT_NEAR(energy(&R), E0 + R.merger_infos[0].dE_collision, 1e-9 * std::abs(E0));
        }
        ketju_free_system(&R);
    }
}

// 33a286a2: the booked energy change of a collision is minus the pair's internal energy plus the change of its
// potential energy with the other particles as it moves to its CoM; it equals the change of the total energy exactly
TEST(Gizmo_collision_energy, includes_other_members_potential) {
    struct ketju_system R;
    setup_stars(&R, 3);
    std::vector<double> radius = {0.5, 0.5, 0.5};
    R.collision_radius = radius.data();
    const double m[3] = {1.0, 3.0, 2.0};
    const double x[3][3] = {{0.3, 0.1, 0}, {-0.4, 0.0, 0.1}, {2.0, 1.5, -0.5}};
    const double v[3][3] = {{0.2, -0.5, 0.1}, {-0.3, 0.4, 0}, {0.05, 0, 0.1}};
    for (int i = 0; i < 3; ++i) {
        R.physical_state->mass[i] = m[i];
        for (int k = 0; k < 3; ++k) R.physical_state->pos[i][k] = x[i][k], R.physical_state->vel[i][k] = v[i][k];
    }
    // the pair's internal energy and its potential-energy change with the third star (computed here independently)
    double r01 = 0, v01 = 0, d0 = 0, d1 = 0, dc = 0;
    for (int k = 0; k < 3; ++k) {
        const double xc = (m[0] * x[0][k] + m[1] * x[1][k]) / (m[0] + m[1]);
        r01 += (x[0][k] - x[1][k]) * (x[0][k] - x[1][k]);
        v01 += (v[0][k] - v[1][k]) * (v[0][k] - v[1][k]);
        d0 += (x[2][k] - x[0][k]) * (x[2][k] - x[0][k]);
        d1 += (x[2][k] - x[1][k]) * (x[2][k] - x[1][k]);
        dc += (x[2][k] - xc) * (x[2][k] - xc);
    }
    const double E_pair = 0.5 * m[0] * m[1] / (m[0] + m[1]) * v01 - m[0] * m[1] / std::sqrt(r01);
    const double dE_others = m[2] * (m[0] / std::sqrt(d0) + m[1] / std::sqrt(d1) - (m[0] + m[1]) / std::sqrt(dc));
    const double E0 = energy(&R);
    ASSERT_EQ(ketju_do_star_collisions(&R, nullptr), 1); // r = 0.71 < R1 + R2 = 1
    ASSERT_EQ(R.num_particles, 2);
    const double dE = R.merger_infos[0].dE_collision;
    EXPECT_NEAR(energy(&R) - E0, dE, 1e-13 * std::abs(E0));
    EXPECT_NEAR(dE, -E_pair + dE_others, 1e-13 * std::abs(E0));
    // negative check: the pair term alone (before 33a286a2) misses the total-energy change by dE_others
    EXPECT_GT(std::abs(dE_others), 1e-3 * std::abs(E_pair));
    EXPECT_GT(std::abs((energy(&R) - E0) - (-E_pair)), 1e8 * 1e-13 * std::abs(E0));
    ketju_free_system(&R);
}

// 04659d40: a sign change of r.v counts as a pericentre passage only if the osculating orbit's pericentre time lies
// since the last check. Bound pair e = 0.99, a = 1, G M = 2: rp = 0.01 < R1 + R2 = 0.02.
TEST(Gizmo_pericentre_guard, osculating_pericentre_time) {
    const double M = 2, a = 1, e = 0.99, p = a * (1 - e * e), rc = 0.02;
    auto run = [&](double r, double dt_check) {
        struct ketju_system R;
        setup_stars(&R, 2);
        std::vector<double> radius = {0.5 * rc, 0.5 * rc};
        R.collision_radius = radius.data();
        const double cf = (p / r - 1) / e, f = std::acos(cf); // true anomaly after pericentre (r.v > 0)
        const double vr = std::sqrt(M / p) * e * std::sin(f), vt = std::sqrt(M / p) * (1 + e * cf);
        const double xr[3] = {r * cf, r * std::sin(f), 0};
        const double vrel[3] = {vr * cf - vt * std::sin(f), vr * std::sin(f) + vt * cf, 0};
        for (int k = 0; k < 3; ++k) {
            R.physical_state->pos[0][k] = 0.5 * xr[k], R.physical_state->pos[1][k] = -0.5 * xr[k];
            R.physical_state->vel[0][k] = 0.5 * vrel[k], R.physical_state->vel[1][k] = -0.5 * vrel[k];
        }
        R.physical_state->mass[0] = R.physical_state->mass[1] = 1;
        R.physical_state->time = 10;
        double rv_prev[5] = {0, -1e-3, 0, 0, 10 - dt_check}; // approaching at the last check, dt_check ago
        const int n = ketju_do_star_collisions(&R, rv_prev);
        ketju_free_system(&R);
        return n;
    };
    // time since pericentre at r (elliptic Kepler equation)
    auto t_peri = [&](double r) {
        const double E = std::acos((1 - r / a) / e);
        return (E - e * std::sin(E)) * std::sqrt(a * a * a / M);
    };
    const double r1 = 0.05;
    EXPECT_EQ(run(r1, 2.0 * t_peri(r1)), 1);  // pericentre since the last check: collision
    EXPECT_EQ(run(r1, 0.5 * t_peri(r1)), 0);  // pericentre before the last check: none
    const double r2 = 1.9;                    // near apocentre: r.v > 0 after a perturbed flip, pericentre P/2 ago
    EXPECT_EQ(run(r2, 0.01), 0);
}

// 04659d40 / 50438568: hierarchical triple (the K-stack K3 IC of test/ketju_star_collision: three 1 Msun stars, a 50 AU
// e = 0.2 binary and a third star at ~1200 AU, radii 1 Rsun) integrated in host-sized calls: no pair comes near contact,
// so no collision; the energy is conserved. Positive control only: without the guard MSTAR alone shows no false collision
// here either (the GIZMO case needs the host's kicks between calls); osculating_pericentre_time is the negative check.
TEST(Gizmo_pericentre_guard, k3_triple_no_false_collision) {
    const double AU_per_kpc = 2.0626480624709636e8, G = 887.1287; // AU (km/s)^2 / Msun; time unit AU / (km/s)
    const double X[3] = {2.0846989099999994e-06, 1.7938106899999997e-06, -3.8785095999999995e-06};
    const double V[3][3] = {{0, 2.8741549917192337, 0.038696562636978307}, {0, -1.9895475219637486, 0.038696562636978307},
                            {0, -0.88460746975548465, -0.077393125273956601}};
    struct ketju_system R;
    setup_stars(&R, 3);
    R.constants->G = G;
    std::vector<double> radius(3, 0.0046504673);           // 1 Rsun in AU
    R.collision_radius = radius.data();
    for (int i = 0; i < 3; ++i) {
        R.physical_state->mass[i] = 1;
        R.physical_state->pos[i][0] = X[i] * AU_per_kpc;
        for (int k = 0; k < 3; ++k) R.physical_state->vel[i][k] = V[i][k];
    }
    const double E0 = energy(&R);
    const double dt_host = 1.25e-7 * 978.46e6 / 4.7405; // GIZMO's MaxSizeTimestep 1.25e-7 kpc/(km/s), in AU/(km/s)
    for (int s = 0; s < 400; ++s) {                      // ~50 kyr: ~2 outer orbits, ~200 inner ones
        ketju_run_integrator(&R, dt_host);
        ASSERT_EQ(R.num_mergers, 0) << "false collision in call " << s;
    }
    EXPECT_EQ(R.num_particles, 3);
    EXPECT_NEAR(energy(&R), E0, 1e-8 * std::abs(E0));
    ketju_free_system(&R);
}

// a3eafd95, efcfb032: linear external tide a = T x. With T = -w^2 I the CoM of a binary moves exactly as
// X = X0 cos(w t); the energy including -1/2 sum m x.T.x is conserved (the tide's work enters B, not U)
TEST(Gizmo_external_tide, harmonic_com_and_energy) {
    const double w = 0.01, X0 = 1, t_end = 0.5 * M_PI / w * 0.6; // ~94: ~21 binary orbits
    double steps_off = 0;
    for (int tide = 0; tide <= 1; ++tide) {
        struct ketju_system R;
        setup_stars(&R, 2);
        R.options->enable_external_tide = tide;
        const double T[6] = {-w * w, -w * w, -w * w, 0, 0, 0};
        for (int k = 0; k < 6; ++k) R.options->external_tidal_tensor[k] = T[k];
        for (int i = 0; i < 2; ++i) { // circular binary, a = 1, G M = 2, CoM at X0 at rest
            R.physical_state->mass[i] = 1;
            R.physical_state->pos[i][0] = X0 + (i ? -0.5 : 0.5);
            R.physical_state->vel[i][1] = i ? -std::sqrt(0.5) : std::sqrt(0.5);
        }
        const double E0 = energy(&R, T), EN0 = energy(&R);
        ketju_run_integrator(&R, t_end);
        const auto *ps = R.physical_state;
        double Xc[3], Vc[3];
        for (int k = 0; k < 3; ++k) {
            Xc[k] = 0.5 * (ps->pos[0][k] + ps->pos[1][k]);
            Vc[k] = 0.5 * (ps->vel[0][k] + ps->vel[1][k]);
        }
        if (!tide) { // feature off: the CoM stays at X0, the Newtonian energy is conserved
            EXPECT_NEAR(Xc[0], X0, 1e-12);
            EXPECT_NEAR(energy(&R), EN0, 1e-10 * std::abs(EN0));
            steps_off = R.perf->successful_steps;
            ketju_free_system(&R);
            continue;
        }
        EXPECT_NEAR(Xc[0], X0 * std::cos(w * t_end), 1e-10);
        EXPECT_NEAR(Vc[0], -X0 * w * std::sin(w * t_end), 1e-12);
        EXPECT_NEAR(Xc[1], 0, 1e-10);
        EXPECT_NEAR(energy(&R, T), E0, 1e-10 * std::abs(E0));
        // negative check: the tide does work on the system (the energy without its potential changes)
        EXPECT_GT(std::abs(energy(&R) - EN0), 1e4 * 1e-10 * std::abs(E0));
        // the tide stays out of the time transformation: no step-count blow-up (max_step_count aborts before efcfb032)
        EXPECT_LT(R.perf->successful_steps, 2 * steps_off);
        ketju_free_system(&R);
    }
}
