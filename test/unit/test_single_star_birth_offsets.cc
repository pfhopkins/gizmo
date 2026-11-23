// Unit test: placement of stars born from one parent in one call (src/galaxy_sf/single_star_birth_offsets.h). Stars born together
// used to keep the parent's position bitwise, so a snapshot restart soon after such a birth stopped on coincident positions (GKS-TI18).
// Bars per group (n = 2..64 stars, masses 0.08..150 Msun, 200 groups): mass-weighted mean offset <= 1e-14 h (the group's centre of
// mass stays at the parent's position); every offset at most 2 SINGLE_STAR_BIRTH_OFFSET_FRACTION h; every pair of stars at least
// 1e-6 h apart (the start-up check flags pairs below ~1e-4 of the local scale); the offsets are a function of the stars' own
// uniforms only (same input, same output).
#include "test_harness.h"
#include "src/galaxy_sf/single_star_birth_offsets.h"
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

TEST_CASE("groups keep their centre of mass, stars sit apart, offsets are small")
{
    std::mt19937_64 g(20261123);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    const double h = 0.37;
    double worst_com = 0, worst_min = 1e300, worst_lo = 1e300, worst_hi = 0;
    for(int grp = 0; grp < 200; grp++) {
        int n = 2 + (int)(U(g) * 63);
        std::vector<double> m(n), u(2 * n), d(3 * n), d2(3 * n);
        double M = 0;
        for(int k = 0; k < n; k++) {m[k] = 0.08 * pow(150. / 0.08, U(g)); M += m[k]; u[2*k] = U(g); u[2*k+1] = U(g);}
        single_star_birth_offsets(n, m.data(), u.data(), h, d.data());
        single_star_birth_offsets(n, m.data(), u.data(), h, d2.data());
        for(int j = 0; j < 3 * n; j++) {CHECK(d[j] == d2[j]);}
        for(int c = 0; c < 3; c++) {double s = 0; for(int k = 0; k < n; k++) {s += m[k] * d[3*k+c];} worst_com = fmax(worst_com, fabs(s) / (M * h));}
        for(int k = 0; k < n; k++) {
            double r = sqrt(d[3*k]*d[3*k] + d[3*k+1]*d[3*k+1] + d[3*k+2]*d[3*k+2]) / h;
            worst_lo = fmin(worst_lo, r); worst_hi = fmax(worst_hi, r);
            for(int l = 0; l < k; l++) {
                double dx = d[3*k]-d[3*l], dy = d[3*k+1]-d[3*l+1], dz = d[3*k+2]-d[3*l+2];
                worst_min = fmin(worst_min, sqrt(dx*dx + dy*dy + dz*dz) / h);
            }
        }
    }
    printf("  centre of mass %.3e h, offsets %.4f..%.4f h, closest pair %.4e h\n", worst_com, worst_lo, worst_hi, worst_min);
    CHECK(worst_com <= 1e-14);
    CHECK(worst_hi <= 2.0 * SINGLE_STAR_BIRTH_OFFSET_FRACTION);
    CHECK(worst_min >= 1e-6);
}

TEST_MAIN()
