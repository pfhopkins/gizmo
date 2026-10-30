// Unit tests for the shared position -> integer coordinate mapping (src/domain/domain_key_functions.h).
//
// The host decomposition used to key with the expression
//     domain_double_to_int(((Pos - DomainCorner) / DomainLen) + 1.0)
// written out at each site, and the device tree build with its own copy plus a clamp.  The shared
// routine has to reproduce the host expression bit for bit for every position inside the extent
// (so existing runs keep their keys), and to pin the clamp semantics the device relied on.
#include "test_harness.h"
#include "../../src/domain/domain_key_functions.h"

#include <cstdint>
#include <cmath>
#include <limits>

static constexpr int BITS = 42;

// The retired host expression, verbatim.
static uint64_t legacy_double_to_int(double d) {
    union { double d; unsigned long long ull; } u;
    u.d = d;
    return (uint64_t)((u.ull & 0xFFFFFFFFFFFFFllu) >> (52 - BITS));
}
static uint64_t legacy_coordinate(double pos, double corner, double len) {
    return legacy_double_to_int(((pos - corner) / len) + 1.0);
}

// Deterministic 64-bit generator (splitmix64), so a failure is reproducible.
static uint64_t rng_state = 0x9E3779B97F4A7C15ull;
static uint64_t next_u64() {
    uint64_t z = (rng_state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
static double uniform01() { return (double)(next_u64() >> 11) * 0x1.0p-53; }

TEST_CASE("in-extent positions reproduce the legacy host expression bit for bit") {
    // Extents spanning the regimes the code sees: unit box, a parsec-scale STARFORGE box with a
    // large offset corner, a cosmological box, and a tiny extent far from the origin.
    const double corners[4][3] = {{0, 0, 0}, {-137.5, 2048.25, 9.0e3}, {0, 0, 0}, {1.0e6, -3.0e5, 7.0e4}};
    const double lens[4]       = {1.0, 275.0, 3.0e5, 1.0e-3};
    long mismatches = 0, n = 0;
    for(int e = 0; e < 4; e++) {
        for(int s = 0; s < 200000; s++) {
            for(int k = 0; k < 3; k++) {
                const double pos = corners[e][k] + lens[e] * uniform01();
                const uint64_t got = domain_key_integer_coordinate(domain_key_scaled_coordinate(pos, corners[e][k], lens[e]), BITS);
                if(got != legacy_coordinate(pos, corners[e][k], lens[e])) {mismatches++;}
                n++;
            }
        }
    }
    CHECK(n == 4L * 200000L * 3L);
    CHECK(mismatches == 0);
}

TEST_CASE("the three-axis form is the per-axis form") {
    uint64_t c[3];
    const double corner[3] = {-137.5, 2048.25, 9.0e3}, len = 275.0;
    for(int s = 0; s < 10000; s++) {
        double pos[3];
        for(int k = 0; k < 3; k++) {pos[k] = corner[k] + len * uniform01();}
        domain_key_integer_coordinates(pos[0], pos[1], pos[2], corner[0], corner[1], corner[2], len, BITS, c);
        for(int k = 0; k < 3; k++) {CHECK(c[k] == legacy_coordinate(pos[k], corner[k], len));}
    }
}

TEST_CASE("edges inside the extent agree with the legacy expression") {
    const double corner = 12.0, len = 3.0;
    const uint64_t last = ((uint64_t)1 << BITS) - 1;
    // The corner is cell 0 on both.
    CHECK(domain_key_integer_coordinate(domain_key_scaled_coordinate(corner, corner, len), BITS) == 0);
    CHECK(legacy_coordinate(corner, corner, len) == 0);
    // The largest representable position below the far face is the last cell on both.
    const double near_top = std::nextafter(corner + len, corner);
    CHECK(domain_key_integer_coordinate(domain_key_scaled_coordinate(near_top, corner, len), BITS) == last);
    CHECK(legacy_coordinate(near_top, corner, len) == last);
}

TEST_CASE("the clamp keys out-of-range and non-finite positions to a valid cell") {
    const double corner = 12.0, len = 3.0;
    const uint64_t last = ((uint64_t)1 << BITS) - 1;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    // At or past the far face the sum reaches 2.0, whose mantissa is zero: the legacy expression
    // keyed that to the FIRST cell; the shared routine keys it to the last.
    CHECK(legacy_coordinate(corner + len, corner, len) == 0);
    CHECK(domain_key_integer_coordinate(domain_key_scaled_coordinate(corner + len, corner, len), BITS) == last);
    CHECK(domain_key_integer_coordinate(domain_key_scaled_coordinate(corner + 10.0 * len, corner, len), BITS) == last);
    // Below the near face, NaN, and no extent yet all take the lower bound.
    CHECK(domain_key_integer_coordinate(domain_key_scaled_coordinate(corner - 1.0, corner, len), BITS) == 0);
    CHECK(domain_key_integer_coordinate(domain_key_scaled_coordinate(nan, corner, len), BITS) == 0);
    CHECK(domain_key_integer_coordinate(domain_key_scaled_coordinate(corner + 1.0, corner, 0.0), BITS) == 0);
    CHECK(domain_key_scaled_coordinate(corner + len, corner, len) < 2.0);
    CHECK(domain_key_scaled_coordinate(nan, corner, len) == 1.0);
}

TEST_CASE("a fraction just below 1.0 whose sum rounds to 2.0 is clamped to the last cell") {
    // f = 1 - 2^-54 is representable; f + 1.0 rounds to exactly 2.0 under round-to-nearest-even.
    const double f = 1.0 - 0x1.0p-54;
    CHECK(f + 1.0 == 2.0);
    const uint64_t last = ((uint64_t)1 << BITS) - 1;
    // corner 0, len 1 makes the fraction the position itself.
    CHECK(legacy_coordinate(f, 0.0, 1.0) == 0);
    CHECK(domain_key_integer_coordinate(domain_key_scaled_coordinate(f, 0.0, 1.0), BITS) == last);
}

TEST_MAIN()
