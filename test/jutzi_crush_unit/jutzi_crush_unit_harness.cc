/* Standalone harness for the Jutzi 2008 P-alpha quadratic crush curve
 * (solids/jutzi_crush_curve.h). Reads (P, alpha_0, P_e, P_s) from stdin, emits
 * alpha. No GIZMO / Kokkos / MPI dependencies.
 *
 *   Build: c++ -std=c++17 -O2 jutzi_crush_unit_harness.cc -o harness
 *   Use:   echo "<P> <alpha_0> <P_e> <P_s>" | ./harness
 *
 * Used by test/jutzi_crush_unit/test_jutzi_crush.py to bit-validate the C++
 * curve against the analytic Eq. 8 reference at sampled pressures. */
#include <cstdio>
#include "../../src/solids/jutzi_crush_curve.h"
int main() {
    double P, a0, Pe, Ps;
    while(std::scanf("%lf %lf %lf %lf", &P, &a0, &Pe, &Ps) == 4) {
        std::printf("%.17g\n", jutzi_distention_eq8(P, a0, Pe, Ps));
    }
    return 0;
}
