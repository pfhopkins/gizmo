// Unit test: Remy-Ruyer+14 dust-to-gas ratio over the linear law (declarations/resolvedism_dgr_rr14.h).
// Table 1, X_CO,Z broken power law: log(G/D) = 2.21 + 1.00 (8.69 - x) above x = 8.10, 0.96 + 3.10 (8.69 - x) below.
// Solar and super-solar gas keep the linear law exactly (factor 1.0, so Zsun runs are unchanged).
#include "test_harness.h"
#include "declarations/resolvedism_dgr_rr14.h"

static double gd_rr14(double x) {return (x > 8.10) ? 2.21 + 1.00 * (8.69 - x) : 0.96 + 3.10 * (8.69 - x);}

TEST_CASE("linear law kept above the break (x > 8.10, Z > 0.257 Zsun)") {
    for(double z : {0.26, 0.5, 1.0, 3.0, 10.0}) {CHECK(resolvedism_dgr_rr14_over_linear(z) == 1.0);}
    CHECK(resolvedism_dgr_rr14_over_linear(0.0) == 1.0);
}

TEST_CASE("broken power law below the break, against Table 1") {
    for(double z : {0.2, 0.1, 1e-2, 1e-3, 1e-4}) {
        double x = 8.69 + log10(z), dgr = z * resolvedism_dgr_rr14_over_linear(z); // solar units
        double expect = pow(10.0, 2.21 - gd_rr14(x));                                // G/D at Zsun = 10^2.21
        std::printf("  Z = %g Zsun: DGR = %.3e (RR14 %.3e), linear %.3e\n", z, dgr, expect, z);
        CHECK_CLOSE(dgr / expect, 1.0, 1e-12);
    }
    CHECK_CLOSE(1e-4 * resolvedism_dgr_rr14_over_linear(1e-4), 7.08e-12, 0.01e-12); // design report: 7e-12 at 1e-4 Zsun
    CHECK_CLOSE(1e-2 * resolvedism_dgr_rr14_over_linear(1e-2), 1.12e-5, 0.01e-5);   // and 1.1e-5 at 1e-2 Zsun
}

TEST_CASE("monotonic in Z on each branch; the Table 1 fits meet within 3% at the break") {
    double zb = pow(10., 8.10 - 8.69), prev = 0;
    for(double lz = -6; lz <= 1; lz += 0.001) {
        double z = pow(10., lz), d = z * resolvedism_dgr_rr14_over_linear(z);
        if(lz > log10(zb) && lz - 0.001 <= log10(zb)) {prev = 0;} // new branch
        CHECK(d > prev); prev = d;
    }
    double lo = resolvedism_dgr_rr14_over_linear(zb * (1 - 1e-9)), hi = resolvedism_dgr_rr14_over_linear(zb * (1 + 1e-9));
    std::printf("  break at Z = %.4f Zsun: factor %.4f below, %.4f above\n", zb, lo, hi);
    CHECK(fabs(lo / hi - 1) < 0.03);
}

TEST_MAIN()
