/* Unit test for brent_rootfind() in system/bracketed_rootfind_functions.h, the Brent root finder used by the
   cooling solve (DoCooling), the H2-fraction solve and the cooling-side dust temperature (rt_ir_lambdadust).

   Self-contained: stubs the few GIZMO symbols the header touches and includes the header directly.
   Each case brackets a root and checks that the solver
     - returns a point it actually evaluated (it once returned an unevaluated interpolation step),
     - lands within the requested tolerance of the known root,
     - stays under the iteration cap and never calls endrun() when a root is bracketed,
     - expands a non-bracketing interval in the right direction for negative variables.
   The function shapes are the ones the dust solvers produce: residuals spanning many decades across the
   bracket, |f| ~ 0 at one end for reasons unrelated to the root, jumps (opacity-table zone edges), flat
   cubic roots, very steep roots, and brackets on a negative variable (x = Tdust - T).

   Build: mpicxx -std=c++17 -O2 -I<repo root> test_rootfind_unit.cc -o test_rootfind_unit   (optionally
   also with the production flags -O3 -ffast-math). Exit status is nonzero if any case fails. */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
using std::isnan; /* the header uses the bare C name */

#define MAXITER 150
#define DMIN(a, b) (((a) < (b)) ? (a) : (b))
#define DMAX(a, b) (((a) > (b)) ? (a) : (b))
#define PRINT_WARNING(...) {g_warn++;}
static int g_warn = 0, g_evals = 0;
static std::vector<double> g_xs;      /* every argument the function was evaluated at */
static void endrun(int code) {throw code;}
#ifndef ROOTFIND_HEADER
#define ROOTFIND_HEADER "src/system/bracketed_rootfind_functions.h"
#endif
#ifndef KOKKOS_INLINE_FUNCTION
#define KOKKOS_INLINE_FUNCTION inline /* so ROOTFIND_HEADER can also point at the gpu_kokkos_unified copy */
#endif
#include ROOTFIND_HEADER

typedef double (*fn)(double);
#define EVAL(expr) {g_evals++; g_xs.push_back(x); return (expr);}
static double f_asym(double x)     EVAL((1. - pow(x/16.3, 4)) * 1e-8 * pow(x/2.73, 3))            /* ~1e-8 at the near end, ~-1e9 at the far end; root 16.3 */
static double f_ceiling(double x)  EVAL((1. - pow(x/618., 4)) * (1e-3 * exp(-x/900.) + 1e-10))     /* root 618, |f| tiny at the 1e4 end */
static double f_cubic(double x)    EVAL(x*x*x - 2*x - 5)
static double f_negx(double x)     EVAL(-(x + 9983.7) * (1e-8 + fabs(x + 9983.7)))                 /* x = Tdust - T with T = 1e4; root -9983.7 */
static double f_jump(double x)     EVAL((x < 100.) ? 1.0 : -1.0)
static double f_flat(double x)     EVAL(-pow((x - 50.)/50., 3))
static double f_endpoint(double x) EVAL(7.0 - x)                                                   /* root exactly at the end point b */
static double f_linear(double x)   EVAL(x - 3.)
static double f_steep(double x)    EVAL((x - 30.8) * 4.5e5)
static double f_edge(double x)     EVAL((x < 275.) ? (275. - x) * 1e-3 + 0.2 : (273.4 - x) * 1e-3)   /* zone-edge jump next to a crossing */
static double f_absroot(double x)  EVAL(5. - fabs(x))                                              /* roots at +-5: used for the expansion path */
static double f_zero_a(double x)   EVAL(x - 1.)                                                    /* root exactly at the end point a */
static double f_logwide(double x)  EVAL(log10(x) - 3.)                                             /* 40-decade positive bracket, root 1e3 */
static double f_huge(double x)     EVAL(1e200 * (x - 3.))                                          /* products of residuals overflow */
static double f_tiny(double x)     EVAL(1e-200 * (x - 3.))                                         /* products of residuals underflow */

struct Case {const char *name; fn f; double a, b, root, tol_rel, tol_abs; int max_evals;};
static Case cases[] = {
    {"asymmetric residual, [2.73, 2223]",            f_asym,     2.73,     2223.,  16.3,          1e-6, 1e-4, 60},
    {"|f|~0 at the far end, [50, 1e4]",              f_ceiling,  50.,      1e4,    618.,          1e-6, 1e-4, 60},
    {"cubic x^3-2x-5, [2, 3]",                       f_cubic,    2.,       3.,     2.0945514815423265, 1e-6, 1e-4, 40},
    {"negative variable, [-9997.27, -7777]",         f_negx,    -9997.27, -7777., -9983.7,        1e-6, 1e-4, 80},
    {"jump at 100, [1, 1000]",                       f_jump,     1.,       1000.,  100.,          1e-6, 1e-4, 60},
    {"flat cubic root at 50, [1, 1000]",             f_flat,     1.,       1000.,  50.,           1e-6, 1e-4, 80},
    {"root at the end point, [1, 7]",                f_endpoint, 1.,       7.,     7.,            1e-6, 1e-4, 10},
    {"linear, [0, 6]",                               f_linear,   0.,       6.,     3.,            1e-6, 1e-4, 10},
    {"steep slope, [2.73, 2223]",                    f_steep,    2.73,     2223.,  30.8,          1e-6, 1e-4, 20},
    {"zone-edge jump beside a crossing, [1, 1000]",  f_edge,     1.,       1000.,  275.,          1e-6, 1e-4, 60},
    {"not bracketed, positive, [1, 2] root 5",       f_absroot,  1.,       2.,     5.,            1e-6, 1e-4, 80},
    {"not bracketed, negative, [-2, -1] root -5",    f_absroot, -2.,      -1.,    -5.,            1e-6, 1e-4, 80},
    {"root exactly at the end point a, [1, 9]",      f_zero_a,   1.,       9.,     1.,            1e-6, 1e-4, 10},
    {"40-decade positive bracket, [1e-20, 1e20]",    f_logwide,  1e-20,    1e20,   1e3,           1e-6, 1e-4, 80},
    {"tight tolerance, cubic, [2, 3]",               f_cubic,    2.,       3.,     2.0945514815423265, 1e-12, 0.,   60},
    {"overflowing residual products, [0, 6]",        f_huge,     0.,       6.,     3.,            1e-6, 1e-4, 60},
    {"underflowing residual products, [0, 6]",       f_tiny,     0.,       6.,     3.,            1e-6, 1e-4, 60},
};

int main()
{
    int nfail = 0;
    printf("%-46s %12s %10s %6s %6s  %s\n", "case", "x", "|x-root|", "evals", "iters", "result");
    for(auto &c : cases)
    {
        g_warn = 0; g_evals = 0; g_xs.clear();
        double x = NAN; int iters = 0; std::string why;
        try {
            double fa = c.f(c.a), fb = c.f(c.b);
            BrentRootfindResult r = brent_rootfind(c.f, c.a, c.b, fa, fb, c.tol_rel, c.tol_abs, MAXITER);
            x = r.x; iters = r.iter;
        } catch(int code) {why += "endrun; ";}
        double tol = c.tol_abs + c.tol_rel * fabs(c.root);
        if(!(fabs(x - c.root) <= 3. * tol)) {why += "off root; ";}
        bool evaluated = false; for(double xe : g_xs) {if(xe == x) {evaluated = true; break;}}
        if(!std::isnan(x) && !evaluated) {why += "returned an unevaluated point; ";}
        if(std::isnan(x)) {why += "returned NaN; ";}
        if(iters >= MAXITER) {why += "hit MAXITER; ";}
        if(g_evals > c.max_evals) {why += "too many evaluations; ";}
        bool ok = why.empty();
        if(!ok) {nfail++;}
        printf("%-46s %12.7g %10.2e %6d %6d  %s%s\n", c.name, x, fabs(x - c.root), g_evals, iters, ok ? "PASS" : "FAILED: ", why.c_str());
    }
    printf("%d of %zu cases failed\n", nfail, sizeof(cases)/sizeof(cases[0]));
    return nfail ? 1 : 0;
}
