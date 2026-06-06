/* bracketed_rootfind_functions.h -- Brent 1973 bracketed rootfinder as a
 * proper template function. Replaces the old #include-fragment
 * system/bracketed_rootfind.h, which used macro monomorphisation (caller
 * #define'd ROOTFIND_FUNCTION and a dozen ROOTFIND_* variables in the local
 * scope, #included the fragment, then #undef'd).
 *
 * The algorithm is identical: inverse-quadratic interpolation with secant
 * fallback and bisection safeguard, plus an initial bracket-expansion step
 * if the caller-supplied bounds do not actually bracket the root. Log-space
 * bisection is used when both endpoints are positive. Behaviour,
 * convergence criteria, and failure modes match the original fragment bit
 * for bit.
 *
 * Usage:
 *   auto func = [&](double x) { return ... };
 *   BrentRootfindResult r = brent_rootfind(func, x_a, x_b, f_a, f_b,
 *                                          rel_tol, abs_tol, MAXITER);
 *   double root = r.x;
 *   if (r.iter > MAXITER) { ... did not converge ... }
 *
 * Nested rootfinds (one brent_rootfind inside another) work naturally — no
 * ROOTFIND_FUNCTION_INNER kludge needed.
 *
 * Requires allvars.h / macros.h (PRINT_WARNING, endrun, DMIN/DMAX) to be
 * included by the caller before this header.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#ifndef BRACKETED_ROOTFIND_FUNCTIONS_H
#define BRACKETED_ROOTFIND_FUNCTIONS_H

struct BrentRootfindResult {
    double x;      /* converged root */
    double error;  /* final |x_b - x_a| */
    int iter;      /* iterations used (> max_iter if hit the cap) */
};

template <typename F>
KOKKOS_INLINE_FUNCTION
BrentRootfindResult brent_rootfind(F&& func,
                                   double x_a, double x_b,
                                   double f_a, double f_b,
                                   double rel_tol, double abs_tol,
                                   int max_iter)
{
    /* If bounds do not bracket the root, expand the interval geometrically until they do. */
    if(f_a * f_b > 0) {
        PRINT_WARNING("ERROR: Bounds supplied to brent_rootfind do not bracket the root. "
                      "x_a=%g x_b=%g f_a=%g f_b=%g Expanding region...",
                      x_a, x_b, f_a, f_b);
        double bracket_fac = 1.1;
        int bracket_iter = 0;
        do {
            double tmp = x_a;
            x_a = DMIN(x_a, x_b) / bracket_fac;
            x_b = DMAX(tmp, x_b) * bracket_fac;
            f_a = func(x_a);
            f_b = func(x_b);
            bracket_iter++;
        } while(f_a * f_b > 0 && bracket_iter < max_iter);
        if((bracket_iter == max_iter) || isnan(f_a) || isnan(f_b)) {
            PRINT_WARNING("ERROR: Could not bracket root. x_a=%g x_b=%g f_a=%g f_b=%g\n",
                          x_a, x_b, f_a, f_b);
            endrun(234528);
        }
    }

    /* Convention: 'a' is the bracket with the larger residual, 'b' the smaller. */
    if(fabs(f_a) < fabs(f_b)) {
        double tmp = f_a; f_a = f_b; f_b = tmp;
        tmp = x_a;        x_a = x_b; x_b = tmp;
    }

    double x_c = x_a, f_c = f_a;
    int used_bisection = 1;
    int do_bisection = 0;
    int iter = 0;
    double x_c_old = x_c, x_new, f_new = f_c;
    double x_error = 1e100, delta_tol = 0.;

    do {
        x_new = 0;
        if((f_a != f_c) && (f_b != f_c)) {
            /* inverse quadratic interpolation */
            x_new += x_a * f_c * f_b / (f_a - f_b) / (f_a - f_c);
            x_new += x_b * f_c * f_a / (f_b - f_a) / (f_b - f_c);
            x_new += x_c * f_a * f_b / (f_c - f_a) / (f_c - f_b);
        } else {
            /* secant method */
            x_new = (x_a * f_b - x_b * f_a) / (f_b - f_a);
        }
        delta_tol = DMAX(fabs(abs_tol), rel_tol * fabs(x_new));

        do_bisection = 0;
        double x_midpoint_a = 0.25 * (3 * x_a + x_b);
        if((x_new < DMIN(x_midpoint_a, x_b)) || (x_new > DMAX(x_midpoint_a, x_b))) {
            do_bisection = 1;
        } else {
            /* accept interpolation and bug out if converged */
            if(fabs(x_new - x_b) < delta_tol) { break; }
        }
        if(used_bisection) {
            if(fabs(x_new - x_b) >= 0.5 * fabs(x_c - x_b)) { do_bisection = 1; }
            if(x_b != x_c) {
                if(fabs(x_b - x_c) < delta_tol) { do_bisection = 1; }
            }
        } else {
            if(fabs(x_new - x_b) >= 0.5 * fabs(x_c_old - x_c)) { do_bisection = 1; }
            if(x_c_old != x_c) {
                if(fabs(x_c_old - x_c) < delta_tol) { do_bisection = 1; }
            }
        }
        if(do_bisection) {
            /* log-space bisection if both endpoints are positive (helps typical use cases) */
            if((x_b > 0) && (x_a > 0)) {
                x_new = sqrt(x_b * x_a);
            } else {
                x_new = 0.5 * (x_b + x_a);
            }
            used_bisection = 1;
        } else {
            used_bisection = 0;
        }

        f_new = func(x_new);
        if(f_new == 0) { break; }

        x_c_old = x_c;
        x_c = x_b;
        f_c = f_b;
        if(f_a * f_new < 0) {
            x_b = x_new; f_b = f_new;
        } else {
            x_a = x_new; f_a = f_new;
        }

        if(fabs(f_a) < fabs(f_b)) {
            double tmp = f_a; f_a = f_b; f_b = tmp;
            tmp = x_a;        x_a = x_b; x_b = tmp;
        }
        x_error = fabs(x_b - x_a);
        iter++;
        if(iter > max_iter) { break; }
    } while(x_error > delta_tol);

    BrentRootfindResult r = {x_new, x_error, iter};
    return r;
}

#endif /* BRACKETED_ROOTFIND_FUNCTIONS_H */
