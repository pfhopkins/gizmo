/* Brent (1973) bracketed root finder as a template function over the residual.

   Usage:
     auto residual = [&](double x) { return ...; };
     BrentRootfindResult r = brent_rootfind(residual, x_a, x_b, f_a, f_b, rel_tol, abs_tol, MAXITER);
     double root = r.x;
     if(r.iter > MAXITER) { ... did not converge ... }

   x_a and x_b should bracket the root, with f_a = residual(x_a) and f_b = residual(x_b). If they do not, the bracket
   is widened (geometrically for a positive variable, by its own width otherwise) until it does, or endrun. Iteration
   stops when the residual is exactly zero or the bracket is no wider than max(abs_tol, rel_tol*|best estimate|); a
   proposed step shorter than that is taken as a minimal step, never as convergence. The returned x is always an
   evaluated point: the best one found. Bisection is in log space when both ends are positive.

   Nested root finds just nest. Needs DMIN/DMAX, PRINT_WARNING and endrun from the caller's includes; device-callable
   (KOKKOS_INLINE_FUNCTION), so the residual functor must be too. test/rootfind_unit exercises it without a GIZMO build. */

#ifndef BRACKETED_ROOTFIND_FUNCTIONS_H
#define BRACKETED_ROOTFIND_FUNCTIONS_H

struct BrentRootfindResult {
    double x;      /* the root: the best evaluated point */
    double error;  /* final bracket width |x_b - x_a| */
    int iter;      /* iterations used (> max_iter if it hit the cap) */
};

template <typename F>
KOKKOS_INLINE_FUNCTION
BrentRootfindResult brent_rootfind(F&& func, double x_a, double x_b, double f_a, double f_b,
                                          double rel_tol, double abs_tol, int max_iter)
{
    if(f_a * f_b > 0) {
        PRINT_WARNING("ERROR: Bounds supplied to brent_rootfind do not bracket the root. x_a=%g x_b=%g f_a=%g f_b=%g Expanding region...",
                      x_a, x_b, f_a, f_b);
        double bracket_fac = 1.1;
        int bracket_iter = 0;
        do {
            double lo = DMIN(x_a, x_b), hi = DMAX(x_a, x_b);
            if(lo > 0) {x_a = lo / bracket_fac; x_b = hi * bracket_fac;} /* positive variable: geometric widening keeps it positive */
            else {double w = hi - lo; x_a = lo - w; x_b = hi + w;} /* sign-agnostic widening */
            f_a = func(x_a);
            f_b = func(x_b);
            bracket_iter++;
        } while(f_a * f_b > 0 && bracket_iter < max_iter);
        if((bracket_iter == max_iter) || isnan(f_a) || isnan(f_b)) {
            PRINT_WARNING("ERROR: Could not bracket root. x_a=%g x_b=%g f_a=%g f_b=%g\n", x_a, x_b, f_a, f_b);
            endrun(234528);
        }
    }

    if(fabs(f_a) < fabs(f_b)) { /* convention: a is the end with the larger residual, b the best estimate */
        double tmp = f_a; f_a = f_b; f_b = tmp;
        tmp = x_a; x_a = x_b; x_b = tmp;
    }

    double x_c = x_a, f_c = f_a, x_c_old = x_c, x_new = x_b, f_new = f_b, x_error = 1e100, delta_tol = 0.;
    int used_bisection = 1, iter = 0;
    if(f_b == 0 || f_a == 0) { /* exact zero at an end point: return it as is */
        if(f_b != 0) {x_new = x_a; f_new = f_a;}
        x_error = 0;
    }

    while(x_error > delta_tol) {
        x_new = 0;
        if((f_a != f_c) && (f_b != f_c)) { /* inverse quadratic interpolation */
            x_new += x_a * f_c * f_b / (f_a - f_b) / (f_a - f_c);
            x_new += x_b * f_c * f_a / (f_b - f_a) / (f_b - f_c);
            x_new += x_c * f_a * f_b / (f_c - f_a) / (f_c - f_b);
        } else { /* secant */
            x_new = (x_a * f_b - x_b * f_a) / (f_b - f_a);
        }
        delta_tol = DMAX(fabs(abs_tol), rel_tol * fabs(x_b)); /* relative to the best estimate, not to the proposed point */

        int do_bisection = 0;
        double x_midpoint_a = 0.25 * (3 * x_a + x_b);
        if((x_new < DMIN(x_midpoint_a, x_b)) || (x_new > DMAX(x_midpoint_a, x_b))) {
            do_bisection = 1;
        } else if(fabs(x_new - x_b) < delta_tol) { /* a step shorter than the tolerance is a minimal step, not an exit */
            x_new = x_b + ((x_a > x_b) ? delta_tol : -delta_tol);
        }
        if(used_bisection) {
            if(fabs(x_new - x_b) >= 0.5 * fabs(x_c - x_b)) {do_bisection = 1;}
            if((x_b != x_c) && (fabs(x_b - x_c) < delta_tol)) {do_bisection = 1;}
        } else {
            if(fabs(x_new - x_b) >= 0.5 * fabs(x_c_old - x_c)) {do_bisection = 1;}
            if((x_c_old != x_c) && (fabs(x_c_old - x_c) < delta_tol)) {do_bisection = 1;}
        }
        if(do_bisection) {
            if((x_b > 0) && (x_a > 0)) {x_new = sqrt(x_b * x_a);} else {x_new = 0.5 * (x_b + x_a);} /* log space when possible */
            used_bisection = 1;
        } else {
            used_bisection = 0;
        }

        f_new = func(x_new);
        if(f_new == 0) {break;}

        x_c_old = x_c; x_c = x_b; f_c = f_b;
        if(f_a * f_new < 0) {x_b = x_new; f_b = f_new;} else {x_a = x_new; f_a = f_new;}
        if(fabs(f_a) < fabs(f_b)) {
            double tmp = f_a; f_a = f_b; f_b = tmp;
            tmp = x_a; x_a = x_b; x_b = tmp;
        }
        x_error = fabs(x_b - x_a);
        iter++;
        if(iter > max_iter) {break;}
    }
    if(f_new != 0 && fabs(f_b) < fabs(f_new)) {x_new = x_b;} /* the best evaluated point, not whichever was evaluated last */

    BrentRootfindResult r = {x_new, x_error, iter};
    return r;
}

#endif /* BRACKETED_ROOTFIND_FUNCTIONS_H */
