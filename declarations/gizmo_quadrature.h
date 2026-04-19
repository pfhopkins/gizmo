/* gizmo_quadrature.h — Simple Gauss-Legendre quadrature for GIZMO.
 *
 * Provides a 20-point Gauss-Legendre rule on [a, b] that replaces
 * gsl_integration_qag(..., GSL_INTEG_GAUSS41, ...) for the smooth
 * integrands in driftfac.cc (cosmological drift/kick tables) and
 * sidm_core.cc (SIDM geometric factor tables).
 *
 * For the smooth functions used in those integrations (1/(H*a^3),
 * 1/(H*a^2), kernel-weighted angular integrals) a 20-point rule
 * delivers machine-precision accuracy — far more than the 1e-8
 * tolerance requested from GSL.  No workspace allocation is needed;
 * the quadrature is a single function call.
 *
 * Integrand signature: double f(double x, void *params)
 * (same as gsl_function.function, so existing integrand bodies can be
 * passed directly after removing gsl_function wrapper boilerplate).
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */
#ifndef GIZMO_QUADRATURE_H
#define GIZMO_QUADRATURE_H

/* Integrand function pointer type — identical to gsl_function.function. */
typedef double (*gizmo_integrand_fn)(double x, void *params);

/* 20-point Gauss-Legendre nodes (positive half) and weights on [-1, 1].
 * Source: Abramowitz & Stegun §25.4 / standard GL tables. */
static const double GL20_X[10] = {
    0.0765265211334973338546831,
    0.2277858511416450780804962,
    0.3737060887154195606725482,
    0.5108670019508270980787851,
    0.6360536807265150029356944,
    0.7463064833401287943893162,
    0.8391169718222188233945291,
    0.9122344282513259058677524,
    0.9639719272779137912676661,
    0.9931285991850949247861224
};
static const double GL20_W[10] = {
    0.1527533871307258506980843,
    0.1491729864726037467878287,
    0.1420961093183820513292983,
    0.1316886384491766268984945,
    0.1181945319615184173123774,
    0.1019301198172404350367501,
    0.0832767415767047487247581,
    0.0626720483341090635695065,
    0.0406014298003869413310400,
    0.0176140071391521183118620
};

/* Integrate f(x, params) over [a, b] using the 20-point Gauss-Legendre rule.
 * Accurate to machine precision for polynomials up to degree 39 and for
 * smooth functions (no singularities or rapid oscillations) in general. */
static inline double gizmo_gl20_integrate(gizmo_integrand_fn f, double a, double b, void *params)
{
    double mid = 0.5 * (a + b);
    double half = 0.5 * (b - a);
    double sum = 0.0;
    int i;
    for(i = 0; i < 10; i++) {
        double dx = half * GL20_X[i];
        sum += GL20_W[i] * (f(mid + dx, params) + f(mid - dx, params));
    }
    return half * sum;
}

#endif /* GIZMO_QUADRATURE_H */
