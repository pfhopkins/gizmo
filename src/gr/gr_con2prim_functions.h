/* gr/gr_con2prim_functions.h -- conserved-to-primitive inversion (Noble et al. 2006, 2D Newton-Raphson in W and v^2), device-callable.
 *
 * gr_get_primitive_variables() returns the Lorentz factor on success and a negative code on failure:
 *   -1 : rho0 <= 0 or u <= 0 after the inversion
 *   -3 : W out of range,  -4 : v^2 >= 1,  -5 : the initial Lorentz factor is not finite
 *   -(100*r + 1) : the Newton-Raphson iteration returned r (1 = not converged, 2 = non-finite residual)
 * It never prints: the callers report a failure.
 *
 * gr_drift_reinvert() is the inversion at fixed specific internal energy (the energy slaved to the momentum), used by the energy-holding
 * drift (GR_DRIFT_HOLD_INTERNAL_ENERGY, the default) for the drifted and predicted states.
 *
 * Port of gr/utoprim.cc of the C/C++ GIZMO branches (after the HARM utoprim_2d of Noble et al.), without the host diagnostics. */
#ifndef GR_CON2PRIM_FUNCTIONS_H
#define GR_CON2PRIM_FUNCTIONS_H
#ifdef GENERAL_RELATIVITY

#define GR_C2P_MAX_NEWT_ITER   (30)       /* maximum number of Newton-Raphson iterations */
#define GR_C2P_NEWT_TOL        (1.0e-10)  /* convergence tolerance */
#define GR_C2P_EXTRA_NEWT_ITER (2)        /* extra iterations once converged */
#define GR_C2P_W_TOO_BIG       (1.e20)    /* gamma^2 (rho0 + u + p) above this is unphysical */
#define GR_C2P_MIN_DV          (1.0e-15)
#define GR_C2P_FAIL_VAL        (MAX_REAL_NUMBER)

KOKKOS_INLINE_FUNCTION double gr_c2p_vsq_calc(double W, const struct kernel_NR *kernel) /* v^2 from W = gamma^2 w */
{
    double Wsq = W*W, Bsq_W = kernel->Bsq + W, Xsq = Bsq_W*Bsq_W;
    return (Wsq*kernel->Qtsq + kernel->QdotBsq*(Bsq_W + W))/(Wsq*Xsq);
}

KOKKOS_INLINE_FUNCTION double gr_c2p_x1_of_x0(double x0, const struct kernel_NR *kernel) /* physical v^2 from W */
{
    double vsq = fabs(gr_c2p_vsq_calc(x0, kernel));
    return (vsq > 1.) ? (1.0 - GR_C2P_MIN_DV) : vsq;
}

/* residuals and Newton step for x = (W, v^2) */
KOKKOS_INLINE_FUNCTION void gr_c2p_func_vsq(const double x[], double dx[], double resid[], double jac[][2], double *f, double *df, const struct kernel_NR *kernel)
{
    double W = x[0], vsq = x[1], dPdvsq, dPdW;
    double p_tmp = gr_eos_info(W, vsq, &dPdW, &dPdvsq, kernel);
    double t2 = -kernel->half_Bsq + dPdvsq, t3 = kernel->Bsq + W, t4 = t3*t3, t23 = 1./W, t16 = kernel->QdotBsq*t23*t23;
    double t11 = kernel->Qtsq - vsq*t4 + t16*(kernel->Bsq + W + W);
    double t18 = -kernel->Qdotn - kernel->half_Bsq*(1.0 + vsq) + 0.5*t16 - W + p_tmp;
    double t24 = t16*t23, t25 = -1.0 + dPdW - t24;
    double t35 = t25*t3 + (kernel->Bsq - 2.0*dPdvsq)*(t16 + vsq*W)*t23;
    double t21 = 1/(t3*t35);
    dx[0] = -(t2*t11 + t4*t18)*t21;
    double t40 = -2*(vsq + t24)*t3;
    dx[1] = -(-t25*t11 + t40*t18)*t21;
    jac[0][0] = t40; jac[0][1] = -t4; jac[1][0] = t25; jac[1][1] = t2;
    resid[0] = t11; resid[1] = t18;
    *df = -resid[0]*resid[0] - resid[1]*resid[1];
    *f = -0.5*(*df);
}

/* 2D Newton-Raphson on (W, v^2): 0 converged, 1 not converged, 2 non-finite residual */
KOKKOS_INLINE_FUNCTION int gr_c2p_twod_newton_raphson(double x[], const struct kernel_NR *kernel)
{
    double f = 1., df = 1., dx[2], x_old[2], resid[2], jac[2][2], errx = 1.;
    int n_iter = 0, i_extra = 0, doing_extra = 0, keep_iterating = 1;
    const double dv = (1. - GR_C2P_MIN_DV);
    while(keep_iterating)
    {
        gr_c2p_func_vsq(x, dx, resid, jac, &f, &df, kernel);
        x_old[0] = x[0]; x_old[1] = x[1];
        x[0] += dx[0]; x[1] += dx[1];
        errx = (x[0] == 0.) ? fabs(dx[0]) : fabs(dx[0]/x[0]);
        if(x[0] < 0.) {x[0] = fabs(x[0]);} else {if(x[0] > GR_C2P_W_TOO_BIG) {x[0] = x_old[0];}}
        if(x[1] < 0.) {x[1] = 0.;} else {if(x[1] > 1.) {x[1] = dv;}}
        if((fabs(errx) <= GR_C2P_NEWT_TOL) && (doing_extra == 0) && (GR_C2P_EXTRA_NEWT_ITER > 0)) {doing_extra = 1;}
        if(doing_extra == 1) {i_extra++;}
        if(((fabs(errx) <= GR_C2P_NEWT_TOL) && (doing_extra == 0)) || (i_extra > GR_C2P_EXTRA_NEWT_ITER) || (n_iter >= (GR_C2P_MAX_NEWT_ITER - 1))) {keep_iterating = 0;}
        n_iter++;
    }
    if((!isfinite(f)) || (!isfinite(df))) {return 2;}
    if(fabs(errx) <= GR_C2P_NEWT_TOL) {return 0;}
    return 1;
}

/* primitives from the (densitized) conserved variables; prims->u and prims->v on input are the initial guess (the previous primitives) */
KOKKOS_INLINE_FUNCTION double gr_get_primitive_variables(const struct gr_conservatives *cons_pred, struct gr_primitives *prims, const struct gr_metric *gr, int gr_limiter)
{
    double x_2d[2], tau, dens, gamma, inv_sqrtg = 1./gr->gdet, rho0, u, p, rhoh, W_last, W, vsq, QdotB, QdB_o_W;
    int i_increase, retval;
    struct kernel_NR kernel;
    prims->u_kept = 0;
    dens = cons_pred->D*inv_sqrtg;
    Vec3<double> Svec = inv_sqrtg*cons_pred->S;          /* S_i */
    tau = cons_pred->tau*inv_sqrtg;
    Vec3<double> uSvec = gr->ugamma.matvec(Svec);        /* S^i */
#ifdef GR_TEST_PARTICLE
    W = (tau + dens)/sqrt((tau + dens)*(tau + dens) - dot(uSvec, Svec));
    prims->rho0 = dens/W;
    prims->v = uSvec/(tau + dens);
    prims->u = 1e-20; prims->p = 0;
    return W;
#endif
    kernel.Bsq = 0.0; QdotB = 0.0; kernel.QdotBsq = 0.0; kernel.half_Bsq = 0.0;
    kernel.Qdotn = -(tau + dens);
    kernel.Qtsq = dot(uSvec, Svec);
    kernel.D = dens;
    /* initial guess from the previous primitives */
    vsq = gr->gamma.quad(prims->v);
    if(vsq > 1) {gamma = lorentz_factor(gr->gamma, prims->v, gr_limiter); vsq = 1. - 1./gamma/gamma;}
    gamma = 1./sqrt(1.0 - vsq);
    if(!isfinite(gamma)) {return -5.;}
    rho0 = kernel.D/gamma;
    u = prims->u*rho0;
    p = gr_pressure_rho0_u(rho0, prims->u);
    rhoh = rho0 + u + p;
    W_last = rhoh*gamma*gamma;
    i_increase = 0;
    while(((W_last*W_last*W_last*(W_last + 2.*kernel.Bsq) - kernel.QdotBsq*(2.*W_last + kernel.Bsq)) <= W_last*W_last*(kernel.Qtsq - kernel.Bsq*kernel.Bsq)) && (i_increase < 10))
    {
        W_last *= 10.; i_increase++;
    }
    x_2d[0] = fabs(W_last);
    x_2d[1] = gr_c2p_x1_of_x0(W_last, &kernel);
    retval = gr_c2p_twod_newton_raphson(x_2d, &kernel);
    W = x_2d[0]; vsq = x_2d[1];
    if((retval != 0) || (W == GR_C2P_FAIL_VAL)) {return -(retval*100. + 1.);}
    if(W <= 0. || W > GR_C2P_W_TOO_BIG) {return -3.;}
    if(vsq >= 1.) {return -4.;}
    if(gr_limiter == 1) {double RMAX = 1.0 - 1./((double)GR_GAMMA_LIMITER*(double)GR_GAMMA_LIMITER); vsq = 0.5*(RMAX + vsq - fabs(vsq - RMAX));}
    gamma = 1./sqrt(1. - vsq);
    rho0 = kernel.D/gamma;
    rhoh = W*(1. - vsq);
    rhoh = fmax(rho0, rhoh);
    p = gr_pressure_rho0_w(rho0, rhoh);
    u = rhoh - (rho0 + p);
    /* (the C/C++ GR code skipped these two lines under EOS_GENERAL, although the GR equation of state is always the ideal gas of gr_functions.h: a bug) */
    if(p <= 0 && fabs(p)/rhoh < 1e-10) {u = prims->u*rho0; p = gr_pressure_rho0_u(rho0, u/rho0); prims->u_kept = 1;} /* (as in the C/C++ GR code: no admissible u, the previous one is kept) */
    if(u <= 0) {u = p/(GAMMA_DEFAULT - 1);}
    prims->rho0 = rho0;
    prims->u = u/rho0;
    prims->p = p;
    double W_inv = 1./(W + kernel.Bsq);
    prims->v = W_inv*uSvec;
    lorentz_factor(gr->gamma, prims->v, gr_limiter); /* limits the velocity in place where required */
    if((rho0 <= 0.) || (u <= 0.)) {return -1;}
    return gamma;
}

/* Inversion with the energy slaved to the momentum: the conserved D, S_j (and sqrt(gamma) B^i) and the specific internal energy u are
   held fixed, and the Lorentz factor, the rest-mass density, the pressure and tau are recomputed with the metric gr. It is used by the
   default drift (the metric changes along the path at fixed S_j and u), the kick removing the corresponding rate from the Valencia energy
   source (gr_drift_tau_path_rate), and, with either drift, by the entropy floor of merged cells (mesh/merge_split.cc). Carrying tau through the drift unchanged leaves (D, S_j, tau) inconsistent with the new metric by
   O(dt), which makes the velocity of the drifted state first-order accurate (circular geodesics in Kerr-Schild coordinates drift);
   for a test particle the modified scheme is the Hamiltonian splitting in the covariant momentum, second order. Ideal gas: h = 1 + Gamma u does not depend on rho0,
   so S_i S^i = (D h)^2 (W^2 - 1) gives W directly, and v^i = S^i/(D h W).
   On input prims->v is the velocity before the drift; returns the Lorentz factor (closed form: it cannot fail). */
KOKKOS_INLINE_FUNCTION double gr_drift_reinvert(const struct gr_conservatives *cons, double u, double gamma_eos, const struct gr_metric *gr, int gr_limiter, struct gr_primitives *prims, struct gr_conservatives *cons_new)
{
    const double D = cons->D, h = 1. + gamma_eos*u, gdet = gr->gdet;
    Vec3<double> v = prims->v;
    double W = lorentz_factor(gr->gamma, v, gr_limiter);
    double S2 = gr->ugamma.quad(cons->S);
    W = sqrt(1. + S2/(D*h*D*h));
    v = gr->ugamma.matvec(cons->S)/(D*h*W);
    lorentz_factor(gr->gamma, v, gr_limiter);
    prims->v = v;
    prims->rho0 = D/(gdet*W);
    prims->u = u;
    prims->p = gr_pressure_rho0_u(prims->rho0, u);
    gr_get_conservative_variables(prims, cons_new, W, gr);
    return W;
}

#endif /* GENERAL_RELATIVITY */
#endif /* GR_CON2PRIM_FUNCTIONS_H */
