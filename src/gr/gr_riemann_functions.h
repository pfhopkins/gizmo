/* gr/gr_riemann_functions.h -- the GR hydro Riemann solvers, device-callable.
 *
 * HLL (GENERAL_RELATIVITY=0) or the HLLC of Mignone & Bodo 2005 (GENERAL_RELATIVITY=1), with the star state of Duez et al.
 * (Lupi 2023).
 *
 * gr_riemann_solver() is the entry point called from hydro_core_meshless_functions.h in place of the Newtonian
 * Riemann_solver(): the states are the primitives of each side (rest-mass density, Eulerian velocity v^i, pressure),
 * the metric at the face is Riemann_vec.gr and the face velocity Riemann_vec.v_frame
 * (Eulerian). The fluxes are returned in the lab frame, per unit face area, already in the coordinate frame of the moving
 * face: Fluxes.rho = F(D), Fluxes.v = F(S_i), Fluxes.p = F(tau). A failed solve leaves
 * P_M < 0, which triggers the lower-order retries of the caller; nothing here prints or stops.
 *
 * Port of gr/gr_riemann_solver.h of the C/C++ GIZMO branches (A. Lupi, G. Fedrigo). */
#ifndef GR_RIEMANN_FUNCTIONS_H
#define GR_RIEMANN_FUNCTIONS_H
#ifdef GENERAL_RELATIVITY


/* components of a symmetric tensor in the rotated basis whose rows are R: g' = R g R^T (R orthonormal) */
KOKKOS_INLINE_FUNCTION SymmetricTensor2<double> gr_rotate_tensor(const SymmetricTensor2<double> &g, const Mat3<double> &R)
{
    SymmetricTensor2<double> gp;
    for(int a = 0; a < 3; a++) {for(int b = a; b < 3; b++) {gp[a][b] = dot(R[a], g.matvec(R[b]));}}
    return gp;
}

/* the left/right state of a face: conserved variables and the quantities of the primitives the fluxes need */
struct gr_face_state
{
    struct gr_conservatives cons;
    Vec3<double> v;        /*!< Eulerian velocity (possibly limited) */
    Vec3<double> vcoord;   /*!< coordinate velocity alpha v - beta */
    double W;              /*!< Lorentz factor */
};

/* ------------------------------------------------------------------ wave speeds */
/* fastest coordinate signal speeds along n of a state with Eulerian velocity v (hydro): lam[0] = lambda_+, lam[1] = lambda_- */
KOKKOS_INLINE_FUNCTION void gr_lambda_hydro(const Vec3<double> &v, double cs, double g_nn, double beta_n, double alpha, const Vec3<double> &n_unit, double Wfac, double lam[2], int gr_limiter)
{
    double vtilde_n = dot(v, n_unit), cs2 = cs*cs;
    double v2max = 1. - MIN_REAL_NUMBER; if(gr_limiter) {v2max = 1. - 1./GR_GAMMA_LIMITER/GR_GAMMA_LIMITER;}
    double v2 = 1. - 1./Wfac/Wfac; if(v2 < 0) {v2 = 0.;}
    double discr = sqrt((1-v2)*(g_nn*(1-v2*cs2) - vtilde_n*vtilde_n*(1-cs2)));
    lam[0] = alpha*DMIN((vtilde_n*(1-cs2) + cs*discr)/(1-v2*cs2),  v2max) - beta_n;
    lam[1] = alpha*DMAX((vtilde_n*(1-cs2) - cs*discr)/(1-v2*cs2), -v2max) - beta_n;
}

/* ------------------------------------------------------------------ face states */
/* conserved variables (and the rest-frame field) of one side from its primitives; v is limited in place where required */
KOKKOS_INLINE_FUNCTION void gr_face_state_from_primitives(const struct Conserved_var_Riemann &side, const struct gr_metric *gr, int gr_limiter, struct gr_face_state *st
                                                          )
{
    struct gr_primitives prims;
    prims.rho0 = side.rho; prims.v = side.v; prims.u = side.u; prims.p = side.p;
    st->W = lorentz_factor(gr->gamma, prims.v, gr_limiter);
    gr_get_conservative_variables(&prims, &(st->cons), st->W, gr);
    st->v = prims.v;
    st->vcoord = Vcoord(prims.v, gr);
}


/* star state (contact wave speed and pressure) of Duez et al. / Mignone & Bodo 2005; returns 0 on success, 1 on failure */
KOKKOS_INLINE_FUNCTION int gr_starstate(const struct gr_conservatives &cons_L, const struct gr_conservatives &cons_R, const Vec3<double> &n_unit, double vline_L, double vline_R,
                                        const struct gr_metric *gr, double beta_n, double g_nn, double S_L, double S_R, double *S_M, double *P_M, int gr_limiter)
{
    double sXR = dot(gr->ugamma.matvec(cons_R.S), n_unit), sXL = dot(gr->ugamma.matvec(cons_L.S), n_unit); /* S^n */
    /* the coefficients of the quadratic equation for lambda* (Duez et al. eqs. 23-26) */
    double AR = (cons_R.tau + cons_R.D)*(S_R + beta_n) - gr->alpha*sXR;
    double AL = (cons_L.tau + cons_L.D)*(S_L + beta_n) - gr->alpha*sXL;
    double CR = gr->alpha*g_nn, CL = CR;
    double BR = CR*gr->gdet*cons_R.p + sXR*(vline_R - S_R);
    double BL = CL*gr->gdet*cons_L.p + sXL*(vline_L - S_L);
    double ER = -(beta_n + S_R), EL = -(beta_n + S_L);
    double a = (EL*AR - AL*ER), b = -(AL*CR + BL*ER - CL*AR - EL*BR), c = BR*CL - BL*CR, lamstar;
    if(a == 0)
    {
        if(b == 0) {return 1;}
        lamstar = -c/b;
    }
    else
    {
        double discr = b*b - 4*a*c;
        if(discr < 0) {return 1;}
        if(discr == 0) {lamstar = -0.5*b/a;}
        else
        {
            double tmp = (b > 0) ? -0.5*(b + sqrt(discr)) : -0.5*(b - sqrt(discr)); /* the negative root (Mignone & Bodo) */
            lamstar = c/tmp;
        }
    }
    /* lambda* is the Eulerian velocity component along the coordinate normal, bounded by sqrt(gamma^nn) (|v^n| <= sqrt(gamma^nn) |v|) */
    if(fabs(lamstar) > sqrt(g_nn)) {if(gr_limiter) {lamstar = copysign(sqrt(g_nn*(1. - 1./GR_GAMMA_LIMITER/GR_GAMMA_LIMITER)), lamstar);} else {return 1;}} /* (keeps the direction of the contact) */
    *P_M = (AL*lamstar + BL)/((CL + EL*lamstar)*gr->gdet); /* the pressure is continuous across the contact */
    *S_M = gr->alpha*lamstar - beta_n;
    return 0;
}

/* HLL fluxes of the hydro system (lab frame, direction n) */
KOKKOS_INLINE_FUNCTION void gr_hydro_hll_flux(const struct gr_conservatives &cons_L, const struct gr_conservatives &cons_R, double vline_L, double vline_R, double S_L, double S_R,
                                              double v_frame_line, const struct gr_metric *gr, double beta_n, const Vec3<double> &n_unit, struct Riemann_outputs *Riemann_out)
{
    double alpha = gr->alpha, gdet = gr->gdet;
#ifdef HYDRO_MESHLESS_FINITE_VOLUME
    if(S_L > v_frame_line)
    {
        Riemann_out->Fluxes.rho = cons_L.D*(vline_L - v_frame_line);
        Riemann_out->Fluxes.p   = cons_L.tau*(vline_L - v_frame_line) + gdet*cons_L.p*(vline_L + beta_n);
        Riemann_out->Fluxes.v   = (vline_L - v_frame_line)*cons_L.S + (alpha*gdet*cons_L.p)*n_unit;
        return;
    }
    if(S_R < v_frame_line)
    {
        Riemann_out->Fluxes.rho = cons_R.D*(vline_R - v_frame_line);
        Riemann_out->Fluxes.p   = cons_R.tau*(vline_R - v_frame_line) + gdet*cons_R.p*(vline_R + beta_n);
        Riemann_out->Fluxes.v   = (vline_R - v_frame_line)*cons_R.S + (alpha*gdet*cons_R.p)*n_unit;
        return;
    }
#endif
    struct gr_conservatives U_hll;
    double facLR = 1./(S_R - S_L);
    U_hll.D   = (S_R*cons_R.D - S_L*cons_L.D + cons_L.D*vline_L - cons_R.D*vline_R)*facLR;
    U_hll.tau = (S_R*cons_R.tau - S_L*cons_L.tau + cons_L.tau*vline_L - cons_R.tau*vline_R + gdet*(cons_L.p*(vline_L + beta_n) - cons_R.p*(vline_R + beta_n)))*facLR;
    U_hll.S   = facLR*(S_R*cons_R.S - S_L*cons_L.S + vline_L*cons_L.S - vline_R*cons_R.S + (alpha*gdet*(cons_L.p - cons_R.p))*n_unit);
    double F_D = (S_R*cons_L.D*vline_L - S_L*cons_R.D*vline_R + S_L*S_R*(cons_R.D - cons_L.D))*facLR;
#ifdef HYDRO_MESHLESS_FINITE_MASS
    v_frame_line = F_D/U_hll.D; /* the HLL equivalent of the zero-mass-flux face velocity */
    Riemann_out->S_M = v_frame_line;
    Riemann_out->Fluxes.rho = 0;
#else
    Riemann_out->Fluxes.rho = F_D - v_frame_line*U_hll.D;
#endif
    Riemann_out->Fluxes.p = facLR*(S_R*(cons_L.tau*(vline_L - S_L) + gdet*cons_L.p*(vline_L + beta_n)) - S_L*(cons_R.tau*(vline_R - S_R) + gdet*cons_R.p*(vline_R + beta_n))) - v_frame_line*U_hll.tau;
    Riemann_out->Fluxes.v = facLR*(S_R*((vline_L - S_L)*cons_L.S + (alpha*gdet*cons_L.p)*n_unit) - S_L*((vline_R - S_R)*cons_R.S + (alpha*gdet*cons_R.p)*n_unit)) - v_frame_line*U_hll.S;
}

#if (GENERAL_RELATIVITY==1)
/* HLLC fluxes of the hydro system (Mignone & Bodo 2005), finite-volume form */
KOKKOS_INLINE_FUNCTION void gr_hydro_hllc_flux(const struct gr_conservatives &cons_L, const struct gr_conservatives &cons_R, double vline_L, double vline_R, double S_L, double S_R,
                                               double v_frame_line, const struct gr_metric *gr, double beta_n, const Vec3<double> &n_unit, struct Riemann_outputs *Riemann_out)
{
    double alpha = gr->alpha, gdet = gr->gdet, S_M = Riemann_out->S_M, P_M = Riemann_out->P_M;
    /* a face at or beyond an edge of the fan takes the outer state; the star branches then always have S_L < v_frame < S_R, so their
       denominators S_M - S_L and S_R - S_M cannot vanish (a contact on an outer wave, S_M = S_L or S_R, needs no special case) */
    if(S_L >= v_frame_line)
    {
        Riemann_out->Fluxes.rho = cons_L.D*(vline_L - v_frame_line);
        Riemann_out->Fluxes.p   = cons_L.tau*(vline_L - v_frame_line) + gdet*cons_L.p*(vline_L + beta_n);
        Riemann_out->Fluxes.v   = (vline_L - v_frame_line)*cons_L.S + (alpha*gdet*cons_L.p)*n_unit;
    }
    else if(v_frame_line <= S_M)
    {
        double v_fac = (S_L - vline_L)/(S_L - S_M); if(v_fac < 0) {v_fac = 0;}
        struct gr_conservatives U;
        U.D   = v_fac*cons_L.D;
        U.S   = v_fac*cons_L.S + (alpha*gdet*(cons_L.p - P_M)/(S_M - S_L))*n_unit;
        U.tau = v_fac*cons_L.tau + gdet*((vline_L + beta_n)*cons_L.p - (S_M + beta_n)*P_M)/(S_M - S_L);
        Riemann_out->Fluxes.rho = U.D*(S_M - v_frame_line);
        Riemann_out->Fluxes.p   = U.tau*(S_M - v_frame_line) + gdet*P_M*(S_M + beta_n);
        Riemann_out->Fluxes.v   = (S_M - v_frame_line)*U.S + (alpha*gdet*P_M)*n_unit;
    }
    else if(v_frame_line < S_R)
    {
        double v_fac = (S_R - vline_R)/(S_R - S_M); if(v_fac < 0) {v_fac = 0;}
        struct gr_conservatives U;
        U.D   = v_fac*cons_R.D;
        U.S   = v_fac*cons_R.S + (alpha*gdet*(P_M - cons_R.p)/(S_R - S_M))*n_unit;
        U.tau = v_fac*cons_R.tau + gdet*((S_M + beta_n)*P_M - (vline_R + beta_n)*cons_R.p)/(S_R - S_M);
        Riemann_out->Fluxes.rho = U.D*(S_M - v_frame_line);
        Riemann_out->Fluxes.p   = U.tau*(S_M - v_frame_line) + gdet*P_M*(S_M + beta_n);
        Riemann_out->Fluxes.v   = (S_M - v_frame_line)*U.S + (alpha*gdet*P_M)*n_unit;
    }
    else
    {
        Riemann_out->Fluxes.rho = cons_R.D*(vline_R - v_frame_line);
        Riemann_out->Fluxes.p   = cons_R.tau*(vline_R - v_frame_line) + gdet*cons_R.p*(vline_R + beta_n);
        Riemann_out->Fluxes.v   = (vline_R - v_frame_line)*cons_R.S + (alpha*gdet*cons_R.p)*n_unit;
    }
}
#endif

/* the hydro solver, in the lab frame along n_unit */
KOKKOS_INLINE_FUNCTION void gr_hydro_riemann(const struct Input_vec_Riemann &Riemann_vec, struct Riemann_outputs *Riemann_out, const Vec3<double> &n_unit, int gr_limiter)
{
    const struct gr_metric *gr = &(Riemann_vec.gr);
    double beta_n = dot(gr->beta, n_unit), gnn = gr->ugamma.quad(n_unit); /* gamma^nn */
    double vline_L = gr->alpha*dot(Riemann_vec.L.v, n_unit) - beta_n, vline_R = gr->alpha*dot(Riemann_vec.R.v, n_unit) - beta_n;
    double v_frame_line = gr->alpha*dot(Riemann_vec.v_frame, n_unit) - beta_n;
    struct gr_face_state L, R;
    gr_face_state_from_primitives(Riemann_vec.L, gr, gr_limiter, &L);
    gr_face_state_from_primitives(Riemann_vec.R, gr, gr_limiter, &R);
    double lam_l[2], lam_r[2];
    gr_lambda_hydro(L.v, Riemann_vec.L.cs, gnn, beta_n, gr->alpha, n_unit, L.W, lam_l, gr_limiter);
    gr_lambda_hydro(R.v, Riemann_vec.R.cs, gnn, beta_n, gr->alpha, n_unit, R.W, lam_r, gr_limiter);
    double S_L = DMIN(lam_r[1], lam_l[1]), S_R = DMAX(lam_r[0], lam_l[0]);
#if (GENERAL_RELATIVITY==1)
    if(gr_starstate(L.cons, R.cons, n_unit, vline_L, vline_R, gr, beta_n, gnn, S_L, S_R, &(Riemann_out->S_M), &(Riemann_out->P_M), gr_limiter) != 0) {Riemann_out->P_M = -1; return;}
    if(Riemann_out->P_M < 0) {return;}
#ifdef HYDRO_MESHLESS_FINITE_MASS
    /* the face moves with the contact (v_frame = S_M): no mass flux, pressure work and force of the star state (the finite-volume
       form below with v_frame_line = S_M; the C/C++ GR code omitted the alpha*sqrt(gamma) factors here) */
    Riemann_out->Fluxes.rho = 0;
    Riemann_out->Fluxes.v = (gr->alpha*gr->gdet*Riemann_out->P_M)*n_unit;
    Riemann_out->Fluxes.p = gr->gdet*Riemann_out->P_M*(Riemann_out->S_M + beta_n);
#else
    gr_hydro_hllc_flux(L.cons, R.cons, vline_L, vline_R, S_L, S_R, v_frame_line, gr, beta_n, n_unit, Riemann_out);
#endif
#else
    /* HLL needs no star state (as the MHD HLL): P_M, the mean face pressure, only tells the caller that the solution is valid, and S_M is
       the mean signal speed or, for MFM, the zero-mass-flux face velocity the flux uses (set in gr_hydro_hll_flux) */
    if(!(S_R > S_L)) /* a degenerate (or non-finite) wave fan: the caller retries at lower order, except for two pressureless states (no flux, as before) */
    {
        const int pressureless_pair = (L.cons.p == 0) && (R.cons.p == 0) && isfinite(S_L) && isfinite(S_R) && (S_L == S_R);
        Riemann_out->P_M = pressureless_pair ? 0 : -1; return;
    }
    Riemann_out->S_M = 0.5*(S_L + S_R);
    Riemann_out->P_M = 0.5*(L.cons.p + R.cons.p);
    gr_hydro_hll_flux(L.cons, R.cons, vline_L, vline_R, S_L, S_R, v_frame_line, gr, beta_n, n_unit, Riemann_out);
    if(!(isfinite(Riemann_out->Fluxes.rho) && isfinite(Riemann_out->Fluxes.p) && isfinite(Riemann_out->Fluxes.v[0]) && isfinite(Riemann_out->Fluxes.v[1]) && isfinite(Riemann_out->Fluxes.v[2])))
        {Riemann_out->P_M = -1;} /* non-finite fluxes: the caller retries at lower order */
#endif
}

/* ------------------------------------------------------------------ entry point */
/* the GR replacement of Riemann_solver(): primitives of both sides in Riemann_vec (lab frame), fluxes back in the lab frame */
KOKKOS_INLINE_FUNCTION void gr_riemann_solver(struct Input_vec_Riemann Riemann_vec, struct Riemann_outputs *Riemann_out, const Vec3<double> &n_unit, int gr_limiter)
{
    if((Riemann_vec.L.p < 0 && Riemann_vec.R.p < 0) || (Riemann_vec.L.rho < 0) || (Riemann_vec.R.rho < 0)) {Riemann_out->P_M = 0; return;}
#ifndef EOS_GENERAL
    /* sound speeds and internal energies from pressure and rest-mass density (ideal gas, relativistic: c_s^2 = Gamma P/(rho0 h)) */
    Riemann_vec.L.u = Riemann_vec.L.p/((GAMMA_DEFAULT - 1.)*Riemann_vec.L.rho);
    Riemann_vec.R.u = Riemann_vec.R.p/((GAMMA_DEFAULT - 1.)*Riemann_vec.R.rho);
    Riemann_vec.L.cs = gr_sound_speed(GAMMA_DEFAULT, Riemann_vec.L.p, Riemann_vec.L.rho, Riemann_vec.L.u);
    Riemann_vec.R.cs = gr_sound_speed(GAMMA_DEFAULT, Riemann_vec.R.p, Riemann_vec.R.rho, Riemann_vec.R.u);
#endif
    gr_hydro_riemann(Riemann_vec, Riemann_out, n_unit, gr_limiter);
}

#endif /* GENERAL_RELATIVITY */
#endif /* GR_RIEMANN_FUNCTIONS_H */
