/* gr/gr_functions.h -- device-callable GR physics: tensor algebra, metric, EOS, conserved variables, rest-frame field.
 *
 * Every function here is a KOKKOS_INLINE_FUNCTION (host and device), works on the math types of the trunk (Vec3,
 * Mat3, SymmetricTensor2) and the gr_* structs of gr/gr.h, and reads simulation parameters only through All (the
 * device sees the synced mirror). No particle indices, no printf, no endrun: a caller that needs to report a failure
 * does it on the host.
 *
 * Conventions: the spatial metric is a SymmetricTensor2 (gamma[i][j] with symmetric indexing), vectors are Vec3<double>;
 * gamma.matvec(v) lowers an index, ugamma.matvec(w) raises it, gamma.quad(v) = gamma_ij v^i v^j. The Eulerian velocity
 * is v = (dx/dt + beta)/alpha (Vphys), the coordinate one dx/dt = alpha v - beta (Vcoord).
 *
 * Included from declarations/cell_data.h (after particle_data.h and the declaration of All), so that the cell
 * methods can use the GR density and sound speed; include it directly in any other header that needs it.
 *
 * Port of gr/metric.cc, gr/gr_eos.cc and parts of gr/gr_utils.cc of the C/C++ GIZMO branches (A. Lupi). */
#ifndef GR_GR_FUNCTIONS_H
#define GR_GR_FUNCTIONS_H
#ifdef GENERAL_RELATIVITY


/* ------------------------------------------------------------------ kinematics */
/* Lorentz factor of the contravariant Eulerian velocity v; with gr_limiter the velocity is limited IN PLACE to the maximum
   Lorentz factor GR_GAMMA_LIMITER. (The C code used long double here; device code uses double.) */
KOKKOS_INLINE_FUNCTION double lorentz_factor(const SymmetricTensor2<double> &gamma, Vec3<double> &v, int gr_limiter)
{
    double denom, R = gamma.quad(v);
    if(gr_limiter == 1)
    {
        double RMAX = 1.0 - 1./((double)GR_GAMMA_LIMITER*(double)GR_GAMMA_LIMITER);
        double Rstar = 0.5*(RMAX + R - fabs(R - RMAX));
        v *= sqrt(Rstar/(R + 1e-70));
        denom = 1. - Rstar;
    }
    else
    {
        denom = 1. - R;
        if(denom < MIN_REAL_NUMBER) {denom = MIN_REAL_NUMBER;}
    }
    return 1./sqrt(denom);
}

/* coordinate velocity dx/dt <-> velocity measured by the Eulerian observer */
KOKKOS_INLINE_FUNCTION Vec3<double> Vphys(const Vec3<double> &vcoord, const struct gr_metric *gr) {return (vcoord + gr->beta)/gr->alpha;}
KOKKOS_INLINE_FUNCTION Vec3<double> Vcoord(const Vec3<double> &vphys, const struct gr_metric *gr) {return gr->alpha*vphys - gr->beta;}

/* ------------------------------------------------------------------ metric */
/* displacement of pos from the black hole of the single-hole metrics: the hole sits at the origin, or at the box centre if the box is
   periodic (nearest image). Every distance from the hole (metric, excision, Lorentz-factor limiter, test-particle cut) uses it */
KOKKOS_INLINE_FUNCTION Vec3<double> gr_hole_displacement(const Vec3<double> &pos)
{
    Vec3<double> dp = pos;
#ifdef BOX_PERIODIC
    dp[0] -= boxHalf_X; dp[1] -= boxHalf_Y; dp[2] -= boxHalf_Z;
    NEAREST_XYZ(dp[0],dp[1],dp[2],1);
#endif
    return dp;
}

/* 1 inside GAMMA_LIMITER_RADIUS of the hole (where the Lorentz factor is limited) */
KOKKOS_INLINE_FUNCTION int gr_limiter_flag(const Vec3<double> &pos)
{
    return (gr_hole_displacement(pos).norm_sq() <= (double)GAMMA_LIMITER_RADIUS*(double)GAMMA_LIMITER_RADIUS) ? 1 : 0;
}


/* covariant 4-metric at pos; the source is at the box centre if the box is periodic */
KOKKOS_INLINE_FUNCTION void get_covariant_metric(const Vec3<double> &pos, struct gr_sym4 *g)
{
    Vec3<double> dp = gr_hole_displacement(pos);
#if (GR_METRIC==0) /* Minkowski */
    g->tt = -1; g->ti = Vec3<double>{0,0,0}; g->ij.set_isotropic(1.);
#elif (GR_METRIC==1) /* Kerr: Boyer-Lindquist (USE_BOYERLINDQUIST) or Kerr-Schild coordinates */
    double M = All.MetricParams[0], a = All.MetricParams[1]*M, a2 = a*a;
    double x2 = dp[0]*dp[0], y2 = dp[1]*dp[1], z2 = dp[2]*dp[2], R2 = x2 + y2 + z2;
    double r2 = (R2 - a2 + sqrt((R2-a2)*(R2-a2) + 4*a2*z2))/2.;
#ifndef USE_KERRSCHILD_COORDINATES /* Boyer-Lindquist in Cartesian components (Liptai & Price 2019): singular on the axis x = y = 0, where the
                                      1/(x^2+y^2) terms cancel only analytically; for particle distributions that avoid the axis */
    double sinth2 = 1. - z2/r2, rho2 = r2 + a2*z2/r2, delta = r2 - 2*M*sqrt(r2) + a2, Att = 2.0*M*sqrt(r2)/rho2, xy2 = x2 + y2;
    g->tt = Att - 1.0;
    double gtp = -Att*a*sinth2, gpp = (r2 + a2 + Att*a2*sinth2)*sinth2;
    g->ti = Vec3<double>{-dp[1]/xy2*gtp, dp[0]/xy2*gtp, 0.};
    double fac1 = r2/rho2/delta, fac2 = gpp/(xy2*xy2), fac3 = z2/rho2/(r2*sinth2), fac4 = (1 - z2*(r2+a2)/r2/rho2);
    g->ij[0][0] = fac1*x2 + fac2*y2 + fac3*x2;
    g->ij[1][1] = fac1*y2 + fac2*x2 + fac3*y2;
    g->ij[2][2] = (1-fac4)*(a2+r2)/delta + rho2/(r2*sinth2)*fac4*fac4;
    g->ij[0][1] = dp[0]*dp[1]*(fac1 - fac2 + fac3);
    g->ij[0][2] = (a2+r2)*dp[0]*dp[2]/rho2/delta - dp[0]*dp[2]/(r2*sinth2)*fac4;
    g->ij[1][2] = (a2+r2)*dp[1]*dp[2]/rho2/delta - dp[1]*dp[2]/(r2*sinth2)*fac4;
#else
    double rho2 = r2*r2 + a2*z2, r = sqrt(r2), Mfac = 2*M*r*r*r/rho2, denom = r2 + a2;
    Vec3<double> l = {(r*dp[0] + a*dp[1])/denom, (r*dp[1] - a*dp[0])/denom, dp[2]/r}; /* spatial part of the null vector l_mu = (1, l_i) */
    g->tt = -1 + Mfac;
    g->ti = Mfac*l;
    g->ij = Mfac*outer_product(l);
    g->ij[0][0] += 1; g->ij[1][1] += 1; g->ij[2][2] += 1;
#endif
#elif (GR_METRIC<0) /* tabulated static spherically symmetric metric (All.GR_MetricTable: r, M(r), -g_tt, r_iso; see gr_load_metric_table) */
    const int N = All.GR_MetricTableN;
    const double *rb = All.GR_MetricTable, *mass = rb + N, *gtt = rb + 2*N;
    double r = dp.norm();
    /* the row is found by bisection (the radii are strictly increasing, checked by the loader): no assumption on the spacing; the radius is
       floored at the first row, or at the second if the first is r = 0 (where the interpolated quantities are singular); beyond the last
       row the metric is the exterior Schwarzschild one with the mass of the last row */
#ifdef USE_ISOTROPIC_RADIUS /* isotropic coordinates: the table is indexed by r_iso, the spatial metric is conformally flat, e^{4 phi} = (r_areal/r_iso)^2 */
    const double *riso = rb + 3*N;
    {double r_floor = (riso[0] > 0) ? riso[0] : riso[1]; if(r <= r_floor) {r = r_floor;}}
    double e4phi, g00;
    if(r < riso[N-1])
    {
        int il = 0, ih = N-1; while(ih > il+1) {int im = (il+ih)/2; if(riso[im] > r) {ih = im;} else {il = im;}}
        double f = (r - riso[il])/(riso[il+1] - riso[il]), r_areal = rb[il] + f*(rb[il+1] - rb[il]);
        e4phi = (r_areal/r)*(r_areal/r); g00 = -(gtt[il] + f*(gtt[il+1] - gtt[il]));
    }
    else {double q = 0.5*mass[N-1]/r, lapse = (1 - q)/(1 + q); e4phi = (1 + q)*(1 + q)*(1 + q)*(1 + q); g00 = -lapse*lapse;} /* exterior Schwarzschild, isotropic */
    g->tt = g00; g->ti = Vec3<double>{0,0,0}; g->ij.set_isotropic(e4phi);
#else /* areal (Schwarzschild-like) radius: g_rr = 1/(1 - 2M(r)/r), gamma_ij = delta_ij + (g_rr - 1) x_i x_j/r^2 */
    {double r_floor = (rb[0] > 0) ? rb[0] : rb[1]; if(r <= r_floor) {r = r_floor;}}
    double Mr, g00;
    if(r < rb[N-1])
    {
        int il = 0, ih = N-1; while(ih > il+1) {int im = (il+ih)/2; if(rb[im] > r) {ih = im;} else {il = im;}}
        double f = (r - rb[il])/(rb[il+1] - rb[il]); Mr = mass[il] + f*(mass[il+1] - mass[il]); g00 = -(gtt[il] + f*(gtt[il+1] - gtt[il]));
    }
    else {Mr = mass[N-1]; g00 = -(1 - 2*mass[N-1]/r);} /* exterior Schwarzschild, areal radius */
    double grr = 1./(1 - 2*Mr/r), omega = 2*Mr/(r*r*r);
    Vec3<double> dp2 = {dp[0]*dp[0], dp[1]*dp[1], dp[2]*dp[2]};
    g->tt = g00; g->ti = Vec3<double>{0,0,0};
    g->ij[0][0] = grr*(1 - omega*(dp2[1] + dp2[2]));
    g->ij[1][1] = grr*(1 - omega*(dp2[0] + dp2[2]));
    g->ij[2][2] = grr*(1 - omega*(dp2[0] + dp2[1]));
    g->ij[0][1] = grr*dp[0]*dp[1]*omega;
    g->ij[0][2] = grr*dp[0]*dp[2]*omega;
    g->ij[1][2] = grr*dp[1]*dp[2]*omega;
#endif
#endif
}

#if (GR_METRIC>0)
/* 1 if the excision region of a black hole lies between the cells at pos_i and pos_j (dp = pos_i - pos_j, periodic-wrapped), so that
   they do not interact. Symmetric under i <-> j (as the test of the C++ branch, mesh/ngb.cc check_BH_intersection): the pair is excluded
   if the point of the segment closest to the hole lies inside the segment and is nearer to the hole than r_H - R_c (the bare horizon,
   not the excision radius), with R_c = 0.5 (h_i + h_j)/DesNumNgb^(1/NUMDIMS) a size common to both cells. The margin is subtracted, so
   that cells at the edge of the excision region keep interacting with their neighbours along the edge (with GR_EXCISION_RADIUS r_H + R_c,
   as in the C++ branch, the cells just outside the excision radius lost their tangential neighbours: Bondi with excision at 1.05 r_H had
   rho0 25% below Michel at r = 2.5) */
KOKKOS_INLINE_FUNCTION int gr_segment_crosses_excised_hole(const Vec3<double> &dp, double normdp2, const Vec3<double> &dpBH, double M, const Vec3<double> &spin, double R_common)
{
    double proj = dot(dpBH, dp), t = proj/normdp2;
    if(t <= 0 || t >= 1) {return 0;} /* the closest point is one of the two cells, which are outside the excised region */
    double perpBH2 = dpBH.norm_sq() - proj*proj/normdp2, R_block = GR_HORIZON_RADIUS(M, spin) - R_common;
    if(R_block <= 0) {return 0;}
    return (perpBH2 < R_block*R_block) ? 1 : 0;
}
KOKKOS_INLINE_FUNCTION int gr_pair_crosses_excised_hole(const Vec3<double> &pos_i, double h_i, const Vec3<double> &dp, double h_j)
{
    double normdp2 = dp.norm_sq();
    if(normdp2 <= 0) {return 0;}
    double R_common = 0.5*(h_i + h_j)/pow((double)All.DesNumNgb, 1./NUMDIMS);
    Vec3<double> dpBH = gr_hole_displacement(pos_i), spin = {0, 0, All.MetricParams[1]};
    return gr_segment_crosses_excised_hole(dp, normdp2, dpBH, All.MetricParams[0], spin, R_common);
}
#endif

/* 3+1 coefficients (spatial metric and its inverse, shift, lapse, sqrt(gamma)) from the covariant 4-metric; limiter is left untouched */
KOKKOS_INLINE_FUNCTION void gr_metric_from_4metric(const struct gr_sym4 *g, struct gr_metric *gr)
{
    gr->gamma = g->ij;
    double det = gr->gamma.invert(gr->ugamma);
    gr->beta = gr->ugamma.matvec(g->ti);            /* beta^i = gamma^ij g_tj */
    gr->alpha = sqrt(dot(gr->beta, g->ti) - g->tt); /* alpha^2 = beta_i beta^i - g_tt */
    gr->gdet = sqrt(det);
}

/* fills the metric structure at pos (potential: Newtonian correction of the lapse, only with self-gravity) */
KOKKOS_INLINE_FUNCTION void set_metric_coefficients(const Vec3<double> &pos, double potential, struct gr_metric *gr)
{
    struct gr_sym4 g;
    get_covariant_metric(pos, &g);
    gr_metric_from_4metric(&g, gr);
    gr->limiter = gr_limiter_flag(pos);
}

/* element (mu,nu) of a symmetric 4-tensor (0 = t) */
KOKKOS_INLINE_FUNCTION double gr_sym4_get(const struct gr_sym4 *g, int mu, int nu)
{
    if(mu == 0 && nu == 0) {return g->tt;}
    if(mu == 0) {return g->ti[nu-1];}
    if(nu == 0) {return g->ti[mu-1];}
    return g->ij[mu-1][nu-1];
}

/* full contraction of two symmetric 4-tensors, A^{mu nu} B_{mu nu} = A.tt B.tt + 2 A.ti.B.ti + A.ij:B.ij */
KOKKOS_INLINE_FUNCTION double gr_sym4_contract(const struct gr_sym4 &A, const struct gr_sym4 &B)
{
    return A.tt*B.tt + 2.*dot(A.ti, B.ti) + A.ij.double_contract(B.ij);
}

/* spatial derivatives of the 4-metric (dg[dir] = d g_{mu nu}/dx^dir), lapse and shift (dbeta[dir][j] = d beta^j/dx^dir) at pos,
   by centred finite differences along the Cartesian directions */
KOKKOS_INLINE_FUNCTION void compute_metric_jacobian(const Vec3<double> &pos, struct gr_sym4 dg[3], Vec3<double> &dalpha, Mat3<double> &dbeta)
{
    const double fac = 0.5/GR_EPS_TOLERANCE;
    for(int dir = 0; dir < 3; dir++)
    {
        Vec3<double> pH = pos, pL = pos; pH[dir] += GR_EPS_TOLERANCE; pL[dir] -= GR_EPS_TOLERANCE;
        struct gr_sym4 gH, gL; struct gr_metric mH, mL;
        get_covariant_metric(pH, &gH); get_covariant_metric(pL, &gL);
        gr_metric_from_4metric(&gH, &mH); gr_metric_from_4metric(&gL, &mL);
        dg[dir].tt = fac*(gH.tt - gL.tt);
        dg[dir].ti = fac*(gH.ti - gL.ti);
        dg[dir].ij = fac*(gH.ij - gL.ij);
        dalpha[dir] = fac*(mH.alpha - mL.alpha);
        dbeta[dir] = fac*(mH.beta - mL.beta);
    }
}


/* extrinsic curvature K_ij = -alpha Gamma^0_ij and the metric derivatives at pos, given the metric there */
KOKKOS_INLINE_FUNCTION void get_metric_derivatives(const Vec3<double> &pos, const struct gr_metric *gr, SymmetricTensor2<double> &Kij, struct gr_sym4 dg[3], Vec3<double> &dalpha, Mat3<double> &dbeta)
{
    double invalpha2 = 1./(gr->alpha*gr->alpha);
    double g00 = -invalpha2; Vec3<double> g0i = invalpha2*gr->beta; /* g^{00}, g^{0i} of the inverse 4-metric */
    compute_metric_jacobian(pos, dg, dalpha, dbeta);
    for(int i = 0; i < 3; i++) {for(int j = i; j < 3; j++)
    {
        double K = 0.5*g00*(gr_sym4_get(&dg[j], 0, i+1) + gr_sym4_get(&dg[i], 0, j+1)); /* the time derivative of the metric is zero (static metrics) */
        for(int w = 1; w < 4; w++) {K += 0.5*g0i[w-1]*(gr_sym4_get(&dg[j], w, i+1) + gr_sym4_get(&dg[i], w, j+1) - gr_sym4_get(&dg[w-1], i+1, j+1));}
        Kij[i][j] = -gr->alpha*K;
    }}
}

/* ------------------------------------------------------------------ equation of state (ideal gas with GAMMA_DEFAULT, as in the con2prim) */
/* rest-mass density from the conserved density D = sqrt(gamma) rho0 W and the coordinate velocity */
KOKKOS_INLINE_FUNCTION double gr_get_primitive_density(double density, const Vec3<double> &vcoord, const struct gr_metric *gr, int gr_limiter)
{
    Vec3<double> vphys = Vphys(vcoord, gr);
    return density/lorentz_factor(gr->gamma, vphys, gr_limiter)/gr->gdet;
}

KOKKOS_INLINE_FUNCTION double gr_pressure_rho0_u(double rho0, double u)
{
#ifdef GR_TEST_PARTICLE
    return 0;
#else
    return (GAMMA_DEFAULT - 1.)*rho0*u;
#endif
}

KOKKOS_INLINE_FUNCTION double gr_pressure_rho0_w(double rho0, double w) /* from w = rho0 + rho0 u + p */
{
#ifdef GR_TEST_PARTICLE
    return 0;
#else
    return (GAMMA_DEFAULT - 1.)*(w - rho0)/GAMMA_DEFAULT;
#endif
}

KOKKOS_INLINE_FUNCTION double gr_get_u_from_p(double rho0, double p)
{
#ifdef GR_TEST_PARTICLE
    return 0;
#else
    return p/rho0/(GAMMA_DEFAULT - 1.);
#endif
}

/* pressure and its derivatives with respect to W = rho h gamma^2 and v^2, for the con2prim iteration */
KOKKOS_INLINE_FUNCTION double gr_eos_info(double W, double vsq, double *dpdw, double *dpdvsq, const struct kernel_NR *kernel)
{
    double ftmp = 1. - vsq, gtmp = sqrt(ftmp), gam_m1_o_gam = (GAMMA_DEFAULT - 1.)/GAMMA_DEFAULT;
    *dpdw = gam_m1_o_gam*ftmp;
    *dpdvsq = gam_m1_o_gam*(0.5*kernel->D/gtmp - W);
    return gam_m1_o_gam*(W*ftmp - kernel->D*gtmp);
}

/* relativistic sound speed of the ideal gas: c_s^2 = Gamma P / (rho0 h), rho0 h = rho0 (1+u) + P */
KOKKOS_INLINE_FUNCTION double gr_sound_speed(double gamma_eos, double pressure, double rho0, double u)
{
    double rho_h = rho0*(1. + u) + pressure;
    if(rho_h > 0 && pressure > 0) {return sqrt(gamma_eos*pressure/rho_h);}
    return 0;
}

/* ------------------------------------------------------------------ conserved variables from the primitives */
KOKKOS_INLINE_FUNCTION void gr_get_conservative_variables(const struct gr_primitives *prims, struct gr_conservatives *cons, double Wlorentz, const struct gr_metric *gr)
{
    double hstar = 1.0 + prims->u + prims->p/prims->rho0, Pstar = prims->p, gdet = gr->gdet;
    Vec3<double> v_cov = gr->gamma.matvec(prims->v);
    cons->D = gdet*prims->rho0*Wlorentz;
    cons->S = Vec3<double>{0,0,0}; cons->tau = 0;
    double DhW = cons->D*Wlorentz*hstar;
    cons->S += DhW*v_cov;
    cons->tau += DhW - gdet*Pstar - cons->D;
    if(cons->tau < 0) {cons->tau = MIN_REAL_NUMBER;} /* (the C code printed a warning here) */
    cons->p = Pstar;
}



#include "gr_con2prim_functions.h"

#endif /* GENERAL_RELATIVITY */
#endif /* GR_GR_FUNCTIONS_H */
