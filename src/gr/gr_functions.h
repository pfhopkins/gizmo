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
   periodic (nearest image). Every distance from the hole (metric, excision, Lorentz-factor limiter) uses it */
KOKKOS_INLINE_FUNCTION Vec3<double> gr_hole_displacement(const Vec3<double> &pos)
{
    Vec3<double> dp = pos;
#ifdef BOX_PERIODIC
    dp[0] -= boxHalf_X; dp[1] -= boxHalf_Y; dp[2] -= boxHalf_Z;
    NEAREST_XYZ(dp[0],dp[1],dp[2],1);
#endif
    return dp;
}

#if (GR_METRIC==1)
/* square of the radial coordinate r of the Kerr metric (the same in Kerr-Schild and Boyer-Lindquist coordinates) at the displacement dp from
   the hole, with a2 = (spin*M)^2: (x^2+y^2)/(r^2+a^2) + z^2/r^2 = 1. The horizon is r = r_H; r = |dp| only without spin */
KOKKOS_INLINE_FUNCTION double gr_kerr_radius_sq(const Vec3<double> &dp, double a2)
{
    double z2 = dp[2]*dp[2], R2 = dp[0]*dp[0] + dp[1]*dp[1] + z2;
    return (R2 - a2 + sqrt((R2-a2)*(R2-a2) + 4*a2*z2))/2.;
}
#endif

/* 1 inside GR_GAMMA_LIMITER_RADIUS of the hole (where the Lorentz factor is limited); never when the radius is not positive (off) */
KOKKOS_INLINE_FUNCTION int gr_limiter_flag(const Vec3<double> &pos)
{
    if(!((double)GR_GAMMA_LIMITER_RADIUS > 0)) {return 0;}
    return (gr_hole_displacement(pos).norm_sq() <= (double)GR_GAMMA_LIMITER_RADIUS*(double)GR_GAMMA_LIMITER_RADIUS) ? 1 : 0;
}


/* the covariant 4-metric from its deviation from Minkowski, g = eta + h, in place */
KOKKOS_INLINE_FUNCTION void gr_add_minkowski(struct gr_sym4 *g)
{
    g->tt -= 1; g->ij[0][0] += 1; g->ij[1][1] += 1; g->ij[2][2] += 1;
}

/* deviation of the covariant 4-metric from Minkowski, h = g - eta, at the displacement dp from the source (gr_hole_displacement), computed
   without forming eta + h: the finite differences of the metric derivatives difference h, so that they keep their relative accuracy where
   h is small (far from the source) instead of losing it in the O(1) components of g */
KOKKOS_INLINE_FUNCTION void get_metric_deviation(const Vec3<double> &dp, struct gr_sym4 *h)
{
#if (GR_METRIC==0) /* Minkowski */
    h->tt = 0; h->ti = Vec3<double>{0,0,0}; h->ij.set_isotropic(0.);
#elif (GR_METRIC==1) /* Kerr: Boyer-Lindquist (GR_METRIC_USE_BOYER_LINDQUIST) or Kerr-Schild coordinates */
#ifndef GR_METRIC_USE_KERR_SCHILD
    /* Boyer-Lindquist, Cartesian components (x = sqrt(r^2+a^2) sin(theta) cos(phi), y = sqrt(r^2+a^2) sin(theta) sin(phi), z = r cos(theta)):
       subtracting the flat metric in the same oblate spheroidal coordinates, the 1/(x^2+y^2) of d phi cancels the sin^2(theta) of g_t phi and
       g_phi phi, (x^2+y^2) = (r^2+a^2) sin^2(theta), so h is regular on the rotation axis (singular only at the horizon and on the ring r = 0):
       h_tt = 2Mr/rho^2, h_ti = -2Mar/(rho^2 (r^2+a^2)) w_i,
       h_ij = 2Mr/(Delta (r^2+a^2) rho^2) v_i v_j + 2Mr a^2/(rho^2 (r^2+a^2)^2) w_i w_j,
       with rho^2 = r^2 + a^2 z^2/r^2, Delta = r^2 - 2Mr + a^2, v = (r x, r y, (r^2+a^2) z/r) (rho^2 times the gradient of r) and
       w = (-y, x, 0) ((x^2+y^2) times the gradient of phi) */
    double M = All.MetricParams[0], a = All.MetricParams[1]*M, a2 = a*a;
    double z2 = dp[2]*dp[2], r2 = gr_kerr_radius_sq(dp, a2), r = sqrt(r2);
    double rho2 = r2 + a2*z2/r2, delta = r2 - 2*M*r + a2, ra2 = r2 + a2, Mr2 = 2*M*r;
    Vec3<double> v = {r*dp[0], r*dp[1], ra2*dp[2]/r}, w = {-dp[1], dp[0], 0.};
    h->tt = Mr2/rho2;
    h->ti = (-Mr2*a/(rho2*ra2))*w;
    h->ij = (Mr2/(delta*ra2*rho2))*outer_product(v) + (Mr2*a2/(rho2*ra2*ra2))*outer_product(w);
#else /* h_{mu nu} = (2 M r^3/(r^4 + a^2 z^2)) l_mu l_nu */
    double M = All.MetricParams[0], a = All.MetricParams[1]*M, a2 = a*a;
    double z2 = dp[2]*dp[2], r2 = gr_kerr_radius_sq(dp, a2);
    double rho2 = r2*r2 + a2*z2, r = sqrt(r2), Mfac = 2*M*r*r*r/rho2, denom = r2 + a2;
    Vec3<double> l = {(r*dp[0] + a*dp[1])/denom, (r*dp[1] - a*dp[0])/denom, dp[2]/r}; /* spatial part of the null vector l_mu = (1, l_i) */
    h->tt = Mfac;
    h->ti = Mfac*l;
    h->ij = Mfac*outer_product(l);
#endif
#elif (GR_METRIC<0) /* tabulated static spherically symmetric metric (All.GR_MetricTable: r, M(r), -g_tt, r_iso; see gr_load_metric_table) */
    const int N = All.GR_MetricTableN;
    const double *rb = All.GR_MetricTable, *mass = rb + N, *gtt = rb + 2*N;
    double r = dp.norm();
    /* the row is found by bisection (the radii are strictly increasing, checked by the loader): no assumption on the spacing; the radius is
       floored at the first row, or at the second if the first is r = 0 (where the interpolated quantities are singular); beyond the last
       row the metric is the exterior Schwarzschild one with the mass of the last row */
#ifdef GR_METRIC_USE_ISOTROPIC_RADIUS /* isotropic coordinates: the table is indexed by r_iso, the spatial metric is conformally flat, e^{4 phi} = (r_areal/r_iso)^2 */
    const double *riso = rb + 3*N;
    {double r_floor = (riso[0] > 0) ? riso[0] : riso[1]; if(r <= r_floor) {r = r_floor;}}
    double e4phi_m1, htt; /* e^{4 phi} - 1 and g_tt + 1 */
    if(r < riso[N-1])
    {
        int il = 0, ih = N-1; while(ih > il+1) {int im = (il+ih)/2; if(riso[im] > r) {ih = im;} else {il = im;}}
        double f = (r - riso[il])/(riso[il+1] - riso[il]), r_areal = rb[il] + f*(rb[il+1] - rb[il]);
        e4phi_m1 = (r_areal - r)*(r_areal + r)/(r*r); htt = 1 - (gtt[il] + f*(gtt[il+1] - gtt[il]));
    }
    else {double q = 0.5*mass[N-1]/r; e4phi_m1 = q*(4 + q*(6 + q*(4 + q))); htt = 4*q/((1 + q)*(1 + q));} /* exterior Schwarzschild, isotropic */
    h->tt = htt; h->ti = Vec3<double>{0,0,0}; h->ij.set_isotropic(e4phi_m1);
#else /* areal (Schwarzschild-like) radius: g_rr = 1/(1 - 2M(r)/r), gamma_ij = delta_ij + (g_rr - 1) x_i x_j/r^2 */
    {double r_floor = (rb[0] > 0) ? rb[0] : rb[1]; if(r <= r_floor) {r = r_floor;}}
    double Mr, htt;
    if(r < rb[N-1])
    {
        int il = 0, ih = N-1; while(ih > il+1) {int im = (il+ih)/2; if(rb[im] > r) {ih = im;} else {il = im;}}
        double f = (r - rb[il])/(rb[il+1] - rb[il]); Mr = mass[il] + f*(mass[il+1] - mass[il]); htt = 1 - (gtt[il] + f*(gtt[il+1] - gtt[il]));
    }
    else {Mr = mass[N-1]; htt = 2*mass[N-1]/r;} /* exterior Schwarzschild, areal radius */
    double grr_m1_o_r2 = 2*Mr/((r - 2*Mr)*r*r); /* (g_rr - 1)/r^2 */
    h->tt = htt; h->ti = Vec3<double>{0,0,0};
    for(int i = 0; i < 3; i++) {for(int j = i; j < 3; j++) {h->ij[i][j] = grr_m1_o_r2*dp[i]*dp[j];}}
#endif
#endif
}

/* covariant 4-metric at pos, eta + h (get_metric_deviation) */
KOKKOS_INLINE_FUNCTION void get_covariant_metric(const Vec3<double> &pos, struct gr_sym4 *g)
{
    get_metric_deviation(gr_hole_displacement(pos), g);
    gr_add_minkowski(g);
}

#if (GR_METRIC>0)
/* 1 if the excision region of a black hole lies between the cells at pos_i and pos_j (dp = pos_i - pos_j, periodic-wrapped), so that
   they do not interact. Symmetric under i <-> j (as the test of the C/C++ GR code, mesh/ngb.cc check_BH_intersection): the pair is excluded
   if the point of the segment closest to the hole lies inside the segment and is nearer to the hole than r_H - R_c (the bare horizon,
   not the excision radius), with R_c = 0.5 (h_i + h_j)/DesNumNgb^(1/NUMDIMS) a size common to both cells. The margin is subtracted, so
   that cells at the edge of the excision region keep interacting with their neighbours along the edge (with GR_EXCISION_RADIUS r_H + R_c,
   as in the C/C++ GR code, the cells just outside the excision radius lost their tangential neighbours: Bondi with excision at 1.05 r_H had
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

/* fills the metric structure at pos */
KOKKOS_INLINE_FUNCTION void set_metric_coefficients(const Vec3<double> &pos, struct gr_metric *gr)
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

/* mass of the metric (G = c = 1), the length unit of the finite-difference steps: the hole mass (Kerr), the mass of the last row
   (tabulated), 1 for Minkowski or a massless metric */
KOKKOS_INLINE_FUNCTION double gr_metric_mass_scale(void)
{
#if (GR_METRIC==1)
    double M = All.MetricParams[0];
#elif (GR_METRIC<0)
    double M = All.GR_MetricTable[2*All.GR_MetricTableN - 1];
#else
    double M = 1;
#endif
    return (M > 0) ? M : 1;
}

/* spatial derivatives of the 4-metric (dg[dir] = d g_{mu nu}/dx^dir), lapse and shift (dbeta[dir][j] = d beta^j/dx^dir) at pos, given the
   metric gr there. dg: centred finite differences of the deviation from Minkowski (get_metric_deviation) along the Cartesian directions
   (the step is taken in the displacement from the hole, after the periodic wrap: the two points never fall on opposite images),
   with the step GR_METRIC_DERIVATIVE_STEP times the mass of the metric; differencing h = g - eta instead of g, the rounding error goes as
   eps (r/M) M/h instead of eps (r/M)^2 M/h. Lapse and shift by the chain rule (differences of alpha ~ 1 would lose the accuracy again):
   d beta^i = gamma^ij (d g_tj - d gamma_jk beta^k), 2 alpha d alpha = d beta^i g_ti + beta^i d g_ti - d g_tt */
KOKKOS_INLINE_FUNCTION void compute_metric_jacobian(const Vec3<double> &pos, const struct gr_metric *gr, struct gr_sym4 dg[3], Vec3<double> &dalpha, Mat3<double> &dbeta)
{
    const double h = GR_METRIC_DERIVATIVE_STEP*gr_metric_mass_scale();
    const Vec3<double> dp = gr_hole_displacement(pos);
    const Vec3<double> g_t = gr->gamma.matvec(gr->beta); /* g_ti = beta_i */
    for(int dir = 0; dir < 3; dir++)
    {
        Vec3<double> pH = dp, pL = dp; pH[dir] += h; pL[dir] -= h;
        const double fac = 1./(pH[dir] - pL[dir]); /* the step actually represented */
        struct gr_sym4 hH, hL;
        get_metric_deviation(pH, &hH); get_metric_deviation(pL, &hL);
        dg[dir].tt = fac*(hH.tt - hL.tt);
        dg[dir].ti = fac*(hH.ti - hL.ti);
        dg[dir].ij = fac*(hH.ij - hL.ij);
        dbeta[dir] = gr->ugamma.matvec(dg[dir].ti - dg[dir].ij.matvec(gr->beta));
        dalpha[dir] = (dot(dbeta[dir], g_t) + dot(gr->beta, dg[dir].ti) - dg[dir].tt)/(2.*gr->alpha);
    }
}

/* extrinsic curvature K_ij = -alpha Gamma^0_ij and the metric derivatives at pos, given the metric there */
KOKKOS_INLINE_FUNCTION void get_metric_derivatives(const Vec3<double> &pos, const struct gr_metric *gr, SymmetricTensor2<double> &Kij, struct gr_sym4 dg[3], Vec3<double> &dalpha, Mat3<double> &dbeta)
{
    double invalpha2 = 1./(gr->alpha*gr->alpha);
    double g00 = -invalpha2; Vec3<double> g0i = invalpha2*gr->beta; /* g^{00}, g^{0i} of the inverse 4-metric */
    compute_metric_jacobian(pos, gr, dg, dalpha, dbeta);
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
