#ifndef RESOLVEDISM_DUST_KERNELS_H
#define RESOLVEDISM_DUST_KERNELS_H
/* Hu+19/23/26 dust microphysics on plain doubles (um, yr, km/s, cm^-3, K); no GIZMO state. */
#include <math.h>

#define DUSTK_AMIN_UM    0.005          /* MRN range (Hu+19/26) */
#define DUSTK_AMAX_UM    0.25
#define DUSTK_RHO_GR     3.0            /* grain material density [g cm^-3] (Hu+19) */
#define DUSTK_SEC_PER_YR 3.15576e7
#define DUSTK_KB         1.380649e-16
#define DUSTK_MP         1.67262192e-24
#define DUSTK_QE         4.80320471e-10
#define DUSTK_EV         1.602176634e-12
#define DUSTK_UMAX_EV    10.0           /* grain potential cap (photoelectric limit) */
#define DUSTK_VCOUPLE    30.0           /* km/s: below this v_rel and m_d decouple (Hu+19 2.5.2) */
#define DUSTK_SHOCK_EPS  0.1            /* smeared-shock limiter: drag rate capped at this x |div v| */

static inline double dustk_poly5_pow10(double x, double a0, double a1, double a2, double a3, double a4, double a5)
{
    return pow(10.0, a0 + x*(a1 + x*(a2 + x*(a3 + x*(a4 + x*a5)))));
}

/* Nozawa+06 erosion rates [um yr^-1 cm^3], Hu+19 Table 1 fits */
static inline double dustk_Yth_C(double T)
{
    if(T < 3.0e4) {return 0;} if(T > 1.0e10) {T = 1.0e10;}
    return dustk_poly5_pow10(log10(T), -2.34333937e+02, 1.38485732e+02, -3.39021615e+01, 4.17705353e+00, -2.58281473e-01, 6.38827523e-03);
}
static inline double dustk_Yth_Sil(double T)
{
    if(T < 3.0e4) {return 0;} if(T > 1.0e10) {T = 1.0e10;}
    return dustk_poly5_pow10(log10(T), -2.34790500e+02, 1.33208637e+02, -3.13027448e+01, 3.71345730e+00, -2.21823668e-01, 5.31746427e-03);
}
static inline double dustk_Ynth_C(double v)
{
    if(v < 20.0) {return 0;} if(v > 3.0e5) {v = 3.0e5;}
    return dustk_poly5_pow10(log10(v), -4.87021030e+01, 5.57114303e+01, -2.88275125e+01, 7.44225761e+00, -9.65469006e-01, 5.00882872e-02);
}
static inline double dustk_Ynth_Sil(double v)
{
    if(v < 20.0) {return 0;} if(v > 3.0e5) {v = 3.0e5;}
    return dustk_poly5_pow10(log10(v), -3.24171434e+01, 2.86540492e+01, -1.15865054e+01, 2.23338917e+00, -2.08259379e-01, 7.36059545e-03);
}

/* Hu+19 eqs 15-21: log bin b of nbin over MRN; effective size and mass weight */
static inline void dustk_size_bin(int nbin, int b, double *a_eff_um, double *mass_weight)
{
    double la = log10(DUSTK_AMIN_UM), lb = log10(DUSTK_AMAX_UM), d = (lb - la) / nbin;
    double alo = pow(10.0, la + b*d), ahi = pow(10.0, la + (b+1)*d);
    *a_eff_um = sqrt(alo * ahi);
    *mass_weight = (sqrt(ahi) - sqrt(alo)) / (sqrt(DUSTK_AMAX_UM) - sqrt(DUSTK_AMIN_UM));
}
static inline double dustk_a_eff_total_um(void) {return sqrt(DUSTK_AMIN_UM * DUSTK_AMAX_UM);} /* <a^3>/<a^2> of MRN */

struct dustk_env {
    double nH;        /* cm^-3 */
    double T;         /* K, held fixed over the step */
    double x_e;       /* n_e/n_H (ions ~ electrons) */
    double G0;        /* Habing */
    double ahyd[3];   /* gas hydro acceleration [km/s/yr] */
    double divv;      /* gas velocity divergence [1/yr] */
    int betatron;     /* 1: Hu+19 eq 11 (B ~ rho) */
    int shock_limit;  /* 1: drag rate capped by the compression rate in converging flow (smeared-shock correction) */
};

static inline double dustk_norm3(const double v[3]) {return sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);}

/* plasma-drag coefficient x_ion phi^2 ln Lambda of Hu+19 eq 9, grain charge from eq 10 */
static inline double dustk_plasma_coeff(const struct dustk_env *e, double a_um)
{
    double ne = e->x_e * e->nH; if(ne <= 0 || e->T <= 0) {return 0;}
    double a_cm = a_um * 1.0e-4, kT = DUSTK_KB * e->T;
    double gamma = e->G0 * sqrt(e->T) / ne;
    double Zgr = 36.0 * (gamma / 1.0e3) * (a_um / 0.1);
    double Zmax = DUSTK_UMAX_EV * DUSTK_EV * a_cm / (DUSTK_QE * DUSTK_QE);
    if(Zgr > Zmax) {Zgr = Zmax;}
    double phi = Zgr * DUSTK_QE * DUSTK_QE / (a_cm * kT);
    if(phi <= 0) {return 0;}
    double Lambda = 3.0 / (2.0 * a_cm * DUSTK_QE * phi) * sqrt(kT / (M_PI * ne));
    double lnL = (Lambda > 1.0) ? log(Lambda) : 0;
    double x_ion = (e->x_e < 1.0) ? e->x_e : 1.0;
    return x_ion * phi * phi * lnL;
}

/* 1/t_rel [1/yr] at |v_rel| = v (Hu+19 eqs 7-9, 12) */
static inline double dustk_trel_inv(const struct dustk_env *e, double a_um, double plasma, double v)
{
    double cs = sqrt(DUSTK_KB * e->T / DUSTK_MP); /* cm/s */
    double s = v * 1.0e5 / (sqrt(2.0) * cs);
    double G = 1.5 * sqrt(1.0 + 0.44*s*s) + plasma / (1.3 + s*s*s);
    double tdrag_s = 2.0 * sqrt(2.0) * (a_um*1.0e-4) * DUSTK_RHO_GR * cs / (3.0 * e->nH * DUSTK_KB * e->T * G);
    double r = DUSTK_SEC_PER_YR / tdrag_s;
    if(e->shock_limit && e->divv < 0 && dustk_norm3(e->ahyd) >= DUSTK_VCOUPLE * (-e->divv)) {r = r / (1.0 + r / (DUSTK_SHOCK_EPS * (-e->divv)));}
    if(e->betatron) {r += 0.5 * e->divv;}
    return r;
}


/* Advance v_rel of one size bin over dt and return int 3 nH Y_nth / a dt for C and silicate (Hu+19 2.5) */
static inline void dustk_step_vrel(const struct dustk_env *e, double a_um, double dt, double v[3],
                                   double *Inth_C, double *Inth_Sil, double fsub, int nmax)
{
    *Inth_C = *Inth_Sil = 0;
    if(dt <= 0 || e->T <= 0 || e->nH <= 0) {return;}
    double plasma = dustk_plasma_coeff(e, a_um), fac = 3.0 * e->nH / a_um;
    double vmag = dustk_norm3(v), amag = dustk_norm3(e->ahyd);
    double rinv = dustk_trel_inv(e, a_um, plasma, vmag);
    double reach = (rinv > 0 && dt * rinv > 1.0) ? amag / rinv : amag * dt; /* max |v| the kick can drive */
    if(vmag < DUSTK_VCOUPLE && reach < DUSTK_VCOUPLE && rinv > 0)
    {
        if(dt * rinv > 3.0) { /* terminal velocity, G at the terminal speed */
            double vt = amag / rinv; int it;
            for(it = 0; it < 3; it++) {double r2 = dustk_trel_inv(e, a_um, plasma, vt); if(r2 <= 0) {break;} rinv = r2; vt = amag / rinv;}
            for(int k = 0; k < 3; k++) {v[k] = -e->ahyd[k] / rinv;}
        } else {
            double cs = sqrt(DUSTK_KB * e->T / DUSTK_MP), s = vmag * 1.0e5 / (sqrt(2.0) * cs);
            if(s < 0.5) { /* linear drag: exact (Hu+19 eq 14) */
                double ex = exp(-dt * rinv);
                for(int k = 0; k < 3; k++) {v[k] = v[k]*ex - e->ahyd[k] / rinv * (1.0 - ex);}
            } else { /* backward Euler: v_rel too low for nonthermal sputtering */
                int n = (int)ceil(dt * rinv); if(n < 1) {n = 1;} if(n > 8) {n = 8;}
                double h = dt / n;
                for(int j = 0; j < n; j++) {
                    double ri = dustk_trel_inv(e, a_um, plasma, dustk_norm3(v));
                    for(int k = 0; k < 3; k++) {v[k] = (v[k] - e->ahyd[k]*h) / (1.0 + h*ri);}
                }
            }
        }
        return;
    }
    /* coupled (m_d, v_rel): Heun subcycling, dt_sub = fsub |t_rel| */
    double t = 0, hmin = dt / nmax; int nit = 0;
    while(t < dt && nit++ < nmax + 8)
    {
        double v0 = dustk_norm3(v), r0 = dustk_trel_inv(e, a_um, plasma, v0);
        double h = (r0 != 0) ? fsub / fabs(r0) : dt;
        if(amag > 0) {double ha = fsub * ((v0 > DUSTK_VCOUPLE) ? v0 : DUSTK_VCOUPLE) / amag; if(ha < h) {h = ha;}} /* resolve the kick */
        if(h < hmin) {h = hmin;} if(h > dt - t) {h = dt - t;}
        double k0C = fac * dustk_Ynth_C(v0), k0S = fac * dustk_Ynth_Sil(v0);
        double vn[3];
        if(h * fabs(r0) > 1.0) { /* subcycle cap hit: stable update */
            double g = (r0 > 0) ? 1.0 / (1.0 + h*r0) : exp(-h*r0);
            for(int k = 0; k < 3; k++) {vn[k] = (v[k] - e->ahyd[k]*h) * g;}
        } else {
            double f0[3], vp[3], f1[3];
            for(int k = 0; k < 3; k++) {f0[k] = -v[k]*r0 - e->ahyd[k]; vp[k] = v[k] + f0[k]*h;}
            double r1 = dustk_trel_inv(e, a_um, plasma, dustk_norm3(vp));
            for(int k = 0; k < 3; k++) {f1[k] = -vp[k]*r1 - e->ahyd[k]; vn[k] = v[k] + 0.5*(f0[k] + f1[k])*h;}
        }
        double v1 = dustk_norm3(vn);
        *Inth_C   += 0.5 * (k0C + fac * dustk_Ynth_C(v1)) * h;
        *Inth_Sil += 0.5 * (k0S + fac * dustk_Ynth_Sil(v1)) * h;
        for(int k = 0; k < 3; k++) {v[k] = vn[k];}
        t += h;
    }
}

/* exact solution of dm/dt = r m (1 - m/K) (Hu+23 eq 4 with fixed pool); no change if m0 >= K */
static inline double dustk_logistic(double m0, double K, double r, double dt)
{
    if(m0 <= 0 || K <= m0 || r <= 0 || dt <= 0) {return m0;}
    return K / (1.0 + (K/m0 - 1.0) * exp(-r * dt));
}

/* 1/t_grow [1/yr] at Z'=1 (Hu+23/26 eq 4/6); zero at T >= 300 K */
static inline double dustk_growth_rate0(double nH, double T, double tgrow_gyr, double coulomb)
{
    if(T >= 300.0 || T <= 0 || nH <= 0) {return 0;}
    double tg = tgrow_gyr * 1.0e9 / nH / sqrt(T / 100.0) * (dustk_a_eff_total_um() / 0.035) / coulomb;
    return 1.0 / tg;
}

/* carbon growth: dm/dt = Z' r0 m (1 - f_C - f_CO), pool C_tot - C_CO (mass fractions) */
static inline double dustk_grow_carbon(double D, double Ctot, double Cco, double Zp, double r0, double dt)
{
    double P = Ctot - Cco;
    if(P <= 0 || Ctot <= 0) {return D;}
    return dustk_logistic(D, P, Zp * r0 * P / Ctot, dt);
}

/* silicate growth: Hu f_Si on the key element Si, capacity from the scarcest of pool[O,Mg,Si,Fe]/frac */
static inline double dustk_grow_silicate(double S, const double pool[4], const double frac[4], double Zp, double r0, double dt)
{
    double K = 1.0e30;
    for(int k = 0; k < 4; k++) {double c = (pool[k] > 0) ? pool[k] / frac[k] : 0; if(c < K) {K = c;}}
    double KSi = pool[2] / frac[2];
    if(K <= 0 || KSi <= 0) {return S;}
    return dustk_logistic(S, K, Zp * r0 * K / KSi, dt);
}

#endif
