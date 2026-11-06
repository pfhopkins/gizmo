/* Unit test of solids/ism_dust_hu_kernels.h against analytic solutions and fine-step references. */
#include <stdio.h>
#include <math.h>
#include "solids/ism_dust_hu_kernels.h"

static int nfail = 0;
static void check(const char *name, double got, double want, double rtol)
{
    double err = fabs(got - want) / (fabs(want) > 0 ? fabs(want) : 1.0);
    int ok = err <= rtol;
    if(!ok) {nfail++;}
    printf("%-44s got=%.6e want=%.6e relerr=%.2e %s\n", name, got, want, err, ok ? "PASS" : "FAIL");
}

static struct dustk_env make_env(double nH, double T, double x_e, double ahx, double divv, int beta)
{
    struct dustk_env e; e.nH = nH; e.T = T; e.x_e = x_e; e.G0 = 1.0;
    e.ahyd[0] = ahx; e.ahyd[1] = e.ahyd[2] = 0; e.divv = divv; e.betatron = beta; return e;
}

/* RK4 reference for (v, I_C, I_Sil) with the same physics, tiny steps */
static void reference(const struct dustk_env *e, double a, double dt, double v[3], double *IC, double *IS, int n)
{
    double plasma = dustk_plasma_coeff(e, a), fac = 3.0 * e->nH / a, h = dt / n;
    *IC = *IS = 0;
    for(int j = 0; j < n; j++) {
        double k[4][3], y[3], kc[4], ks[4];
        for(int st = 0; st < 4; st++) {
            double c = (st == 0) ? 0 : (st == 3 ? 1.0 : 0.5);
            for(int q = 0; q < 3; q++) {y[q] = v[q] + (st ? c * h * k[st-1][q] : 0);}
            double vm = dustk_norm3(y), r = dustk_trel_inv(e, a, plasma, vm);
            for(int q = 0; q < 3; q++) {k[st][q] = -y[q]*r - e->ahyd[q];}
            kc[st] = fac * dustk_Ynth_C(vm); ks[st] = fac * dustk_Ynth_Sil(vm);
        }
        for(int q = 0; q < 3; q++) {v[q] += h/6.0 * (k[0][q] + 2*k[1][q] + 2*k[2][q] + k[3][q]);}
        *IC += h/6.0 * (kc[0] + 2*kc[1] + 2*kc[2] + kc[3]);
        *IS += h/6.0 * (ks[0] + 2*ks[1] + 2*ks[2] + ks[3]);
    }
}

int main(void)
{
    /* 1. size distribution */
    double a1, w1; dustk_size_bin(1, 0, &a1, &w1);
    check("MRN single-bin a_eff = <a3>/<a2>", a1, sqrt(0.005*0.25), 1e-12);
    check("MRN single-bin weight", w1, 1.0, 1e-12);
    double wsum = 0; for(int b = 0; b < 8; b++) {double a, w; dustk_size_bin(8, b, &a, &w); wsum += w;}
    check("MRN 8-bin weights sum", wsum, 1.0, 1e-12);

    /* 2. thermal sputtering e-folding time a/(3 n Y) (Hu+19 eq 3 prefactor 0.33 Myr) */
    double Y = dustk_Yth_C(1.0e7), a = 1.0, tsp = a / (3.0 * 1.0 * Y);
    check("t_sput(a=1um,n=1) vs 0.33Myr/(Y/1e-6)", tsp, 0.33e6 / (Y / 1.0e-6), 0.02);
    double m = 1.0, T0 = 1.0e7, nH = 0.5, dt = 3.0e4, aeff = dustk_a_eff_total_um();
    double ex = exp(-dt * 3.0 * nH * dustk_Yth_Sil(T0) / aeff), mr = m;
    int nr = 100000; for(int j = 0; j < nr; j++) {mr -= mr * 3.0 * nH * dustk_Yth_Sil(T0) / aeff * (dt/nr);}
    check("thermal sputter factor vs fine Euler", ex, mr, 1e-4);

    /* 3. drag time at n=1, T=1e6, a=0.035 um (Hu+26 eq 3: 21 kyr, G omitted) */
    struct dustk_env e0 = make_env(1.0, 1.0e6, 0.0, 0.0, 0.0, 0);
    double tdG = 1.0 / dustk_trel_inv(&e0, 0.035, 0.0, 0.0) * 1.5;
    check("t_drag*G(0) at n=1,T=1e6,a=0.035", tdG, 0.59e6 * 0.035, 0.02);

    /* 4. linear drag + constant a_hydro (Hu+19 Fig 2 upper): exact eq 14 */
    for(int ia = 0; ia < 2; ia++) {
        double ag = ia ? 0.028 : 0.2;
        struct dustk_env e = make_env(10.0, 1.0e4, 0.0, -4000.0/1.0e9, 0.0, 0);
        double r = dustk_trel_inv(&e, ag, 0.0, 0.0), v[3] = {1.0, 0, 0}, I1, I2;
        double tend = 0.1 / r; /* well inside the linear branch */
        dustk_step_vrel(&e, ag, tend, v, &I1, &I2, 0.1, 100000);
        double want = 1.0*exp(-tend*r) + 4000.0/1.0e9/r*(1.0 - exp(-tend*r));
        check(ia ? "linear drag exact, a=0.028" : "linear drag exact, a=0.2", v[0], want, 1e-3);
    }

    /* 5. supersonic coupled branch vs RK4: v0=1000 km/s, hot plasma, with/without betatron */
    for(int ib = 0; ib < 2; ib++) {
        struct dustk_env e = make_env(1.0, 1.0e6, 1.2, 0.0, ib ? -1.0e-5 : 0.0, ib);
        double v[3] = {1000.0, 0, 0}, vr[3] = {1000.0, 0, 0}, IC, IS, ICr, ISr, dtt = 5.0e4;
        dustk_step_vrel(&e, 0.035, dtt, v, &IC, &IS, 0.1, 100000);
        reference(&e, 0.035, dtt, vr, &ICr, &ISr, 200000);
        check(ib ? "coupled v_rel (betatron) vs RK4" : "coupled v_rel vs RK4", v[0], vr[0], 0.02);
        check(ib ? "coupled I_nth,C (betatron) vs RK4" : "coupled I_nth,C vs RK4", IC, ICr, 0.02);
        check(ib ? "coupled I_nth,Sil (betatron) vs RK4" : "coupled I_nth,Sil vs RK4", IS, ISr, 0.02);
    }

    /* 6. shock kick: dust lags a sudden gas acceleration (v_rel -> -dv_gas) */
    {
        struct dustk_env e = make_env(1.0, 1.0e4, 1.0e-3, 300.0/100.0, 0.0, 0); /* 300 km/s over 100 yr */
        double v[3] = {0,0,0}, IC, IS;
        dustk_step_vrel(&e, 0.035, 100.0, v, &IC, &IS, 0.1, 100000);
        double vr[3] = {0,0,0}, ICr, ISr; reference(&e, 0.035, 100.0, vr, &ICr, &ISr, 200000);
        check("shock kick v_rel vs RK4", v[0], vr[0], 0.02);
        check("shock kick I_nth,C vs RK4", IC, ICr, 0.03);
    }

    /* 7. logistic growth: exact vs fine RK4 of dm/dt = r m (1-m/K) */
    {
        double m0 = 1e-4, K = 2e-3, r = 3.0e-6, tt = 2.0e6, mm = m0; int n = 200000; double h = tt / n;
        for(int j = 0; j < n; j++) {
            double k1 = r*mm*(1-mm/K), k2 = r*(mm+0.5*h*k1)*(1-(mm+0.5*h*k1)/K), k3 = r*(mm+0.5*h*k2)*(1-(mm+0.5*h*k2)/K), k4 = r*(mm+h*k3)*(1-(mm+h*k3)/K);
            mm += h/6*(k1+2*k2+2*k3+k4);
        }
        check("logistic exact vs RK4", dustk_logistic(m0, K, r, tt), mm, 1e-8);
        check("logistic saturates at K", dustk_logistic(m0, K, r, 1e12), K, 1e-12);
        check("logistic no change above K", dustk_logistic(3e-3, K, r, tt), 3e-3, 0);
    }

    /* 8. growth law (Hu+26 eq 4) and temperature cut */
    check("t_grow(n=1,T=100,Z'=1) = 1.5 Gyr (a_eff/0.035)", 1.0/dustk_growth_rate0(1.0, 100.0, 1.5, 1.0), 1.5e9*dustk_a_eff_total_um()/0.035, 1e-12);
    check("no growth at T=300 K", dustk_growth_rate0(1.0e3, 300.0, 1.5, 1.0), 0.0, 0);
    check("no growth at T=1e4 K", dustk_growth_rate0(1.0e3, 1.0e4, 1.5, 1.0), 0.0, 0);
    {
        double Ctot = 2.48e-3, Cco = 1.0e-3, D = 1.0e-4;
        double Dn = dustk_grow_carbon(D, Ctot, Cco, 1.0, 1.0e-6, 1.0e9);
        check("carbon growth capped at C_tot - C_CO", Dn, Ctot - Cco, 1e-10);
        double pool[4] = {6.0e-3, 7.4e-4, 6.96e-4, 1.35e-3}, frac[4] = {64./172, 24./172, 28./172, 56./172};
        double S = dustk_grow_silicate(1.0e-4, pool, frac, 1.0, 1.0e-6, 1.0e12), K = 1e30;
        for(int k = 0; k < 4; k++) {if(pool[k]/frac[k] < K) {K = pool[k]/frac[k];}}
        check("silicate growth capped at scarcest pool", S, K, 1e-10);
        int ok = 1; for(int k = 0; k < 4; k++) {if(S*frac[k] > pool[k]*(1+1e-12)) {ok = 0;}}
        check("silicate elements within pools", ok, 1, 0);
    }

    printf("%s (%d failures)\n", nfail ? "DUST_HU_KERNELS FAIL" : "DUST_HU_KERNELS PASS", nfail);
    return nfail ? 1 : 0;
}
