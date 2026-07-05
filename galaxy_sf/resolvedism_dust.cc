#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../declarations/allvars.h"
#include "../core/proto.h"

/*
 * Resolved-ISM dust evolution model for single-star galaxy formation.
 *
 * Tracks 5 dust mass fractions per gas cell:
 *   Dust[0] = carbonaceous dust (C)
 *   Dust[1] = O in silicate (FeMgSiO4)
 *   Dust[2] = Mg in silicate
 *   Dust[3] = Si in silicate
 *   Dust[4] = Fe in silicate
 *
 * Silicate stoichiometry: FeMgSiO4 (M_w = 172 amu)
 *   O4 = 64/172, Mg = 24/172, Si = 28/172, Fe = 56/172
 *
 * Physics:
 *   - Thermal sputtering: Nozawa et al. 2006 polynomial fits
 *   - Dust production: condensation fractions applied to SN/AGB/Ia metal yields (in resolvedism_fb.cc)
 *   - SN shock destruction: McKee 1989 (in resolvedism_fb.cc)
 *   - CHEMCOOL coupling: per-particle DGR, gas-phase metal depletion (in chemcool.cc)
 *
 * This file handles the ISM dust processing (sputtering) which is operator-split
 * from the chemistry solver, called after cooling_parent_routine() each timestep.
 */

#ifdef GALSF_RESOLVEDISM_DUST

/* Silicate mass fractions within FeMgSiO4 (M_w = 172 amu) */
#define DUST_SIL_FRAC_O  (64.0 / 172.0)
#define DUST_SIL_FRAC_MG (24.0 / 172.0)
#define DUST_SIL_FRAC_SI (28.0 / 172.0)
#define DUST_SIL_FRAC_FE (56.0 / 172.0)


/* ---- Nozawa et al. 2006 thermal sputtering yield fits ---- */
/* Returns sputtering rate in units of [micron yr^-1 cm^3] */

static double get_yield_therSput_C(double T)
{
    if(T < 3e4) return 0;
    if(T > 1e10) T = 1e10;
    double x = log10(T);
    double x2 = x*x, x3 = x2*x, x4 = x3*x, x5 = x4*x;
    return pow(10, -2.34333937e+02 + 1.38485732e+02*x - 3.39021615e+01*x2
                 + 4.17705353e+00*x3 - 2.58281473e-01*x4 + 6.38827523e-03*x5);
}

static double get_yield_therSput_Sil(double T)
{
    if(T < 3e4) return 0;
    if(T > 1e10) T = 1e10;
    double x = log10(T);
    double x2 = x*x, x3 = x2*x, x4 = x3*x, x5 = x4*x;
    return pow(10, -2.34790500e+02 + 1.33208637e+02*x - 3.13027448e+01*x2
                 + 3.71345730e+00*x3 - 2.21823668e-01*x4 + 5.31746427e-03*x5);
}


/* ---- Main ISM dust evolution routine ---- */
/* Called once per timestep after cooling, operator-split from chemistry */

void resolvedism_dust_evolve(void)
{
    int i;
    double grain_size = 0.1; /* fiducial grain radius in microns */
    /* conversion factor: sputtering rate is in yr^-1, need code time^-1 */
    double yr_to_code = SECONDS_PER_YEAR / UNIT_TIME_IN_CGS;

    for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    {
        if(P[i].Type != 0) continue;

        /* total dust content — skip if no dust */
        double dust_total = 0;
        int k;
        for(k = 0; k < NUM_RESOLVEDISM_DUST; k++) dust_total += CellP[i].Dust[k];
        if(dust_total <= 0) continue;

        /* physical quantities */
        double rho_phys = CellP[i].Density * All.cf_a3inv;
        double nH = rho_phys * UNIT_DENSITY_IN_NHCGS;
#ifdef CHEMCOOL
        double T = CellP[i].Temp;
#else
        double ne_dummy = 1, nh0_dummy = 0, nHe0_dummy, nHepp_dummy, nhp_dummy, nHeII_dummy, mu_dummy = 1;
        double T = ThermalProperties(CellP[i].InternalEnergyPred, rho_phys, i, &mu_dummy, &ne_dummy, &nh0_dummy, &nhp_dummy, &nHe0_dummy, &nHeII_dummy, &nHepp_dummy);
#endif

        /* particle timestep in code units */
        double dt = GET_PARTICLE_TIMESTEP_IN_PHYSICAL(i);

        /* ---- Thermal sputtering (Nozawa+ 2006) ---- */
        /* tau_sput_inv = 3 * nH / a * Y(T), in yr^-1 */
        double tau_sput_C_inv   = 3.0 * nH / grain_size * get_yield_therSput_C(T);
        double tau_sput_Sil_inv = 3.0 * nH / grain_size * get_yield_therSput_Sil(T);

        /* convert from yr^-1 to code time^-1 */
        tau_sput_C_inv   *= yr_to_code;
        tau_sput_Sil_inv *= yr_to_code;

        /* exponential integrator for unconditional stability:
           Dust_new = Dust_old * exp(-dt * tau_inv) */
        if(tau_sput_C_inv > 0) {
            double fac_C = exp(-dt * tau_sput_C_inv);
            CellP[i].Dust[0] *= fac_C;
        }
        if(tau_sput_Sil_inv > 0) {
            double fac_Sil = exp(-dt * tau_sput_Sil_inv);
            CellP[i].Dust[1] *= fac_Sil;
            CellP[i].Dust[2] *= fac_Sil;
            CellP[i].Dust[3] *= fac_Sil;
            CellP[i].Dust[4] *= fac_Sil;
        }

#ifdef GALSF_RESOLVEDISM_DUST_SHOCK_DESTRUCTION
        /* ---- Shock-tracked SN dust destruction (2026-07-05) ----
         * Replaces the injection-kernel McKee 1989 hook (see fb_thermal.cc), which is
         * an UNRESOLVED-SNR estimator: in Stella's resolved regime it saturates at
         * frac=1 over the ~32-cell kernel (measured 0.032 Msun vs ~1 Msun McKee intent
         * in the n=1 A/B test), while thermal sputtering alone captures only ~0.008
         * Msun (the bubble goes radiative in ~1 Myr; the real destroyer is KINETIC/
         * betatron sputtering + shattering in the radiative shock, not modeled by the
         * Nozawa thermal rates). Here we destroy grains IN CELLS THE RESOLVED SHOCK
         * ACTUALLY SWEEPS: compression (divv<0) + signal-velocity excess over sound.
         * Efficiency per shock crossing: saturating Slavin+2015-like form,
         *   eps_sil(v_s) = 0.4 v_s^2/(v_s^2+100^2),  eps_C = 0.5 eps_sil
         * applied as a rate over the kernel-crossing time h/v_s. Total destroyed =
         * swept mass x DGR x eps -> the McKee-intended budget emerges from the
         * resolved blast, with the correct spatial/anisotropic distribution. */
        {
            double cs_kms   = Get_Gas_effective_soundspeed_i(i) * UNIT_VEL_IN_KMS;
            double vsig_kms = CellP[i].MaxSignalVel * UNIT_VEL_IN_KMS;
            double divv     = P[i].Particle_DivVel;   /* code 1/time; <0 = compression */
            double v_shock  = 0.5 * (vsig_kms - 2.0 * cs_kms); /* pairwise approach speed */
            if(divv < 0 && v_shock > 50.0)
            {
                double v2 = v_shock * v_shock;
                double eps_sil = 0.4 * v2 / (v2 + 100.0*100.0);
                double eps_C   = 0.5 * eps_sil;
                double h_code  = Get_Particle_Size(i) * All.cf_atime;
                double t_cross = h_code / (v_shock / UNIT_VEL_IN_KMS); /* code time for shock to cross the cell */
                double fac_sil = exp(-dt * eps_sil / t_cross);
                double fac_C   = exp(-dt * eps_C   / t_cross);
                CellP[i].Dust[0] *= fac_C;
                CellP[i].Dust[1] *= fac_sil; CellP[i].Dust[2] *= fac_sil;
                CellP[i].Dust[3] *= fac_sil; CellP[i].Dust[4] *= fac_sil;
            }
        }
#endif

        /* ---- ISM grain growth (accretion of gas-phase metals onto grains) ---- */
        /* Sticking efficiency S(T_gas, T_dust) = 1 / (1 + 0.04*sqrt(T_gas + T_dust))
           (Hirashita 2000).  Smoothly transitions from S~1 at low T to S~0 at high T. */
        /* Rates from Dwek 1998 / Draine 2009, same form as gizmo2017 evolve_abundances_dust.F */
        double T_dust = CellP[i].DustTemp;
        double sticking = 1.0 / (1.0 + 0.04 * sqrt(T + T_dust));
        if(sticking > 1e-4 && dust_total > 0) {
            double sqrtT = sqrt(T);
            /* gas-phase metal mass fractions */
            double Z_C  = DMAX(P[i].Metallicity[MET_OF(ELEM_C)]  - CellP[i].Dust[0], 0);
            double Z_O  = DMAX(P[i].Metallicity[MET_OF(ELEM_O)]  - CellP[i].Dust[1], 0);
            double Z_Mg = DMAX(P[i].Metallicity[MET_OF(ELEM_Mg)] - CellP[i].Dust[2], 0);
            double Z_Si = DMAX(P[i].Metallicity[MET_OF(ELEM_Si)] - CellP[i].Dust[3], 0);
            double Z_Fe = DMAX(P[i].Metallicity[MET_OF(ELEM_Fe)] - CellP[i].Dust[4], 0);

            /* condensation fractions: fraction of element already in dust */
            double f_C  = (P[i].Metallicity[MET_OF(ELEM_C)]  > 0) ? CellP[i].Dust[0] / P[i].Metallicity[MET_OF(ELEM_C)]  : 1.0;
            double f_Si = (P[i].Metallicity[MET_OF(ELEM_Si)] > 0) ? CellP[i].Dust[3] / P[i].Metallicity[MET_OF(ELEM_Si)] : 1.0;
            f_C = DMIN(f_C, 1.0); f_Si = DMIN(f_Si, 1.0);

            /* accretion timescale^{-1} [yr^-1] (gizmo2017 formula) */
            /* t_accr_inv = prefactor * nH / grain_size * sqrt(T) * (Z_elem / Z_solar) * sticking * (1 - f) */
            double t_accr_C_inv  = 2.15e-12 * nH / grain_size * sqrtT * (Z_C  / 2.07e-3) * sticking * (1.0 - f_C);
            double t_accr_Si_inv = 2.01e-12 * nH / grain_size * sqrtT * (Z_Si / 6.83e-4) * sticking * (1.0 - f_Si);

            /* convert to code time^-1 */
            t_accr_C_inv  *= yr_to_code;
            t_accr_Si_inv *= yr_to_code;

            /* Carbonaceous: grow C dust from gas-phase C */
            if(t_accr_C_inv > 0 && Z_C > 0) {
                double dDust_C = CellP[i].Dust[0] * (exp(dt * t_accr_C_inv) - 1.0);
                dDust_C = DMIN(dDust_C, Z_C); /* can't exceed available gas-phase C */
                CellP[i].Dust[0] += dDust_C;
            }

            /* Silicate: grow from gas-phase O, Mg, Si, Fe in stoichiometric ratios */
            /* Si is key element; grow silicate limited by Si, then check other elements */
            if(t_accr_Si_inv > 0 && Z_Si > 0) {
                double dDust_Si = CellP[i].Dust[3] * (exp(dt * t_accr_Si_inv) - 1.0);
                dDust_Si = DMIN(dDust_Si, Z_Si); /* can't exceed available gas-phase Si */

                /* stoichiometric amounts of other elements needed */
                double dSil_total = dDust_Si / DUST_SIL_FRAC_SI; /* total silicate mass */
                double dDust_O  = dSil_total * DUST_SIL_FRAC_O;
                double dDust_Mg = dSil_total * DUST_SIL_FRAC_MG;
                double dDust_Fe = dSil_total * DUST_SIL_FRAC_FE;

                /* limit by available gas-phase metals */
                if(dDust_O  > Z_O)  { dSil_total *= Z_O  / dDust_O;  }
                if(dDust_Mg > Z_Mg) { dSil_total = DMIN(dSil_total, Z_Mg / DUST_SIL_FRAC_MG * DUST_SIL_FRAC_SI / dDust_Si * dSil_total); }
                if(dDust_Fe > Z_Fe) { dSil_total = DMIN(dSil_total, Z_Fe / DUST_SIL_FRAC_FE * DUST_SIL_FRAC_SI / dDust_Si * dSil_total); }

                /* re-derive stoichiometric amounts from limited total */
                if(dSil_total > 0) {
                    CellP[i].Dust[1] += dSil_total * DUST_SIL_FRAC_O;
                    CellP[i].Dust[2] += dSil_total * DUST_SIL_FRAC_MG;
                    CellP[i].Dust[3] += dSil_total * DUST_SIL_FRAC_SI;
                    CellP[i].Dust[4] += dSil_total * DUST_SIL_FRAC_FE;
                }
            }
        }

        /* floor: ensure non-negative and don't exceed total element abundance */
        for(k = 0; k < NUM_RESOLVEDISM_DUST; k++) {
            if(CellP[i].Dust[k] < 0) CellP[i].Dust[k] = 0;
        }
        /* conservation: dust can't exceed total metal content */
        CellP[i].Dust[0] = DMIN(CellP[i].Dust[0], DMAX(P[i].Metallicity[MET_OF(ELEM_C)],  0));
        CellP[i].Dust[1] = DMIN(CellP[i].Dust[1], DMAX(P[i].Metallicity[MET_OF(ELEM_O)],  0));
        CellP[i].Dust[2] = DMIN(CellP[i].Dust[2], DMAX(P[i].Metallicity[MET_OF(ELEM_Mg)], 0));
        CellP[i].Dust[3] = DMIN(CellP[i].Dust[3], DMAX(P[i].Metallicity[MET_OF(ELEM_Si)], 0));
        CellP[i].Dust[4] = DMIN(CellP[i].Dust[4], DMAX(P[i].Metallicity[MET_OF(ELEM_Fe)], 0));
    }
}


/* ---- Dust condensation fractions for stellar feedback ---- */
/* Called from resolvedism_fb.cc to compute dust yields from metal yields */

void resolvedism_dust_condensation(int sne_flag, double *metal_yields, double *dust_yields)
{
    /* metal_yields[0..14] = yields for 15 elements (H,He,C,N,O,F,Ne,Na,Mg,Al,Si,S,Ca,Ti,Fe)
     * dust_yields[0..4]   = output dust yields (C, O_sil, Mg_sil, Si_sil, Fe_sil)
     * sne_flag: 1=CCSN, 2=AGB, 4=TypeIa
     */
    int k;
    for(k = 0; k < NUM_RESOLVEDISM_DUST; k++) dust_yields[k] = 0;

    /* Per-channel condensation efficiencies — VERIFIED against Choban+2022 Table 1
     * ("Species" implementation; arXiv:2201.12369) on 2026-07-05:
     *   delta_SNII = 0.15 (carbon), 0.00035 (silicate), 0.001 (iron); Ia = 0.005 iron;
     *   AGB C/O>1: 1.0 C; AGB C/O<1: 0.8 O/Mg/Si/Fe.   All match this code.
     * The tiny SN silicate efficiency is DELIBERATE (calibrated to presolar-grain
     * abundances: SN silicate grains are rare in meteorites).  Silicates are built
     * by ISM accretion in cold clouds (resolvedism_dust_evolve growth term), NOT by
     * SN condensation.  (Known literature tension: ejecta-condensation models and
     * SN 1987A suggest higher; we follow Choban+22.)
     * SN subtypes have their own named constants (flags 5=PISN/PPISN, 6=ECSN),
     * currently = CCSN values; Choban+22 has no PISN/ECSN — future lit pass. */
    double CCSN_C_cond  = 0.15,  CCSN_Si_cond  = 0.00035, CCSN_Fe_cond  = 0.001;
    double PISN_C_cond  = 0.15,  PISN_Si_cond  = 0.00035, PISN_Fe_cond  = 0.001; /* no PISN in Choban+22 */
    double ECSN_C_cond  = 0.15,  ECSN_Si_cond  = 0.00035, ECSN_Fe_cond  = 0.001; /* no ECSN in Choban+22 */

    if(sne_flag == 1 || sne_flag == 5 || sne_flag == 6) {
        /* ---- SN channels: CCSN (1), PISN/PPISN (5), ECSN (6) ---- */
        double C_cond  = (sne_flag==5) ? PISN_C_cond  : (sne_flag==6) ? ECSN_C_cond  : CCSN_C_cond;
        double Si_cond = (sne_flag==5) ? PISN_Si_cond : (sne_flag==6) ? ECSN_Si_cond : CCSN_Si_cond;
        double Fe_cond = (sne_flag==5) ? PISN_Fe_cond : (sne_flag==6) ? ECSN_Fe_cond : CCSN_Fe_cond;

        /* Carbonaceous dust */
        dust_yields[0] = C_cond * DMAX(metal_yields[ELEM_C], 0);

        /* Silicate: limited by Si (key element), stoichiometric FeMgSiO4 */
        double Si_dust = Si_cond * DMAX(metal_yields[ELEM_Si], 0);
        double Sil_total = Si_dust / DUST_SIL_FRAC_SI;  /* total silicate mass from Si budget */

        /* check that O, Mg, Fe can support this amount of silicate */
        double O_needed  = Sil_total * DUST_SIL_FRAC_O;
        double Mg_needed = Sil_total * DUST_SIL_FRAC_MG;
        double Fe_needed = Sil_total * DUST_SIL_FRAC_FE;

        if(O_needed  > DMAX(metal_yields[ELEM_O], 0))  { Sil_total = DMAX(metal_yields[ELEM_O], 0)  / DUST_SIL_FRAC_O; }
        if(Mg_needed > DMAX(metal_yields[ELEM_Mg], 0)) { Sil_total = DMIN(Sil_total, DMAX(metal_yields[ELEM_Mg], 0) / DUST_SIL_FRAC_MG); }
        if(Fe_needed > DMAX(metal_yields[ELEM_Fe], 0)) { Sil_total = DMIN(Sil_total, DMAX(metal_yields[ELEM_Fe], 0) / DUST_SIL_FRAC_FE); }

        dust_yields[1] = Sil_total * DUST_SIL_FRAC_O;
        dust_yields[2] = Sil_total * DUST_SIL_FRAC_MG;
        dust_yields[3] = Sil_total * DUST_SIL_FRAC_SI;
        dust_yields[4] = DMAX(Fe_cond * DMAX(metal_yields[ELEM_Fe], 0), Sil_total * DUST_SIL_FRAC_FE);

    } else if(sne_flag == 2 || sne_flag == 3) {
        /* ---- AGB death (2) and slow/cool WINDS (3): C/O-driven condensation ----
         * DUST FIX #1 (2026-07-05): the wind channel previously produced ZERO dust.
         * With WINDS on, the AGB superwind AND the at-death envelope force-dump all
         * leave through flag 3, so the flag-2 C/O logic below only ever saw the last
         * <=1e-6 Msun scraps -> AGB dust (the dominant galactic dust source) was
         * effectively zero in production-config builds.  The caller gates flag-3
         * condensation on wind speed (v_w < 100 km/s: AGB/RSG regime); fast hot
         * MS/WR winds do not condense here (WC carbon dust: known omission, lit pass). */
        double C_yield = DMAX(metal_yields[ELEM_C], 0);
        double O_yield = DMAX(metal_yields[ELEM_O], 0);

        /* number abundances for C/O ratio */
        double n_C = C_yield / 12.0;
        double n_O = O_yield / 16.0;

        if(n_C > n_O) {
            /* Carbon-rich AGB: excess C (after CO formation) condenses into carbonaceous dust */
            dust_yields[0] = C_yield - 0.75 * O_yield;  /* C - (12/16)*O = excess C mass */
            if(dust_yields[0] < 0) dust_yields[0] = 0;
        } else {
            /* Oxygen-rich AGB: silicate dust with 0.8 condensation efficiency */
            double cond_eff = 0.8;
            double Mg_yield = DMAX(metal_yields[ELEM_Mg], 0);
            double Si_yield = DMAX(metal_yields[ELEM_Si], 0);
            double Fe_yield = DMAX(metal_yields[ELEM_Fe], 0);

            /* Si is typically the key element */
            double Si_dust = cond_eff * Si_yield;
            double Sil_total = Si_dust / DUST_SIL_FRAC_SI;

            /* limit by available elements */
            double O_avail = O_yield - (16.0/12.0) * C_yield;  /* O not locked in CO */
            if(O_avail < 0) O_avail = 0;
            if(Sil_total * DUST_SIL_FRAC_O  > O_avail)        { Sil_total = O_avail / DUST_SIL_FRAC_O; }
            if(Sil_total * DUST_SIL_FRAC_MG > cond_eff * Mg_yield) { Sil_total = DMIN(Sil_total, cond_eff * Mg_yield / DUST_SIL_FRAC_MG); }
            if(Sil_total * DUST_SIL_FRAC_FE > cond_eff * Fe_yield) { Sil_total = DMIN(Sil_total, cond_eff * Fe_yield / DUST_SIL_FRAC_FE); }

            dust_yields[1] = Sil_total * DUST_SIL_FRAC_O;
            dust_yields[2] = Sil_total * DUST_SIL_FRAC_MG;
            dust_yields[3] = Sil_total * DUST_SIL_FRAC_SI;
            dust_yields[4] = Sil_total * DUST_SIL_FRAC_FE;
        }

    } else if(sne_flag == 4) {
        /* ---- Type Ia: small amount of metallic iron ---- */
        double Fe_cond = 0.005;
        dust_yields[4] = Fe_cond * DMAX(metal_yields[ELEM_Fe], 0);
    }
    /* sne_flag == 3 handled above with the AGB C/O logic (caller gates on v_wind) */
}


#ifdef GALSF_RESOLVEDISM_DUST_SELFTEST
/* Physics self-test for the resolvedism dust model (2026-07-05).
 * Feeds REAL stellar-table metal yields into the condensation/destruction/rate
 * functions and dumps a table the python checkers validate against the analytic
 * physics.  Mirrors the code's own channel-flag logic (SN->flag 1, AGB->2, Ia->4),
 * so it exposes what the model ACTUALLY does — including PISN/ECSN using the CCSN
 * condensation prescription.  Runs at startup (after tables + units are up), gated. */
void resolvedism_dust_selftest(void)
{
    if(ThisTask != 0) return;
    printf("DUSTTEST_BEGIN\n");
    printf("DUSTTEST_SILFRAC O=%.6f Mg=%.6f Si=%.6f Fe=%.6f\n", DUST_SIL_FRAC_O, DUST_SIL_FRAC_MG, DUST_SIL_FRAC_SI, DUST_SIL_FRAC_FE);
    printf("DUSTTEST_CONDEFF CCSN_C=0.15 CCSN_Si=0.00035 CCSN_Fe=0.001 AGB_Osil=0.8 Ia_Fe=0.005\n");
    /* --- condensation with real table yields, across the mass spectrum --- */
    double masses[] = {1.2, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 40.0, 60.0, 100.0, 150.0, 200.0, 300.0};
    double Zs[] = {0.001, 0.02};
    int nm = sizeof(masses)/sizeof(double), nz = sizeof(Zs)/sizeof(double), im, iz, k;
    for(iz = 0; iz < nz; iz++) for(im = 0; im < nm; im++) {
        double M = masses[im], Z = Zs[iz], logM = log10(M), logZ = log10(Z);
        int rt = stellar_remnant_type(logM, logZ);
        double rem = stellar_remnant_mass(logM, logZ);
        double my[STBL_NELEM]; double Mej = 0, Zmet = 0;
        int flag = (rt == REM_WD) ? 2 : (rt == REM_PISN || rt == REM_PPISN) ? 5 : (rt == REM_ECSN) ? 6 : 1; /* mirror fb_thermal/fb_momentum mapping */
        for(k = 0; k < STBL_NELEM; k++) {
            my[k] = (flag == 2) ? stellar_elem_ej_AGB(logM, logZ, k) : stellar_elem_ej_SN(logM, logZ, k);
            if(my[k] < 0) my[k] = 0; Mej += my[k]; if(k >= ELEM_C) Zmet += my[k]; }
        double dy[NUM_RESOLVEDISM_DUST]; resolvedism_dust_condensation(flag, my, dy);
        double dust_tot = 0; for(k = 0; k < NUM_RESOLVEDISM_DUST; k++) dust_tot += dy[k];
        printf("DUSTTEST_COND M=%.1f Z=%.4f remtype=%d flag=%d Mej=%.4e Zmet=%.4e Cy=%.4e Oy=%.4e Mgy=%.4e Siy=%.4e Fey=%.4e dC=%.4e dOsil=%.4e dMg=%.4e dSi=%.4e dFe=%.4e dusttot=%.4e d2m=%.5f\n",
            M, Z, rt, flag, Mej, Zmet, my[ELEM_C], my[ELEM_O], my[ELEM_Mg], my[ELEM_Si], my[ELEM_Fe],
            dy[0], dy[1], dy[2], dy[3], dy[4], dust_tot, (Zmet > 0 ? dust_tot/Zmet : 0));
    }
    /* --- SN shock destruction fraction sweep --- */
    double E51 = 1.0e51 / UNIT_ENERGY_IN_CGS;
    double n0s[] = {0.1, 1.0, 100.0, 1.0e4}; double Ms[] = {1.0, 10.0, 100.0};
    for(int a = 0; a < 4; a++) for(int b = 0; b < 3; b++) {
        double rho_code = n0s[a] / (UNIT_DENSITY_IN_NHCGS * All.cf_a3inv);
        double frac = resolvedism_dust_sn_destruction_frac(rho_code, E51, Ms[b] / UNIT_MASS_IN_SOLAR);
        printf("DUSTTEST_DESTR n0=%.4e Mcell=%.1f frac=%.6e\n", n0s[a], Ms[b], frac);
    }
    /* --- sputtering + growth yield functions over T --- */
    double Ts[] = {1.0e4, 3.0e4, 1.0e5, 3.0e5, 1.0e6, 1.0e7, 1.0e8};
    for(int a = 0; a < 7; a++) {
        double T = Ts[a], stick = 1.0/(1.0 + 0.04*sqrt(T + 20.0));
        printf("DUSTTEST_RATE T=%.4e sputY_C=%.6e sputY_Sil=%.6e sticking=%.6e\n",
            T, get_yield_therSput_C(T), get_yield_therSput_Sil(T), stick);
    }
    printf("DUSTTEST_END\n"); fflush(stdout);
}
#endif


/* ---- SN shock destruction fraction (McKee 1989 + Cioffi 1988) ---- */
/* Returns fraction of pre-existing dust destroyed in a cell */

double resolvedism_dust_sn_destruction_frac(double rho_code, double E_sne_code, double mass_code)
{
    double dest_eff = 0.4;  /* dust destruction efficiency */
    double n0 = rho_code * All.cf_a3inv * UNIT_DENSITY_IN_NHCGS;
    double E_51 = E_sne_code * UNIT_ENERGY_IN_CGS / 1e51;
    double vs7 = 1.0;  /* minimum shock velocity in 10^7 cm/s that destroys dust */
    /* mass shocked to >100 km/s (McKee 1989 eq.) */
    double mass_shocked_Msun = dest_eff * 2460.0 * E_51 / (pow(n0, 0.1) * pow(vs7, 9.0/7.0));
    double frac = mass_shocked_Msun / (mass_code * UNIT_MASS_IN_SOLAR);
    return DMIN(frac, 1.0);
}


#endif /* GALSF_RESOLVEDISM_DUST */
