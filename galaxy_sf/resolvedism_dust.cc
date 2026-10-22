#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "resolvedism_dust_kernels.h"
#include "../declarations/resolvedism_dgr.h"
#ifdef CHEMCOOL
#include "../cooling/chemcool/chemcool_consts.h"
#endif

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
 * Physics (Hu+19/23/26, see galaxy_sf/resolvedism_dust_kernels.h):
 *   - thermal + nonthermal sputtering (Nozawa+06), dust v_rel from drag, hydro acceleration, betatron
 *   - ISM growth in cold gas
 *   - stellar condensation yields (resolvedism_fb_thermal.cc / resolvedism_fb_momentum.cc)
 *   - CHEMCOOL coupling via resolvedism_dgr.h and gas-phase C/O/Si
 */

#ifdef GALSF_RESOLVEDISM_DUST

/* Silicate mass fractions within FeMgSiO4 (M_w = 172 amu) */
#define DUST_SIL_FRAC_O  (64.0 / 172.0)
#define DUST_SIL_FRAC_MG (24.0 / 172.0)
#define DUST_SIL_FRAC_SI (28.0 / 172.0)
#define DUST_SIL_FRAC_FE (56.0 / 172.0)


/* ---- ISM dust evolution (Hu+19/23/26), operator-split after chemistry ---- */

#define DUST_FSUB 0.1           /* subcycle factor (Hu+19 eq 13) */
#define DUST_NSUB_MAX 100000

static double dust_ledger[7];   /* cumulative code mass: sput_th C/Sil, sput_nth C/Sil, grow C/Sil, clamp */

static void resolvedism_dust_ledger_write(void)
{
    if(All.NumCurrentTiStep % 16) {return;}
    double tot[7], mloc[2] = {0, 0}, mtot[2];
    int i;
    for(i = 0; i < NumPart; i++) {if(P[i].Type == 0 && P[i].Mass > 0) {
        mloc[0] += P[i].Mass * CellP[i].Dust[0];
        mloc[1] += P[i].Mass * (CellP[i].Dust[1] + CellP[i].Dust[2] + CellP[i].Dust[3] + CellP[i].Dust[4]);}}
    MPI_Reduce(dust_ledger, tot, 7, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(mloc, mtot, 2, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if(ThisTask != 0) {return;}
    char path[DEFAULT_PATH_BUFFERSIZE_TOUSE];
    snprintf(path, DEFAULT_PATH_BUFFERSIZE_TOUSE, "%s/dust_processes.txt", All.OutputDir);
    FILE *f = fopen(path, "a"); if(!f) {return;}
    if(ftell(f) == 0) {fprintf(f, "# step time M_dust_C M_dust_Sil | cumulative [Msun]: sput_th_C sput_th_Sil sput_nth_C sput_nth_Sil grow_C grow_Sil clamp\n");}
    fprintf(f, "%lld %.10e %.8e %.8e %.8e %.8e %.8e %.8e %.8e %.8e %.8e\n", (long long)All.NumCurrentTiStep, All.Time,
            mtot[0]*UNIT_MASS_IN_SOLAR, mtot[1]*UNIT_MASS_IN_SOLAR, tot[0]*UNIT_MASS_IN_SOLAR, tot[1]*UNIT_MASS_IN_SOLAR,
            tot[2]*UNIT_MASS_IN_SOLAR, tot[3]*UNIT_MASS_IN_SOLAR, tot[4]*UNIT_MASS_IN_SOLAR, tot[5]*UNIT_MASS_IN_SOLAR, tot[6]*UNIT_MASS_IN_SOLAR);
    fclose(f);
}

static void resolvedism_dust_env(int i, struct dustk_env *e)
{
    double rho_phys = CellP[i].Density * All.cf_a3inv;
    e->nH = rho_phys * UNIT_DENSITY_IN_NHCGS;
#ifdef CHEMCOOL
    e->T = CellP[i].Temp;
    e->x_e = DMAX(CellP[i].TracAbund[IHP], 0) + 1.4e-4 * DMAX(P[i].Metallicity[0], 0) / All.SolarAbundances[0]; /* H+ plus C+ */
#else
    double ne = 1, nh0 = 0, nHe0, nHepp, nhp, nHeII, mu = 1;
    e->T = ThermalProperties(CellP[i].InternalEnergyPred, rho_phys, i, &mu, &ne, &nh0, &nhp, &nHe0, &nHeII, &nHepp);
    e->x_e = ne;
#endif
#if defined(GALSF_RESOLVEDISM_G0_VARIABLE) || defined(RADTRANSFER)
    e->G0 = CellP[i].G0;
#elif defined(CHEMCOOL)
    e->G0 = All.G0;
#else
    e->G0 = 1.0;
#endif
    double a_conv = UNIT_VEL_IN_KMS * SECONDS_PER_YEAR / UNIT_TIME_IN_CGS; /* code accel -> km/s/yr (non-comoving) */
    for(int k = 0; k < 3; k++) {e->ahyd[k] = CellP[i].HydroAccel[k] * a_conv;}
    e->divv = P[i].Particle_DivVel * All.cf_a2inv * SECONDS_PER_YEAR / UNIT_TIME_IN_CGS;
#ifdef GALSF_RESOLVEDISM_DUST_NO_BETATRON
    e->betatron = 0;
#else
    e->betatron = 1;
#endif
}

void resolvedism_dust_evolve(void)
{
    for(int i : ActiveParticleList)
    {
        if(P[i].Type != 0) continue;

        double dust_total = 0;
        int k;
        for(k = 0; k < NUM_RESOLVEDISM_DUST; k++) dust_total += CellP[i].Dust[k];
        if(dust_total <= 0) {memset(CellP[i].DustVrel, 0, sizeof(CellP[i].DustVrel)); continue;}

        struct dustk_env env; resolvedism_dust_env(i, &env);
        double nH = env.nH, T = env.T;
        double dt = get_particle_timestep_in_physical(i);
        double dt_yr = dt * UNIT_TIME_IN_CGS / SECONDS_PER_YEAR;

        /* ---- thermal + nonthermal sputtering per MRN size bin (Hu+19 eqs 2, 12, 20) ---- */
        double YthC = dustk_Yth_C(T), YthS = dustk_Yth_Sil(T);
        double fC = 0, fS = 0, dthC = 0, dthS = 0, dnthC = 0, dnthS = 0;
        for(int b = 0; b < NUM_RESOLVEDISM_DUST_BINS; b++)
        {
            double a_b, w_b; dustk_size_bin(NUM_RESOLVEDISM_DUST_BINS, b, &a_b, &w_b);
            double v[3] = {CellP[i].DustVrel[b][0], CellP[i].DustVrel[b][1], CellP[i].DustVrel[b][2]};
            double InC, InS;
            dustk_step_vrel(&env, a_b, dt_yr, v, &InC, &InS, DUST_FSUB, DUST_NSUB_MAX);
            for(k = 0; k < 3; k++) {CellP[i].DustVrel[b][k] = v[k];}
            double IthC = dt_yr * 3.0 * nH * YthC / a_b, IthS = dt_yr * 3.0 * nH * YthS / a_b;
            double gC = exp(-(IthC + InC)), gS = exp(-(IthS + InS));
            fC += w_b * gC; fS += w_b * gS;
            if(IthC + InC > 0) {double lost = w_b * (1.0 - gC); dthC += lost * IthC / (IthC + InC); dnthC += lost * InC / (IthC + InC);}
            if(IthS + InS > 0) {double lost = w_b * (1.0 - gS); dthS += lost * IthS / (IthS + InS); dnthS += lost * InS / (IthS + InS);}
        }
        double DSil = CellP[i].Dust[1] + CellP[i].Dust[2] + CellP[i].Dust[3] + CellP[i].Dust[4];
        dust_ledger[0] += P[i].Mass * CellP[i].Dust[0] * dthC; dust_ledger[2] += P[i].Mass * CellP[i].Dust[0] * dnthC;
        dust_ledger[1] += P[i].Mass * DSil * dthS;              dust_ledger[3] += P[i].Mass * DSil * dnthS;
        CellP[i].Dust[0] *= fC;
        for(k = 1; k <= 4; k++) {CellP[i].Dust[k] *= fS;}

        /* ---- ISM growth (Hu+23/26 eq 4): T < 300 K only, total-Z single (1-f), CO-bound C/O excluded, exact logistic ---- */
        double r0 = dustk_growth_rate0(nH, T, All.DustGrowthTimescaleGyr, All.DustGrowthCoulombFactor);
        if(r0 > 0)
        {
            double Zp = DMAX(P[i].Metallicity[0], 0) / All.SolarAbundances[0];
            double C_CO = 0, O_CO = 0;
#if defined(CHEMCOOL) && (ICO != UNUSED)
            {
#ifdef GALSF_CHEMCOOL_VARIABLE_XH_AND_ABHE
                double X_H = DMAX(P[i].Metallicity[MET_OF(ELEM_H)], 1e-10);
#else
                double X_H = 1.0 / (1.0 + 4.0 * ABHE);
#endif
                double xco = DMAX(CellP[i].TracAbund[ICO], 0);
                C_CO = 12.0 * xco * X_H; O_CO = 16.0 * xco * X_H; /* same X_H convention as chem_interface.cc */
            }
#endif
            double D0 = CellP[i].Dust[0];
            double D1 = dustk_grow_carbon(D0, DMAX(P[i].Metallicity[MET_OF(ELEM_C)], 0), C_CO, Zp, r0, dt_yr);
            CellP[i].Dust[0] = D1;
            dust_ledger[4] += P[i].Mass * (D1 - D0);

            double S0 = CellP[i].Dust[3] / DUST_SIL_FRAC_SI; /* stoichiometric silicate, keyed on Si */
            double Fe_free = DMAX(CellP[i].Dust[4] - S0 * DUST_SIL_FRAC_FE, 0); /* non-silicate Fe */
            double pool[4] = {DMAX(P[i].Metallicity[MET_OF(ELEM_O)] - O_CO, 0), DMAX(P[i].Metallicity[MET_OF(ELEM_Mg)], 0),
                              DMAX(P[i].Metallicity[MET_OF(ELEM_Si)], 0), DMAX(P[i].Metallicity[MET_OF(ELEM_Fe)] - Fe_free, 0)};
            double frac[4] = {DUST_SIL_FRAC_O, DUST_SIL_FRAC_MG, DUST_SIL_FRAC_SI, DUST_SIL_FRAC_FE};
            double dS = dustk_grow_silicate(S0, pool, frac, Zp, r0, dt_yr) - S0;
            if(dS > 0) {
                for(k = 0; k < 4; k++) {CellP[i].Dust[k+1] += dS * frac[k];}
                dust_ledger[5] += P[i].Mass * dS;
            }
        }

        double Dpre = 0; for(k = 0; k < NUM_RESOLVEDISM_DUST; k++) {Dpre += CellP[i].Dust[k];}
        resolvedism_dust_clamp_cell(i);
        double Dpost = 0; for(k = 0; k < NUM_RESOLVEDISM_DUST; k++) {Dpost += CellP[i].Dust[k];}
        dust_ledger[6] += P[i].Mass * (Dpre - Dpost);
    }
    resolvedism_dust_ledger_write();
}

/* Dust cannot exceed its element pool; silicate is scaled as a whole to keep stoichiometry. */
void resolvedism_dust_clamp_cell(int i)
{
    int k;
    for(k = 0; k < NUM_RESOLVEDISM_DUST; k++) if(CellP[i].Dust[k] < 0) CellP[i].Dust[k] = 0;
    CellP[i].Dust[0] = DMIN(CellP[i].Dust[0], DMAX(P[i].Metallicity[MET_OF(ELEM_C)], 0));
    double f = 1;
    if(CellP[i].Dust[1] > 0) f = DMIN(f, DMAX(P[i].Metallicity[MET_OF(ELEM_O)],  0) / CellP[i].Dust[1]);
    if(CellP[i].Dust[2] > 0) f = DMIN(f, DMAX(P[i].Metallicity[MET_OF(ELEM_Mg)], 0) / CellP[i].Dust[2]);
    if(CellP[i].Dust[3] > 0) f = DMIN(f, DMAX(P[i].Metallicity[MET_OF(ELEM_Si)], 0) / CellP[i].Dust[3]);
    if(CellP[i].Dust[4] > 0) f = DMIN(f, DMAX(P[i].Metallicity[MET_OF(ELEM_Fe)], 0) / CellP[i].Dust[4]);
    if(f < 1) for(k = 1; k <= 4; k++) CellP[i].Dust[k] *= f;
}

#define DUST_SEED_MAXDEPL 0.9 /* max fraction of a cell's element pool locked in seed dust */
/* Fresh-start seed: MW dust (C 1.9e-3 + silicate 3.5e-3, Hu+23) x Z/Zsun, capped by the cell's element pools. */
void resolvedism_dust_seed_cell(int i)
{
    double Zp = DMAX(P[i].Metallicity[0], 0) / All.SolarAbundances[0];
    double C_dust = DMIN(RESOLVEDISM_DGR_MW_C_ABS * Zp, DUST_SEED_MAXDEPL * DMAX(P[i].Metallicity[MET_OF(ELEM_C)], 0));
    double sil = RESOLVEDISM_DGR_MW_SIL_ABS * Zp;
    sil = DMIN(sil, DUST_SEED_MAXDEPL * DMAX(P[i].Metallicity[MET_OF(ELEM_O)],  0) / DUST_SIL_FRAC_O);
    sil = DMIN(sil, DUST_SEED_MAXDEPL * DMAX(P[i].Metallicity[MET_OF(ELEM_Mg)], 0) / DUST_SIL_FRAC_MG);
    sil = DMIN(sil, DUST_SEED_MAXDEPL * DMAX(P[i].Metallicity[MET_OF(ELEM_Si)], 0) / DUST_SIL_FRAC_SI);
    sil = DMIN(sil, DUST_SEED_MAXDEPL * DMAX(P[i].Metallicity[MET_OF(ELEM_Fe)], 0) / DUST_SIL_FRAC_FE);
    CellP[i].Dust[0] = C_dust;
    CellP[i].Dust[1] = sil * DUST_SIL_FRAC_O;
    CellP[i].Dust[2] = sil * DUST_SIL_FRAC_MG;
    CellP[i].Dust[3] = sil * DUST_SIL_FRAC_SI;
    CellP[i].Dust[4] = sil * DUST_SIL_FRAC_FE;
    memset(CellP[i].DustVrel, 0, sizeof(CellP[i].DustVrel));
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
        double T = Ts[a], v = T / 1.0e3; /* v_rel sweep 10-1e5 km/s alongside T */
        printf("DUSTTEST_RATE T=%.4e sputY_C=%.6e sputY_Sil=%.6e v=%.4e nthY_C=%.6e nthY_Sil=%.6e\n",
            T, dustk_Yth_C(T), dustk_Yth_Sil(T), v, dustk_Ynth_C(v), dustk_Ynth_Sil(v));
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
