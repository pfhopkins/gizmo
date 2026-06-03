#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../mesh/kernel.h"
#include "resolvedism_fb_shared.h"

/* Resolved-ISM momentum feedback injection: winds, AGB, and radiation pressure.
 * Injects momentum kicks + ejecta mass + metal yields + dust (AGB only).
 * Uses ngb_treefind_pairs_threads (mutual visibility for momentum conservation).
 *
 * Called from resolvedism_inject_fb_energy() BEFORE the thermal pass.
 * Pass 0: wind + AGB ejecta (mass + metals + momentum)
 * Pass 1: radiation pressure (momentum only, no mass) */

#ifdef GALSF_RESOLVEDISM_FB

struct kernel_resolvedismFB_momentum {double dp[3], r, wk, dwk, hinv, hinv3, hinv4;};

#define CORE_FUNCTION_NAME resolvedismFB_momentum_evaluate
#define INPUTFUNCTION_NAME particle2in_resolvedismFB_momentum
#define OUTPUTFUNCTION_NAME out2particle_resolvedismFB_momentum
#define CONDITIONFUNCTION_FOR_EVALUATION if(resolvedismFB_momentum_active_check(i,loop_iteration))
#include "../system/code_block_xchange_initialize.h"

struct INPUT_STRUCT_NAME
{
    MyDouble Pos[3], KernelRadius, Mej;
    /* Bit-exact normalizer measured by the weighting pre-pass; see resolvedism_fb_thermal.cc */
    MyDouble FB_Area_weighted_sum_in;
    MyDouble WindMomentum;
    MyDouble MetalMass;
    MyIDType StarID;
    MyDouble StarMass; /* initial mass in Msun, for diagnostics */
    int fb_channel; /* 1=AGB, 2=wind */
#ifdef GALSF_RESOLVEDISM_METALS_INDIVIDUAL
    MyDouble ElemYields[NUM_RESOLVEDISM_ELEMENTS];
#endif
#ifdef GALSF_RESOLVEDISM_DUST
    MyDouble DustYields[NUM_RESOLVEDISM_DUST];
#endif
    int NodeList[NODELISTLENGTH];
}
*DATAIN_NAME, *DATAGET_NAME;

void particle2in_resolvedismFB_momentum(struct INPUT_STRUCT_NAME *in, int i, int loop_iteration)
{
    int k; for(k=0;k<3;k++) {in->Pos[k]=P[i].Pos[k];}
    in->KernelRadius = P[i].KernelRadius;
    in->Mej = 0; in->MetalMass = 0; in->WindMomentum = 0;
    /* Load measured AWS from prior weighting pass (or 0 if this IS the weighting pass). */
    in->FB_Area_weighted_sum_in = (loop_iteration >= 0) ? (MyDouble)P[i].FB_Area_weighted_sum : 0;
    in->StarID = P[i].ID;
    in->StarMass = P[i].MstarSampleIMF[0];
    in->fb_channel = DMAX(P[i].SNe_ThisTimeStep - 1, 0);
#ifdef GALSF_RESOLVEDISM_METALS_INDIVIDUAL
    for(k=0; k<NUM_RESOLVEDISM_ELEMENTS; k++) in->ElemYields[k] = 0;
#endif
#ifdef GALSF_RESOLVEDISM_DUST
    for(k=0; k<NUM_RESOLVEDISM_DUST; k++) in->DustYields[k] = 0;
#endif
    if(P[i].Mass <= 0) return;
#ifdef DO_DENSITY_AROUND_NONGAS_PARTICLES
    if(P[i].DensityAroundParticle <= 0) return;
#endif

    double Mstar, logM, logZ;
    if(!get_star_info(i, &Mstar, &logM, &logZ)) return;

#ifdef GALSF_RESOLVEDISM_RADPRESSURE
    /* ---- Radiation pressure (loop_iteration == 1) ---- */
    if(loop_iteration == 1) {
        double star_age_yr = evaluate_stellar_age_Gyr(i) * 1.0e9;
        if(star_age_yr <= 0) return;
        double lifetime_yr = get_star_lifetime(Mstar, logM, logZ);
        if(star_age_yr >= lifetime_yr) return;
        double table_age = get_star_table_age(star_age_yr, logM, logZ);
        if(table_age <= 0) return; /* PMS: no radiation pressure */

        double log_age = log10(DMAX(table_age, 100.0));
        double log_Lbol = stellar_log_L_bol(logM, logZ, log_age);
        double Lbol_cgs = pow(10.0, log_Lbol);
        if(Lbol_cgs <= 0) return;

        double dt = GET_PARTICLE_FEEDBACK_TIMESTEP_IN_PHYSICAL(i);
        if(dt <= 0) return;
        double dt_cgs = dt * UNIT_TIME_IN_CGS;

#ifdef GALSF_RESOLVEDISM_DUST
        double DGR = P[i].DGR_around / 0.01;
#else
        double DGR = All.DGRnormalized;
#ifdef DGR_SCALE_WITH_Z
        DGR = All.DGRnormalized * All.InitialMetallicity;
#endif
#endif

        double rho_phys = P[i].DensityAroundParticle * All.cf_a3inv;
        double h_phys = P[i].KernelRadius * All.cf_atime;
        double NH = rho_phys * UNIT_DENSITY_IN_NHCGS * h_phys * UNIT_LENGTH_IN_CGS;
        double sigma_dust = 2.0e-21 * DGR;
        double tau_UV = sigma_dust * NH;
        double f_abs = 1.0 - exp(-tau_UV);
        if(f_abs < 1.0e-6) return;

        double dp_cgs = f_abs * Lbol_cgs * dt_cgs / C_LIGHT_CGS;
        double Sigma_cgs = rho_phys * UNIT_DENSITY_IN_CGS * h_phys * UNIT_LENGTH_IN_CGS;
        double kappa_IR = 5.0 * DGR;
        double tau_IR = kappa_IR * Sigma_cgs;
        if(tau_IR > 0) dp_cgs += tau_IR * Lbol_cgs * dt_cgs / C_LIGHT_CGS;

        in->WindMomentum = dp_cgs / (UNIT_MASS_IN_CGS * All.UnitVelocity_in_cm_per_s);
        RadPressure_dp_thisStep += dp_cgs;
        return;
    }
#endif

    /* ---- Pass 0: AGB and wind injection ---- */
    if(P[i].SNe_ThisTimeStep != 2 && P[i].SNe_ThisTimeStep != 3) return;

    /* ---- Wind injection (SNe_ThisTimeStep == 3) ---- */
    /* Per-element ejecta from telescoping cumulative table:
     *   delta_k = elem_ej_wind_cumulative(now, k) - elem_ej_wind_cumulative(prev, k)
     * sums exactly to the table's total wind ejecta over the star's life.
     * Momentum still uses WindMomentumAccum (v_wind × dM integrated externally). */
#ifdef GALSF_RESOLVEDISM_WINDS
    if(P[i].SNe_ThisTimeStep == 3) {
        /* Mass-conservation principle: gas-injected = star-removed = P[i].WindMassAccum,
         * which was set in the accumulator (fb.cc) to (cum_table - cum_injected_so_far).
         * Star Mass is reduced by exactly WindMassAccum in the cleanup loop, so we
         * use the same value here. Per-element composition comes from table delta,
         * rescaled so Σ ElemYields == WindMassAccum exactly. */
        double dM_wind_solar = P[i].WindMassAccum;
        double dp_cgs = P[i].WindMomentumAccum * SOLAR_MASS_CGS * 1.0e5;
        in->WindMomentum = dp_cgs / (UNIT_MASS_IN_CGS * All.UnitVelocity_in_cm_per_s);

        double star_age_yr = evaluate_stellar_age_Gyr(i) * 1.0e9;
        double table_age = get_star_table_age(star_age_yr, logM, logZ);
        double log_age_now  = log10(DMAX(table_age, 100.0));
        double log_age_prev = P[i].last_wind_log_age;

        /* Per-element delta from table (cum_now - cum_prev) gives composition; rescale to dM_wind_solar. */
        double delta_sum = 0;
        double delta_solar[STBL_NELEM];
        for(int k = 0; k < STBL_NELEM; k++) {
            double cum_now  = stellar_elem_ej_wind_cumulative(logM, logZ, log_age_now,  k);
            double cum_prev = stellar_elem_ej_wind_cumulative(logM, logZ, log_age_prev, k);
            delta_solar[k] = cum_now - cum_prev;
            delta_sum += delta_solar[k];
        }
        double scale = (delta_sum > 0 && dM_wind_solar > 0) ? dM_wind_solar / delta_sum : 0;

        double metal_mass_solar = 0;
        for(int k = 0; k < STBL_NELEM; k++) {
            double m_k = delta_solar[k] * scale;
#ifdef GALSF_RESOLVEDISM_METALS_INDIVIDUAL
            in->ElemYields[k] = m_k / UNIT_MASS_IN_SOLAR;
#endif
            if(k >= ELEM_C) metal_mass_solar += m_k;
        }
        in->Mej = dM_wind_solar / UNIT_MASS_IN_SOLAR;
        in->MetalMass = metal_mass_solar / UNIT_MASS_IN_SOLAR;
#ifdef GALSF_RESOLVEDISM_ISOLATED_FB_TEST
        {
            double sum_y_code = 0;
            for(int kk = 0; kk < NUM_RESOLVEDISM_ELEMENTS; kk++) sum_y_code += in->ElemYields[kk];
            double sum_delta = 0;
            for(int kk = 0; kk < STBL_NELEM; kk++) sum_delta += delta_solar[kk];
            printf("FBDBG_WIND_DONOR star=%llu(%.1fMsun) dM_wind=%.6e delta_sum=%.6e scale=%.6e in.Mej=%.6e Msun SumY=%.6e Msun Y[H]=%.6e Y[He]=%.6e Y[C]=%.6e Y[O]=%.6e Y[Fe]=%.6e log_age_prev=%.4f log_age_now=%.4f\n",
                (unsigned long long)P[i].ID, P[i].MstarSampleIMF[0],
                dM_wind_solar, delta_sum, scale,
                in->Mej*UNIT_MASS_IN_SOLAR, sum_y_code*UNIT_MASS_IN_SOLAR,
                in->ElemYields[0]*UNIT_MASS_IN_SOLAR, in->ElemYields[1]*UNIT_MASS_IN_SOLAR,
                in->ElemYields[2]*UNIT_MASS_IN_SOLAR, in->ElemYields[4]*UNIT_MASS_IN_SOLAR,
                in->ElemYields[10]*UNIT_MASS_IN_SOLAR,
                log_age_prev, log_age_now);
            fflush(stdout);
        }
#endif
        return;
    }
#endif

    /* ---- AGB death (SNe_ThisTimeStep == 2) ---- */
#ifdef GALSF_RESOLVEDISM_STELLAR_TABLES
    int rem_type = stellar_remnant_type(logM, logZ);
    double rem_mass = stellar_remnant_mass(logM, logZ);

    double M_particle_solar = P[i].Mass * UNIT_MASS_IN_SOLAR;
    double Mej_solar = M_particle_solar - rem_mass;
    if(Mej_solar < 0) Mej_solar = 0;
    in->Mej = Mej_solar / UNIT_MASS_IN_SOLAR;

    /* AGB momentum: planetary nebula ejection at 30 km/s */
    if(rem_type == REM_WD && Mej_solar > 0) {
        in->WindMomentum = Mej_solar * 30.0 * SOLAR_MASS_CGS * 1.0e5 / (UNIT_MASS_IN_CGS * All.UnitVelocity_in_cm_per_s);
    }

    /* AGB yields: absolute element ejecta from table (all ≥0, Σ_k = Mej by construction) */
    double metal_mass_solar = 0;
    double Mej_table_solar = 0;
    for(k = 0; k < STBL_NELEM; k++) {
        double m_k = stellar_elem_ej_AGB(logM, logZ, k);
#ifdef GALSF_RESOLVEDISM_METALS_INDIVIDUAL
        in->ElemYields[k] = m_k / UNIT_MASS_IN_SOLAR;
#endif
        Mej_table_solar += m_k;
        if(k >= ELEM_C) metal_mass_solar += m_k;
    }
    in->Mej = Mej_table_solar / UNIT_MASS_IN_SOLAR;
    in->MetalMass = metal_mass_solar / UNIT_MASS_IN_SOLAR;

#ifdef GALSF_RESOLVEDISM_ISOLATED_FB_TEST
    {
        double sum_y_code = 0;
        for(int kk = 0; kk < NUM_RESOLVEDISM_ELEMENTS; kk++) sum_y_code += in->ElemYields[kk];
        printf("FBDBG_AGB_DONOR star=%llu(%.1fMsun) rem_type=%d M_pre=%.6e rem=%.6e Mej_solar(unused)=%.6e Mej_tab=%.6e in.Mej=%.6e Msun SumY=%.6e Msun Y[H]=%.6e Y[He]=%.6e Y[C]=%.6e Y[O]=%.6e Y[Fe]=%.6e\n",
            (unsigned long long)P[i].ID, P[i].MstarSampleIMF[0],
            rem_type, M_particle_solar, rem_mass, Mej_solar, Mej_table_solar,
            in->Mej*UNIT_MASS_IN_SOLAR, sum_y_code*UNIT_MASS_IN_SOLAR,
            in->ElemYields[0]*UNIT_MASS_IN_SOLAR, in->ElemYields[1]*UNIT_MASS_IN_SOLAR,
            in->ElemYields[2]*UNIT_MASS_IN_SOLAR, in->ElemYields[4]*UNIT_MASS_IN_SOLAR,
            in->ElemYields[10]*UNIT_MASS_IN_SOLAR);
        fflush(stdout);
    }
#endif

#ifdef GALSF_RESOLVEDISM_DUST
    {
        double metal_yields_solar[STBL_NELEM], dust_yields_solar[NUM_RESOLVEDISM_DUST];
        for(k = 0; k < STBL_NELEM; k++) metal_yields_solar[k] = in->ElemYields[k] * UNIT_MASS_IN_SOLAR;
        resolvedism_dust_condensation(2, metal_yields_solar, dust_yields_solar);
        for(k = 0; k < NUM_RESOLVEDISM_DUST; k++) in->DustYields[k] = dust_yields_solar[k] / UNIT_MASS_IN_SOLAR;
    }
#endif
#endif /* GALSF_RESOLVEDISM_STELLAR_TABLES */
}

struct OUTPUT_STRUCT_NAME
{
    /* Injection-pass accumulators: momentum delivered to neighbors (for star recoil)
     * and total mass deposited (for star mass reduction). */
    MyFloat MomentumInjected[3];
    MyFloat M_coupled;
    /* Weighting-pass accumulator: Σ_j (Mass_j × kernel.wk) over neighbors walked. */
    MyDouble FB_Area_weighted_sum_accum;
}
*DATARESULT_NAME, *DATAOUT_NAME;

void out2particle_resolvedismFB_momentum(struct OUTPUT_STRUCT_NAME *out, int i, int mode, int loop_iteration)
{
    if(loop_iteration < 0) {
        /* Weighting pass: accumulate measured kernel sum onto the star. */
        P[i].FB_Area_weighted_sum += out->FB_Area_weighted_sum_accum;
        return;
    }
    /* Injection pass: recoil + mass subtraction.  With Σwk=1 from the weighting pre-pass,
     * the TOTAL across all out2particle calls equals the intended Mej / p_ejecta exactly. */
    if(P[i].Mass > 0) {
        int k;
        for(k = 0; k < 3; k++) {
            P[i].Vel[k] -= out->MomentumInjected[k] * All.cf_atime / P[i].Mass;
        }
        P[i].Mass -= out->M_coupled;
        if((P[i].Mass < 0) || (isnan(P[i].Mass))) P[i].Mass = 0;
    }
}

int resolvedismFB_momentum_active_check(int i, int loop_iteration);
int resolvedismFB_momentum_active_check(int i, int loop_iteration)
{
    if(P[i].Type != 4) return 0;
    if(P[i].KernelRadius <= 0) return 0;
    if(P[i].NumNgb <= 0) return 0;

    /* Pair (-1, 0) for wind/AGB; pair (-2, 1) for radpressure.  STARFORGE pattern:
     * the weighting pre-pass uses the same active_check criterion as its injection. */
    if(loop_iteration == 0 || loop_iteration == -1) {
        /* AGB (2) and wind (3) */
        if(P[i].SNe_ThisTimeStep == 2 || P[i].SNe_ThisTimeStep == 3) return 1;
    }
#ifdef GALSF_RESOLVEDISM_RADPRESSURE
    if(loop_iteration == 1 || loop_iteration == -2) {
        if(P[i].Mass <= 0) return 0;
        double Mstar = 0;
#ifdef GALSF_RESOLVEDISM_SAMPLE_IMF
        if(P[i].sampled) Mstar = P[i].MstarSampleIMF[0];
#endif
#ifdef GALSF_RESOLVEDISM_STOCHASTIC_IMF
        Mstar = P[i].Mstar;
#endif
        if(Mstar >= 2.0) return 1;
    }
#endif
    return 0;
}


int resolvedismFB_momentum_evaluate(int target, int mode, int *exportflag, int *exportnodecount, int *exportindex, int *ngblist, int loop_iteration)
{
    int startnode, numngb_inbox, listindex = 0, j, k, n;
    double u, r2, h2;
    struct kernel_resolvedismFB_momentum kernel;
    struct INPUT_STRUCT_NAME local;
    struct OUTPUT_STRUCT_NAME out;
    memset(&out, 0, sizeof(struct OUTPUT_STRUCT_NAME));

    if(mode == 0) {particle2in_resolvedismFB_momentum(&local, target, loop_iteration);} else {local = DATAGET_NAME[target];}
    if(local.Mej <= 0 && local.WindMomentum <= 0) return 0;
    if(local.KernelRadius <= 0) return 0;
    /* Injection pass needs the AWS measured by the weighting pre-pass. */
    if(loop_iteration >= 0 && local.FB_Area_weighted_sum_in <= 0) return 0;
    h2 = local.KernelRadius * local.KernelRadius;
    kernel_hinv(local.KernelRadius, &kernel.hinv, &kernel.hinv3, &kernel.hinv4);

    if(mode == 0) {startnode = All.MaxPart;}
    else {startnode = DATAGET_NAME[target].NodeList[0]; startnode = Nodes[startnode].u.d.nextnode;}

    while(startnode >= 0)
    {
        while(startnode >= 0)
        {
            numngb_inbox = ngb_treefind_pairs_threads(local.Pos, local.KernelRadius, target, &startnode, mode, exportflag, exportnodecount, exportindex, ngblist);
            if(numngb_inbox < 0) {return -2;}
            for(n = 0; n < numngb_inbox; n++)
            {
                j = ngblist[n];
                if(P[j].Type != 0) {continue;}
                double Mass_j;
                #pragma omp atomic read
                Mass_j = P[j].Mass;
                if(Mass_j <= 0) {continue;}

                for(k=0;k<3;k++) {kernel.dp[k] = local.Pos[k] - P[j].Pos[k];}
                NEAREST_XYZ(kernel.dp[0],kernel.dp[1],kernel.dp[2],1);
                r2=0; for(k=0;k<3;k++) {r2 += kernel.dp[k]*kernel.dp[k];}
                if(r2 <= 0 || r2 >= h2) {continue;}
                kernel.r = sqrt(r2);
                if(kernel.r <= 0) {continue;}
                u = kernel.r * kernel.hinv;
                if(u<1) {kernel_main(u, kernel.hinv3, kernel.hinv4, &kernel.wk, &kernel.dwk, 0);} else {kernel.wk=kernel.dwk=0;}
                if((kernel.wk <= 0)||(isnan(kernel.wk))) {continue;}

                /* STARFORGE two-pass pattern: weighting pre-pass (loop_iteration<0) just
                 * measures Σ_j (Mass_j × kernel.wk); injection pass uses that measured sum
                 * as the bit-exact normalizer → Σ wk_j = 1 by construction. */
                if(loop_iteration < 0) {
                    out.FB_Area_weighted_sum_accum += Mass_j * kernel.wk;
                    continue;
                }
                /* STARFORGE-style normalizer with MIN_REAL_NUMBER+fabs() guard. */
                double wk = (Mass_j * kernel.wk) / (MIN_REAL_NUMBER + fabs(local.FB_Area_weighted_sum_in));

                /* ---- Mass + metals injection (wind and AGB ejecta) ---- */
                if(local.Mej > 0) {
                    double dM = wk * local.Mej;
#ifdef METALS
                    {
                        double Z_old, M_old = Mass_j;
                        #pragma omp atomic read
                        Z_old = P[j].Metallicity[0];   /* total Z in FIRE-pattern layout */
                        double dMZ = wk * local.MetalMass;
                        double dZ = (dMZ - Z_old * dM) / (M_old + dM);
                        #pragma omp atomic
                        P[j].Metallicity[0] += dZ;
                    }
                    {
                        int ch = local.fb_channel;
                        double Mnew_j = Mass_j + dM;
                        for(int c = 0; c < 4; c++) {
                            double F_old;
                            #pragma omp atomic read
                            F_old = CellP[j].MetalMassFrom[c];
                            double dMZ_c = (c == ch) ? wk * local.MetalMass : 0;
                            double dF = (dMZ_c - F_old * dM) / Mnew_j;
                            #pragma omp atomic
                            CellP[j].MetalMassFrom[c] += dF;
                        }
                    }
#endif
#ifdef GALSF_RESOLVEDISM_METALS_INDIVIDUAL
                    /* 28-slot layout: write yield Y[k] into Met[MET_OF(k)] via
                     * mass-fraction blend (k=0..26 includes H).  Met[0] (total Z)
                     * is updated separately above.  Σ Met[1..27] stays 1 since
                     * Σ Y[k] = Mej. */
#ifdef GALSF_RESOLVEDISM_ISOLATED_FB_TEST
                    double dbg_sumX_pre = 0;
                    double dbg_sumdMX = 0;
                    for(int kk = 1; kk < NUM_METAL_SPECIES; kk++) dbg_sumX_pre += P[j].Metallicity[kk];
                    for(int kk = 0; kk < NUM_RESOLVEDISM_ELEMENTS; kk++) dbg_sumdMX += wk * local.ElemYields[kk];
#endif
                    for(k = 0; k < NUM_RESOLVEDISM_ELEMENTS; k++) {
                        int m = MET_OF(k);
                        double X_old;
                        #pragma omp atomic read
                        X_old = P[j].Metallicity[m];
                        double dMX = wk * local.ElemYields[k];
                        double dX = (dMX - X_old * dM) / (Mass_j + dM);
                        #pragma omp atomic
                        P[j].Metallicity[m] += dX;
                    }
                    /* Force ΣMet = 1 by construction (see fb_thermal for rationale). */
                    {
                        double Z_now, Y_now;
                        #pragma omp atomic read
                        Z_now = P[j].Metallicity[0];
                        #pragma omp atomic read
                        Y_now = P[j].Metallicity[MET_OF(ELEM_He)];
                        double X_H_new = 1.0 - Z_now - Y_now;
                        if(X_H_new < 0) X_H_new = 0;
                        #pragma omp atomic write
                        P[j].Metallicity[MET_OF(ELEM_H)] = (MyFloat)X_H_new;
                    }
#ifdef GALSF_RESOLVEDISM_ISOLATED_FB_TEST
                    {
                        double dbg_sumX_post = 0;
                        for(int kk = 1; kk < NUM_METAL_SPECIES; kk++) dbg_sumX_post += P[j].Metallicity[kk];
                        printf("FBDBG_MOM_RECV star=%llu(ch=%d) cell=%llu wk=%.6e Mass_pre=%.6e dM=%.6e SumdMX=%.6e ratio_SumdMX_dM=%.6e SumX_pre=%.9e SumX_post=%.9e\n",
                            (unsigned long long)local.StarID, local.fb_channel,
                            (unsigned long long)P[j].ID, wk, Mass_j, dM, dbg_sumdMX,
                            (dM>0?dbg_sumdMX/dM:0.0), dbg_sumX_pre, dbg_sumX_post);
                        fflush(stdout);
                    }
#endif
#endif
                    #pragma omp atomic
                    P[j].Mass += dM;
                    /* Accumulate actual deposition for measured-coupling reduction
                     * on the dying star (FIRE pattern); see out2particle. */
                    out.M_coupled += dM;
                    P[j].wakeup = -1;
                    NeedToWakeupParticles_local = 1;
                }

#ifdef GALSF_RESOLVEDISM_DUST
                /* Inject dust from AGB ejecta (no SN destruction here — that's in the thermal file) */
                if(local.Mej > 0) {
                    double dM = wk * local.Mej;
                    double Mnew_j = Mass_j + dM;
                    for(k = 0; k < NUM_RESOLVEDISM_DUST; k++) {
                        double D_old;
                        #pragma omp atomic read
                        D_old = CellP[j].Dust[k];
                        double dMD = wk * local.DustYields[k];
                        double dD = (dMD - D_old * dM) / Mnew_j;
                        #pragma omp atomic
                        CellP[j].Dust[k] += dD;
                    }
                }
#endif

                /* ---- Momentum kick (wind, AGB, radpressure) ---- */
                if(local.WindMomentum > 0) {
                    double dp_share = wk * local.WindMomentum;
                    double dM_wind = (local.Mej > 0) ? wk * local.Mej : 0;
                    double dv_mag2 = 0;
                    for(k = 0; k < 3; k++) {
                        double dp_k = dp_share * (-kernel.dp[k] / kernel.r);
                        double dv_k = dp_k * All.cf_atime / (Mass_j + dM_wind);
                        #pragma omp atomic
                        P[j].Vel[k] += dv_k;
                        #pragma omp atomic
                        CellP[j].VelPred[k] += dv_k;
                        #pragma omp atomic
                        P[j].dp[k] += dp_k * All.cf_atime;
                        out.MomentumInjected[k] += dp_k;
                        dv_mag2 += dv_k * dv_k;
                    }
                    /* diagnostic: flag extreme kicks (>500 km/s in code velocity units) */
                    {
                        double dv_mag = sqrt(dv_mag2);
                        double vk0, vk1, vk2;
                        #pragma omp atomic read
                        vk0 = P[j].Vel[0];
                        #pragma omp atomic read
                        vk1 = P[j].Vel[1];
                        #pragma omp atomic read
                        vk2 = P[j].Vel[2];
                        double vmag = sqrt(vk0*vk0 + vk1*vk1 + vk2*vk2);
                        if(dv_mag > 500.0 || vmag > 5000.0)
                            printf("WIND_KICK_WARN: Task=%d star=%llu(%.1fMsun) -> cell=%llu dv=%.1f |v|=%.1f Nngb=%.2f rho=%.3e wk=%.4f r=%.4f ch=%d\n",
                                ThisTask, (unsigned long long)local.StarID, local.StarMass,
                                (unsigned long long)P[j].ID, dv_mag, vmag,
                                P[j].NumNgb, CellP[j].Density, wk, kernel.r, local.fb_channel);
                    }
                    P[j].wakeup = -1;
                    NeedToWakeupParticles_local = 1;
                }

                /* NaN sanity check */
                {
                    for(k=0;k<3;k++) {
                        double vk;
                        #pragma omp atomic read
                        vk = P[j].Vel[k];
                        if(!isfinite(vk)) {
                            printf("NAN_CHECK_FB_MOMENTUM: Task=%d neighbor ID=%llu Vel[%d]=%.6e after injection from star target=%d\n",
                                ThisTask, (unsigned long long)P[j].ID, k, vk, target);
                        }
                    }
                }

            }
        }
        if(mode == 1)
        {
            listindex++;
            if(listindex < NODELISTLENGTH)
            {
                startnode = DATAGET_NAME[target].NodeList[listindex];
                if(startnode >= 0) {startnode = Nodes[startnode].u.d.nextnode;}
            }
        }
    }

    if(mode == 0) {out2particle_resolvedismFB_momentum(&out, target, 0, loop_iteration);} else {DATARESULT_NAME[target] = out;}
    return 0;
}


void resolvedism_fb_momentum_calc(int fb_loop_iteration)
{
    #include "../system/code_block_xchange_perform_ops_malloc.h"
    loop_iteration = fb_loop_iteration;
    #include "../system/code_block_xchange_perform_ops.h"
    #include "../system/code_block_xchange_perform_ops_demalloc.h"
}

#include "../system/code_block_xchange_finalize.h"

#endif /* GALSF_RESOLVEDISM_FB */
