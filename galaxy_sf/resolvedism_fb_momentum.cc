#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../declarations/allvars.h"
#include "../core/proto.h"
#ifdef GALSF_RESOLVEDISM_FB_HEALPIX
#include "../gravity/healpix_utils.h"
#endif
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
#ifdef GALSF_RESOLVEDISM_FB_HEALPIX
    MyFloat HpxCount[12];
#endif
}
*DATAIN_NAME, *DATAGET_NAME;

void particle2in_resolvedismFB_momentum(struct INPUT_STRUCT_NAME *in, int i, int loop_iteration)
{
    int k; for(k=0;k<3;k++) {in->Pos[k]=P[i].Pos[k];}
    in->KernelRadius = P[i].KernelRadius;
    in->Mej = 0; in->MetalMass = 0; in->WindMomentum = 0;
    /* Load measured AWS from prior weighting pass (or 0 if this IS the weighting pass). */
    in->FB_Area_weighted_sum_in = (loop_iteration >= 0) ? (MyDouble)P[i].FB_Area_weighted_sum : 0;
#ifdef GALSF_RESOLVEDISM_FB_HEALPIX
    {int hp_; for(hp_=0;hp_<12;hp_++) {in->HpxCount[hp_] = P[i].FB_HpxCount[hp_];}}
#endif
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
#ifdef GALSF_RESOLVEDISM_ISOLATED_FB_TEST
        /* throttled gate-trace: which return fires? (FBDBG_RP, parsed by test suite) */
        static int rp_dbg_count = 0; int rp_dbg = (P[i].ID <= 8 && rp_dbg_count < 40); if(rp_dbg) rp_dbg_count++;
        #define RP_TRACE(reason, val) do { if(rp_dbg) {printf("FBDBG_RP_GATE star=%llu gate=%s val=%.6e\n", (unsigned long long)P[i].ID, reason, (double)(val)); fflush(stdout);} } while(0)
#else
        #define RP_TRACE(reason, val) do {} while(0)
#endif
        double star_age_yr = evaluate_stellar_age_Gyr(i) * 1.0e9;
        if(star_age_yr <= 0) {RP_TRACE("age<=0", star_age_yr); return;}
        double lifetime_yr = get_star_lifetime(Mstar, logM, logZ);
        if(star_age_yr >= lifetime_yr) {RP_TRACE("dead", star_age_yr); return;}
        double table_age = get_star_table_age(star_age_yr, logM, logZ);
        if(table_age <= 0) {RP_TRACE("pms", table_age); return;} /* PMS: no radiation pressure */

        double log_age = log10(DMAX(table_age, 100.0));
        double log_Lbol = stellar_log_L_bol(logM, logZ, log_age);
        double Lbol_cgs = pow(10.0, log_Lbol);
        if(Lbol_cgs <= 0) {RP_TRACE("Lbol<=0", Lbol_cgs); return;}

        double dt = GET_PARTICLE_FEEDBACK_TIMESTEP_IN_PHYSICAL(i);
        if(dt <= 0) {RP_TRACE("dt<=0", dt); return;}
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
        if(f_abs < 1.0e-6) {RP_TRACE("f_abs", f_abs); RP_TRACE("f_abs.DGR", DGR); RP_TRACE("f_abs.NH", NH); return;}

        double dp_cgs = f_abs * Lbol_cgs * dt_cgs / C_LIGHT_CGS;
        RP_TRACE("FIRING.dp_cgs", dp_cgs);
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

#ifdef GALSF_RESOLVEDISM_DUST
        /* DUST FIX #1 (2026-07-05): condense dust in the WIND channel (was zero).
         * C/O-driven condensation (flag 3 == AGB logic) applied to the wind-ejecta
         * composition, gated on wind speed: only slow cool winds condense
         * (v_w < 100 km/s: AGB superwind / RSG).  This resurrects the dominant
         * AGB dust channel in WINDS-on builds, where the superwind AND the at-death
         * envelope force-dump leave via flag 3.  Fast hot winds (MS, WR) do not
         * condense here — WC carbon dust is a documented omission pending lit pass. */
        {
            double v_w_eff = (P[i].WindMassAccum > 0) ? (P[i].WindMomentumAccum / P[i].WindMassAccum) : 1.0e10; /* km/s */
            if(v_w_eff < 100.0) {
                double metal_yields_solar[STBL_NELEM], dust_yields_solar[NUM_RESOLVEDISM_DUST];
                int kk; for(kk = 0; kk < STBL_NELEM; kk++) metal_yields_solar[kk] = in->ElemYields[kk] * UNIT_MASS_IN_SOLAR;
                resolvedism_dust_condensation(3, metal_yields_solar, dust_yields_solar);
                for(kk = 0; kk < NUM_RESOLVEDISM_DUST; kk++) in->DustYields[kk] = dust_yields_solar[kk] / UNIT_MASS_IN_SOLAR;
#ifdef GALSF_RESOLVEDISM_ISOLATED_FB_TEST
                { double dsum=0; for(kk=0;kk<NUM_RESOLVEDISM_DUST;kk++) dsum+=dust_yields_solar[kk];
                  double nC=metal_yields_solar[ELEM_C]/12.0, nO=metal_yields_solar[ELEM_O]/16.0;
                  printf("FBDBG_WINDDUST star=%llu vW=%.2fkm/s dM=%.4e C=%.4e O=%.4e nC/nO=%.3f branch=%s dust:C=%.4e sil=%.4e tot=%.4e\n",
                    (unsigned long long)P[i].ID, v_w_eff, P[i].WindMassAccum,
                    metal_yields_solar[ELEM_C], metal_yields_solar[ELEM_O], (nO>0?nC/nO:99.),
                    (nC>nO?"C-rich":"O-rich"), dust_yields_solar[0], dust_yields_solar[1]+dust_yields_solar[2]+dust_yields_solar[3]+dust_yields_solar[4], dsum); fflush(stdout); }
#endif
            }
        }
#endif
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

    /* same export-repack freeze as the SN thermal path (2026-07-05) */
    double M_particle_solar = (P[i].M_at_SN_trigger > 0) ? (double)P[i].M_at_SN_trigger
                                                         : P[i].Mass * UNIT_MASS_IN_SOLAR;
    double Mej_solar = M_particle_solar - rem_mass;
    if(Mej_solar < 0) Mej_solar = 0;
    in->Mej = Mej_solar / UNIT_MASS_IN_SOLAR;

    /* AGB momentum: planetary nebula ejection at 30 km/s */
    if(rem_type == REM_WD && Mej_solar > 0) {
        in->WindMomentum = Mej_solar * 30.0 * SOLAR_MASS_CGS * 1.0e5 / (UNIT_MASS_IN_CGS * All.UnitVelocity_in_cm_per_s);
    }

    /* AGB yields: the table gives the ejecta COMPOSITION; rescale it to the physical
     * ejecta mass Mej_solar = M_particle - rem_mass so we never remove more than the
     * star currently has. BUGFIX 2026-07-02: previously in->Mej was overwritten with the
     * ZAMS-envelope table sum Mej_table_solar, which for a wind-stripped star exceeds the
     * current mass -> out2particle clip -> mass non-conservation (the test_FULL leak).
     * Mirror the SN thermal path: current-mass Mej, table composition. */
    double metal_mass_solar = 0;
    double Mej_table_solar = 0;
    for(k = 0; k < STBL_NELEM; k++) {
        double m_k = stellar_elem_ej_AGB(logM, logZ, k);
#ifdef GALSF_RESOLVEDISM_METALS_INDIVIDUAL
        in->ElemYields[k] = m_k / UNIT_MASS_IN_SOLAR; /* table-absolute; rescaled below */
#endif
        Mej_table_solar += m_k;
        if(k >= ELEM_C) metal_mass_solar += m_k;
    }
    double zscale = (Mej_table_solar > 0) ? (Mej_solar / Mej_table_solar) : 0.0;
#ifdef GALSF_RESOLVEDISM_METALS_INDIVIDUAL
    for(k = 0; k < STBL_NELEM; k++) in->ElemYields[k] *= zscale;
#endif
    in->Mej = Mej_solar / UNIT_MASS_IN_SOLAR;                     /* current-mass ejecta (not table sum) */
    in->MetalMass = metal_mass_solar * zscale / UNIT_MASS_IN_SOLAR;

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
#ifdef GALSF_RESOLVEDISM_FB_HEALPIX
    MyFloat HpxCount[12];
#endif
}
*DATARESULT_NAME, *DATAOUT_NAME;

void out2particle_resolvedismFB_momentum(struct OUTPUT_STRUCT_NAME *out, int i, int mode, int loop_iteration)
{
    if(loop_iteration < 0) {
        /* Weighting pass: accumulate measured kernel sum onto the star. */
        P[i].FB_Area_weighted_sum += out->FB_Area_weighted_sum_accum;
#ifdef GALSF_RESOLVEDISM_FB_HEALPIX
        {int hp_; for(hp_=0;hp_<12;hp_++) {P[i].FB_HpxCount[hp_] += out->HpxCount[hp_];}}
#endif
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

#ifdef GALSF_RESOLVEDISM_WINDS_CONTINUOUS
extern int FB_WindBatchOnly;
#endif
int resolvedismFB_momentum_active_check(int i, int loop_iteration);
int resolvedismFB_momentum_active_check(int i, int loop_iteration)
{
#ifdef GALSF_RESOLVEDISM_WINDS_CONTINUOUS
    if(FB_WindBatchOnly && P[i].SNe_ThisTimeStep != 3) {return 0;}   /* batch pass: winds only */
#endif
    if(P[i].Type != 4) return 0;
    if(P[i].KernelRadius <= 0) return 0;
    if(P[i].NumNgb <= 0) return 0;

    /* Pair (-1, 0) for wind/AGB; pair (-2, 1) for radpressure.  STARFORGE pattern:
     * the weighting pre-pass uses the same active_check criterion as its injection. */
    if(loop_iteration == 0 || loop_iteration == -1) {
        /* per-event serialization (see resolvedism_fb_serialized_pass in resolvedism_fb.cc):
         * when the token is set, only that single event is active. Radpressure pair
         * (-2,1) below is deliberately NOT gated — it stays batched (momentum-only,
         * many low-rate sources; runs with the token released). */
        if(FB_SerialEventID != 0 && P[i].ID != FB_SerialEventID) return 0;
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
#ifdef GALSF_RESOLVEDISM_FB_HEALPIX
                {double v_hp[3]; long ip_hp; v_hp[0]=-kernel.dp[0]; v_hp[1]=-kernel.dp[1]; v_hp[2]=-kernel.dp[2];
                 vec2pix_ring(1, v_hp, &ip_hp); out.HpxCount[(int)ip_hp] += 1;}
#endif
                    continue;
                }
                /* STARFORGE-style normalizer with MIN_REAL_NUMBER+fabs() guard. */
                double wk = (Mass_j * kernel.wk) / (MIN_REAL_NUMBER + fabs(local.FB_Area_weighted_sum_in));
#ifdef GALSF_RESOLVEDISM_FB_HEALPIX
                /* solid-angle-uniform weight: 1/(N_occupied_pixels * N_gas_in_this_pixel) */
                {double v_hp[3]; long ip_hp; int hp_, nocc_=0;
                 v_hp[0]=-kernel.dp[0]; v_hp[1]=-kernel.dp[1]; v_hp[2]=-kernel.dp[2];
                 vec2pix_ring(1, v_hp, &ip_hp);
                 for(hp_=0;hp_<12;hp_++) {if(local.HpxCount[hp_] > 0) {nocc_++;}}
                 if(nocc_ > 0 && local.HpxCount[(int)ip_hp] > 0) {wk = 1.0/((double)nocc_ * (double)local.HpxCount[(int)ip_hp]);}
                 else {wk = 0;}}
#endif

                /* ---- Mass + metals injection (wind and AGB ejecta) ---- */
                if(local.Mej > 0) {
                    double dM = wk * local.Mej;
#ifdef GALSF_RESOLVEDISM_METALS_INDIVIDUAL
                    /* SINGLE SOURCE OF TRUTH for total-Z: the summed METAL part of the
                     * SAME ElemYields[] the per-element blend consumes below.  Using the
                     * donor's independently-packed MetalMass scalar here let any
                     * MetalMass-vs-ΣElemYields mismatch (~0.6% in the wind channel) leak
                     * into ΣEA≠1 through the X_H=1−Z−He force (pisn200, 2026-07-04). */
                    double yield_metals = 0; for(k = 2; k < NUM_RESOLVEDISM_ELEMENTS; k++) {yield_metals += local.ElemYields[k];}
#else
                    double yield_metals = local.MetalMass;
#endif
#ifdef METALS
                    {
                        double Z_old, M_old = Mass_j;
                        #pragma omp atomic read
                        Z_old = P[j].Metallicity[0];   /* total Z in FIRE-pattern layout */
                        double dMZ = wk * yield_metals;
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
                            double dMZ_c = (c == ch) ? wk * yield_metals : 0;
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
                        if(X_H_new >= 0) {
                            #pragma omp atomic write
                            P[j].Metallicity[MET_OF(ELEM_H)] = (MyFloat)X_H_new;
                        } else {
                            /* metal-dominated pathological case (massive ejecta >> cell
                             * mass): the old X_H=0 clamp left ΣX = He+Σmetals > 1 (the
                             * April ΣEA=1.83 clip bug).  Renormalize all non-H slots AND
                             * total Z so ΣX = 1 with X_H = 0: element ratios and the
                             * slot0 == Σmetals invariant are both preserved. */
                            int mH = MET_OF(ELEM_H), mm;
                            double snorm = 0;
                            for(mm = 1; mm < NUM_METAL_SPECIES; mm++) {
                                if(mm == mH) continue;
                                double xv;
                                #pragma omp atomic read
                                xv = P[j].Metallicity[mm];
                                snorm += xv;
                            }
                            if(snorm > 0) {
                                double inv = 1.0 / snorm;
                                for(mm = 1; mm < NUM_METAL_SPECIES; mm++) {
                                    if(mm == mH) continue;
                                    #pragma omp atomic
                                    P[j].Metallicity[mm] *= inv;
                                }
                                #pragma omp atomic
                                P[j].Metallicity[0] *= inv;
                                #pragma omp atomic write
                                P[j].Metallicity[mH] = 0;
                            }
                        }
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
#ifdef GALSF_RESOLVEDISM_WINDS_THERMAL_ONLY
                /* PURE THERMAL wind mode (bracketing experiment, GRIFFIN fig.3 right
                 * column; 2026-07-05): NO momentum kick — the FULL wind kinetic
                 * luminosity wk * 1/2 Mej v_w^2 is deposited as heat (mass+metals
                 * still deposited as usual). Brackets the physical answer from the
                 * opposite side of momentum-only. Mutually exclusive w/ WINDS_THERMAL. */
                if(local.WindMomentum > 0 && local.Mej > 0) {
                    double dM_w0 = wk * local.Mej;
                    double v_w0  = local.WindMomentum / local.Mej;
                    double dE_full = 0.5 * (wk * local.Mej) * v_w0 * v_w0;
                    double du_full = dE_full / (Mass_j + dM_w0);
                    #pragma omp atomic
                    CellP[j].InternalEnergy += du_full;
                    #pragma omp atomic
                    CellP[j].InternalEnergyPred += du_full;
                    P[j].wakeup = -1;
                    NeedToWakeupParticles_local = 1;
                }
#else
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
#ifdef GALSF_RESOLVEDISM_WINDS_THERMAL
                    /* GRIFFIN-style inelastic residual (Lahen+23 eq. 3, 2026-07-05):
                     * the momentum kick transfers only ~dM/m of the wind KE to bulk
                     * motion; the remainder physically thermalizes at the (unresolved)
                     * reverse shock. Deposit it as heat. Default OFF: momentum-only is
                     * the deliberate anti-overcooling choice; this flag enables the
                     * energy-conserving variant for A/B testing (wind35 Weaver slope,
                     * cluster pre-SN pressurization). At Z~0.1 Zsun this is ~1% of an
                     * SN for 12-25 Msun stars but 0.2-0.7 SN-equiv for 60-100 Msun. */
                    if(local.Mej > 0 && local.WindMomentum > 0) {
                        double dM_w = wk * local.Mej;
                        double v_w  = local.WindMomentum / local.Mej;  /* wind speed, code units */
                        double dvrel2 = 0; int kk_w;
                        for(kk_w = 0; kk_w < 3; kk_w++) {
                            double vw_k = v_w * (-kernel.dp[kk_w] / kernel.r);  /* radial wind vector */
                            double dvr = vw_k - P[j].Vel[kk_w]/All.cf_atime;
                            dvrel2 += dvr*dvr;
                        }
                        double dE_th = 0.5 * (Mass_j*dM_w/(Mass_j+dM_w)) * dvrel2; /* inelastic residual */
                        double du_th = dE_th / (Mass_j + dM_w);
                        #pragma omp atomic
                        CellP[j].InternalEnergy += du_th;
                        #pragma omp atomic
                        CellP[j].InternalEnergyPred += du_th;
                    }
#endif
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
#endif /* GALSF_RESOLVEDISM_WINDS_THERMAL_ONLY (kick block ends here) */

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
