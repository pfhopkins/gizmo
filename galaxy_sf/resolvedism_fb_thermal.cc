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

/* Resolved-ISM thermal feedback injection: SN (CCSN/ECSN/PISN/PPISN) and Type Ia.
 * Injects thermal energy, ejecta mass, metal yields, and dust (destruction + injection).
 * Uses ngb_treefind_variable_threads (one-way search) — no momentum conservation needed.
 *
 * Called from resolvedism_inject_fb_energy() AFTER the momentum pass. */

#ifdef GALSF_RESOLVEDISM_FB

struct kernel_resolvedismFB_thermal {double dp[3], r, wk, dwk, hinv, hinv3, hinv4;};

#define CORE_FUNCTION_NAME resolvedismFB_thermal_evaluate
#define INPUTFUNCTION_NAME particle2in_resolvedismFB_thermal
#define OUTPUTFUNCTION_NAME out2particle_resolvedismFB_thermal
#define CONDITIONFUNCTION_FOR_EVALUATION if(resolvedismFB_thermal_active_check(i))
#include "../system/code_block_xchange_initialize.h"

struct INPUT_STRUCT_NAME
{
    MyDouble Pos[3], KernelRadius, Esne, Mej;
    /* Bit-exact normalizer measured by the weighting pre-pass (loop_iteration<0):
     * FB_Area_weighted_sum_in = Σ_j (Mass_j × kernel.wk).  In the injection pass
     * (loop_iteration>=0) we use wk_j = Mass_j × kernel.wk / FB_Area_weighted_sum_in
     * so Σwk = 1 by construction. */
    MyDouble FB_Area_weighted_sum_in;
    MyDouble MetalMass;
    int fb_channel; /* 0=SN, 3=Ia */
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

void particle2in_resolvedismFB_thermal(struct INPUT_STRUCT_NAME *in, int i, int loop_iteration)
{
    int k; for(k=0;k<3;k++) {in->Pos[k]=P[i].Pos[k];}
    in->KernelRadius = P[i].KernelRadius;
    in->Esne = 0; in->Mej = 0; in->MetalMass = 0;
    /* Load measured AWS from prior weighting pass (or 0 if this is the weighting pass). */
    in->FB_Area_weighted_sum_in = (loop_iteration >= 0) ? (MyDouble)P[i].FB_Area_weighted_sum : 0;
#ifdef GALSF_RESOLVEDISM_FB_HEALPIX
    {int hp_; for(hp_=0;hp_<12;hp_++) {in->HpxCount[hp_] = P[i].FB_HpxCount[hp_];}}
#endif
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

    /* ---- Type Ia ---- */
#ifdef GALSF_RESOLVEDISM_TYPE_IA
    if(P[i].SNe_ThisTimeStep == 4) {
        in->Esne = IA_ENERGY_ERG / UNIT_ENERGY_IN_CGS;
        /* Eject the entire WD progenitor mass.  In our single-star sampling the
         * WD's P[i].Mass evolved through prior wind/AGB phases and may not equal
         * the canonical Chandrasekhar mass (1.378 Msun); injecting the actual
         * particle mass guarantees the star is fully disrupted to 0 and no excess
         * mass is created.  Yields are rescaled by the actual-to-Chandrasekhar
         * ratio so Σ yields = actual ejecta mass (preserves Fe-peak composition). */
        /* same export-repack freeze as the SN path (2026-07-05) */
        double MIa_solar = (P[i].M_at_SN_trigger > 0) ? (double)P[i].M_at_SN_trigger
                                                      : P[i].Mass * UNIT_MASS_IN_SOLAR;
        in->Mej = MIa_solar / UNIT_MASS_IN_SOLAR;
        double Mej_actual_solar = MIa_solar;
        double yield_rescale = Mej_actual_solar / IA_EJECTA_MASS;
        double metal_mass_solar = 0;
        for(int k = 0; k < STBL_NELEM; k++) {
            double y_k = stellar_type_ia_yield(k) * yield_rescale;
#ifdef GALSF_RESOLVEDISM_METALS_INDIVIDUAL
            in->ElemYields[k] = y_k / UNIT_MASS_IN_SOLAR;
#endif
            if(k >= ELEM_C) metal_mass_solar += y_k;
        }
        in->MetalMass = metal_mass_solar / UNIT_MASS_IN_SOLAR;
#ifdef GALSF_RESOLVEDISM_DUST
        {
            double metal_yields_solar[STBL_NELEM], dust_yields_solar[NUM_RESOLVEDISM_DUST];
            for(int kk = 0; kk < STBL_NELEM; kk++) metal_yields_solar[kk] = in->ElemYields[kk] * UNIT_MASS_IN_SOLAR;
            resolvedism_dust_condensation(4, metal_yields_solar, dust_yields_solar);
            for(int kk = 0; kk < NUM_RESOLVEDISM_DUST; kk++) in->DustYields[kk] = dust_yields_solar[kk] / UNIT_MASS_IN_SOLAR;
        }
#endif
        return;
    }
#endif

    /* ---- SN (CCSN/ECSN/PISN/PPISN) ---- */
    if(P[i].SNe_ThisTimeStep != 1) return; /* only SN in this file */

    double Mstar, logM, logZ;
    if(!get_star_info(i, &Mstar, &logM, &logZ)) return;

#ifdef GALSF_RESOLVEDISM_STELLAR_TABLES
    int rem_type = stellar_remnant_type(logM, logZ);
    double rem_mass = stellar_remnant_mass(logM, logZ);

    switch(rem_type) {
        case REM_PISN:  in->Esne = 1.0e51 / UNIT_ENERGY_IN_CGS; rem_mass = 0; break; /* uniform 1e51 (was 1e52) */
        case REM_PPISN: in->Esne = 1.0e51 / UNIT_ENERGY_IN_CGS; break;
        case REM_ECSN:  in->Esne = 1.0e51 / UNIT_ENERGY_IN_CGS; break; /* uniform 1e51 (was 5e50) */
        case REM_FSN:
        case REM_DBH:   in->Esne = 0; break; /* no explosion energy */
        case REM_CCSN:
        default:        in->Esne = 1.0e51 / UNIT_ENERGY_IN_CGS; break;
    }

    /* SN ejecta: read absolute element ejecta from table directly.
     * Table guarantees: all values ≥0, Σ_k elem_ej_SN_mass = post-wind envelope ejecta.
     *
     * When WINDS is OFF at compile time, the wind ejecta from the star's lifetime
     * was never injected anywhere — it would silently vanish from the mass ledger.
     * To preserve mass conservation, fold the cumulative wind ejecta (at end of life)
     * into the SN injection so the entire returned mass is delivered in one event. */
#ifndef GALSF_RESOLVEDISM_WINDS
    double t_end_yr = stellar_lifetime(logM, logZ);
    double log_age_end = log10(DMAX(t_end_yr, 100.0));
#endif
    /* Mass-conservation fix: inject Mej_actual = (current particle mass) - rem_mass.
       This makes mass-removed-from-star = mass-injected-to-gas exactly.
       Per-element shape comes from the table; rescale so total = Mej_actual. */
    /* EXPORT-REPACK FIX (2026-07-05, found by check_twin_sn + FBOUT ledger):
     * particle2in is called AGAIN when packing the EXPORT buffer, AFTER out2particle
     * has already reduced P[i].Mass by the locally-deposited mass -> every REMOTE
     * receiver got Mej scaled by (1 - local_fraction) (measured: all imports at
     * exactly 0.9166x). Any event with remote receivers under-deposited and the
     * clip parked the shortfall on the star (remnants 2-6 Msun instead of 1.62 in
     * 7 edge cases). Freeze to the pre-walk snapshot so every pack is identical. */
    double M_pre_solar = (P[i].M_at_SN_trigger > 0) ? (double)P[i].M_at_SN_trigger
                                                    : P[i].Mass * UNIT_MASS_IN_SOLAR;
    double Mej_actual = M_pre_solar - rem_mass;
    if(Mej_actual < 0) Mej_actual = 0;

    double Mej_table = 0;
    double m_k_table[STBL_NELEM];
    for(k = 0; k < STBL_NELEM; k++) {
        m_k_table[k] = stellar_elem_ej_SN(logM, logZ, k);
#ifndef GALSF_RESOLVEDISM_WINDS
        m_k_table[k] += stellar_elem_ej_wind_cumulative(logM, logZ, log_age_end, k);
#endif
        Mej_table += m_k_table[k];
    }
    double scale = (Mej_table > 0) ? Mej_actual / Mej_table : 0;

    double metal_mass_solar = 0;
    for(k = 0; k < STBL_NELEM; k++) {
        double m_k = m_k_table[k] * scale;
#ifdef GALSF_RESOLVEDISM_METALS_INDIVIDUAL
        in->ElemYields[k] = m_k / UNIT_MASS_IN_SOLAR;
#endif
        if(k >= ELEM_C) metal_mass_solar += m_k;
    }
    in->Mej = Mej_actual / UNIT_MASS_IN_SOLAR;
    in->MetalMass = metal_mass_solar / UNIT_MASS_IN_SOLAR;

#ifdef GALSF_RESOLVEDISM_ISOLATED_FB_TEST
    {
        double sum_y_code = 0;
        for(int kk = 0; kk < NUM_RESOLVEDISM_ELEMENTS; kk++) sum_y_code += in->ElemYields[kk];
        printf("FBDBG_TH_DONOR star=%llu rem_type=%d M_pre=%.6e rem=%.6e Mej_act=%.6e Mej_tab=%.6e scale=%.6e in.Mej=%.6e Msun SumY=%.6e Msun Y[H]=%.6e Y[He]=%.6e Y[C]=%.6e Y[O]=%.6e Y[Fe]=%.6e\n",
            (unsigned long long)P[i].ID, rem_type, M_pre_solar, rem_mass, Mej_actual, Mej_table, scale,
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
        /* DUST FIX #3: per-subtype condensation (was: all subtypes -> CCSN flag 1) */
        int dust_flag = (rem_type == REM_PISN || rem_type == REM_PPISN) ? 5 : (rem_type == REM_ECSN) ? 6 : 1;
        resolvedism_dust_condensation(dust_flag, metal_yields_solar, dust_yields_solar);
        for(k = 0; k < NUM_RESOLVEDISM_DUST; k++) in->DustYields[k] = dust_yields_solar[k] / UNIT_MASS_IN_SOLAR;
    }
#endif

#else
    in->Esne = 1.0e51 / UNIT_ENERGY_IN_CGS;
#endif
}

struct OUTPUT_STRUCT_NAME
{
    /* Mass actually deposited to gas neighbors during the injection walk.
     * out2particle subtracts this from the star → bit-exact mass conservation. */
    MyFloat M_coupled;
    /* Weighting-pass accumulator: out.FB_Area_weighted_sum_accum = Σ_j (Mass_j × kernel.wk)
     * over neighbors walked on this rank/iteration.  Out2particle sums these across all
     * mode=0/mode=1 calls into P[i].FB_Area_weighted_sum for the next (injection) pass. */
    MyDouble FB_Area_weighted_sum_accum;
#ifdef GALSF_RESOLVEDISM_SN_SPAWN
    MyDouble KernelGasMass;      /* plain gas mass in kernel (spawn-criterion input) */
#endif
#ifdef GALSF_RESOLVEDISM_ISOLATED_FB_TEST
    MyDouble SumWK; int Nrecv;   /* injection-walk diagnostics (2026-07-05) */
#endif
#ifdef GALSF_RESOLVEDISM_FB_HEALPIX
    MyFloat HpxCount[12];
#endif
}
*DATARESULT_NAME, *DATAOUT_NAME;

void out2particle_resolvedismFB_thermal(struct OUTPUT_STRUCT_NAME *out, int i, int mode, int loop_iteration)
{
    if(loop_iteration < 0) {
        /* Weighting pass: accumulate measured kernel sum onto the star.  Called for
         * mode=0 (local walk) AND mode=1 (per-remote-rank import return) — both
         * contributions sum into P[i].FB_Area_weighted_sum for the upcoming injection. */
        P[i].FB_Area_weighted_sum += out->FB_Area_weighted_sum_accum;
#ifdef GALSF_RESOLVEDISM_FB_HEALPIX
        {int hp_; for(hp_=0;hp_<12;hp_++) {P[i].FB_HpxCount[hp_] += out->HpxCount[hp_];}}
#endif
#ifdef GALSF_RESOLVEDISM_SN_SPAWN
        P[i].FB_KernelGasMass += out->KernelGasMass;
#endif
        return;
    }
    /* Injection pass: subtract M_coupled.  With Σwk=1 from the weighting pre-pass,
     * the SUM of M_coupled across all calls (1 local + N remote) equals Mej exactly,
     * so star ends at M_pre − Mej = rem_mass (table value) to FP precision. */
    if(P[i].Mass > 0) {
        P[i].Mass -= out->M_coupled;
        if((P[i].Mass < 0) || (isnan(P[i].Mass))) P[i].Mass = 0;
    }
#ifdef GALSF_RESOLVEDISM_ISOLATED_FB_TEST
    if(out->M_coupled != 0 || out->Nrecv > 0) {
        printf("FBOUT task=%d donor=%llu mode=%d Nrecv=%d SumWK=%.8e Mcoupled=%.8e Mass_now=%.8e\n",
               ThisTask, (unsigned long long)P[i].ID, mode, out->Nrecv,
               (double)out->SumWK, (double)out->M_coupled, (double)P[i].Mass); fflush(stdout);
    }
#endif
}

int resolvedismFB_thermal_active_check(int i);
int resolvedismFB_thermal_active_check(int i)
{
    if(P[i].Type != 4) return 0;
    if(P[i].KernelRadius <= 0) return 0;
    if(P[i].NumNgb <= 0) return 0;
    /* per-event serialization (see resolvedism_fb_serialized_pass in resolvedism_fb.cc):
     * when the token is set, only that single event is active in this walk */
    if(FB_SerialEventID != 0 && P[i].ID != FB_SerialEventID) return 0;
    if(P[i].SNe_ThisTimeStep == 1 || P[i].SNe_ThisTimeStep == 4) return 1;
    return 0;
}


int resolvedismFB_thermal_evaluate(int target, int mode, int *exportflag, int *exportnodecount, int *exportindex, int *ngblist, int loop_iteration)
{
    int startnode, numngb_inbox, listindex = 0, j, k, n;
    double u, r2, h2;
    struct kernel_resolvedismFB_thermal kernel;
    struct INPUT_STRUCT_NAME local;
    struct OUTPUT_STRUCT_NAME out;
    memset(&out, 0, sizeof(struct OUTPUT_STRUCT_NAME));

    if(mode == 0) {particle2in_resolvedismFB_thermal(&local, target, loop_iteration);} else {local = DATAGET_NAME[target];}
    if(local.Esne <= 0 && local.Mej <= 0) return 0;
    if(local.KernelRadius <= 0) return 0;
    /* Injection pass requires non-zero AWS from prior weighting pass to normalize. */
    if(loop_iteration >= 0 && local.FB_Area_weighted_sum_in <= 0) return 0;
    h2 = local.KernelRadius * local.KernelRadius;
    kernel_hinv(local.KernelRadius, &kernel.hinv, &kernel.hinv3, &kernel.hinv4);

    if(mode == 0) {startnode = All.MaxPart;}
    else {startnode = DATAGET_NAME[target].NodeList[0]; startnode = Nodes[startnode].u.d.nextnode;}

    while(startnode >= 0)
    {
        while(startnode >= 0)
        {
            numngb_inbox = ngb_treefind_variable_threads(local.Pos, local.KernelRadius, target, &startnode, mode, exportflag, exportnodecount, exportindex, ngblist);
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
                 * measures Σ_j (Mass_j × kernel.wk); injection pass (loop_iteration>=0)
                 * uses that measured sum as denominator → Σ wk_j = 1 by construction. */
                if(loop_iteration < 0) {
                    out.FB_Area_weighted_sum_accum += Mass_j * kernel.wk;
#ifdef GALSF_RESOLVEDISM_FB_HEALPIX
                {double v_hp[3]; long ip_hp; v_hp[0]=-kernel.dp[0]; v_hp[1]=-kernel.dp[1]; v_hp[2]=-kernel.dp[2];
                 vec2pix_ring(1, v_hp, &ip_hp); out.HpxCount[(int)ip_hp] += 1;}
#endif
#ifdef GALSF_RESOLVEDISM_SN_SPAWN
                out.KernelGasMass += Mass_j;
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

                /* ---- Thermal energy injection ---- */
                if(local.Esne > 0) {
#ifdef COSMIC_RAY_FLUID
                    double cr_frac = All.CosmicRay_SNeFraction;
                    double dE = wk * local.Esne * (1.0 - cr_frac) / Mass_j;
#else
                    double dE = wk * local.Esne / Mass_j;
#endif
                    #pragma omp atomic
                    CellP[j].InternalEnergy += dE;
                    #pragma omp atomic
                    CellP[j].InternalEnergyPred += dE;
                    /* update pressure immediately after energy injection (gizmo2017 pattern) */
                    {
                        double u_pred, rho_j;
                        #pragma omp atomic read
                        u_pred = CellP[j].InternalEnergyPred;
                        #pragma omp atomic read
                        rho_j = CellP[j].Density;
                        #pragma omp atomic write
                        CellP[j].Pressure = (GAMMA_DEFAULT - 1.0) * u_pred * rho_j;
                    }
                    P[j].wakeup = -1;
                    NeedToWakeupParticles_local = 1;
#ifdef COSMIC_RAY_FLUID
                    double dEcr = wk * local.Esne * cr_frac;
                    double v_ej = (local.Mej > 0) ? sqrt(2.0 * local.Esne / local.Mej) : 3000.0/UNIT_VEL_IN_KMS;
                    double crdir[3]; for(k=0;k<3;k++) {crdir[k] = -kernel.dp[k] / kernel.r;}
#ifdef GALSF_RESOLVEDISM_ISOLATED_FB_TEST
                    { static int crdbg=0; if(crdbg<40 && dEcr>0) { crdbg++;
                        double ecr_before=CellP[j].CosmicRayEnergy[0];
                        inject_cosmic_rays(dEcr, v_ej, 0, j, crdir);
                        printf("CRDBG cell=%llu Esne=%.4e cr_frac=%.4e wk=%.4e dEcr_in=%.4e v_ej=%.4e | CR_egy[0] %.4e->%.4e (d=%.4e)\n",
                            (unsigned long long)P[j].ID, local.Esne, cr_frac, wk, dEcr, v_ej,
                            ecr_before, CellP[j].CosmicRayEnergy[0], CellP[j].CosmicRayEnergy[0]-ecr_before);
                        fflush(stdout); } else { inject_cosmic_rays(dEcr, v_ej, 0, j, crdir); } }
#else
                    inject_cosmic_rays(dEcr, v_ej, 0, j, crdir);
#endif
#endif
                }

#ifdef GALSF_RESOLVEDISM_ISOLATED_FB_TEST
                { static long fbw_ev=-1; static double fbw_sum=0; static int fbw_n=0;
                  if((long)FB_SerialEventID != fbw_ev) { if(fbw_ev>=0 && fbw_n>0) {printf("FBWK event=%ld task=%d Sum_wk=%.8e n_recv=%d\n", fbw_ev, ThisTask, fbw_sum, fbw_n); fflush(stdout);} fbw_ev=(long)FB_SerialEventID; fbw_sum=0; fbw_n=0; }
                  fbw_sum += wk; fbw_n++; }
#endif
                /* ---- Mass + metals injection ---- */
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
                    /* 28-slot layout: write yield Y[k] for table element k (0..26)
                     * into Met[MET_OF(k)] (= k+1) via mass-fraction blend.  Met[0]
                     * (total Z) is updated separately above.  After the loop,
                     * Met[1] (H) gets blended from yields like everything else
                     * — Σ Met[1..27] stays at 1 since Σ Y[k] = Mej. */
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
                    /* Force ΣMet = 1 by construction: recompute H from conservation
                     * X_H = 1 − Z − Y.  Absorbs float32 roundoff that accumulates
                     * across the 27-element mass-fraction blend above. */
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
                        printf("FBDBG_TH_RECV cell=%llu wk=%.6e Mass_pre=%.6e dM=%.6e SumdMX=%.6e ratio_SumdMX_dM=%.6e SumX_pre=%.9e SumX_post=%.9e\n",
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
#ifdef GALSF_RESOLVEDISM_ISOLATED_FB_TEST
                    out.SumWK += wk; out.Nrecv++;
#endif
                    P[j].wakeup = -1;
                    NeedToWakeupParticles_local = 1;
                }

#ifdef GALSF_RESOLVEDISM_DUST
#if !defined(GALSF_RESOLVEDISM_DUST_NO_MCKEE) && !defined(GALSF_RESOLVEDISM_DUST_SHOCK_DESTRUCTION)
                /* SN shock destruction of pre-existing dust.
                 * REGIME NOTE (2026-07-05): McKee 1989 is an SNR-lifetime-integrated
                 * estimator for an UNRESOLVED remnant. Stella resolves the Sedov phase,
                 * and the resolved hot bubble sputters dust explicitly (dust_evolve)
                 * -> potential double-count. Also: per-cell application saturates at
                 * frac=1 in ~Msun cells (kernel holds far less than the 984*wk Msun the
                 * formula wants to shock). GALSF_RESOLVEDISM_DUST_NO_MCKEE disables this
                 * hook for the destruction-accounting A/B test (dust_mckee_on/off). */
                if(local.Esne > 0) {
                    double E_into_cell = wk * local.Esne;
                    double Rho_j;
                    #pragma omp atomic read
                    Rho_j = CellP[j].Density;
                    double frac_dest = resolvedism_dust_sn_destruction_frac(Rho_j, E_into_cell, Mass_j);
                    if(frac_dest > 0) {
                        for(k = 0; k < NUM_RESOLVEDISM_DUST; k++) {
                            #pragma omp atomic
                            CellP[j].Dust[k] *= (1.0 - frac_dest);
                        }
                    }
                }
#endif /* !GALSF_RESOLVEDISM_DUST_NO_MCKEE */
                /* Inject new dust from SN/Ia ejecta */
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

                /* NaN sanity check */
                {
                    double u_j;
                    #pragma omp atomic read
                    u_j = CellP[j].InternalEnergy;
                    if(!isfinite(u_j) || u_j < 0) {
                        printf("NAN_CHECK_FB_THERMAL: Task=%d neighbor ID=%llu u=%.6e after injection from star target=%d\n",
                            ThisTask, (unsigned long long)P[j].ID, u_j, target);
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

    if(mode == 0) {out2particle_resolvedismFB_thermal(&out, target, 0, loop_iteration);} else {DATARESULT_NAME[target] = out;}
    return 0;
}


void resolvedism_fb_thermal_calc(int fb_loop_iteration)
{
    if(fb_loop_iteration < 0) {
        PRINT_STATUS(" ..SN/Ia weighting pre-pass (measures Σ Mass_j*kernel.wk on each star)");
    } else {
        PRINT_STATUS(" ..injecting single-star SN/Ia thermal energy + mass + metals");
    }
    #include "../system/code_block_xchange_perform_ops_malloc.h"
    loop_iteration = fb_loop_iteration;  /* STARFORGE pattern: malloc.h declared loop_iteration=0; override */
    #include "../system/code_block_xchange_perform_ops.h"
    #include "../system/code_block_xchange_perform_ops_demalloc.h"
}

#include "../system/code_block_xchange_finalize.h"

#endif /* GALSF_RESOLVEDISM_FB */
