#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../mesh/kernel.h"
#include "resolvedism_fb_shared.h"

/* Resolved-ISM thermal feedback injection: SN (CCSN/ECSN/PISN/PPISN) and Type Ia.
 * Injects thermal energy, ejecta mass, metal yields, and dust (destruction + injection).
 * Uses ngb_treefind_variable_threads (one-way search) — no momentum conservation needed.
 *
 * Called from resolvedism_inject_sn_energy() AFTER the momentum pass. */

#ifdef GALSF_RESOLVEDISM_FB

struct kernel_resolvedismFB_thermal {double dp[3], r, wk, dwk, hinv, hinv3, hinv4;};

#define CORE_FUNCTION_NAME resolvedismFB_thermal_evaluate
#define INPUTFUNCTION_NAME particle2in_resolvedismFB_thermal
#define OUTPUTFUNCTION_NAME out2particle_resolvedismFB_thermal
#define CONDITIONFUNCTION_FOR_EVALUATION if(resolvedismFB_thermal_active_check(i))
#include "../system/code_block_xchange_initialize.h"

struct INPUT_STRUCT_NAME
{
    MyDouble Pos[3], KernelRadius, Esne, Mej, wt_sum;
    MyDouble MetalMass;
    int fb_channel; /* 0=SN, 3=Ia */
#ifdef GALSF_RESOLVEDISM_METALS_INDIVIDUAL
    MyDouble ElemYields[NUM_RESOLVEDISM_ELEMENTS];
#endif
#ifdef GALSF_RESOLVEDISM_DUST
    MyDouble DustYields[NUM_RESOLVEDISM_DUST];
#endif
    int NodeList[NODELISTLENGTH];
}
*DATAIN_NAME, *DATAGET_NAME;

void particle2in_resolvedismFB_thermal(struct INPUT_STRUCT_NAME *in, int i, int loop_iteration)
{
    int k; for(k=0;k<3;k++) {in->Pos[k]=P[i].Pos[k];}
    in->KernelRadius = P[i].KernelRadius;
    in->wt_sum = 0; in->Esne = 0; in->Mej = 0; in->MetalMass = 0;
    in->fb_channel = DMAX(P[i].SNe_ThisTimeStep - 1, 0);
#ifdef GALSF_RESOLVEDISM_METALS_INDIVIDUAL
    for(k=0; k<NUM_RESOLVEDISM_ELEMENTS; k++) in->ElemYields[k] = 0;
#endif
#ifdef GALSF_RESOLVEDISM_DUST
    for(k=0; k<NUM_RESOLVEDISM_DUST; k++) in->DustYields[k] = 0;
#endif
    if(P[i].Mass <= 0) return;
#ifdef DO_DENSITY_AROUND_NONGAS_PARTICLES
    in->wt_sum = P[i].DensityAroundParticle;
    if(P[i].DensityAroundParticle <= 0) return;
#endif

    /* ---- Type Ia ---- */
#ifdef GALSF_RESOLVEDISM_TYPE_IA
    if(P[i].SNe_ThisTimeStep == 4) {
        in->Esne = IA_ENERGY_ERG / UNIT_ENERGY_IN_CGS;
        in->Mej = IA_EJECTA_MASS / UNIT_MASS_IN_SOLAR;
        double metal_mass_solar = 0;
        for(int k = 0; k < STBL_NELEM; k++) {
            double y_k = stellar_type_ia_yield(k);
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
        case REM_PISN:  in->Esne = 1.0e52 / UNIT_ENERGY_IN_CGS; rem_mass = 0; break;
        case REM_PPISN: in->Esne = 1.0e51 / UNIT_ENERGY_IN_CGS; break;
        case REM_ECSN:  in->Esne = 5.0e50 / UNIT_ENERGY_IN_CGS; break;
        case REM_CCSN:
        default:        in->Esne = 1.0e51 / UNIT_ENERGY_IN_CGS; break;
    }

    double M_particle_solar = P[i].Mass * UNIT_MASS_IN_SOLAR;
    double Mej_solar = M_particle_solar - rem_mass;
    if(Mej_solar < 0) Mej_solar = 0;
    in->Mej = Mej_solar / UNIT_MASS_IN_SOLAR;

    double metal_mass_solar = 0;
    for(k = 0; k < STBL_NELEM; k++) {
        double net_y = stellar_sn_yield(logM, logZ, k);
        double X_birth = 0;
#ifdef GALSF_RESOLVEDISM_METALS_INDIVIDUAL
        X_birth = P[i].ElementAbundance[k];
#else
        if(k == ELEM_H) X_birth = 0.74;
        else if(k == ELEM_He) X_birth = 0.24;
        else {
            double Z_birth = DMAX(P[i].BirthMetallicity, 1e-10);
            double solar_frac[STBL_NELEM] = {
                0.7381, 0.2485, 2.36e-3, 6.91e-4, 5.72e-3, 3.26e-7, 1.25e-3,
                2.98e-5, 5.91e-4, 5.57e-5, 6.65e-4, 5.16e-6, 3.10e-4, 3.15e-6,
                7.37e-5, 2.93e-6, 6.44e-5, 3.48e-8, 3.59e-6, 2.30e-7, 1.37e-5,
                9.17e-6, 1.17e-3, 3.30e-6, 6.99e-5, 7.20e-7, 1.67e-6};
            X_birth = solar_frac[k] * (Z_birth / 0.014);
        }
#endif
        double M_elem_ej = net_y + X_birth * Mej_solar;
        if(M_elem_ej < 0) M_elem_ej = 0;
#ifdef GALSF_RESOLVEDISM_METALS_INDIVIDUAL
        in->ElemYields[k] = M_elem_ej / UNIT_MASS_IN_SOLAR;
#endif
        if(k >= ELEM_C) metal_mass_solar += M_elem_ej;
    }
    in->MetalMass = metal_mass_solar / UNIT_MASS_IN_SOLAR;

#ifdef GALSF_RESOLVEDISM_DUST
    {
        double metal_yields_solar[STBL_NELEM], dust_yields_solar[NUM_RESOLVEDISM_DUST];
        for(k = 0; k < STBL_NELEM; k++) metal_yields_solar[k] = in->ElemYields[k] * UNIT_MASS_IN_SOLAR;
        resolvedism_dust_condensation(1, metal_yields_solar, dust_yields_solar);
        for(k = 0; k < NUM_RESOLVEDISM_DUST; k++) in->DustYields[k] = dust_yields_solar[k] / UNIT_MASS_IN_SOLAR;
    }
#endif

#else
    in->Esne = 1.0e51 / UNIT_ENERGY_IN_CGS;
#endif
}

struct OUTPUT_STRUCT_NAME
{
    MyFloat dummy; /* thermal injection has no star-side return data */
}
*DATARESULT_NAME, *DATAOUT_NAME;

void out2particle_resolvedismFB_thermal(struct OUTPUT_STRUCT_NAME *out, int i, int mode, int loop_iteration)
{
    /* No recoil needed for thermal energy injection */
}

int resolvedismFB_thermal_active_check(int i);
int resolvedismFB_thermal_active_check(int i)
{
    if(P[i].Type != 4) return 0;
    if(P[i].KernelRadius <= 0) return 0;
    if(P[i].NumNgb <= 0) return 0;
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
    if(local.wt_sum <= 0) return 0;
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

                double wk = Mass_j * kernel.wk / local.wt_sum;

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
                    P[j].wakeup = 1;
                    NeedToWakeupParticles_local = 1;
#ifdef COSMIC_RAY_FLUID
                    double dEcr = wk * local.Esne * cr_frac;
                    double v_ej = (local.Mej > 0) ? sqrt(2.0 * local.Esne / local.Mej) : 3000.0/UNIT_VEL_IN_KMS;
                    double crdir[3]; for(k=0;k<3;k++) {crdir[k] = -kernel.dp[k] / kernel.r;}
                    inject_cosmic_rays(dEcr, v_ej, 0, j, crdir);
#endif
                }

                /* ---- Mass + metals injection ---- */
                if(local.Mej > 0) {
                    double dM = wk * local.Mej;
#ifdef METALS
                    {
                        double Z_old, M_old = Mass_j;
                        #pragma omp atomic read
                        Z_old = P[j].Metallicity[0];
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
                    for(k = 0; k < NUM_RESOLVEDISM_ELEMENTS; k++) {
                        double X_old;
                        #pragma omp atomic read
                        X_old = P[j].ElementAbundance[k];
                        double dMX = wk * local.ElemYields[k];
                        double dX = (dMX - X_old * dM) / (Mass_j + dM);
                        #pragma omp atomic
                        P[j].ElementAbundance[k] += dX;
                    }
#endif
                    #pragma omp atomic
                    P[j].Mass += dM;
                    P[j].wakeup = 1;
                    NeedToWakeupParticles_local = 1;
                }

#ifdef GALSF_RESOLVEDISM_DUST
                /* SN shock destruction of pre-existing dust */
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


void resolvedism_fb_thermal_calc(void)
{
    PRINT_STATUS(" ..injecting single-star SN/Ia thermal energy + mass + metals");
    #include "../system/code_block_xchange_perform_ops_malloc.h"
    #include "../system/code_block_xchange_perform_ops.h"
    #include "../system/code_block_xchange_perform_ops_demalloc.h"
}

#include "../system/code_block_xchange_finalize.h"

#endif /* GALSF_RESOLVEDISM_FB */
