#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../mesh/kernel.h"

/* Resolved-ISM supernova feedback for individually sampled single stars.
 * Each star particle represents ONE star drawn from the Kroupa IMF.
 * When a massive star (>=8 Msun) reaches the end of its lifetime, it
 * explodes as a single SN event injecting E_SN = 10^51 erg as thermal
 * energy into neighboring gas cells, kernel-weighted.
 *
 * Uses the code_block_xchange framework for MPI-parallel tree walks.
 * KernelRadius and DensityAroundParticle come from the density loop. */

#ifdef GALSF_RESOLVEDISM_FB

#ifndef DO_DENSITY_AROUND_NONGAS_PARTICLES
#error "GALSF_RESOLVEDISM_FB requires DO_DENSITY_AROUND_NONGAS_PARTICLES for kernel weights (wt_sum)"
#endif

/* =========================================================================== */
/*  Check the single star in each active star particle for a SN event.         */
/*  Each particle has exactly one sampled star in MstarSampleIMF[0].           */
/*  If stellar age > lifetime and mass >= 8 Msun, the star dies as a SN.       */
/* =========================================================================== */
void resolvedism_determine_SNe(void)
{
    if(All.Time <= 0) return;
    int i;
    int n_sne_local = 0, n_sne_total = 0;

    /* first pass: count local SNe so we can allocate buffers */
    int n_sne_count = 0;
    for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    {
        if(P[i].Type != 4 || P[i].Mass <= 0 || P[i].SNe_ThisTimeStep != 0) continue;
        double star_age_yr = evaluate_stellar_age_Gyr(i) * 1.0e9;
        if(star_age_yr <= 0) continue;
        double Mstar = 0;
#ifdef GALSF_RESOLVEDISM_STOCHASTIC_IMF
        Mstar = P[i].Mstar;
#endif
#ifdef GALSF_RESOLVEDISM_SAMPLE_IMF
        if(P[i].sampled) {Mstar = P[i].MstarSampleIMF[0];}
#endif
        if(Mstar < 8.0 || Mstar <= 0) continue;
        double lifetime_yr = get_lifetime(Mstar);
#ifdef GALSF_RESOLVEDISM_INSTANT_SN
        if(All.Time < All.TimeInstantSN) {lifetime_yr = 0;}
#endif
        if(star_age_yr > lifetime_yr) {n_sne_count++;}
    }

    /* allocate local SN info buffers */
    double *sn_x = (double *)mymalloc("sn_x", n_sne_count * sizeof(double));
    double *sn_y = (double *)mymalloc("sn_y", n_sne_count * sizeof(double));
    double *sn_z = (double *)mymalloc("sn_z", n_sne_count * sizeof(double));
    double *sn_u = (double *)mymalloc("sn_u", n_sne_count * sizeof(double));
    double *sn_rho = (double *)mymalloc("sn_rho", n_sne_count * sizeof(double));
    double *sn_mstar = (double *)mymalloc("sn_mstar", n_sne_count * sizeof(double));
    long long *sn_id = (long long *)mymalloc("sn_id", n_sne_count * sizeof(long long));

    /* second pass: flag SNe and collect info */
    for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    {
        if(P[i].Type != 4) continue;
        if(P[i].Mass <= 0) continue;

        /* Only check virgin particles (aligned with gizmo2017 update_weights.c:
         * if(P[i].flag_SNII == 0)). Stars with SNe_ThisTimeStep != 0 already
         * exploded or are marked done (-1). */
        if(P[i].SNe_ThisTimeStep != 0) continue;

        double star_age_yr = evaluate_stellar_age_Gyr(i) * 1.0e9;
        if(star_age_yr <= 0) continue;

#ifdef GALSF_RESOLVEDISM_STOCHASTIC_IMF
        /* Single star with mass P[i].Mstar */
        {
            double Mstar = P[i].Mstar;
            if(Mstar < 8.0) continue; /* only massive stars produce core-collapse SNe */
            double lifetime_yr = get_lifetime(Mstar);
#ifdef GALSF_RESOLVEDISM_INSTANT_SN
            if(All.Time < All.TimeInstantSN) {lifetime_yr = 0;}
#endif
            if(star_age_yr > lifetime_yr) {
                P[i].SNe_ThisTimeStep = 1; /* this single star dies as SN */
                sn_x[n_sne_local] = P[i].Pos[0];
                sn_y[n_sne_local] = P[i].Pos[1];
                sn_z[n_sne_local] = P[i].Pos[2];
                sn_u[n_sne_local] = P[i].InternalEnergyAroundParticle;
                sn_rho[n_sne_local] = P[i].DensityAroundParticle;
                sn_mstar[n_sne_local] = Mstar;
                sn_id[n_sne_local] = (long long)P[i].ID;
                P[i].Mstar = 0; /* prevent re-trigger on snapshot restart */
                n_sne_local++;
            }
        }
#endif

#ifdef GALSF_RESOLVEDISM_SAMPLE_IMF
        /* Single star stored in MstarSampleIMF[0] */
        if(P[i].sampled) {
            double Mstar = P[i].MstarSampleIMF[0];
            if(Mstar < 8.0) continue; /* not massive enough for core-collapse */
            if(Mstar <= 0) continue;  /* already exploded or empty */
            double lifetime_yr = get_lifetime(Mstar);
#ifdef GALSF_RESOLVEDISM_INSTANT_SN
            if(All.Time < All.TimeInstantSN) {lifetime_yr = 0;}
#endif
            if(star_age_yr > lifetime_yr) {
                P[i].SNe_ThisTimeStep = 1; /* this single star dies as SN */
                sn_x[n_sne_local] = P[i].Pos[0];
                sn_y[n_sne_local] = P[i].Pos[1];
                sn_z[n_sne_local] = P[i].Pos[2];
                sn_u[n_sne_local] = P[i].InternalEnergyAroundParticle;
                sn_rho[n_sne_local] = P[i].DensityAroundParticle;
                sn_mstar[n_sne_local] = Mstar;
                sn_id[n_sne_local] = (long long)P[i].ID;
                P[i].MstarSampleIMF[0] = 0; /* remove the dead star */
                n_sne_local++;
            }
        }
#endif
    }

    /* MPI_Gatherv to rank 0 for file output */
    MPI_Allreduce(&n_sne_local, &n_sne_total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if(n_sne_total > 0)
    {
        if(ThisTask == 0) {printf("ResolvedISM: %d single-star SNe this timestep at t=%g\n", n_sne_total, All.Time); fflush(stdout);}

        int *recvcounts = NULL, *displs = NULL;
        double *all_x = NULL, *all_y = NULL, *all_z = NULL, *all_u = NULL, *all_rho = NULL, *all_mstar = NULL;
        long long *all_id = NULL;

        if(ThisTask == 0)
        {
            recvcounts = (int *)mymalloc("recvcounts", NTask * sizeof(int));
            displs = (int *)mymalloc("displs", NTask * sizeof(int));
        }
        MPI_Gather(&n_sne_local, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if(ThisTask == 0)
        {
            displs[0] = 0;
            for(i = 1; i < NTask; i++) {displs[i] = displs[i-1] + recvcounts[i-1];}
            all_x = (double *)mymalloc("all_x", n_sne_total * sizeof(double));
            all_y = (double *)mymalloc("all_y", n_sne_total * sizeof(double));
            all_z = (double *)mymalloc("all_z", n_sne_total * sizeof(double));
            all_u = (double *)mymalloc("all_u", n_sne_total * sizeof(double));
            all_rho = (double *)mymalloc("all_rho", n_sne_total * sizeof(double));
            all_mstar = (double *)mymalloc("all_mstar", n_sne_total * sizeof(double));
            all_id = (long long *)mymalloc("all_id", n_sne_total * sizeof(long long));
        }
        MPI_Gatherv(sn_x, n_sne_local, MPI_DOUBLE, all_x, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(sn_y, n_sne_local, MPI_DOUBLE, all_y, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(sn_z, n_sne_local, MPI_DOUBLE, all_z, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(sn_u, n_sne_local, MPI_DOUBLE, all_u, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(sn_rho, n_sne_local, MPI_DOUBLE, all_rho, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(sn_mstar, n_sne_local, MPI_DOUBLE, all_mstar, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(sn_id, n_sne_local, MPI_LONG_LONG, all_id, recvcounts, displs, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

        if(ThisTask == 0)
        {
            for(i = 0; i < n_sne_total; i++) {
                fprintf(FdSNinfo, "%.16g %g %g %g %g %g %lld %g\n",
                    All.Time, all_x[i], all_y[i], all_z[i], all_u[i], all_rho[i], all_id[i], all_mstar[i]);
            }
            fflush(FdSNinfo);
            myfree(all_id);
            myfree(all_mstar);
            myfree(all_rho);
            myfree(all_u);
            myfree(all_z);
            myfree(all_y);
            myfree(all_x);
            myfree(displs);
            myfree(recvcounts);
        }
    }
    myfree(sn_id);
    myfree(sn_mstar);
    myfree(sn_rho);
    myfree(sn_u);
    myfree(sn_z);
    myfree(sn_y);
    myfree(sn_x);
}


/* =========================================================================== */
/*  Single-star SN thermal energy injection via code_block_xchange framework.  */
/*  Each exploding star dumps exactly E_SN = 10^51 erg into its gas neighbors, */
/*  kernel-weighted. One star = one SN = one energy injection event.           */
/* =========================================================================== */

struct kernel_resolvedismFB {double dp[3], r, wk, dwk, hinv, hinv3, hinv4;};

#define CORE_FUNCTION_NAME resolvedismFB_evaluate
#define INPUTFUNCTION_NAME particle2in_resolvedismFB
#define OUTPUTFUNCTION_NAME out2particle_resolvedismFB
#define CONDITIONFUNCTION_FOR_EVALUATION if(resolvedismFB_active_check(i))
#include "../system/code_block_xchange_initialize.h"

struct INPUT_STRUCT_NAME
{
    MyDouble Pos[3], KernelRadius, Esne, wt_sum;
    int NodeList[NODELISTLENGTH];
}
*DATAIN_NAME, *DATAGET_NAME;

void particle2in_resolvedismFB(struct INPUT_STRUCT_NAME *in, int i, int loop_iteration)
{
    int k; for(k=0;k<3;k++) {in->Pos[k]=P[i].Pos[k];}
    in->KernelRadius = P[i].KernelRadius;
    in->wt_sum = 0; in->Esne = 0;
#ifdef DO_DENSITY_AROUND_NONGAS_PARTICLES
    in->wt_sum = P[i].DensityAroundParticle;
    if((P[i].SNe_ThisTimeStep != 1)||(P[i].DensityAroundParticle<=0)||(P[i].Mass<=0)) return;
#else
    if((P[i].SNe_ThisTimeStep != 1)||(P[i].Mass<=0)) return;
#endif
    /* Single star SN: exactly 10^51 erg, no multiplier */
    in->Esne = 1.0e51 / UNIT_ENERGY_IN_CGS;
}

struct OUTPUT_STRUCT_NAME
{
    MyFloat dummy;
}
*DATARESULT_NAME, *DATAOUT_NAME;

void out2particle_resolvedismFB(struct OUTPUT_STRUCT_NAME *out, int i, int mode, int loop_iteration)
{
    /* pure thermal dump: no star particle update needed */
}

int resolvedismFB_active_check(int i);
int resolvedismFB_active_check(int i)
{
    if(P[i].Type != 4) {return 0;}
    if(P[i].KernelRadius <= 0) {return 0;}
    if(P[i].NumNgb <= 0) {return 0;}
    if(P[i].SNe_ThisTimeStep == 1) {return 1;} /* single star flagged for SN */
    return 0;
}


/*!   -- this subroutine writes to shared memory [updating the neighbor values]: need to protect these writes for openmp below */
int resolvedismFB_evaluate(int target, int mode, int *exportflag, int *exportnodecount, int *exportindex, int *ngblist, int loop_iteration)
{
    int startnode, numngb_inbox, listindex = 0, j, k, n;
    double u, r2, h2;
    struct kernel_resolvedismFB kernel;
    struct INPUT_STRUCT_NAME local;
    struct OUTPUT_STRUCT_NAME out;
    memset(&out, 0, sizeof(struct OUTPUT_STRUCT_NAME));

    if(mode == 0) {particle2in_resolvedismFB(&local, target, loop_iteration);} else {local = DATAGET_NAME[target];}
    if(local.Esne <= 0) return 0;
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
            numngb_inbox = ngb_treefind_pairs_threads(local.Pos, local.KernelRadius, target, &startnode, mode, exportflag, exportnodecount, exportindex, ngblist);
            if(numngb_inbox < 0) {return -2;}
            for(n = 0; n < numngb_inbox; n++)
            {
                j = ngblist[n]; /* since we use the -threaded- version above of ngb-finding, its super-important this is the lower-case ngblist here! */
                if(P[j].Type != 0) {continue;}
                double Mass_j;
                #pragma omp atomic read
                Mass_j = P[j].Mass;
                if(Mass_j <= 0) {continue;}

                for(k=0;k<3;k++) {kernel.dp[k] = local.Pos[k] - P[j].Pos[k];}
                NEAREST_XYZ(kernel.dp[0],kernel.dp[1],kernel.dp[2],1);
                r2=0; for(k=0;k<3;k++) {r2 += kernel.dp[k]*kernel.dp[k];}
                if(r2 <= 0) {continue;}
                if(r2 >= h2) {continue;}
                kernel.r = sqrt(r2);
                if(kernel.r <= 0) {continue;}
                u = kernel.r * kernel.hinv;
                if(u<1) {kernel_main(u, kernel.hinv3, kernel.hinv4, &kernel.wk, &kernel.dwk, 0);} else {kernel.wk=kernel.dwk=0;}
                if((kernel.wk <= 0)||(isnan(kernel.wk))) {continue;}

                double wk = Mass_j * kernel.wk / local.wt_sum; /* normalized weight function */
                double dE = wk * local.Esne / Mass_j; /* specific energy increment [per unit mass] */

                #pragma omp atomic
                CellP[j].InternalEnergy += dE;
                #pragma omp atomic
                CellP[j].InternalEnergyPred += dE;

            } /* for(n = 0; n < numngb; n++) */
        } /* while(startnode >= 0) inner */
        if(mode == 1)
        {
            listindex++;
            if(listindex < NODELISTLENGTH)
            {
                startnode = DATAGET_NAME[target].NodeList[listindex];
                if(startnode >= 0) {startnode = Nodes[startnode].u.d.nextnode;}
            }
        }
    } /* while(startnode >= 0) outer */

    if(mode == 0) {out2particle_resolvedismFB(&out, target, 0, loop_iteration);} else {DATARESULT_NAME[target] = out;}
    return 0;
}


void resolvedism_inject_sn_energy(void)
{
    PRINT_STATUS(" ..injecting single-star SN thermal energy");
    #include "../system/code_block_xchange_perform_ops_malloc.h"
    #include "../system/code_block_xchange_perform_ops.h"
    #include "../system/code_block_xchange_perform_ops_demalloc.h"

    /* Latch after injection (aligned with gizmo2017 stellarfeedback.c:
     * P[i].flag_SNII = -1; //unmark it!) */
    int i;
    for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
        if(P[i].SNe_ThisTimeStep == 1) P[i].SNe_ThisTimeStep = -1;
}
#include "../system/code_block_xchange_finalize.h"


#endif /* GALSF_RESOLVEDISM_FB */
