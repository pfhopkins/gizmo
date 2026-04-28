#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <map>
#include <vector>
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../mesh/kernel.h"
#include "../mesh/ghost_symlist_lifecycle.h"
#include "dm_fuzzy_gpu.h"

/*! \file dm_fuzzy.c
 *  \brief routines needed for fuzzy-DM implementation
 *         This file was written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#ifdef DM_FUZZY



/* do_dm_fuzzy_flux_computation / _old moved to sidm/dm_fuzzy_functions.h
   as KOKKOS_INLINE_FUNCTION so both the CPU tree-walk and the B2 AGSForce
   GPU kernel can share a single body. */




/* kicks for fuzzy-dm integration: just put relevant drift-kick operators here to keep the code clean
 mode=0 -> 'kick', mode=1 -> 'drift' */
void do_dm_fuzzy_drift_kick(int i, double dt, int mode)
{
    if(mode==0)
    {
        // calculate various energies: quantum potential QP0, 'stored' numerical pressure NQ0, kinetic energy KE0
        double dNQ=P[i].AGS_Dt_Numerical_QuantumPotential*dt, NQ0=P[i].AGS_Numerical_QuantumPotential, NQ1=NQ0+dNQ, KE0=0.5*P[i].Mass*P[i].Vel.norm_sq()*All.cf_a2inv;
        double f00 = 0.5 * All.ScalarField_hbar_over_mass; // this encodes the coefficient with the mass of the particle: units vel*L = hbar / particle_mass
        double d2rho = P[i].AGS_Gradients2_Density[0][0] + P[i].AGS_Gradients2_Density[1][1] + P[i].AGS_Gradients2_Density[2][2]; // laplacian
        double drho2 = P[i].AGS_Gradients_Density.norm_sq();
        double QP0 = (f00*f00 / P[i].AGS_Density) * (d2rho - 0.5*drho2/P[i].AGS_Density); // quantum 'potential'
        NQ1 = DMAX(0,DMAX(NQ1,0.1*NQ0)); NQ1 = DMIN(NQ1,1.1*DMAX(DMAX(KE0+NQ0,fabs(QP0)),KE0+NQ0+QP0)); // limit kick to not produce unphysical energy over-or-under-shoot
        P[i].AGS_Numerical_QuantumPotential = NQ1;
    }

#if (DM_FUZZY > 0) /* if using direct-wavefunction integration methods */
    double vol_inv = P[i].AGS_Density / P[i].Mass;
    if(mode == 0)
    {
        //double psimag_mass_old = (P[i].AGS_Psi_Re*P[i].AGS_Psi_Re + P[i].AGS_Psi_Im*P[i].AGS_Psi_Im) * vol_inv;
        P[i].AGS_Psi_Re += P[i].AGS_Dt_Psi_Re * dt;
        P[i].AGS_Psi_Im += P[i].AGS_Dt_Psi_Im * dt;
        double mass_old = P[i].Mass, dmass = P[i].AGS_Dt_Psi_Mass * dt, mass_new = mass_old + dmass;
        dmass = DMIN(DMAX(dmass,-0.5*mass_old),0.5*mass_old);
        mass_new = mass_old + dmass;
        double psimag_mass_new = (P[i].AGS_Psi_Re*P[i].AGS_Psi_Re + P[i].AGS_Psi_Im*P[i].AGS_Psi_Im) * vol_inv;
#if (DM_FUZZY == 2)
        mass_new = psimag_mass_new; /* uses direct [NON-MASS-CONSERVING] integration of psi field */
#endif
        double psi_corr_fac = sqrt(mass_new / (MIN_REAL_NUMBER + psimag_mass_new));
        P[i].Mass = mass_new; P[i].AGS_Psi_Re *= psi_corr_fac; P[i].AGS_Psi_Im *= psi_corr_fac;

        P[i].AGS_Density = P[i].Mass * vol_inv;
        P[i].AGS_Psi_Re_Pred = P[i].AGS_Psi_Re;
        P[i].AGS_Psi_Im_Pred = P[i].AGS_Psi_Im;
    } else {
        /* in drift mode, AGS_Density should automatically be drifted already by the predictor step, but not the other quantities here */
        P[i].AGS_Psi_Re_Pred += P[i].AGS_Dt_Psi_Re * dt;
        P[i].AGS_Psi_Im_Pred += P[i].AGS_Dt_Psi_Im * dt;
        P[i].AGS_Density *= 1. + DMIN(DMAX(P[i].AGS_Dt_Psi_Mass*dt/P[i].Mass,-0.5),0.5);
    }
#endif
}


/* initialize wavefunction values in the code ICs */
void do_dm_fuzzy_initialization(void)
{
#if (DM_FUZZY > 0)
    int i;
    for(i = 0; i < NumPart; i++)
    {
        double volume = P[i].AGS_Density / P[i].Mass, psimag = sqrt(P[i].AGS_Density), phase = 0;
        /* approximation for initial phase below is fine for slowly-varying k, otherwise not ideal */
        Vec3<double> pos_d = {(double)P[i].Pos[0], (double)P[i].Pos[1], (double)P[i].Pos[2]};
        Vec3<double> vel_d = {(double)P[i].Vel[0], (double)P[i].Vel[1], (double)P[i].Vel[2]};
        phase = dot(pos_d, vel_d) / All.ScalarField_hbar_over_mass;

        P[i].AGS_Psi_Re = psimag * volume * cos(phase); /* remember, we evolve the volume-integrated value of psi */
        P[i].AGS_Psi_Im = psimag * volume * sin(phase);

        P[i].AGS_Dt_Psi_Mass = 0; P[i].AGS_Dt_Psi_Re = 0; P[i].AGS_Dt_Psi_Im = 0;
        P[i].AGS_Psi_Re_Pred = P[i].AGS_Psi_Re; P[i].AGS_Psi_Im_Pred = P[i].AGS_Psi_Im;
    }
#endif
}



/* dm_fuzzy_reconstruct_and_slopelimit{,_sub} moved to sidm/dm_fuzzy_functions.h
   as KOKKOS_INLINE_FUNCTION. */




/* --------------------------------------------------------------------------
 Everything below here is a giant block to define the sub-routines needed
 to calculate the higher-order matrix gradient estimators for the density
 field, around each DM element (based on its interacting neighbor set,
 within the AGS_KernelRadius volume). This will give the density gradients
 AGS_Gradients_Density needed to actually compute the quantum pressure tensor
 -------------------------------------------------------------------------- */


/* define a common 'gradients' structure to hold everything we're going to take derivatives of */
struct Quantities_for_Gradients_DM
{
    MyDouble AGS_Density, AGS_Gradients_Density[3];
#if (DM_FUZZY > 0)
    MyDouble AGS_Psi_Re, AGS_Gradients_Psi_Re[3], AGS_Psi_Im, AGS_Gradients_Psi_Im[3];
#endif
};

/* this is a temporary structure for quantities used ONLY in the loop below, for example for computing the slope-limiters (for the Reimann problem) */
static struct temporary_dmgradients_data_topass
{
    struct Quantities_for_Gradients_DM Maxima;
    struct Quantities_for_Gradients_DM Minima;
}
*DMGradDataPasser;

struct kernel_DMGrad {Vec3<double> dp; double r,wk_i, wk_j, dwk_i, dwk_j,h_i;};



#define CORE_FUNCTION_NAME DMGrad_evaluate /* name of the 'core' function doing the actual inter-neighbor operations. this MUST be defined somewhere as "int CORE_FUNCTION_NAME(int target, int mode, int *exportflag, int *exportnodecount, int *exportindex, int *ngblist, int loop_iteration)" */
#define INPUTFUNCTION_NAME particle2in_DMGrad    /* name of the function which loads the element data needed (for e.g. broadcast to other processors, neighbor search) */
#define OUTPUTFUNCTION_NAME out2particle_DMGrad  /* name of the function which takes the data returned from other processors and combines it back to the original elements */
#define CONDITIONFUNCTION_FOR_EVALUATION if(ags_density_isactive(i)) /* function for which elements will be 'active' and allowed to undergo operations. can be a function call, e.g. 'density_is_active(i)', or a direct function call like 'if(P[i].Mass>0)' */
#include "../system/code_block_xchange_initialize.h" /* pre-define all the ALL_CAPS variables we will use below, so their naming conventions are consistent and they compile together, as well as defining some of the function calls needed */


/* this structure defines the variables that need to be sent -from- the 'searching' element */
struct INPUT_STRUCT_NAME
{
    Vec3<MyDouble> Pos; MyDouble AGS_KernelRadius;
    struct Quantities_for_Gradients_DM GQuant;
    int NodeList[NODELISTLENGTH], Type;
}
*DATAIN_NAME, *DATAGET_NAME;

/* this subroutine assigns the values to the variables that need to be sent -from- the 'searching' element */
static inline void particle2in_DMGrad(struct INPUT_STRUCT_NAME *in, int i, int loop_iteration)
{
    in->Pos = P[i].Pos;
    in->AGS_KernelRadius = P[i].AGS_KernelRadius;
    in->Type = P[i].Type;
    in->GQuant.AGS_Density = P[i].AGS_Density;
    for(int k=0;k<3;k++) {in->GQuant.AGS_Gradients_Density[k] = P[i].AGS_Gradients_Density[k];}
#if (DM_FUZZY > 0)
    in->GQuant.AGS_Psi_Re = P[i].AGS_Psi_Re_Pred * P[i].AGS_Density / P[i].Mass;
    for(int k=0;k<3;k++) {in->GQuant.AGS_Gradients_Psi_Re[k] = P[i].AGS_Gradients_Psi_Re[k];}
    in->GQuant.AGS_Psi_Im = P[i].AGS_Psi_Im_Pred * P[i].AGS_Density / P[i].Mass;
    for(int k=0;k<3;k++) {in->GQuant.AGS_Gradients_Psi_Im[k] = P[i].AGS_Gradients_Psi_Im[k];}
#endif
}


/* this structure defines the variables that need to be sent -back to- the 'searching' element */
struct OUTPUT_STRUCT_NAME
{
    struct Quantities_for_Gradients_DM Gradients[3];
    struct Quantities_for_Gradients_DM Maxima;
    struct Quantities_for_Gradients_DM Minima;
}
*DATARESULT_NAME, *DATAOUT_NAME;

#define ASSIGN_ADD_PRESET(x,y,mode) (mode == 0 ? (x=y) : (x+=y))
#define MINMAX_CHECK(x,xmin,xmax) ((x<xmin)?(xmin=x):((x>xmax)?(xmax=x):(1)))
#define MAX_ADD(x,y,mode) ((y > x) ? (x = y) : (1)) // simpler definition now used
#define MIN_ADD(x,y,mode) ((y < x) ? (x = y) : (1))

/* this subroutine assigns the values to the variables that need to be sent -back to- the 'searching' element */
static inline void out2particle_DMGrad(struct OUTPUT_STRUCT_NAME *out, int i, int mode, int loop_iteration)
{
    if(loop_iteration <= 0)
    {
        int k;
        MAX_ADD(DMGradDataPasser[i].Maxima.AGS_Density,out->Maxima.AGS_Density,mode);
        MIN_ADD(DMGradDataPasser[i].Minima.AGS_Density,out->Minima.AGS_Density,mode);
        for(k=0;k<3;k++) {ASSIGN_ADD_PRESET(P[i].AGS_Gradients_Density[k],out->Gradients[k].AGS_Density,mode);}
#if (DM_FUZZY > 0)
        for(k=0;k<3;k++) {ASSIGN_ADD_PRESET(P[i].AGS_Gradients_Psi_Re[k],out->Gradients[k].AGS_Psi_Re,mode);}
        for(k=0;k<3;k++) {ASSIGN_ADD_PRESET(P[i].AGS_Gradients_Psi_Im[k],out->Gradients[k].AGS_Psi_Im,mode);}
#endif
    } else {
        int k,k2;
        for(k=0;k<3;k++)
        {
            MAX_ADD(DMGradDataPasser[i].Maxima.AGS_Gradients_Density[k],out->Maxima.AGS_Gradients_Density[k],mode);
            MIN_ADD(DMGradDataPasser[i].Minima.AGS_Gradients_Density[k],out->Minima.AGS_Gradients_Density[k],mode);
            for(k2=0;k2<3;k2++) {ASSIGN_ADD_PRESET(P[i].AGS_Gradients2_Density[k2][k],out->Gradients[k].AGS_Gradients_Density[k2],mode);}
#if (DM_FUZZY > 0)
            for(k2=0;k2<3;k2++) {ASSIGN_ADD_PRESET(P[i].AGS_Gradients2_Psi_Re[k2][k],out->Gradients[k].AGS_Gradients_Psi_Re[k2],mode);}
            for(k2=0;k2<3;k2++) {ASSIGN_ADD_PRESET(P[i].AGS_Gradients2_Psi_Im[k2][k],out->Gradients[k].AGS_Gradients_Psi_Im[k2],mode);}
#endif
        }
        // do we need limiters here for the density gradients? Not clear if this all needs computing
    }
}

/* this actually builds the final gradients out of the data passed */
template<typename T>
void construct_gradient_DMGrad(Vec3<T>& grad, int i)
{
    /* use the NV_T matrix-based gradient estimator */
    double v_tmp[3] = {(double)grad[0], (double)grad[1], (double)grad[2]};
    for(int k=0;k<3;k++) {grad[k] = (T)(P[i].NV_T[k][0]*v_tmp[0] + P[i].NV_T[k][1]*v_tmp[1] + P[i].NV_T[k][2]*v_tmp[2]);}
}


/* this subroutine does the actual neighbor-element calculations (this is the 'core' of the loop, essentially) */
/*!   -- this subroutine contains no writes to shared memory -- */
int DMGrad_evaluate(int target, int mode, int *exportflag, int *exportnodecount, int *exportindex, int *ngblist, int loop_iteration)
{
    /* define variables */
    int startnode, numngb_inbox, listindex = 0, j, k, n;
    double hinv, hinv3, hinv4, r2, u;
    struct kernel_DMGrad kernel;
    struct INPUT_STRUCT_NAME local;
    struct OUTPUT_STRUCT_NAME out;
    /* zero memory and import data for local target */
    memset(&out, 0, sizeof(struct OUTPUT_STRUCT_NAME));
    memset(&kernel, 0, sizeof(struct kernel_DMGrad));
    if(mode == 0) {particle2in_DMGrad(&local, target, loop_iteration);} else {local = DATAGET_NAME[target];}
    /* check if we should bother doing a neighbor loop */
    if(local.AGS_KernelRadius <= 0) return 0;
    if(local.GQuant.AGS_Density <= 0) return 0;
    /* now set particle-i centric quantities so we don't do it inside the loop */
    kernel.h_i = local.AGS_KernelRadius;
    double h2_i = kernel.h_i*kernel.h_i;
    kernel_hinv(kernel.h_i, &hinv, &hinv3, &hinv4);
    int AGS_kernel_shared_BITFLAG = ags_gravity_kernel_shared_BITFLAG(local.Type); // determine allowed particle types for search for adaptive gravitational softening terms

    /* Now start the actual neighbor computation for this particle */
    if(mode == 0) {startnode = All.MaxPart; /* root node */} else {startnode = DATAGET_NAME[target].NodeList[0]; startnode = Nodes[startnode].u.d.nextnode;    /* open it */}
    while(startnode >= 0)
    {
        while(startnode >= 0)
        {
            numngb_inbox = ngb_treefind_variable_threads_targeted(local.Pos.data_ptr(), local.AGS_KernelRadius, target, &startnode, mode, exportflag, exportnodecount, exportindex, ngblist, AGS_kernel_shared_BITFLAG);
            if(numngb_inbox < 0) {return -2;} /* no neighbors! */
            for(n = 0; n < numngb_inbox; n++) /* neighbor loop */
            {
                j = ngblist[n]; /* since we use the -threaded- version above of ngb-finding, its super-important this is the lower-case ngblist here! */
                if((P[j].Mass <= 0)||(P[j].AGS_Density <= 0)) {continue;} /* make sure neighbor is valid */
                /* calculate position relative to target */
                for(int kk=0;kk<3;kk++) {kernel.dp[kk] = (double)(local.Pos[kk] - P[j].Pos[kk]);}
                nearest_xyz(kernel.dp);
                r2 = kernel.dp.norm_sq();
                if((r2 <= 0) || (r2 >= h2_i)) continue;
                /* calculate kernel quantities needed below */
                kernel.r = sqrt(r2); u = kernel.r * hinv;
                kernel_main(u, hinv3, hinv4, &kernel.wk_i, &kernel.dwk_i, -1);
                /* DIFFERENCE & SLOPE LIMITING: need to check maxima and minima of particle values in the kernel, to avoid 'overshoot' with our gradient estimators. this check should be among all interacting pairs */
                if(loop_iteration <= 0)
                {
                    double d_rho = P[j].AGS_Density - local.GQuant.AGS_Density;
                    MINMAX_CHECK(d_rho,out.Minima.AGS_Density,out.Maxima.AGS_Density);
                    for(k=0;k<3;k++) {out.Gradients[k].AGS_Density += -kernel.wk_i * kernel.dp[k] * d_rho;} /* sign is important here! */
#if (DM_FUZZY > 0)
                    d_rho = P[j].AGS_Psi_Re_Pred * P[j].AGS_Density / P[j].Mass - local.GQuant.AGS_Psi_Re;
                    for(k=0;k<3;k++) {out.Gradients[k].AGS_Psi_Re += -kernel.wk_i * kernel.dp[k] * d_rho;}
                    d_rho = P[j].AGS_Psi_Im_Pred * P[j].AGS_Density / P[j].Mass - local.GQuant.AGS_Psi_Im;
                    for(k=0;k<3;k++) {out.Gradients[k].AGS_Psi_Im += -kernel.wk_i * kernel.dp[k] * d_rho;}
#endif
                } else {
                    int k2; double d_grad_rho;
                    for(k=0;k<3;k++)
                    {
                        d_grad_rho = P[j].AGS_Gradients_Density[k] - local.GQuant.AGS_Gradients_Density[k];
                        MINMAX_CHECK(d_grad_rho,out.Minima.AGS_Gradients_Density[k],out.Maxima.AGS_Gradients_Density[k]);
                        for(k2=0;k2<3;k2++) {out.Gradients[k2].AGS_Gradients_Density[k] += -kernel.wk_i * kernel.dp[k2] * d_grad_rho;}
#if (DM_FUZZY > 0)
                        d_grad_rho = P[j].AGS_Gradients_Psi_Re[k] - local.GQuant.AGS_Gradients_Psi_Re[k];
                        for(k2=0;k2<3;k2++) {out.Gradients[k2].AGS_Gradients_Psi_Re[k] += -kernel.wk_i * kernel.dp[k2] * d_grad_rho;}
                        d_grad_rho = P[j].AGS_Gradients_Psi_Im[k] - local.GQuant.AGS_Gradients_Psi_Im[k];
                        for(k2=0;k2<3;k2++) {out.Gradients[k2].AGS_Gradients_Psi_Im[k] += -kernel.wk_i * kernel.dp[k2] * d_grad_rho;}
#endif
                    }
                } // loop_iteration
            } // numngb_inbox loop
        } // while(startnode)
        /* continue to open leaves if needed */
        if(mode == 1)
        {
            listindex++;
            if(listindex < NODELISTLENGTH)
            {
                startnode = DATAGET_NAME[target].NodeList[listindex];
                if(startnode >= 0) {startnode = Nodes[startnode].u.d.nextnode;    /* open it */}
            }
        }
    }
    /* Collect the result at the right place */
    if(mode == 0) {out2particle_DMGrad(&out, target, 0, loop_iteration);} else {DATARESULT_NAME[target] = out;}
    return 0;
}




void DMGrad_gradient_calc(void)
{
    CPU_Step[CPU_MISC] += measure_time(); double t00_truestart = my_second();
    PRINT_STATUS(" ..calculating higher-order gradients for DM density field\n");
    /* initialize data, if needed */
    if(All.Time==All.TimeBegin) {int i; for (int i : ActiveParticleList) {P[i].AGS_Numerical_QuantumPotential=0;}}

    /* allocate memory shared across all loops */
    DMGradDataPasser = (struct temporary_dmgradients_data_topass *) mymalloc("DMGradDataPasser",NumPart * sizeof(struct temporary_dmgradients_data_topass));
    /* GPU neighbor-list path: import DM ghosts once, keep alive across both
       gradient iterations (ghost content doesn't change between passes). */
    int loop_iteration = 0;
    double timeall=0, timecomp=0, timecomm=0, timewait=0, t0;
    CPU_Step[CPU_MISC] += measure_time(); t0 = my_second();
    double dmgrad_ghost_safety = gizmo_ghost_safety_factor();
    gizmo_density_prep_ghosts(dmgrad_ghost_safety);

    /* loop over the number of iterations needed to actually compute the gradients fully */
    for(loop_iteration=0; loop_iteration<2; loop_iteration++) // need 2 iterations to compute gradients-of-gradients
    {
        /* Partition active DMGrad particles by shared AGS neighbor-type bitmask.
           DM_FUZZY activates only for Type==1, so this yields at most one group. */
        std::map<int, std::vector<int>> bitmask_groups;
        for (int ii : ActiveParticleList) {
            if(ags_density_isactive(ii)) {
                int bm = ags_gravity_kernel_shared_BITFLAG(P[ii].Type);
                if(bm != 0) bitmask_groups[bm].push_back(ii);
            }
        }
        /* Zero per-iteration accumulators on i-side for all active particles */
        for(auto& kv : bitmask_groups) {
            for(int ii : kv.second) {
                if(loop_iteration <= 0) {
                    DMGradDataPasser[ii].Maxima.AGS_Density = 0; DMGradDataPasser[ii].Minima.AGS_Density = 0;
                    for(int k=0;k<3;k++) P[ii].AGS_Gradients_Density[k] = 0;
#if (DM_FUZZY > 0)
                    for(int k=0;k<3;k++) {P[ii].AGS_Gradients_Psi_Re[k] = 0; P[ii].AGS_Gradients_Psi_Im[k] = 0;}
#endif
                } else {
                    for(int k=0;k<3;k++) {
                        DMGradDataPasser[ii].Maxima.AGS_Gradients_Density[k] = 0;
                        DMGradDataPasser[ii].Minima.AGS_Gradients_Density[k] = 0;
                        for(int k2=0;k2<3;k2++) P[ii].AGS_Gradients2_Density[k2][k] = 0;
#if (DM_FUZZY > 0)
                        for(int k2=0;k2<3;k2++) {P[ii].AGS_Gradients2_Psi_Re[k2][k] = 0; P[ii].AGS_Gradients2_Psi_Im[k2][k] = 0;}
#endif
                    }
                }
            }
        }
        /* Launch one GPU kernel per bitmask group */
        for(auto& kv : bitmask_groups) {
            int bm = kv.first;
            std::vector<int>& ilist = kv.second;
            int nl_num_active = (int)ilist.size();
            int *nl_active = (int *) mymalloc("dmg_nl_active", (nl_num_active > 0 ? nl_num_active : 1) * sizeof(int));
            double *nl_radii = (double *) mymalloc("dmg_nl_radii", (nl_num_active > 0 ? nl_num_active : 1) * sizeof(double));
            struct dmgrad_gpu_in *nl_in = (struct dmgrad_gpu_in *) mymalloc("dmg_nl_in",
                (nl_num_active > 0 ? nl_num_active : 1) * sizeof(struct dmgrad_gpu_in));
            for(int a=0;a<nl_num_active;a++) {
                int ii = ilist[a];
                nl_active[a] = ii;
                nl_radii[a] = P[ii].AGS_KernelRadius;
                nl_in[a].AGS_Density = P[ii].AGS_Density;
                for(int k=0;k<3;k++) nl_in[a].AGS_Gradients_Density[k] = P[ii].AGS_Gradients_Density[k];
#if (DM_FUZZY > 0)
                nl_in[a].AGS_Psi_Re = P[ii].AGS_Psi_Re_Pred * P[ii].AGS_Density / P[ii].Mass;
                for(int k=0;k<3;k++) nl_in[a].AGS_Gradients_Psi_Re[k] = P[ii].AGS_Gradients_Psi_Re[k];
                nl_in[a].AGS_Psi_Im = P[ii].AGS_Psi_Im_Pred * P[ii].AGS_Density / P[ii].Mass;
                for(int k=0;k<3;k++) nl_in[a].AGS_Gradients_Psi_Im[k] = P[ii].AGS_Gradients_Psi_Im[k];
#endif
            }
            struct dmgrad_gpu_out *nl_outs = (struct dmgrad_gpu_out *) mymalloc("dmg_nl_outs",
                (nl_num_active > 0 ? nl_num_active : 1) * sizeof(struct dmgrad_gpu_out));
            dmgrad_evaluate_gpu(P, NumPart, nl_active, nl_num_active, nl_radii, nl_in, bm, loop_iteration, nl_outs);
            /* Scatter back */
            for(int a=0;a<nl_num_active;a++) {
                int ii = nl_active[a];
                if(loop_iteration <= 0) {
                    if(nl_outs[a].max_rho > DMGradDataPasser[ii].Maxima.AGS_Density) DMGradDataPasser[ii].Maxima.AGS_Density = nl_outs[a].max_rho;
                    if(nl_outs[a].min_rho < DMGradDataPasser[ii].Minima.AGS_Density) DMGradDataPasser[ii].Minima.AGS_Density = nl_outs[a].min_rho;
                    for(int k=0;k<3;k++) P[ii].AGS_Gradients_Density[k] += nl_outs[a].grad_rho[k];
#if (DM_FUZZY > 0)
                    for(int k=0;k<3;k++) {P[ii].AGS_Gradients_Psi_Re[k] += nl_outs[a].grad_psi_re[k]; P[ii].AGS_Gradients_Psi_Im[k] += nl_outs[a].grad_psi_im[k];}
#endif
                } else {
                    for(int k=0;k<3;k++) {
                        if(nl_outs[a].max_grho[k] > DMGradDataPasser[ii].Maxima.AGS_Gradients_Density[k]) DMGradDataPasser[ii].Maxima.AGS_Gradients_Density[k] = nl_outs[a].max_grho[k];
                        if(nl_outs[a].min_grho[k] < DMGradDataPasser[ii].Minima.AGS_Gradients_Density[k]) DMGradDataPasser[ii].Minima.AGS_Gradients_Density[k] = nl_outs[a].min_grho[k];
                        for(int k2=0;k2<3;k2++) P[ii].AGS_Gradients2_Density[k2][k] += nl_outs[a].grad2_rho[k2][k];
#if (DM_FUZZY > 0)
                        for(int k2=0;k2<3;k2++) {P[ii].AGS_Gradients2_Psi_Re[k2][k] += nl_outs[a].grad2_psi_re[k2][k]; P[ii].AGS_Gradients2_Psi_Im[k2][k] += nl_outs[a].grad2_psi_im[k2][k];}
#endif
                    }
                }
            }
            myfree(nl_outs); myfree(nl_in); myfree(nl_radii); myfree(nl_active);
        }

        /* do post-loop operations on the results */
        int i;
        for (int i : ActiveParticleList)
        {
            if(loop_iteration <= 0)
            {
                /* now we can properly calculate (second-order accurate) gradients of hydrodynamic quantities from this loop */
                construct_gradient_DMGrad(P[i].AGS_Gradients_Density,i);
#if (DM_FUZZY > 0)
                construct_gradient_DMGrad(P[i].AGS_Gradients_Psi_Re,i);
                construct_gradient_DMGrad(P[i].AGS_Gradients_Psi_Im,i);
#endif
                /* finally, we need to apply a sensible slope limiter to the gradients, to prevent overshooting */
                /* (actually not clear that we need to slope-limit these, because we are not using the gradients for reconstruction.
                    testing this now. if not, we can remove the limiter information entirely and save some time in these computations) */
            } else {
                int k;
                for(k=0;k<3;k++)
                {
                    /* construct the gradient-of-gradient */
                    construct_gradient_DMGrad(P[i].AGS_Gradients2_Density[k],i);
#if (DM_FUZZY > 0)
                    construct_gradient_DMGrad(P[i].AGS_Gradients2_Psi_Re[k],i);
                    construct_gradient_DMGrad(P[i].AGS_Gradients2_Psi_Im[k],i);
#endif
                }
                /* symmetrize the gradients */
                int k0[3]={0,0,1},k1[3]={1,2,2}; double tmp;
                for(k=0;k<3;k++)
                {
                    tmp = 0.5 * (P[i].AGS_Gradients2_Density[k0[k]][k1[k]] + P[i].AGS_Gradients2_Density[k1[k]][k0[k]]);
                    P[i].AGS_Gradients2_Density[k0[k]][k1[k]] = P[i].AGS_Gradients2_Density[k1[k]][k0[k]] = tmp;
#if (DM_FUZZY > 0)
                    tmp = 0.5 * (P[i].AGS_Gradients2_Psi_Re[k0[k]][k1[k]] + P[i].AGS_Gradients2_Psi_Re[k1[k]][k0[k]]);
                    P[i].AGS_Gradients2_Psi_Re[k0[k]][k1[k]] = P[i].AGS_Gradients2_Psi_Re[k1[k]][k0[k]] = tmp;
                    tmp = 0.5 * (P[i].AGS_Gradients2_Psi_Im[k0[k]][k1[k]] + P[i].AGS_Gradients2_Psi_Im[k1[k]][k0[k]]);
                    P[i].AGS_Gradients2_Psi_Im[k0[k]][k1[k]] = P[i].AGS_Gradients2_Psi_Im[k1[k]][k0[k]] = tmp;
#endif
                }
            }
        }
    } // end of loop_iteration

    /* de-allocate memory and collect timing information */
    if(NTask > 1) {ghost_exchange_cleanup();}
    myfree(DMGradDataPasser); /* free the temporary structure we created for the MinMax and additional data passing */
    double t1; t1 = WallclockTime = my_second(); timeall = timediff(t00_truestart, t1);
    CPU_Step[CPU_AGSDENSCOMPUTE] += timecomp; CPU_Step[CPU_AGSDENSWAIT] += timewait; CPU_Step[CPU_AGSDENSCOMM] += timecomm;
    CPU_Step[CPU_AGSDENSMISC] += timeall - (timecomp + timewait + timecomm);
}
#include "../system/code_block_xchange_finalize.h" /* de-define the relevant variables and macros to avoid compilation errors and memory leaks */




#endif // DM_FUZZY
