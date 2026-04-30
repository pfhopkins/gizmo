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




#endif // DM_FUZZY
