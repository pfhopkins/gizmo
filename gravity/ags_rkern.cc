#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <map>
#include <vector>
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../sidm/dm_fuzzy_flux_functions.h"
#include "../sidm/sidm_core_flux_functions.h"
#include "../sidm/cbe_integrator_flux_functions.h"
#ifdef GRAIN_COLLISIONS
#include "../solids/grain_helper_functions.h"
#endif
#include "../mesh/kernel.h"
#include "../mesh/ghost_symlist_lifecycle.h"
#include "ags_gpu_decls.h"
#include "ags_functions.h"
#include "../mesh/ghost_writeback.h"

/*! \file ags_rkern.c
 *  \brief kernel length determination for non-gas particles
 *
 *  This file contains a loop modeled on the gas density computation which 
 *    determines softening lengths (and appropriate correction terms) 
 *    for all particle types, to make softenings fully adaptive
 */
/*
 * This file was written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */


#define AGS_DSOFT_TOL (0.5)    // amount by which softening lengths are allowed to vary in single timesteps //

/*! this routine is called by the adaptive gravitational softening neighbor search and forcetree (for application 
    of the appropriate correction terms), to determine which particle types "talk to" which other particle types 
    (i.e. which particle types you search for to determine the softening radii for gravity). For effectively volume-filling
    fluids like gas or dark matter, it makes sense for this to be 'matched' to particles of the same type. For other 
    particle types like stars or sink particles, it's more ambiguous, and requires some judgement on the part of the user. 
    The routine specifically returns a bitflag which defines all valid particles to which a particle of type 'primary' 
    can 'see': i.e. SUM(2^n), where n are all the particle types desired for neighbor finding,
    so e.g. if you want particle types 0 and 4, set the bitmask = 17 = 1 + 16 = 2^0 + 2^4
 */
int ags_gravity_kernel_shared_BITFLAG(short int particle_type_primary)
{
#ifdef ADAPTIVE_GRAVSOFT_FORALL
    if(!((1 << particle_type_primary) & (ADAPTIVE_GRAVSOFT_FORALL))) {return 0;} /* particle is NOT one of the designated 'adaptive' types */
#endif

#ifdef ADAPTIVE_GRAVSOFT_FROM_TIDAL_CRITERION
    if(!((1 << particle_type_primary) & (ADAPTIVE_GRAVSOFT_FROM_TIDAL_CRITERION))) {return ADAPTIVE_GRAVSOFT_FROM_TIDAL_CRITERION;} /* particle is NOT one of the designated 'adaptive' types */
#endif

    if(particle_type_primary == 0) {return 1;} /* gas particles see gas particles */

#if (ADAPTIVE_GRAVSOFT_FORALL & 32) && defined(SINK_PARTICLES)
    if(particle_type_primary == 5) {return 1;} /* sink particle particles are AGS-active, but using sink physics, they see only gas */
#endif
    
#if defined(GALSF) && ( (ADAPTIVE_GRAVSOFT_FORALL & 16) || (ADAPTIVE_GRAVSOFT_FORALL & 8) || (ADAPTIVE_GRAVSOFT_FORALL & 4) )
    if(All.ComovingIntegrationOn) /* stars [4 for cosmo runs, 2+3+4 for non-cosmo runs] are AGS-active and see baryons (any type) */
    {
        if(particle_type_primary == 4) {return 17;} // 2^0+2^4
    } else {
        if((particle_type_primary == 4)||(particle_type_primary == 2)||(particle_type_primary == 3)) {return 29;} // 2^0+2^2+2^3+2^4
    }
#endif
    
#ifdef DM_SIDM
    if((1 << particle_type_primary) & (DM_SIDM)) {return DM_SIDM;} /* SIDM particles see other SIDM particles, regardless of type/mass */
#endif
    
#ifdef AGS_KERNELRADIUS_CALCULATION_IS_ACTIVE
    return (1 << particle_type_primary); /* if we haven't been caught by one of the above checks, we simply return whether or not we see 'ourselves' */
#endif
    
    return 0;
}



#ifdef AGS_KERNELRADIUS_CALCULATION_IS_ACTIVE
/* routine to determine if we need to use ags_density to calculate KernelRadius */
int ags_density_isactive(int i)
{
    int default_to_return = 0; // default to not being active - needs to be pro-actively 'activated' by some physics
#ifdef ADAPTIVE_GRAVSOFT_FORALL
    default_to_return = 1;
    if(!((1 << P[i].Type) & (ADAPTIVE_GRAVSOFT_FORALL))) /* particle is NOT one of the designated 'adaptive' types */
    {
        P[i].AGS_KernelRadius = All.ForceSoftening[P[i].Type];
        P[i].AGS_zeta = 0;
        default_to_return = 0;
    } else {default_to_return = 1;} /* particle is AGS-active */
#endif
#if defined(ADAPTIVE_GRAVSOFT_FORGAS) || (ADAPTIVE_GRAVSOFT_FORALL & 1)
    if(P[i].Type==0)
    {
        P[i].AGS_KernelRadius = P[i].KernelRadius; // gas sees gas, these are identical
        default_to_return = 0; // don't actually need to do the loop //
    }
#endif
#ifdef DM_SIDM
    if((1 << P[i].Type) & (DM_SIDM)) {default_to_return = 1;}
#endif
#if defined(DM_FUZZY) || defined(CBE_INTEGRATOR)
    if(P[i].Type == 1) {default_to_return = 1;}
#endif
    if(P[i].TimeBin < 0) {default_to_return = 0;} /* check our 'marker' for particles which have finished iterating to an KernelRadius solution (if they have, dont do them again) */
    return default_to_return;
}
    

/* routine to return the maximum allowed softening */
double ags_return_maxsoft(int i)
{
    double maxsoft = All.MaxKernelRadius; // user-specified maximum: nothing is allowed to exceed this
#ifdef PMGRID /* Maximum allowed gravitational softening when using the TreePM method. The quantity is given in units of the scale used for the force split (PM_ASMTH) */
    maxsoft = DMIN(maxsoft, 1e3 * 0.5 * All.Asmth[0]); /* no more than 1/2 the size of the largest PM cell, times a 'safety factor' which can be pretty big */
#endif
#if (ADAPTIVE_GRAVSOFT_FORALL & 32) && defined(SINK_PARTICLES) && !defined(SINGLE_STAR_SINK_DYNAMICS)
    if(P[i].Type == 5) {maxsoft = All.SinkMaxAccretionRadius  / All.cf_atime;}   // MaxAccretionRadius is now defined in params.txt in PHYSICAL units
#endif
    return maxsoft;
}

    
/* routine to return the minimum allowed softening */
double ags_return_minsoft(int i)
{
    double minsoft = All.ForceSoftening[P[i].Type]; // this is the user-specified minimum
#if !defined(ADAPTIVE_GRAVSOFT_FORALL)
    minsoft = DMIN(All.MinKernelRadius, minsoft);
#endif
    return minsoft;
}


/* CPU wrappers around the GPU-callable _P forms in ags_functions.h. The
   wrappers exist so existing CPU call sites that rely on the global P stay
   untouched, while GPU kernels and the flux_functions.h templates call the
   _P forms with an explicit particle_data pointer. */
double INLINE_FUNC Get_Particle_Size_AGS(int i) { return Get_Particle_Size_AGS_P(i, P); }
double get_particle_volume_ags(int j) { return get_particle_volume_ags_P(j, P); }


#ifdef AGS_FACE_CALCULATION_IS_ACTIVE

/* --------------------------------------------------------------------------
 Subroutine here exists to calculate the MFM-like effective faces for purposes of face-interaction evaluation
 -------------------------------------------------------------------------- */

/* routine to invert the NV_T matrix after neighbor pass */
double do_cbe_nvt_inversion_for_faces(int i)
{
    /* initialize the matrix to be inverted */
    MyDouble NV_T[3][3], Tinv[3][3]; int j,k; for(j=0;j<3;j++) {for(k=0;k<3;k++) {NV_T[j][k]=P[i].NV_T[j][k];}}
    /* want to work in dimensionless units for defining certain quantities robustly, so normalize out the units */
    double dimensional_NV_T_normalizer = pow( P[i].KernelRadius , 2-NUMDIMS ); /* this has the same dimensions as NV_T here */
    for(j=0;j<3;j++) {for(k=0;k<3;k++) {NV_T[j][k] /= dimensional_NV_T_normalizer;}} /* now NV_T should be dimensionless */
    /* Also, we want to be able to calculate the condition number of the matrix to be inverted, since
        this will tell us how robust our procedure is (and let us know if we need to improve the conditioning) */
    double ConditionNumber=0, ConditionNumber_threshold = 10. * CONDITION_NUMBER_DANGER; /* set a threshold condition number - above this we will 'pre-condition' the matrix for better behavior */
    double trace_initial = NV_T[0][0] + NV_T[1][1] + NV_T[2][2]; /* initial trace of this symmetric, positive-definite matrix; used below as a characteristic value for adding the identity */
    double conditioning_term_to_add = 1.05 * (trace_initial / NUMDIMS) / ConditionNumber_threshold; /* this will be added as a test value if the code does not reach the desired condition number */
    /* now enter an iterative loop to arrive at a -well-conditioned- inversion to use */
    while(1)
    {
        /* initialize the matrix this will go into */
        ConditionNumber = matrix_invert_ndims(NV_T, Tinv); // compute the matrix inverse, and return the condition number
        if(ConditionNumber < ConditionNumber_threshold) {break;} // end loop if we have reached target conditioning for the matrix
        for(j=0;j<NUMDIMS;j++) {NV_T[j][j] += conditioning_term_to_add;} /* add the conditioning term which should make the matrix better-conditioned for subsequent use: this is a normalization times the identity matrix in the relevant number of dimensions */
        conditioning_term_to_add *= 1.2; /* multiply the conditioning term so it will grow and eventually satisfy our criteria */
    } // end of loop broken when condition number is sufficiently small
    for(j=0;j<3;j++) {for(k=0;k<3;k++) {P[i].NV_T[j][k] = Tinv[j][k] / dimensional_NV_T_normalizer;}} // now P[i].NV_T holds the inverted matrix elements //
    return ConditionNumber;
}

#endif





/* ------------------------------------------------------------------------------------------------------
 Everything below here is a giant block to define the sub-routines needed to calculate additional force
  terms for particle types that do not fall into the 'hydro' category.
 -------------------------------------------------------------------------------------------------------- */
int AGSForce_isactive(int i);
int AGSForce_isactive(int i)
{
    if(P[i].TimeBin < 0) return 0; /* check our 'marker' for particles which have finished iterating to an KernelRadius solution (if they have, dont do them again) */
#ifdef DM_SIDM
    if((1 << P[i].Type) & (DM_SIDM)) return 1;
#endif
#if defined(DM_FUZZY) || defined(CBE_INTEGRATOR)
    if(P[i].Type == 1) return 1;
#endif
    return 0; // default to no-action, need to affirm calculation above //
}


void AGSForce_calc(void)
{
    CPU_Step[CPU_MISC] += measure_time(); double t00_truestart = my_second();
    PRINT_STATUS(" ..entering AGS-Force calculation [as hydro loop for non-gas elements]\n");
    /* before doing any operations, need to zero the appropriate memory so we can correctly do pair-wise operations */
#if defined(DM_SIDM)
    {int i; for (int i : ActiveParticleList) {P[i].dtime_sidm = 10.*get_particle_timestep_in_physical(i);}}
#endif
#ifdef CBE_INTEGRATOR
    /* need to zero values for active particles (which will be re-calculated) before they are added below */
    //for (int i : ActiveParticleList) {int k1,k2; for(k1=0;k1<CBE_INTEGRATOR_NBASIS;k1++) {for(k2=0;k2<CBE_INTEGRATOR_NMOMENTS;k2++) {P[i].CBE_basis_moments_dt[k1][k2] = 0;}}}
#endif
    /* GPU neighbor-list path for AGSForce_calc. Partition active particles
       (isactive == 1) by their shared neighbor-type bitmask and launch the
       GPU kernel once per group, same pattern as ags_density(). */
    double timeall = 0, timecomp = 0, timecomm = 0, timewait = 0, t0 = 0;
    CPU_Step[CPU_MISC] += measure_time(); t0 = my_second();
    double ags_ghost_safety = gizmo_ghost_safety_factor();
    gizmo_density_prep_ghosts(ags_ghost_safety);

    std::map<int, std::vector<int>> bitmask_groups;
    uint64_t local_bm_presence_f = 0;
    for (int ii : ActiveParticleList) {
        if(AGSForce_isactive(ii)) {
            int bm = ags_gravity_kernel_shared_BITFLAG(P[ii].Type);
            if(bm > 0 && bm < 64) { bitmask_groups[bm].push_back(ii); local_bm_presence_f |= (1ULL << bm); }
        }
    }
    /* Symmetrise across ranks so all ranks call ghost_writeback_agsforce the same number of times. */
    uint64_t global_bm_presence_f = local_bm_presence_f;
    if(NTask > 1) MPI_Allreduce(&local_bm_presence_f, &global_bm_presence_f, 1, MPI_UINT64_T, MPI_BOR, MPI_COMM_WORLD);

    /* Zero per-iteration i-side accumulators for active AGSForce particles.
       These correspond to the OUTPUTFUNCTION_NAME fields that use mode==0
       ASSIGN (not ASSIGN_ADD). */
    for(auto& kv : bitmask_groups) {
        for(int ii : kv.second) {
#ifdef DM_FUZZY
            P[ii].AGS_Dt_Numerical_QuantumPotential = 0;
#if (DM_FUZZY > 0)
            P[ii].AGS_Dt_Psi_Re = P[ii].AGS_Dt_Psi_Im = P[ii].AGS_Dt_Psi_Mass = 0;
#endif
#endif
#if defined(CBE_INTEGRATOR)
            P[ii].AGS_vsig = 0;
            for(int k1 = 0; k1 < CBE_INTEGRATOR_NBASIS; k1++) {
                for(int k2 = 0; k2 < CBE_INTEGRATOR_NMOMENTS; k2++) {
                    P[ii].CBE_basis_moments_dt[k1][k2] = 0;
                }
            }
#endif
        }
    }

    /* Iterate global bitmask union so all ranks call ghost_writeback_agsforce the same number of times. */
    for(int bm = 1; bm < 64; bm++) {
        if(!(global_bm_presence_f & (1ULL << bm))) continue;
        std::vector<int>& ilist = bitmask_groups[bm];  /* empty if rank has none */
        int nl_num_active = (int)ilist.size();
        int *nl_active = (int *) mymalloc("agsforce_nl_active", (nl_num_active > 0 ? nl_num_active : 1) * sizeof(int));
        double *nl_radii = (double *) mymalloc("agsforce_nl_radii", (nl_num_active > 0 ? nl_num_active : 1) * sizeof(double));
        for(int a = 0; a < nl_num_active; a++) { nl_active[a] = ilist[a]; nl_radii[a] = P[ilist[a]].AGS_KernelRadius; }
        struct ags_force_gpu_out *nl_outs = (struct ags_force_gpu_out *) mymalloc(
            "agsforce_nl_outs", (nl_num_active > 0 ? nl_num_active : 1) * sizeof(struct ags_force_gpu_out));

        /* Snapshot ghost Vel/dp/NInteractions + zero wakeup so post-kernel
           values become pure deltas to reverse-communicate. */
        ghost_write_detector_begin("ags_force");
        ghost_writeback_zero_agsforce();
        ags_force_evaluate_gpu(P, NumPart, nl_active, nl_num_active, nl_radii, bm, nl_outs);
        ghost_writeback_agsforce();
        ghost_write_detector_end();

        /* Scatter i-side accumulators into P[ii] (match CPU OUTPUT semantics). */
        for(int a = 0; a < nl_num_active; a++) {
            int ii = nl_active[a];
#if defined(DM_SIDM)
            for(int k = 0; k < 3; k++) {
                P[ii].Vel[k] += nl_outs[a].sidm_kick[k];
                P[ii].dp[k]  += nl_outs[a].sidm_kick[k] * P[ii].Mass;
            }
            if(nl_outs[a].dtime_sidm < P[ii].dtime_sidm) P[ii].dtime_sidm = nl_outs[a].dtime_sidm;
            P[ii].NInteractions += nl_outs[a].si_count;
#endif
#if defined(GRAIN_EVOLUTION) && (GRAIN_EVOLUTION & 7)
            /* Phase 17b pairwise-outcome scatter. COAG: absorbed mass +
             * per-species composition mass go to ii. Mass conservation:
             * Σ_i Grain_DeltaCoagMass equals Σ_j (M_j set to 0), since
             * each pair is processed exactly once by SIDM dedup.
             * FRAG/SHAT (C8/C9): erosion-fraction multiplier applied as
             * Grain_Size *= factor (1.0 means no event). */
            if(P[ii].Mass > 0) {
                if(nl_outs[a].Grain_DeltaCoagMass > 0) {
                    double M_old = (double)P[ii].Mass;
                    double M_new = M_old + nl_outs[a].Grain_DeltaCoagMass;
                    /* Composition mixing: per-species mass on absorber +
                     * per-species mass absorbed from j-neighbors. */
                    for(int s = 0; s < GRAIN_NUM_SPECIES; s++) {
                        double M_species_s = M_old * (double)P[ii].Composition[s] + nl_outs[a].Grain_DeltaCoag_CompositionMass[s];
                        if(M_species_s < 0) { M_species_s = 0; }
                        P[ii].Composition[s] = (MyFloat)(M_species_s / M_new);
                    }
                    /* Size update: monodisperse mass-conserving rule with
                     * N_phys preserved on the absorber (each absorber
                     * grain takes ~one j-grain worth of mass on average).
                     * a_new = a_old * (M_new / M_old)^(1/3). */
                    P[ii].Grain_Size = (MyFloat)((double)P[ii].Grain_Size * pow(M_new / M_old, 1.0 / 3.0));
                    P[ii].Mass       = (MyDouble)M_new;
                }
                if(nl_outs[a].Grain_DeltaErosionFrac != 1.0 && nl_outs[a].Grain_DeltaErosionFrac > 0.0) {
                    /* C8/C9 size shrinkage applied multiplicatively. */
                    P[ii].Grain_Size = (MyFloat)((double)P[ii].Grain_Size * nl_outs[a].Grain_DeltaErosionFrac);
                }
            }
#endif
#ifdef DM_FUZZY
            for(int k = 0; k < 3; k++) P[ii].GravAccel[k] += nl_outs[a].acc[k];
            P[ii].AGS_Dt_Numerical_QuantumPotential += nl_outs[a].AGS_Dt_Numerical_QuantumPotential;
#if (DM_FUZZY > 0)
            P[ii].AGS_Dt_Psi_Re   += nl_outs[a].AGS_Dt_Psi_Re;
            P[ii].AGS_Dt_Psi_Im   += nl_outs[a].AGS_Dt_Psi_Im;
            P[ii].AGS_Dt_Psi_Mass += nl_outs[a].AGS_Dt_Psi_Mass;
#endif
#endif
#if defined(CBE_INTEGRATOR)
            if(nl_outs[a].AGS_vsig > P[ii].AGS_vsig) P[ii].AGS_vsig = nl_outs[a].AGS_vsig;
            for(int k1 = 0; k1 < CBE_INTEGRATOR_NBASIS; k1++) {
                for(int k2 = 0; k2 < CBE_INTEGRATOR_NMOMENTS; k2++) {
                    P[ii].CBE_basis_moments_dt[k1][k2] += nl_outs[a].CBE_basis_moments_dt[k1][k2];
                }
            }
#endif
        }
        myfree(nl_outs); myfree(nl_radii); myfree(nl_active);
    }

    if(NTask > 1) { ghost_exchange_cleanup(); }
    timecomp += timediff(t0, my_second());
    /* do final operations on results: these are operations that can be done after the complete set of iterations */
#ifdef CBE_INTEGRATOR
        for (int i : ActiveParticleList) {do_postgravity_cbe_calcs(i);} // do any final post-tree-walk calcs from the CBE integrator here //
#endif
    /* collect timing information */
    double t1; t1 = WallclockTime = my_second(); timeall = timediff(t00_truestart, t1);
    CPU_Step[CPU_AGSDENSCOMPUTE] += timecomp; CPU_Step[CPU_AGSDENSWAIT] += timewait;
    CPU_Step[CPU_AGSDENSCOMM] += timecomm; CPU_Step[CPU_AGSDENSMISC] += timeall - (timecomp + timewait + timecomm);
}


#endif // AGS_KERNELRADIUS_CALCULATION_IS_ACTIVE
