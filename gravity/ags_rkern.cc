#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../mesh/kernel.h"
#include "ags_functions.h"

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


/* AGSForce_calc() lives in gravity/ags_force_loop.cc (Wave 3 close-out port).
 * Moved out of this TU to keep ags_rkern.cc host-only (carries unrelated
 * host helpers AGSForce_isactive, ags_gravity_kernel_shared_BITFLAG,
 * ags_return_minsoft/maxsoft, do_cbe_nvt_inversion_for_faces); see
 * reference_runner_port_checklist.md §5. */


#endif // AGS_KERNELRADIUS_CALCULATION_IS_ACTIVE
