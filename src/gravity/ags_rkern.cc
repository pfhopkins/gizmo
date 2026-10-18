#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../mesh/kernel.h"
#include "gravtree_force_kernel.h"   /* shared-bitflag SSOT (gravtree_ags_kernel_shared_bitflag) */
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


/* AGS_DSOFT_TOL (amount by which softening lengths may vary per step) is defined
 * globally in precompiler_logic.h, shared with the AGS density loop. */

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
    return gravtree_ags_kernel_shared_bitflag(particle_type_primary); /* body moved verbatim to gravtree_force_kernel.h (shared with the GPU gravity walk) */
}



#ifdef AGS_KERNELRADIUS_CALCULATION_IS_ACTIVE
/* routine to determine if we need to use ags_density to calculate KernelRadius */
int ags_density_isactive(int i) { return ags_density_isactive_P(i, P); }
    

/* routine to return the maximum allowed softening */
double ags_return_maxsoft(int i) { return ags_return_maxsoft_P(i, P); }

    
/* routine to return the minimum allowed softening */
double ags_return_minsoft(int i) { return ags_return_minsoft_P(i, P); }


/* CPU wrappers around the GPU-callable _P forms in ags_functions.h. The
   wrappers exist so existing CPU call sites that rely on the global P stay
   untouched, while GPU kernels and the flux_functions.h templates call the
   _P forms with an explicit particle_data pointer. */
double INLINE_FUNC Get_Particle_Size_AGS(int i) { return Get_Particle_Size_AGS_P(i, P); }
double get_particle_volume_ags(int j) { return get_particle_volume_ags_P(j, P); }






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
#if defined(CBE_INTEGRATOR)
    if(CBE_INTEGRATOR_DOES_TYPE(P[i].Type)) return 1;
#elif defined(DM_FUZZY)
    if(P[i].Type == 1) return 1;
#endif
    return 0; // default to no-action, need to affirm calculation above //
}


/* AGSForce_calc() lives in gravity/ags_force_loop.cc. Moved out of this TU
 * to keep ags_rkern.cc host-only (carries unrelated host helpers
 * AGSForce_isactive, ags_gravity_kernel_shared_BITFLAG,
 * ags_return_minsoft/maxsoft, ags_invert_nvt_for_faces). */


#endif // AGS_KERNELRADIUS_CALCULATION_IS_ACTIVE
