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
#if defined(CBE_INTEGRATOR)
    if(CBE_INTEGRATOR_DOES_TYPE(P[i].Type)) {default_to_return = 1;}
#elif defined(DM_FUZZY)
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
    /* MaxAccretionRadius is defined in params.txt in PHYSICAL units. Enforce the
     * recommended-policy invariant "nothing exceeds MaxKernelRadius" via DMIN
     * (never > All.MaxKernelRadius, already the line-1 seed): keeps the Mode-B
     * per-type node band (capped at MaxKernelRadius) a valid upper bound on this
     * sink's AGS leaf reach. No effect where SinkMaxAccretionRadius < MaxKernelRadius
     * (the recommended + every-tested config). */
    if(P[i].Type == 5) {maxsoft = DMIN(maxsoft, All.SinkMaxAccretionRadius / All.cf_atime);}
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

/* Invert the accumulated raw moment matrix into the symmetric, fully-populated
 * face operator used by the CBE/dm_fuzzy face construction and the DMGrad
 * gradient estimator. Single source of truth for AGS face NV_T inversion;
 * mirrors the hydro density path (hydro/density_loop.cc:508-561) including the
 * diagonal-loading preconditioning loop.
 *
 * Inputs : sym6 = upper-triangle raw moment components {00,01,02,11,12,22}
 *          h    = AGS kernel size (AGS_KernelRadius), used to normalize the
 *                 matrix into dimensionless units so the regularization scale
 *                 matches the matrix scale (it divides out of the result).
 * Output : Tinv_out = full 3x3 inverse of the raw moment matrix.
 * Returns: cn_expansion (neighbor-expansion signal), cn_final + status (diag).
 *
 * Robustness: matrix_invert_ndims() returns Tinv=0 and
 * ConditionNumber=1 for an exactly-singular matrix (Mat3::invert sets inv=0 on
 * det==0). So acceptance CANNOT key off the condition number alone — we also
 * require a finite, non-zero inverse, and we report a large cn_expansion for
 * the singular case so the neighbor iteration expands (a singular matrix needs
 * MORE neighbors, the opposite of what CN=1 would imply). */
AgsNvtInversion ags_invert_nvt_for_faces(const double sym6[6], double h, double Tinv_out[3][3])
{
    AgsNvtInversion result; result.cn_expansion = 1.0; result.cn_final = 1.0; result.status = 0;
    const double ConditionNumber_threshold = 10. * CONDITION_NUMBER_DANGER;

    /* Normalize to dimensionless units (divides out of the final inverse). */
    double dimensional_NV_T_normalizer = pow(h, 2 - NUMDIMS);
    double inv_norm = ((dimensional_NV_T_normalizer != 0) && !isnan(dimensional_NV_T_normalizer))
                      ? 1.0 / dimensional_NV_T_normalizer : 1.0;
    MyDouble NV_T[3][3], Tinv[3][3];
    NV_T[0][0] = sym6[0]*inv_norm; NV_T[0][1] = sym6[1]*inv_norm; NV_T[0][2] = sym6[2]*inv_norm;
    NV_T[1][0] = sym6[1]*inv_norm; NV_T[1][1] = sym6[3]*inv_norm; NV_T[1][2] = sym6[4]*inv_norm;
    NV_T[2][0] = sym6[2]*inv_norm; NV_T[2][1] = sym6[4]*inv_norm; NV_T[2][2] = sym6[5]*inv_norm;

    double trace_initial = NV_T[0][0] + NV_T[1][1] + NV_T[2][2];
    /* Absolute floor on the trace so the conditioning step makes progress even
     * for a (near-)zero raw matrix (no usable neighbors). */
    double trace_for_cond = DMAX(trace_initial, MIN_REAL_NUMBER);
    double conditioning_term_to_add = 1.05 * (trace_for_cond / NUMDIMS) / ConditionNumber_threshold;
    if(!(conditioning_term_to_add > 0)) {conditioning_term_to_add = MIN_REAL_NUMBER;}

    const int MAXIT = 50; int ok = 0, regularized = 0; int j,k;
    for(int it = 0; it < MAXIT; it++)
    {
        double ConditionNumber = matrix_invert_ndims(NV_T, Tinv);
        double frob_inv = 0; int finite_inv = 1;
        for(j=0;j<3;j++) {for(k=0;k<3;k++) {double e = Tinv[j][k]; if(!isfinite(e)) {finite_inv = 0;} frob_inv += e*e;}}
        int inversion_ok = finite_inv && (frob_inv > 0) && isfinite(ConditionNumber);
        if(it == 0) {result.cn_expansion = inversion_ok ? ConditionNumber : ConditionNumber_threshold;}
        if(inversion_ok && (ConditionNumber < ConditionNumber_threshold)) {
            result.cn_final = ConditionNumber;
            result.status = regularized ? 1 : 0;
            ok = 1;
            break;
        }
        regularized = 1;
        for(j=0;j<NUMDIMS;j++) {NV_T[j][j] += conditioning_term_to_add;}
        conditioning_term_to_add *= 1.2;
        if(!(conditioning_term_to_add > 0) || !isfinite(conditioning_term_to_add)) {conditioning_term_to_add = MIN_REAL_NUMBER;}
    }
    /* Un-normalize: stored = Tinv / normalizer = inverse of the raw moment
     * matrix. matrix_invert_ndims returns the FULL inverse, so all 9 entries
     * are populated (no upper-triangular-only state). */
    for(j=0;j<3;j++) {for(k=0;k<3;k++) {Tinv_out[j][k] = Tinv[j][k] * inv_norm;}}
    if(!ok)
    {
        result.status = 2; result.cn_final = result.cn_expansion;
        PRINT_WARNING("AGS NV_T inversion could not be conditioned after %d diagonal-load steps "
                      "(raw trace=%g); storing best-effort regularized inverse. The CBE/DMGrad "
                      "faces for this particle are degraded -- check its neighbor count.",
                      MAXIT, trace_initial);
    }
    return result;
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
