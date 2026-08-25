/* ags_functions.h — GPU-callable AGS helpers (particle size/volume) and
 * AGS_ATOMIC_{ADD,STORE} dispatch macros for j-side writes in the
 * AGSForce GPU port.
 *
 * Get_Particle_Size_AGS / get_particle_volume_ags used to be host-only
 * with implicit reliance on the global `P`. They are now KOKKOS_INLINE_FUNCTION
 * and take P explicitly so both the CPU tree-walk and the GPU parallel_for
 * lambda can call them uniformly. Inside a GPU kernel the caller passes the
 * SharedSpace P mirror; inside CPU code the caller passes the global P.
 *
 * AGS_ATOMIC_{ADD,STORE} selects backend atomics at compile time. On the GPU
 * path (Kokkos+GPU) we use Kokkos::atomic_{add,store}. On the CPU
 * tree-walk path there is no macro that captures `#pragma omp atomic` (pragmas
 * can't live inside macros), so we provide a plain assignment and the CPU
 * caller is expected to wrap the statement with `#pragma omp atomic` directly.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#ifndef AGS_FUNCTIONS_H
#define AGS_FUNCTIONS_H

#include "../declarations/allvars.h"

/* KOKKOS_INLINE_FUNCTION falls back to `inline` outside GPU TUs. */
#ifndef KOKKOS_INLINE_FUNCTION
#define KOKKOS_INLINE_FUNCTION inline
#endif


#ifdef AGS_KERNELRADIUS_CALCULATION_IS_ACTIVE

/* Effective particle size based on AGS_KernelRadius and NumNgb. */
KOKKOS_INLINE_FUNCTION
double Get_Particle_Size_AGS_P(int i, const struct particle_data *P_arr)
{
#if (NUMDIMS == 1)
    return 2.00000 * P_arr[i].AGS_KernelRadius / P_arr[i].NumNgb;
#endif
#if (NUMDIMS == 2)
    return 1.77245 * P_arr[i].AGS_KernelRadius / P_arr[i].NumNgb;
#endif
#if (NUMDIMS == 3)
    return 1.61199 * P_arr[i].AGS_KernelRadius / P_arr[i].NumNgb;
#endif
}


/* Particle volume = L^NUMDIMS where L = Get_Particle_Size_AGS. */
KOKKOS_INLINE_FUNCTION
double get_particle_volume_ags_P(int j, const struct particle_data *P_arr)
{
    double L_j = Get_Particle_Size_AGS_P(j, P_arr);
#if (NUMDIMS==1)
    return L_j;
#elif (NUMDIMS==2)
    return L_j*L_j;
#else
    return L_j*L_j*L_j;
#endif
}

/* Whether ags_density must solve for this particle's KernelRadius.
 * Also sets AGS_KernelRadius and AGS_zeta for particles that do not iterate --
 * a safety default, idempotent, and touching only this particle's own fields. */
KOKKOS_INLINE_FUNCTION
int ags_density_isactive_P(int i, struct particle_data *P_arr)
{
    int default_to_return = 0; // default to not being active - needs to be pro-actively 'activated' by some physics
#ifdef ADAPTIVE_GRAVSOFT_FORALL
    default_to_return = 1;
    if(!((1 << P_arr[i].Type) & (ADAPTIVE_GRAVSOFT_FORALL))) /* particle is NOT one of the designated 'adaptive' types */
    {
        P_arr[i].AGS_KernelRadius = All.ForceSoftening[P_arr[i].Type];
        P_arr[i].AGS_zeta = 0;
        default_to_return = 0;
    } else {default_to_return = 1;} /* particle is AGS-active */
#endif
#if defined(ADAPTIVE_GRAVSOFT_FORGAS) || (ADAPTIVE_GRAVSOFT_FORALL & 1)
    if(P_arr[i].Type==0)
    {
        P_arr[i].AGS_KernelRadius = P_arr[i].KernelRadius; // gas sees gas, these are identical
        default_to_return = 0; // don't actually need to do the loop //
    }
#endif
#ifdef DM_SIDM
    if((1 << P_arr[i].Type) & (DM_SIDM)) {default_to_return = 1;}
#endif
#if defined(CBE_INTEGRATOR)
    if(CBE_INTEGRATOR_DOES_TYPE(P_arr[i].Type)) {default_to_return = 1;}
#elif defined(DM_FUZZY)
    if(P_arr[i].Type == 1) {default_to_return = 1;}
#endif
    if(P_arr[i].TimeBin < 0) {default_to_return = 0;} /* check our 'marker' for particles which have finished iterating to an KernelRadius solution (if they have, dont do them again) */
    return default_to_return;
}


/* maximum allowed softening */
KOKKOS_INLINE_FUNCTION
double ags_return_maxsoft_P(int i, const struct particle_data *P_arr)
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
    if(P_arr[i].Type == 5) {maxsoft = DMIN(maxsoft, All.SinkMaxAccretionRadius / All.cf_atime);}
#endif
    return maxsoft;
}


/* minimum allowed softening */
KOKKOS_INLINE_FUNCTION
double ags_return_minsoft_P(int i, const struct particle_data *P_arr)
{
    double minsoft = All.ForceSoftening[P_arr[i].Type]; // this is the user-specified minimum
#if !defined(ADAPTIVE_GRAVSOFT_FORALL)
    minsoft = DMIN(All.MinKernelRadius, minsoft);
#endif
    return minsoft;
}

#endif /* AGS_KERNELRADIUS_CALCULATION_IS_ACTIVE */


/* Kokkos atomics inside a device lambda. */
#define AGS_ATOMIC_ADD(ptr, val)   Kokkos::atomic_add((ptr), (val))
#define AGS_ATOMIC_STORE(ptr, val) Kokkos::atomic_store((ptr), (val))

#endif /* AGS_FUNCTIONS_H */
