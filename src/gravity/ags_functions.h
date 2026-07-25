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

#endif /* AGS_KERNELRADIUS_CALCULATION_IS_ACTIVE */


/* Kokkos atomics inside a device lambda. */
#define AGS_ATOMIC_ADD(ptr, val)   Kokkos::atomic_add((ptr), (val))
#define AGS_ATOMIC_STORE(ptr, val) Kokkos::atomic_store((ptr), (val))

#endif /* AGS_FUNCTIONS_H */
