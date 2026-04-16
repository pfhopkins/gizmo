/* gpu_all_mirror.h — single-include boilerplate for GPU TUs that need All.
 *
 * On CUDA: defines a per-TU static __managed__ POINTER (All_ptr) that
 * points to a SINGLE shared UVM allocation.  #define All (*All_ptr) so all
 * existing All.field syntax works on both host and device.  The pointer is
 * set once at startup by gizmo_gpu_alloc_all() via the GPU_ALL_INIT_FUNC
 * setter defined in each TU.
 *
 * On OpenMP (Kokkos host backend): no-op.  All is the regular extern global
 * from allvars.h.
 *
 * MUST be included BEFORE allvars.h so the #define All suppresses the extern
 * declaration in allvars.h (which has #ifndef All guard).
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */
#ifndef GPU_ALL_MIRROR_H
#define GPU_ALL_MIRROR_H

#if defined(OPENMP_GPU_OFFLOAD) && defined(GIZMO_GPU_COMPILER)

#include "global_data_all_struct.h"

/* Per-TU managed pointer to the shared UVM allocation of All.
   __managed__ makes this 8-byte pointer accessible from both host and device
   code within this translation unit.  All TU pointers are set to the SAME
   shared allocation by gizmo_gpu_alloc_all() at startup.  There is only ONE
   copy of the struct data in UVM — not one per TU. */
static __managed__ struct global_data_all_processes *All_ptr = nullptr;
#define All (*All_ptr)

/* Macro to generate the per-TU setter that gizmo_gpu_alloc_all() calls.
   Usage: place GPU_ALL_INIT_FUNC(cooling) at file scope in each GPU TU. */
#define GPU_ALL_INIT_FUNC(name) \
    void gizmo_gpu_init_all_##name(struct global_data_all_processes *p) { All_ptr = p; }

#elif defined(OPENMP_GPU_OFFLOAD)

/* Kokkos OpenMP backend — All is the regular extern from allvars.h.
   Still need the init function stub so cooling.cc can call it. */
#define GPU_ALL_INIT_FUNC(name) \
    void gizmo_gpu_init_all_##name(struct global_data_all_processes *p) { (void)p; }

#else

/* No GPU offload at all — no-op. */
#define GPU_ALL_INIT_FUNC(name)

#endif

#endif /* GPU_ALL_MIRROR_H */
