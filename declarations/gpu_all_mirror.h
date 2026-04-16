/* gpu_all_mirror.h — single-include boilerplate for GPU TUs that need All.
 *
 * On CUDA: defines a per-TU static __managed__ copy of All (All_dev) and
 * redirects All -> All_dev so all existing All.field syntax works on both
 * host and device code.  Each TU gets its own managed copy, synced from
 * the host All by gizmo_gpu_sync_all() each timestep.
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

/* Include global_data_all_struct.h FIRST — it defines GIZMO_GPU_COMPILER
   (via __CUDACC__/__HIPCC__ detection) which we need for the guard below,
   and provides the struct type needed for the __managed__ declaration. */
#include "global_data_all_struct.h"

#if defined(OPENMP_GPU_OFFLOAD) && defined(GIZMO_GPU_COMPILER)

/* Per-TU managed copy of the global All struct.  __managed__ makes it
   accessible from both host and device code within this translation unit.
   The #define redirects all All.field accesses to the managed copy. */
static __managed__ struct global_data_all_processes All_dev;
#define All All_dev

/* Macro to generate the per-TU sync function that copies host All -> All_dev.
   gizmo_gpu_sync_all() in cooling.cc calls each TU's sync function.
   Usage: place GPU_ALL_SYNC_FUNC(cooling) at file scope in each GPU TU. */
#define GPU_ALL_SYNC_FUNC(name) \
    void gizmo_gpu_sync_all_##name(struct global_data_all_processes *host_all) { \
        All_dev = *host_all; \
    }

#elif defined(OPENMP_GPU_OFFLOAD)

/* Kokkos OpenMP backend — All is the regular extern from allvars.h.
   Still need the sync function stub so cooling.cc can call it. */
#define GPU_ALL_SYNC_FUNC(name) \
    void gizmo_gpu_sync_all_##name(struct global_data_all_processes *host_all) { (void)host_all; }

#else

/* No GPU offload at all — no-op. */
#define GPU_ALL_SYNC_FUNC(name)

#endif

#endif /* GPU_ALL_MIRROR_H */
