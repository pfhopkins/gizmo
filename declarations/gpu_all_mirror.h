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

#if defined(GIZMO_GPU_COMPILER)

/* Per-TU managed copy of the global All struct.  __managed__ makes it
   accessible from both host and device code within this translation unit.
   The #define redirects all All.field accesses to the managed copy. */
static __managed__ struct global_data_all_processes All_dev;
#define All All_dev

/* Helper: return a pointer to the host-side All (bypassing the #define All All_dev
   redirect in this TU).  Uses push_macro/pop_macro to locally undo the redirect so
   the extern declaration resolves to the real global symbol, then restores it. */
static inline struct global_data_all_processes *gizmo_gpu_host_all_ptr(void) {
#pragma push_macro("All")
#undef All
    extern struct global_data_all_processes All;
    struct global_data_all_processes *p = &All;
#pragma pop_macro("All")
    return p;
}

/* Macro to generate the per-TU sync function that copies host All -> All_dev.
   gizmo_gpu_sync_all() in cooling.cc calls each TU's sync function.
   Usage: place GPU_ALL_SYNC_FUNC(cooling) at file scope in each GPU TU. */
#define GPU_ALL_SYNC_FUNC(name) \
    void gizmo_gpu_sync_all_##name(struct global_data_all_processes *host_all) { \
        All_dev = *host_all; \
    }

/* Same name, no body — for the `#else` (feature-disabled) branch of a GPU TU
   so cooling.cc's `extern void gizmo_gpu_sync_all_<name>(...);` call still
   resolves.  Step 5 Phase E2 (2026-04-30) — replaces ~14 hand-written
   `void gizmo_gpu_sync_all_<X>(...) { (void)p; }` boilerplate sites. */
#define GPU_ALL_SYNC_FUNC_STUB(name) \
    void gizmo_gpu_sync_all_##name(struct global_data_all_processes *host_all) { (void)host_all; }

/* Freshness guard for the top of a GPU dispatch function.  Ensures this TU's
   All_dev mirror matches host All right now (idempotent; trivial cost — a single
   struct copy).  Use this instead of assuming gizmo_gpu_sync_all() at the top of
   the timestep covers all mid-step All.* mutations; the guard makes the
   invariant local and self-enforcing.
   The extern declaration at block scope lets the guard appear above the TU's
   GPU_ALL_SYNC_FUNC(name) definition (which lives at end of file). */
#define GIZMO_GPU_ENSURE_ALL_FRESH(name) do { \
    extern void gizmo_gpu_sync_all_##name(struct global_data_all_processes *); \
    gizmo_gpu_sync_all_##name(gizmo_gpu_host_all_ptr()); \
} while(0)

#else

/* Kokkos OpenMP backend — All is the regular extern from allvars.h.
   Still need the sync function stub so cooling.cc can call it. */
#define GPU_ALL_SYNC_FUNC(name) \
    void gizmo_gpu_sync_all_##name(struct global_data_all_processes *host_all) { (void)host_all; }

/* Stub form — same as GPU_ALL_SYNC_FUNC on the host backend (no real sync
   needed since All is already host-live). */
#define GPU_ALL_SYNC_FUNC_STUB(name) GPU_ALL_SYNC_FUNC(name)

/* Freshness guard — no-op on host Kokkos backend (All is already live). */
#define GIZMO_GPU_ENSURE_ALL_FRESH(name) ((void)0)

#endif

#endif /* GPU_ALL_MIRROR_H */
