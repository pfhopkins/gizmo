/* gpu_all_mirror.h — per-TU device-visible mirror of `All`, with central
 * auto-registered sync.
 *
 * Each GPU TU that includes this header gets its own private
 * `static __managed__ AllDeviceMirror` plus an automatic, three-layer
 * registration of its address with a central registry owned by
 * cooling/cooling.cc:
 *
 *   1. __attribute__((constructor)) function — fires at C++ runtime
 *      startup (before main), broad safety net registering every
 *      TU's mirror.
 *   2. Dispatch-boundary forced registration — every call to
 *      GIZMO_GPU_ENSURE_ALL_FRESH() runs gizmo_all_device_mirror_register_this_tu()
 *      first, so any TU whose constructor was elided still registers
 *      on its first GPU dispatch. Correctness belt.
 *   3. Registry de-duplication — double registration is harmless.
 *
 * During the device compilation pass (__CUDA_ARCH__ /
 * __HIP_DEVICE_COMPILE__), `#define All AllDeviceMirror` redirects
 * device-side `All.*` reads to the TU's local mirror, which the central
 * `gizmo_gpu_sync_all()` copies from host `All` before every GPU
 * dispatch. Host-pass reads in the same TU see the regular host extern
 * `All` (no host-wrapper macro trap).
 *
 * Why per-TU mirrors rather than one shared symbol:
 * Cross-TU `extern __managed__` and `extern __device__` both require
 * CUDA Relocatable Device Code (`-rdc=true` / `-dc`), which this
 * Makefile does not enable. Without RDC, every cross-TU device-symbol
 * declaration becomes a per-TU definition with its own backing
 * allocation (`cudaGetSymbolAddress` returned err=cudaErrorInvalidSymbol
 * in non-defining TUs on Vista's nvcc/Kokkos toolchain — measured
 * 2026-05-17). Per-TU mirrors with auto-registered central sync gives
 * the same correctness and the same author-facing API as a single
 * shared mirror would, without the RDC requirement. Cost: ~30 × 33KB =
 * ~1MB managed memory per rank, all identical post-sync. Negligible.
 *
 * On host-only Kokkos backends (GIZMO_GPU_COMPILER undef): `All` is the
 * regular host extern; no mirror exists.
 *
 * Conventionally included before allvars.h in GPU TUs.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */
#ifndef GPU_ALL_MIRROR_H
#define GPU_ALL_MIRROR_H

#include "global_data_all_struct.h"

/* Host extern, declared in both compilation passes so the host pass of a
   GPU TU sees the real host extern even when the device-pass macro fires
   below. */
extern struct global_data_all_processes All;

#if defined(GIZMO_GPU_COMPILER)

#ifdef __cplusplus
extern "C" {
#endif
/* Central registry entry point. Defined in cooling/cooling.cc using a
 * Meyer's-singleton vector so static-init-order across TUs is safe.
 * De-duplicates on call so multiple registrations of the same address
 * are harmless. */
void gizmo_register_all_device_mirror(struct global_data_all_processes *mirror);
#ifdef __cplusplus
}
#endif

/* Per-TU private mirror in managed memory. `static __managed__` gives
   each TU its own internal-linkage symbol. */
static __managed__ struct global_data_all_processes AllDeviceMirror;

/* Per-TU forced registration helper. Static inline + function-local
 * static bool means each TU has its own copy of both function and flag;
 * the flag fast-paths the no-op case after first call. Called by both
 * the auto-registration constructor below and by GIZMO_GPU_ENSURE_ALL_FRESH()
 * for defense-in-depth against constructor elision. */
static inline void gizmo_all_device_mirror_register_this_tu(void) {
    static bool registered = false;
    if(!registered) {
        /* HIP: &AllDeviceMirror (a static __managed__ global) is NULL until the
         * runtime's managed-var init constructor has run, which may fire AFTER
         * this __attribute__((constructor)). Only mark registered once a valid
         * (non-NULL) address is captured, so the post-init call from
         * GIZMO_GPU_ENSURE_ALL_FRESH() registers the real address. On CUDA the
         * address is a valid link-time constant, so this registers on first call. */
        struct global_data_all_processes *p = &AllDeviceMirror;
        if(p) {
            gizmo_register_all_device_mirror(p);
            registered = true;
        }
    }
}

/* Broad safety net. Fires before main() in every GPU TU that
 * includes this header. Per-TU anonymous-namespace + static keeps the
 * function file-local. __attribute__((constructor)) is harder to elide
 * than an anonymous-namespace static object's constructor (which was
 * observed silently dropped in ~28 of ~30 TUs on Vista's nvcc/Kokkos
 * toolchain). */
namespace {
    __attribute__((constructor))
    static void _gizmo_all_device_mirror_autoreg(void) {
        gizmo_all_device_mirror_register_this_tu();
    }
}

/* Device-compilation-pass-only redirect. `__CUDA_ARCH__` is defined by
   nvcc only during the device-side compile pass; `__HIP_DEVICE_COMPILE__`
   is the HIP equivalent. On the host pass of the same GPU TU, the
   redirect is NOT defined, so `All.foo` resolves to the host extern
   declared above. */
#if defined(__CUDA_ARCH__) || __HIP_DEVICE_COMPILE__
#define All AllDeviceMirror
#endif

/* Declared with normal C++ linkage in core/proto.h; do NOT add extern
   "C" here — a second declaration with different linkage would conflict. */
void gizmo_gpu_sync_all(void);

/* Central no-arg freshness guard. Layer 2: forces registration of this
 * TU's mirror (idempotent after first call), then runs the central
 * sync. Cheap and idempotent; safe to call at the top of any GPU
 * dispatch function. */
#define GIZMO_GPU_ENSURE_ALL_FRESH() do { \
    gizmo_all_device_mirror_register_this_tu(); \
    gizmo_gpu_sync_all(); \
} while(0)

#else  /* GIZMO_GPU_COMPILER undef — pure host build */

#define GIZMO_GPU_ENSURE_ALL_FRESH() ((void)0)

#endif

#endif /* GPU_ALL_MIRROR_H */
