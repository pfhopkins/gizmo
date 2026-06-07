/* gpu_error_check.h — single-source helper for post-kernel GPU error check.
 *
 * Replaces the inline #if defined(__CUDACC__) / __HIPCC__ block that was
 * previously copy-pasted after every Kokkos::parallel_for + fence in the
 * GPU TUs.  Usage:
 *
 *     Kokkos::parallel_for("mykernel", N, KOKKOS_LAMBDA(int j) { ... });
 *     Kokkos::fence();
 *     gizmo_gpu_check_last_error("mykernel", N);
 *
 * For batched dispatches that want to log the active batch range, pass the
 * batch_start as the third argument:
 *
 *     gizmo_gpu_check_last_error("cooling", batch_n, batch_start);
 *
 * Pure host-side helper — cudaGetLastError / hipGetLastError are host APIs.
 * No-op on pure-CPU builds.
 */
#ifndef GIZMO_GPU_ERROR_CHECK_H
#define GIZMO_GPU_ERROR_CHECK_H

#include <stdio.h>

#ifdef GIZMO_GPU_COMPILER

#include "gpu_device_error_sentinel.h"

/* Forward-declared rather than pulling in proto.h (same approach gpu_all_mirror.h
   uses for gizmo_gpu_sync_all). Host-side controlled-stop request: no MPI / alloc
   / Kokkos, safe to call from a dispatcher's host context after the fence. */
extern void gizmo_request_controlled_stop(int code, const char *reason,
                                          const char *file, int line, const char *func);

/* Post-kernel error check. Consumes any device-side endrun (recorded in the
   per-TU sentinel before the device trap) AND any CUDA/HIP runtime error, then
   routes a single graceful controlled-stop request -- the next all-rank poll
   drains it to a clean finalize (no MPI_Abort). NOTE: this only runs if host
   control returns after Kokkos::fence() following a trapped kernel; that
   behavior is the Vista-CUDA runtime validation gate. */
static inline void gizmo_gpu_check_last_error(const char *tag, int N, int batch_start = -1)
{
    int dev_code = 0, dev_line = 0;
    int have_sentinel = gizmo_gpu_consume_device_error(&dev_code, &dev_line);
    int have_rt_err = 0;
    const char *rt_msg = "";
#if defined(__CUDACC__)
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess) { have_rt_err = 1; rt_msg = cudaGetErrorString(err); }
#elif defined(__HIPCC__)
    hipError_t err = hipGetLastError();
    if(err != hipSuccess) { have_rt_err = 1; rt_msg = hipGetErrorString(err); }
#endif
    if(have_sentinel || have_rt_err) {
        char reason[512];
        snprintf(reason, sizeof(reason),
                 "GPU kernel '%s' device error (N=%d, batch_start=%d): runtime='%s' [device endrun code=%d line=%d]",
                 tag, N, batch_start, rt_msg, dev_code, dev_line);
        fflush(stdout); printf("[GPU] %s\n", reason); fflush(stdout);
        int route_code = have_sentinel ? dev_code : 90009001;  /* generic GPU runtime error if no sentinel */
        gizmo_request_controlled_stop(route_code, reason, __FILE__, __LINE__, "gizmo_gpu_check_last_error");
    }
}

#else  /* !GIZMO_GPU_COMPILER */

static inline void gizmo_gpu_check_last_error(const char *tag, int N, int batch_start = -1)
{
    (void)tag; (void)N; (void)batch_start;
}

#endif /* GIZMO_GPU_COMPILER */

#endif /* GIZMO_GPU_ERROR_CHECK_H */
