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

static inline void gizmo_gpu_check_last_error(const char *tag, int N, int batch_start = -1)
{
#if defined(__CUDACC__)
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess) {
        if(batch_start >= 0) {
            printf("[GPU] %s error batch [%d..%d] N=%d: %s\n",
                   tag, batch_start, batch_start + N - 1, N, cudaGetErrorString(err));
        } else {
            printf("[GPU] %s error N=%d: %s\n", tag, N, cudaGetErrorString(err));
        }
        fflush(stdout);
    }
#elif defined(__HIPCC__)
    hipError_t err = hipGetLastError();
    if(err != hipSuccess) {
        if(batch_start >= 0) {
            printf("[GPU] %s error batch [%d..%d] N=%d: %s\n",
                   tag, batch_start, batch_start + N - 1, N, hipGetErrorString(err));
        } else {
            printf("[GPU] %s error N=%d: %s\n", tag, N, hipGetErrorString(err));
        }
        fflush(stdout);
    }
#else
    (void)tag; (void)N; (void)batch_start;
#endif
}

#else  /* !GIZMO_GPU_COMPILER */

static inline void gizmo_gpu_check_last_error(const char *tag, int N, int batch_start = -1)
{
    (void)tag; (void)N; (void)batch_start;
}

#endif /* GIZMO_GPU_COMPILER */

#endif /* GIZMO_GPU_ERROR_CHECK_H */
