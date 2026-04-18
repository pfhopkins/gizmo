/* gpu_dispatch_templates.h — inline helpers for GPU kernel dispatch.
 *
 * gizmo_gpu_kernel_launch wraps the `Kokkos::parallel_for + Kokkos::fence +
 * gizmo_gpu_check_last_error` trio that appears after every Kokkos::parallel_for
 * in the GPU TUs.  Condenses 3 lines per call site to 1 and provides a single
 * place to add profiling / logging hooks later.
 *
 * Usage:
 *     gizmo_gpu_kernel_launch("density_kernel", N, KOKKOS_LAMBDA(int aa) {
 *         // per-particle work
 *     });
 *
 * For batched dispatches that want to log the batch range on error:
 *     gizmo_gpu_kernel_launch("cooling_loop", batch_n, KOKKOS_LAMBDA(int j) {
 *         do_the_cooling_for_particle(j, kp, kc);
 *     }, batch_start);
 *
 * Include AFTER <Kokkos_Core.hpp>.
 */
#ifndef GIZMO_GPU_DISPATCH_TEMPLATES_H
#define GIZMO_GPU_DISPATCH_TEMPLATES_H

#ifdef OPENMP_GPU_OFFLOAD

#include <utility>
#include "gpu_error_check.h"

template<typename F>
inline void gizmo_gpu_kernel_launch(const char *tag, int N, F&& f, int batch_start = -1)
{
    Kokkos::parallel_for(tag, N, std::forward<F>(f));
    Kokkos::fence();
    gizmo_gpu_check_last_error(tag, N, batch_start);
}

#endif /* OPENMP_GPU_OFFLOAD */

#endif /* GIZMO_GPU_DISPATCH_TEMPLATES_H */
