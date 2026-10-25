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


#include <utility>
#include <type_traits>
#include "gpu_error_check.h"

template<typename F>
inline void gizmo_gpu_kernel_launch(const char *tag, int N, F&& f, int batch_start = -1)
{
    Kokkos::parallel_for(tag, N, std::forward<F>(f));
    Kokkos::fence();
    gizmo_gpu_check_last_error(tag, N, batch_start);
}


/* Sibling of gizmo_gpu_kernel_launch for a kernel that also counts something: the functor takes
 * (i, int &count) and increments count; the total over all items is returned. */
template<typename F>
inline int gizmo_gpu_kernel_launch_count(const char *tag, int N, F&& f, int batch_start = -1)
{
    int total = 0;
    Kokkos::parallel_reduce(tag, N, std::forward<F>(f), total);
    Kokkos::fence();
    gizmo_gpu_check_last_error(tag, N, batch_start);
    return total;
}


/* Team-policy sibling of gizmo_gpu_kernel_launch, for kernels whose per-item
 * work is itself parallel (a neighbor-list row, a node's children, a frequency
 * bin sweep). `league` teams of `team_size` lanes each; the functor receives the
 * team member, takes its item index from member.league_rank(), and divides the
 * item's work with TeamThreadRange.
 *
 * vector_length is fixed at 1, which is a portability requirement rather than a
 * preference: Kokkos caps vector_length at the backend warp size -- 32 on CUDA,
 * 64 on HIP -- and SILENTLY clamps an over-request, so a width carried on the
 * vector dimension cannot express a 64-lane division on NVIDIA. Carrying it on
 * the team dimension expresses both widths on both backends, and makes the
 * nested reduction single-level. It also keeps the item's memory stream
 * coalesced: with vector_length 1 the team rank is the fastest-varying thread
 * index, so lane t of a team reads element t of the item's contiguous run.
 *
 * gizmo_gpu_team_size_max reports the largest LAUNCHABLE team size for a
 * functor. It is an occupancy/register-derived legality bound and nothing more
 * -- in particular it does NOT shrink when the functor's reduction value grows,
 * so it must not be used as a memory or spill budget.
 */
template<typename F>
inline void gizmo_gpu_team_kernel_launch(const char *tag, int league, int team_size, F&& f)
{
    Kokkos::TeamPolicy<> policy(league, team_size, 1);
    Kokkos::parallel_for(tag, policy, std::forward<F>(f));
    Kokkos::fence();
    gizmo_gpu_check_last_error(tag, league, -1);
}

template<typename F>
inline int gizmo_gpu_team_size_max(const F& f)
{
    Kokkos::TeamPolicy<> probe(1, 1, 1);
    return probe.team_size_max(f, Kokkos::ParallelForTag());
}

/* The same pair for a kernel whose teams ask for level-0 scratch.  The bound is asked of a probe
 * carrying the same scratch request as the launch, because that request is part of what limits the
 * team: a probe without it reports a team the launch cannot have.  (A policy constructed at an
 * illegal team size throws before it can be asked anything, hence the probe.) */
template<typename F>
inline void gizmo_gpu_team_kernel_launch(const char *tag, int league, int team_size, size_t scratch_bytes, F&& f)
{
    Kokkos::TeamPolicy<> policy(league, team_size, 1);
    policy.set_scratch_size(0, Kokkos::PerTeam(scratch_bytes));
    Kokkos::parallel_for(tag, policy, std::forward<F>(f));
    Kokkos::fence();
    gizmo_gpu_check_last_error(tag, league, -1);
}

template<typename F>
inline int gizmo_gpu_team_size_max(const F& f, size_t scratch_bytes)
{
    Kokkos::TeamPolicy<> probe(1, 1, 1);
    probe.set_scratch_size(0, Kokkos::PerTeam(scratch_bytes));
    return probe.team_size_max(f, Kokkos::ParallelForTag());
}

/* How many work items the default execution space can have in flight at once:
 * the hardware's lane count, as Kokkos reports it for the backend in use.
 *
 * It answers one question -- are there more lanes than there are items to give
 * them -- which is what decides whether a loop should divide ONE item's work
 * among several lanes instead of giving each lane its own. Keyed on the machine
 * rather than on a tuned item count, so the same code makes the right choice on
 * a laptop's thread pool and on a GPU three orders of magnitude wider. */
inline int gizmo_gpu_lane_count()
{
    const int n = (int) Kokkos::DefaultExecutionSpace().concurrency();
    return (n > 0) ? n : 1;
}

/* True when the default execution space runs on the host, where a kernel's
 * items are already spread across the available threads and there are no SIMT
 * lanes left to divide an item's work among. Team-per-item policies are a
 * device idiom; on a host backend the flat launch above is the right shape. */
inline constexpr bool gizmo_gpu_default_space_is_host()
{
    return std::is_same<Kokkos::DefaultExecutionSpace::memory_space,
                        Kokkos::HostSpace>::value;
}


/* Work that runs on its own execution-space instance while the host carries on, and the means to learn
 * when it has finished.
 *
 * gizmo_gpu_kernel_launch_async enqueues a kernel on `space` and returns at once: no fence.  Whether the
 * launch itself was accepted is checked at once; a refused launch means the device can no longer be
 * trusted, so the run stops through the reviewed hard exit (no further launches, copies or fences).
 *
 * GizmoGpuEvent marks a point in an instance's queue.  record() places it after the work enqueued so far;
 * query() reports NOT_READY, COMPLETE, or ERROR -- any backend status other than those two, which is how a
 * fault in the device work behind it surfaces (it never turns into an endless wait).  An event is
 * recorded again only after its previous completion was seen.  Created once and kept; destroyed before
 * Kokkos is finalized.
 *
 * This is the one place GIZMO calls the CUDA / HIP runtime for stream completion: Kokkos has no portable
 * query of whether an instance's work has finished, only a blocking fence.  It covers completion only --
 * allocation, copies and kernels still go through Kokkos.  Host backends run every launch to completion
 * before it returns, so there an event is always COMPLETE. */
[[noreturn]] void gizmo_fatal_hard_exit_reviewed(int code, const char *reason, const char *file, int line, const char *func);

template<typename F>
inline void gizmo_gpu_kernel_launch_async(const Kokkos::DefaultExecutionSpace &space, const char *tag, int N, F&& f)
{
    Kokkos::parallel_for(tag, Kokkos::RangePolicy<>(space, 0, N), std::forward<F>(f));
#if defined(__CUDACC__)
    if(cudaPeekAtLastError() != cudaSuccess) {gizmo_fatal_hard_exit_reviewed(90009002, "an asynchronous kernel launch was refused", __FILE__, __LINE__, tag);}
#elif defined(__HIPCC__)
    if(hipPeekAtLastError() != hipSuccess) {gizmo_fatal_hard_exit_reviewed(90009002, "an asynchronous kernel launch was refused", __FILE__, __LINE__, tag);}
#endif
}

enum GizmoGpuEventStatus {GIZMO_GPU_EVENT_NOT_READY = 0, GIZMO_GPU_EVENT_COMPLETE = 1, GIZMO_GPU_EVENT_ERROR = 2};

struct GizmoGpuEvent {
#if defined(KOKKOS_ENABLE_CUDA)
    cudaEvent_t event = nullptr;
    int create() {return (cudaEventCreateWithFlags(&event, cudaEventDisableTiming) == cudaSuccess) ? 0 : 1;}
    void record(const Kokkos::DefaultExecutionSpace &space) {cudaEventRecord(event, space.cuda_stream());}
    int query() const {
        const cudaError_t e = cudaEventQuery(event);
        return (e == cudaSuccess) ? GIZMO_GPU_EVENT_COMPLETE : (e == cudaErrorNotReady) ? GIZMO_GPU_EVENT_NOT_READY : GIZMO_GPU_EVENT_ERROR;
    }
    void destroy() {if(event) {cudaEventDestroy(event); event = nullptr;}}
#elif defined(KOKKOS_ENABLE_HIP)
    hipEvent_t event = nullptr;
    int create() {return (hipEventCreateWithFlags(&event, hipEventDisableTiming) == hipSuccess) ? 0 : 1;}
    void record(const Kokkos::DefaultExecutionSpace &space) {(void) hipEventRecord(event, space.hip_stream());}
    int query() const {
        const hipError_t e = hipEventQuery(event);
        return (e == hipSuccess) ? GIZMO_GPU_EVENT_COMPLETE : (e == hipErrorNotReady) ? GIZMO_GPU_EVENT_NOT_READY : GIZMO_GPU_EVENT_ERROR;
    }
    void destroy() {if(event) {(void) hipEventDestroy(event); event = nullptr;}}
#else
    int create() {return 0;}
    void record(const Kokkos::DefaultExecutionSpace &) {}
    int query() const {return GIZMO_GPU_EVENT_COMPLETE;}
    void destroy() {}
#endif
};


#endif /* GIZMO_GPU_DISPATCH_TEMPLATES_H */
