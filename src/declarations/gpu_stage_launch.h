/* gpu_stage_launch.h -- one Kokkos kernel for a per-element stage, on the device or on the host execution space.
 * Include after <Kokkos_Core.hpp> and core/proto.h. */
#ifndef GIZMO_GPU_STAGE_LAUNCH_H
#define GIZMO_GPU_STAGE_LAUNCH_H
#include <type_traits>
#include "gpu_dispatch_templates.h"

/* One kernel for a per-element stage, run on the device or on the host execution space (Stella stages; OUTPUT_LOOP_ROUTES
 * names the space).  The device takes it when the particle storage is device-visible in place (shared space, as the drift
 * requires), the device is not the host, and the per-particle-loop offload count admits n; otherwise the same functor runs
 * on Kokkos::DefaultHostExecutionSpace (Serial in the GPU builds: the order of the former plain loop, so a host reduction
 * sums in the same order).  Kernels read no global: everything is captured by value, and a kernel that reads All goes
 * after GIZMO_GPU_ENSURE_ALL_FRESH() in its translation unit. */
inline int gizmo_stage_on_device(long long n)
{
    if(!Kokkos::SpaceAccessibility<Kokkos::DefaultExecutionSpace, Kokkos::SharedSpace>::accessible) {return 0;}
    if(std::is_same<Kokkos::DefaultExecutionSpace, Kokkos::DefaultHostExecutionSpace>::value) {return 0;}
    return gizmo_gpu_offload_takes(GPU_OFFLOAD_PARTICLE_LOOP, n);
}
template<typename F>
inline void gizmo_stage_for(const char *tag, int on_device, int N, F&& f)
{
    if(N <= 0) {return;}
    if(on_device) {Kokkos::parallel_for(tag, Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, N), f); Kokkos::fence(); gizmo_gpu_check_last_error(tag, N, -1);}
    else {Kokkos::parallel_for(tag, Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, N), f); Kokkos::DefaultHostExecutionSpace().fence();}
}
/* the same with a sum: the functor takes (i, T &partial) */
template<typename T, typename F>
inline T gizmo_stage_sum(const char *tag, int on_device, int N, F&& f)
{
    T total = 0;
    if(N <= 0) {return total;}
    if(on_device) {Kokkos::parallel_reduce(tag, Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, N), f, total); Kokkos::fence(); gizmo_gpu_check_last_error(tag, N, -1);}
    else {Kokkos::parallel_reduce(tag, Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, N), f, total);}
    return total;
}

/* an int buffer of n entries for a stage's index list: shared (device-visible) memory when *on_device, else host memory; when the
 * shared allocation fails *on_device is cleared and the host space takes the stage.  Free with gizmo_stage_index_free. */
inline int *gizmo_stage_index_alloc(int *on_device, long long n, const char *label)
{
    const size_t bytes = (size_t) (n > 0 ? n : 1) * sizeof(int);
    int *p = *on_device ? (int *) gizmo_gpu_alloc_shared(bytes, label) : NULL;
    if(!p) {*on_device = 0; p = (int *) malloc(bytes);}
    return p;
}
inline void gizmo_stage_index_free(int on_device, int *p) {if(!p) {return;} if(on_device) {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(p);} else {free(p);}}

/* the indices i < N with pred(i), in increasing order, into out (room for N; device-visible when on_device): a Kokkos scan, so a
 * host loop over out[] afterwards visits them in the order of a plain loop over i.  Returns their number. */
template<typename F>
inline int gizmo_stage_select(const char *tag, int on_device, int N, const F &pred, int *out)
{
    int total = 0;
    if(N <= 0) {return total;}
    const F p = pred;
    auto scan = KOKKOS_LAMBDA(int i, int &offset, const bool final) {if(p(i)) {if(final) {out[offset] = i;} offset++;}};
    if(on_device) {Kokkos::parallel_scan(tag, Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, N), scan, total); Kokkos::fence(); gizmo_gpu_check_last_error(tag, N, -1);}
    else {Kokkos::parallel_scan(tag, Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, N), scan, total);}
    return total;
}
/* the same reduction with a Kokkos reducer object (e.g. a compensated sum): the functor takes (i, value_type &partial) */
template<typename R, typename F>
inline void gizmo_stage_reduce(const char *tag, int on_device, int N, const F &f, const R &reducer)
{
    if(N <= 0) {return;}
    if(on_device) {Kokkos::parallel_reduce(tag, Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, N), f, reducer); Kokkos::fence(); gizmo_gpu_check_last_error(tag, N, -1);}
    else {Kokkos::parallel_reduce(tag, Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, N), f, reducer);}
}

#endif /* GIZMO_GPU_STAGE_LAUNCH_H */
