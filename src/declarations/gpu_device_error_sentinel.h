/* gpu_device_error_sentinel.h -- device->host bridge for a device-side endrun.
 *
 * A device endrun cannot touch host MPI or the controlled-stop flag. Instead it
 * records the numeric code + source line into a per-TU managed sentinel and then
 * traps; the host post-kernel error check (gpu_error_check.h) consumes the
 * sentinel after the fence and routes to gizmo_request_controlled_stop, which
 * drains gracefully at the next all-rank poll. Device code stays MPI-free; the
 * sentinel is the only device write.
 *
 * Storage is per-TU `static __managed__`, the same idiom and rationale as
 * gpu_all_mirror.h: this Makefile builds without -rdc, so a cross-TU shared
 * device symbol is not available. Per-TU is correct here because a kernel and
 * its own post-fence error check always live in the SAME translation unit, so
 * the check consumes the same TU's sentinel that the kernel wrote. This is NOT
 * a global cross-TU device error bus.
 *
 * No Kokkos dependency: __managed__ and device atomics are GPU-compiler
 * builtins, so this header is safe to include from the universally-included
 * macros.h.
 */
#ifndef GPU_DEVICE_ERROR_SENTINEL_H
#define GPU_DEVICE_ERROR_SENTINEL_H

#if defined(GIZMO_GPU_COMPILER)

struct gizmo_gpu_err_sentinel_t { int set; int code; int line; };
static __managed__ struct gizmo_gpu_err_sentinel_t gizmo_gpu_err_sentinel = {0, 0, 0};

/* HD (not __device__-only): host-side management functions that call endrun still
 * get compiled in the device pass, and clang rejects a __device__-only callee from
 * a __host__ caller (nvcc prunes it silently). The sentinel write is device-only;
 * on host this is a no-op (the host endrun path is the soft-stop request, which
 * does not route through here). first-set-wins; we trap immediately after, and the
 * host-side request is itself first-set-wins, so a benign race is fine. */
static __host__ __device__ inline void gizmo_gpu_device_record_error(int code, int line) {
#if defined(__CUDA_ARCH__) || __HIP_DEVICE_COMPILE__
    if(atomicCAS(&gizmo_gpu_err_sentinel.set, 0, 1) == 0) {
        gizmo_gpu_err_sentinel.code = code;
        gizmo_gpu_err_sentinel.line = line;
    }
#else
    (void)code; (void)line;
#endif
}

/* Host pass: consume (read + clear) this TU's sentinel. Returns 1 if a device
 * endrun was recorded since the last consume, else 0. Clearing avoids a later
 * check on the same TU re-reporting a stale device error. */
static inline int gizmo_gpu_consume_device_error(int *code, int *line) {
    if(gizmo_gpu_err_sentinel.set) {
        if(code) { *code = gizmo_gpu_err_sentinel.code; }
        if(line) { *line = gizmo_gpu_err_sentinel.line; }
        gizmo_gpu_err_sentinel.set = 0;
        return 1;
    }
    return 0;
}

#else  /* host-only build (e.g. Mac OpenMP): no device sentinel */

static inline int gizmo_gpu_consume_device_error(int *code, int *line) {
    (void)code; (void)line; return 0;
}

#endif /* GIZMO_GPU_COMPILER */

#endif /* GPU_DEVICE_ERROR_SENTINEL_H */
