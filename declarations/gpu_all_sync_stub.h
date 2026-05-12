/* gpu_all_sync_stub.h — minimal sync-function stub for host-only TUs.
 *
 * Provides GPU_ALL_SYNC_FUNC / GPU_ALL_SYNC_FUNC_STUB without the
 * `#define All All_dev` macro redirect that gpu_all_mirror.h carries.
 *
 * Use this in a TU when BOTH of the following hold:
 *   (a) The TU needs to register a per-TU sync function so
 *       cooling.cc::gizmo_gpu_sync_all() resolves its
 *       `extern void gizmo_gpu_sync_all_<name>(...)` call, AND
 *   (b) The TU runs no device kernels reading `All.*` — typically a
 *       Spec `_loop.cc` whose only Spec hook touching All is host-side
 *       `populate_call_scalars`.
 *
 * Why a separate header from gpu_all_mirror.h: that header includes a
 * file-scope `#define All All_dev` redirect. In a host-only TU the
 * matching per-TU `__managed__ All_dev` is zero-initialised and never
 * synced (the registered function is a STUB), so any bare `All.foo`
 * read in such a TU silently reads zero. See commit `781ecfb2`
 * (NLR host All accessor) for the bug this caused.
 *
 * Bare `All.foo` in a TU that includes THIS header reads the real host
 * extern. Spec authors are still encouraged to route through
 * `nlr_host_all_ptr()` (mesh/neighbor_loop_runner.h) so the intent is
 * durable even if someone later re-adds the gpu_all_mirror.h include.
 *
 * Idempotent: if gpu_all_mirror.h was already included (real GPU TU),
 * the GPU_ALL_SYNC_FUNC / _STUB macros are already defined and this
 * header is a no-op.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */
#ifndef GPU_ALL_SYNC_STUB_H
#define GPU_ALL_SYNC_STUB_H

#ifndef GPU_ALL_SYNC_FUNC

#include "global_data_all_struct.h"

#define GPU_ALL_SYNC_FUNC(name) \
    void gizmo_gpu_sync_all_##name(struct global_data_all_processes *host_all) { (void)host_all; }

#define GPU_ALL_SYNC_FUNC_STUB(name) GPU_ALL_SYNC_FUNC(name)

#endif /* GPU_ALL_SYNC_FUNC */

#endif /* GPU_ALL_SYNC_STUB_H */
