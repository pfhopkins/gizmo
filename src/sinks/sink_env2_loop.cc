/* sinks/sink_env2_loop.cc — host-only hooks for SinkEnv2Spec.
 *
 * Mechanical-after-pattern port mirroring sinks/sink_env1_loop.cc. sink_env2
 * is pure i-side (no j-writes); merge_accum is two ACCUM_ADD lines; no
 * ghost-writeback bundle. populate_device_context stages per-active
 * Jgas/Jstar from host into UVM.
 *
 * Replaces the SINK_GRAVACCRETION==0 branch of
 * sinks/sink_environment_gpu.cc::sink_environment_second_evaluate_gpu.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <Kokkos_Core.hpp>

#include "../declarations/gpu_all_mirror.h"  /* MUST precede allvars.h: installs device-pass `#define All AllDeviceMirror` redirect before cell_data.h is parsed */
#include "../declarations/allvars.h"
#include "../declarations/gpu_numeric_macros.h"
#include "../core/proto.h"
#include "../mesh/kernel.h"               /* MUST precede sink_env2_loop.h */
#include "../mesh/ghost_writeback.h"      /* detector */
#include "sink_env2_loop.h"

#ifdef SINK_PARTICLES
#if defined(SINK_GRAVACCRETION) && (SINK_GRAVACCRETION == 0)

/* ============================================================================
 * PHYSICS HOOKS
 * ========================================================================== */

double SinkEnv2Spec::search_radius(const neighbor_loop_args& args,
                                    int /*active_slot*/, int i)
{
    return (double)args.P[i].KernelRadius;
}

SinkEnv2Spec::CallScalars
SinkEnv2Spec::populate_call_scalars(const neighbor_loop_args& /*args*/)
{
    return nlr_common_scalars_from_all();
}

void SinkEnv2Spec::apply_active_writeback(const neighbor_loop_args& args,
                                           int active_slot, int /*i*/,
                                           const AccumData& accum)
{
    Aux *aux = nlr_aux<SinkEnv2Spec>(args);
    aux->per_active_accum[active_slot] = accum;
}


/* ============================================================================
 * DEVICE CONTEXT LIFECYCLE
 *
 * Stage host-side per-active Jgas/Jstar (from SinkTempInfo[t].* on the
 * caller side) into a UVM array for device-side load_active. Same pattern
 * as sink_feed's per_active_local.
 * ========================================================================== */

void SinkEnv2Spec::populate_device_context(const neighbor_loop_args& args,
                                            DeviceContext& ctx)
{
    Aux *aux = nlr_aux<SinkEnv2Spec>(args);
    const int N = args.num_active;
    if(N <= 0) {
        ctx.per_active_in = nullptr;
        return;
    }
    SinkEnv2PerActiveIn *uvm = (SinkEnv2PerActiveIn *)
        Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(N * sizeof(SinkEnv2PerActiveIn));
    std::memcpy(uvm, aux->host_inputs, N * sizeof(SinkEnv2PerActiveIn));
    ctx.per_active_in = uvm;
}

void SinkEnv2Spec::cleanup_device_context(const neighbor_loop_args& /*args*/,
                                           DeviceContext& ctx)
{
    if(ctx.per_active_in) {
        Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(
            const_cast<SinkEnv2PerActiveIn *>(ctx.per_active_in));
        ctx.per_active_in = nullptr;
    }
}

/* ghost_write_detector_begin/end: runner default fires
 * ::ghost_write_detector_begin("sink_environment_second") /
 * ::ghost_write_detector_end() via the ghost_write_detector_name override
 * in sink_env2_loop.h. sink_env2 has no ghost_writeback bundle
 * (uses_ghost_writeback = false). */

#endif /* SINK_GRAVACCRETION == 0 */
#endif /* SINK_PARTICLES */
