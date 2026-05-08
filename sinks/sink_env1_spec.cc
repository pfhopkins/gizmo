/* sinks/sink_env1_spec.cc — host-side hooks for SinkEnv1Spec.
 *
 * The KOKKOS_INLINE_FUNCTION hooks (load_active, load_neighbor, pair_kernel,
 * zero_accum) live in sink_env1_spec.h so they are inlinable from both
 * device kernels (Mode A) and host walkers (Mode B/Brute). This .cc
 * contains the host-only hooks: search_radius_host (pre-arena radii),
 * populate_call_scalars_host (per-call globals snapshot),
 * apply_active_writeback (post-dispatch host scatter into the caller's
 * nl_outs buffer), and compare_accum (oracle helper for 3c.2).
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) and Claude for GIZMO.
 */

#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../mesh/kernel.h"           /* MUST precede sink_env1_spec.h */
#include "sink_env1_spec.h"

#ifdef SINK_PARTICLES

/* ----------------------------------------------------------------------------
 * Host hook: search_radius_host
 *
 * Per-active search radius from external state (P[i].KernelRadius).
 * Pre-arena, pre-drift epoch — matches sinks/sink_environment.cc:73:
 *     nl_radii[aa] = P[i].KernelRadius;
 * The runner stages radii_uvm[num_active] from this hook and passes it
 * to gpu_ngb_list_build (Mode A) and Mode B's per-query walker.
 * --------------------------------------------------------------------------*/
double SinkEnv1Spec::search_radius_host(const neighbor_loop_args& args,
                                         int /*active_slot*/, int i)
{
    return (double)args.P[i].KernelRadius;
}

/* ----------------------------------------------------------------------------
 * Host hook: populate_call_scalars_host
 *
 * Capture per-call cosmology + gravity scalars into a POD. Matches the
 * legacy capture in sinks/sink_environment_gpu.cc:108-113:
 *     double cf_atime    = All.cf_atime;
 *     double cf_a2inv    = All.cf_a2inv;
 *     double cf_a3inv    = All.cf_a3inv;
 *     double G_val       = All.G;
 *     double sink_radius_grav = SinkParticle_GravityKernelRadius;
 * --------------------------------------------------------------------------*/
SinkEnv1Spec::CallScalars
SinkEnv1Spec::populate_call_scalars_host(const neighbor_loop_args& /*args*/)
{
    CallScalars cs;
    cs.cf_atime         = All.cf_atime;
    cs.cf_a2inv         = All.cf_a2inv;
    cs.cf_a3inv         = All.cf_a3inv;
    cs.G_val            = All.G;
    cs.sink_radius_grav = SinkParticle_GravityKernelRadius;
    return cs;
}

/* ----------------------------------------------------------------------------
 * Host writeback: apply_active_writeback
 *
 * Copy AccumData into args.aux->nl_outs[active_slot]. The caller
 * (sinks/sink_environment.cc) reads nl_outs in its scatter loop
 * (lines 144-185) and applies physics-specific reductions into
 * SinkTempInfo. Byte-equivalent to the legacy memcpy at
 * sinks/sink_environment_gpu.cc:198 followed by the same caller-side
 * scatter — the runner just routes the AccumData into the same buffer.
 * --------------------------------------------------------------------------*/
void SinkEnv1Spec::apply_active_writeback(const neighbor_loop_args& args,
                                           int active_slot, int /*i*/,
                                           const AccumData& out)
{
    SinkEnv1Aux *aux = (SinkEnv1Aux *)args.aux;
    /* Byte-copy: AccumData == sink_env_gpu_out is POD; matches legacy
     * memcpy(out_host, d_out, num_active * sizeof(struct sink_env_gpu_out))
     * shape, just per-slot. */
    aux->nl_outs[active_slot] = out;
}

/* ----------------------------------------------------------------------------
 * Oracle compare: compare_accum
 *
 * Returns max-relative residual across all AccumData fields. Used by the
 * 3c.2 oracle gate; in 3c.1 the function compiles but is never called
 * (oracle path aborts early in run_neighbor_loop). Conservative
 * implementation: walk the byte representation and compute the maximum
 * relative difference per double-sized chunk. AccumData layout is
 * MyFloat-based (currently double per declarations/typedefs.h:23) — for
 * a future MyFloat=float build this still produces a meaningful relative
 * residual.
 *
 * Note: this is a host-only oracle helper. The pair kernel itself is
 * deterministic per active under fixed neighbor order, so under Mode B
 * vs Brute (same neighbor set, possibly different order) the worst case
 * is FP-floor reordering; compare_accum returns the max relative diff
 * which the runner gates against Spec::accum_tolerance.
 * --------------------------------------------------------------------------*/
double SinkEnv1Spec::compare_accum(const AccumData& a, const AccumData& b)
{
    double max_rel = 0.0;
    /* Flatten as MyFloat array. AccumData layout is exclusively MyFloat
     * scalars and MyFloat[3] arrays (see sinks/sinks_gpu_decls.h:50-86). */
    const MyFloat *pa = reinterpret_cast<const MyFloat*>(&a);
    const MyFloat *pb = reinterpret_cast<const MyFloat*>(&b);
    const size_t n = sizeof(AccumData) / sizeof(MyFloat);
    static_assert(sizeof(AccumData) % sizeof(MyFloat) == 0,
        "SinkEnv1Spec::AccumData must be MyFloat-aligned for byte-walk compare");
    for(size_t k = 0; k < n; k++) {
        double va = (double)pa[k], vb = (double)pb[k];
        double denom = std::fmax(std::fabs(va), std::fabs(vb));
        double diff  = std::fabs(va - vb);
        double rel   = (denom > 0.0) ? (diff / denom) : diff;
        if(rel > max_rel) max_rel = rel;
    }
    return max_rel;
}

#endif /* SINK_PARTICLES */
