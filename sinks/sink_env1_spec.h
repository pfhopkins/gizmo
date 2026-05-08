/* sinks/sink_env1_spec.h — NeighborLoopSpec for the sink_env1 environment loop.
 *
 * First Spec migrated onto the runner contract. Reuses the SSOT pair body in
 * sinks/sink_env1_pair_kernel.h (commit 98b8be5a) — both Mode A device kernel
 * and Mode B host walker call SinkEnv1Spec::pair_kernel, which forwards to
 * sink_env1_pair_kernel.
 *
 * Three-epoch staging (matches legacy timing exactly):
 *   (1) search_radius_host         host pre-arena   — equivalent to
 *                                                      sink_environment.cc:73
 *                                                      nl_radii[aa] = P[i].KernelRadius
 *   (2) populate_call_scalars_host host pre-arena   — equivalent to
 *                                                      sink_environment_gpu.cc:108-113
 *   (3) load_active                device, post-NGL — same UVM/P_gpu epoch as
 *                                                      sink_environment_gpu.cc:135-165
 *                                                      lambda's q-packing block.
 *
 * Architecture binding contract:
 *   ~/.claude/memory/reference_neighbor_loop_contract.md
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) and Claude for GIZMO.
 */
#ifndef SINK_ENV1_SPEC_H
#define SINK_ENV1_SPEC_H

#include "../declarations/allvars.h"

#ifdef SINK_PARTICLES

/* ----------------------------------------------------------------------------
 * Compile-time guard: SINGLE_STAR_SINK_DYNAMICS + SINK_GRAVCAPTURE_GAS j-side
 * SwallowTime atomic write is not yet wired through SymmetricPairScatter
 * writeback. Same compile-failure surface as the existing #error in
 * sinks/sink_environment_gpu.cc:179-181 (kept duplicated until 3c.5
 * retires the legacy evaluator). Codex caution: do not narrow the guard.
 * --------------------------------------------------------------------------*/
#if defined(SINK_GRAVCAPTURE_GAS) && defined(SINGLE_STAR_SINK_DYNAMICS)
#error "SinkEnv1Spec: SSD + SINK_GRAVCAPTURE_GAS j-side SwallowTime atomic write is not yet wired through the runner's SymmetricPairScatter channel. This guard mirrors the existing #error in sinks/sink_environment_gpu.cc:179-181. Restored in 3c.5 alongside spike retirement; do not enable SSD under runner-Mode-A until then."
#endif

#include "../mesh/neighbor_loop_runner.h"
#include "../mesh/mode_b_local_walker.h"      /* MODE_B_SEARCH_*, MODE_B_RADIUS_* */
#include "sinks_gpu_decls.h"                  /* struct sink_env_gpu_out */
#include "sink_environment_mode_b.h"          /* sink_env1_query_t */
/* NOTE: caller TUs must include "../mesh/kernel.h" BEFORE this header.
 * kernel.h has no include guards (defines static inline kernel_main).
 * sink_env1_pair_kernel.h (included below) uses kernel_main + NEAREST_XYZ.
 * mesh/neighbor_loop_runner.cc and sinks/sink_env1_spec.cc both include
 * kernel.h before this header. Same convention as sink_env1_pair_kernel.h. */
#include "sink_env1_pair_kernel.h"            /* SSOT pair body + sink_env1_scalars_t */

/* ----------------------------------------------------------------------------
 * Per-active host-side aux struct, replaces void* casts on neighbor_loop_args::aux.
 * Caller (sink_environment.cc) populates one and passes &aux into args.aux.
 * --------------------------------------------------------------------------*/
struct SinkEnv1Aux {
    sink_env_gpu_out *nl_outs;   /* [num_active] — host buffer to receive accumulator */
};

/* ----------------------------------------------------------------------------
 * SinkEnv1Spec
 *
 * NeighborData lifetime contract:
 *   nb.kp_j MUST point into ctx.P    (= P_gpu UVM array under Mode A;
 *                                       = P_host directly in Mode B walker).
 *   nb.kc_j MUST be either nullptr OR point into ctx.CellP (UVM under Mode A).
 *   Pointers are valid only for the duration of one pair_kernel call.
 *   DO NOT store; DO NOT cross arena scope.
 * --------------------------------------------------------------------------*/
struct SinkEnv1Spec {
    /* ---- Identifier ---- */
    static constexpr const char *loop_name = "sink_env1";

    /* ---- Types ---- */
    using CallScalars = sink_env1_scalars_t;       /* per-call cosmology+gravity scalars */

    struct ActiveData {
        sink_env1_query_t   q;
        sink_env1_scalars_t sc;
    };

    struct NeighborData {
        const struct particle_data *kp_j;
        const struct gas_cell_data *kc_j;          /* nullptr for non-gas / when no CellP */
    };

    using AccumData      = struct sink_env_gpu_out;
    using ScatterData    = NoScatter;
    using IdentityFields = NoIdentity;
    using IterControl    = NotIterative;

    /* ---- Search / writeback constexprs ---- */
    static constexpr int                     search_mode        = MODE_B_SEARCH_SYMMETRIC;
    static constexpr unsigned int            neighbor_type_mask = (unsigned int)SINK_NEIGHBOR_BITFLAG;
    static constexpr mode_b_radius_policy_t  radius_policy      = MODE_B_RADIUS_DEFAULT;
    static constexpr WritePattern            write_pattern      = WritePattern::ActiveReduceOnly;
    static constexpr SidxCacheKind           sidx_cache_kind    = SidxCacheKind::AllTypes;

    /* ---- Oracle compare ----
     * Mode B vs Brute oracle: same pair_kernel, same candidate SET, but
     * DIFFERENT iteration order (tree-walk returns walk-order; brute returns
     * ascending P[] index order). Per-active accumulators sum the same
     * MyFloat contributions in different orders → double-precision floor of
     * ~N·eps_machine relative residual, plus possible cancellation
     * amplification on small-magnitude fields. 1e-10 is one decade tighter
     * than the legacy precedent (sinks/sink_environment_mode_b.cc:360 uses
     * 1e-9 for the identical tree-vs-brute comparison) — sharp enough to
     * catch real bugs, loose enough to not trip on summation reorder noise.
     * (Mode A bit-equivalence vs legacy is checked separately in 3c.1 with
     * byte-exact target.) */
    static constexpr double accum_tolerance = 1e-10;
    static double compare_accum(const AccumData& a, const AccumData& b);

    /* ============================================================================
     * Host hooks (pre-arena epoch)
     * ========================================================================== */

    /* Per-active search radius from P[i].KernelRadius (matches
     * sinks/sink_environment.cc:73 staging). Pre-arena, pre-drift. */
    static double search_radius_host(const neighbor_loop_args& args,
                                      int active_slot, int i);

    /* Per-call scalar globals captured into a POD (matches
     * sinks/sink_environment_gpu.cc:108-113 capture). */
    static CallScalars populate_call_scalars_host(const neighbor_loop_args& args);

    /* ============================================================================
     * Device hooks (KOKKOS_INLINE_FUNCTION; called from device in Mode A,
     * host in Mode B/Brute walker).
     * ========================================================================== */

    /* Build ActiveData[slot] from device-visible state (post-NGL-build).
     *
     * BYTE-EXACT MAPPING to the legacy GPU lambda's q-packing block at
     * sinks/sink_environment_gpu.cc:145-165:
     *   q.pos      = kp[ii].Pos                           [line 146]
     *   q.vel      = kp[ii].Vel                           [line 147]
     *   q.id       = kp[ii].ID                            [line 148]
     *   q.h_search = h_i (= radii[aa])                    [lines 137,149]
     *   q.ags_h    = (ADAPTIVE_GRAVSOFT_FORALL & 32)
     *                 ? kp[ii].AGS_KernelRadius
     *                 : sink_radius_grav                  [lines 150-154]
     *   q.mass     = kp[ii].Mass    (SINK_GRAVCAPTURE_GAS
     *                                || SINK_GRAVACCRETION==8) [lines 155-157]
     *   q.sink_radius = kp[ii].SinkRadius
     *                  (SINK_GRAVCAPTURE_FIXEDSINKRADIUS) [lines 158-160]
     *   q.sink_angmom = kp[ii].Sink_Specific_AngMom
     *                  (SINK_RETURN_ANGMOM_TO_GAS)        [lines 161-163]
     *   q.origin_local_idx = aa                           [line 164]
     *   q.origin_rank      = -1  (lambda sets -1; pack_query in mode_b
     *                             walker sets ThisTask — Mode A contract is
     *                             -1, pair_kernel does not read this field) [line 165]
     */
    KOKKOS_INLINE_FUNCTION
    static ActiveData load_active(const NeighborLoopDeviceContextBase& ctx,
                                   int active_slot, int i,
                                   double h_search,
                                   const CallScalars& cs)
    {
        ActiveData a;
        a.q.pos      = ctx.P[i].Pos;
        a.q.vel      = ctx.P[i].Vel;
        a.q.id       = ctx.P[i].ID;
        a.q.h_search = h_search;
#if (ADAPTIVE_GRAVSOFT_FORALL & 32)
        a.q.ags_h    = (double)ctx.P[i].AGS_KernelRadius;
#else
        a.q.ags_h    = cs.sink_radius_grav;
#endif
#if defined(SINK_GRAVCAPTURE_GAS) || (SINK_GRAVACCRETION == 8)
        a.q.mass     = (double)ctx.P[i].Mass;
#endif
#if defined(SINK_GRAVCAPTURE_FIXEDSINKRADIUS)
        a.q.sink_radius = (double)ctx.P[i].SinkRadius;
#endif
#if defined(SINK_RETURN_ANGMOM_TO_GAS)
        for(int kv = 0; kv < 3; kv++) a.q.sink_angmom[kv] = ctx.P[i].Sink_Specific_AngMom[kv];
#endif
        a.q.origin_local_idx = active_slot;
        a.q.origin_rank      = -1;            /* device — rank N/A in lambda; matches old line 165 */
        a.sc                 = cs;
        return a;
    }

    /* Zero accumulator. Matches the legacy lambda's
     * `memset(&kout[aa], 0, sizeof(struct sink_env_gpu_out))` at
     * sinks/sink_environment_gpu.cc:138 byte-for-byte (sink_env_gpu_out
     * is POD; raw zero is byte-equivalent). */
    KOKKOS_INLINE_FUNCTION
    static void zero_accum(AccumData& out)
    {
        for(size_t b = 0; b < sizeof(out); b++) ((char*)&out)[b] = 0;
    }

    /* Build NeighborData for j. ctx.P/ctx.CellP are UVM under Mode A; in
     * Mode B walker they are P_host/CellP_host — same shape. */
    KOKKOS_INLINE_FUNCTION
    static NeighborData load_neighbor(const NeighborLoopDeviceContextBase& ctx,
                                       int j,
                                       const IdentitySidecar& /*id*/,
                                       const ActiveData& /*a*/)
    {
        NeighborData nb;
        nb.kp_j = &ctx.P[j];
        nb.kc_j = (ctx.CellP && ctx.P[j].Type == 0) ? &ctx.CellP[j] : nullptr;
        return nb;
    }

    /* The physics — forwards to the SSOT pair body. */
    KOKKOS_INLINE_FUNCTION
    static void pair_kernel(const ActiveData& a,
                             const NeighborData& nb,
                             AccumData& accum,
                             NoScatter& /*nb_out*/)
    {
        sink_env1_pair_kernel(a.q, *nb.kp_j, nb.kc_j, a.sc, accum);
    }

    /* ============================================================================
     * Host writeback (post-dispatch)
     * ========================================================================== */

    /* Copy AccumData into args.aux->nl_outs[slot]. The caller's
     * SinkTempInfo scatter loop (sinks/sink_environment.cc:144-185) reads
     * nl_outs and applies its own physics-specific reductions. */
    static void apply_active_writeback(const neighbor_loop_args& args,
                                        int active_slot, int i,
                                        const AccumData& out);

    /* Per-field merge of a peer-rank reply (src) into a local accumulator
     * (dst). Used by run_mode_b_remote at the cross-rank boundary; per-field
     * reduction op matches the pair_kernel writes (sum for additive fields,
     * MAX for DF_mmax_particles). Lifted byte-exact from the legacy
     * sinks/sink_environment_mode_b.cc:446-486 merge_into helper, including
     * all #ifdef-gated optional fields. */
    static void merge_accum(AccumData& dst, const AccumData& src);
};

#endif /* SINK_PARTICLES */

#endif /* SINK_ENV1_SPEC_H */
