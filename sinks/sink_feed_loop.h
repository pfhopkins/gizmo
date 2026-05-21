/* sinks/sink_feed_loop.h — sink "feed" (mark-for-accretion) neighbor loop module.
 *
 * Defines SinkFeedSpec for the neighbor-loop runner (mesh/neighbor_loop_runner.h).
 * Per the standard 2-file convention (see sink_env1_loop.h), this header
 * carries the Spec contract + inline (host+device) physics that the runner
 * must inline into the device kernel; sinks/sink_feed_loop.cc carries the
 * host-only hooks.
 *
 * Replaces sinks/sink_feed_gpu.cc + sinks/sink_feed_functions.h. The latter's
 * structs (SinkFeedLocalIn, SinkFeedOut) and inline pair body
 * (sink_feed_pair_kernel) are absorbed below — the pair body's per-call
 * scalars now travel through SinkFeedCallScalars instead of direct All.*
 * reads (TRAP 1 compliance), and per-active host-staged inputs travel
 * through the SinkFeedDeviceContext extension wired in Phase 4.A.0
 * (mesh/neighbor_loop_runner.h §DeviceContext extension trait).
 *
 * Phase 4 port 3d.1.  Written by Phil Hopkins (phopkins@caltech.edu) and
 * Claude for GIZMO.
 */
#ifndef SINK_FEED_LOOP_H
#define SINK_FEED_LOOP_H

/* Kokkos_Core MUST come BEFORE declarations/allvars.h. allvars.h pulls in
 * declarations/macros.h which #defines `terminate(...)`; that macro mangles
 * the C++ stdlib `<exception>` header's `std::terminate()` declaration that
 * Kokkos_Core.hpp transitively pulls in. Same pattern as the GPU TUs
 * (sinks/sink_environment_gpu.cc, hydro/density_gpu.cc, etc.). */
#include <Kokkos_Core.hpp>

#include "../declarations/allvars.h"

#ifdef SINK_PARTICLES

/* (Kokkos_Core.hpp pulled in above the SINK_PARTICLES guard for stdlib-
 * header ordering — see comment at top of file.) */

#include "../declarations/gpu_rng.h"
#include "../mesh/neighbor_loop_runner.h"
#include "../mesh/mode_b_local_walker.h"   /* MODE_B_SEARCH_*, MODE_B_RADIUS_* */
#include "sink_functions.h"                /* sink_vesc_gpu, sink_lum_bol, ... */
/* NOTE: caller TUs must include "../mesh/kernel.h" BEFORE this header.
 * kernel.h has no include guards; defines static inline kernel_main used
 * by the inline pair body below. The runner and this loop's .cc both
 * include kernel.h first; new callers must do the same. */

#ifndef KOKKOS_INLINE_FUNCTION
#define KOKKOS_INLINE_FUNCTION inline
#endif

/* Forward decls. */
int  sink_isactive(int i);
int  sink_feed_is_active(int i);

/* ============================================================================
 * Per-pair physics types (file scope; PascalCase). SinkFeedSpec re-exports
 * via `using CallScalars = ...; using ActiveData = ...; using AccumData = ...`.
 * ========================================================================== */

/* Per-call cosmology + sink globals captured once from All.* on the host
 * by populate_call_scalars. Routed through ActiveData::scalars to the inline
 * pair body so the body never reads All.* directly (TRAP 1).
 *
 * rng_step uses a sink_feed-unique constant XOR shift on top of
 * NumCurrentTiStep. This is load-bearing: if multiple sink loops in the
 * same timestep used identical rng_step bases, a per-particle draw
 * "fires" simultaneously across loops. See feedback_rng_loop_uniqueness.md
 * in project memory. */
struct SinkFeedCallScalars {
    NlrCommonScalars common;             /* cf_atime, cf_a2inv, newton_G, ... */
    double           sink_radius_grav;   /* SinkParticle_GravityKernelRadius */
    double           sink_accreted_fraction;  /* All.Sink_accreted_fraction */
    uint64_t         rng_step;           /* (uint64_t)NumCurrentTiStep ^ 0xfeed5117ULL */
};

/* Per-active host-fill struct, populated by populate_device_context() on
 * the host (allocates UVM, populates each slot from
 * P_host/CellP_host/SinkTempInfo + sink_eddington_mdot/sink_lum_bol etc.,
 * all host-only). Read on the device by Spec::load_active via
 * SinkFeedDeviceContext::per_active_local. Trivially copyable for byte-
 * level MPI transfer in the Mode B remote path. Conditional fields mirror
 * the pair body's #ifdefs. */
struct SinkFeedLocalIn {
    Vec3<MyDouble> Pos;
    Vec3<MyFloat>  Vel;
    MyFloat KernelRadius;
    MyFloat Mass;
    MyFloat Sink_Mass;
    MyFloat Density;
    MyFloat Mdot;
    MyFloat Dt;
    MyIDType ID;
#ifdef SINK_GRAVCAPTURE_GAS
    MyFloat mass_to_swallow_edd;
#endif
#if defined(SINK_GRAVCAPTURE_GAS) && defined(SINK_ENFORCE_EDDINGTON_LIMIT) && !defined(SINK_ALPHADISK_ACCRETION)
    MyFloat edd_p;
#endif
#if defined(SINK_SWALLOWGAS) && !defined(SINK_GRAVCAPTURE_GAS)
    MyFloat Sink_AccretionDeficit;
#endif
#ifdef SINK_GRAVCAPTURE_FIXEDSINKRADIUS
    MyFloat SinkRadius;
#endif
#if (ADAPTIVE_GRAVSOFT_FORALL & 32)
    MyFloat AGS_KernelRadius;
#endif
#ifdef SINK_ALPHADISK_ACCRETION
    MyFloat Sink_Mass_Reservoir;
#endif
#ifdef SINGLE_STAR_MERGE_AWAY_CLOSE_BINARIES
    /* Per-source eligibility for binary-merge-away. Filled host-side in
     * sink_feed_fill_local via is_star_eligible_for_binary_merge_away(i). */
    int Sink_eligible_for_binary_merge_away;
#endif
#ifdef SINK_CALC_LOCAL_ANGLEWEIGHTS
    Vec3<MyFloat> Jgas_in_Kernel;
#endif
#ifdef SINK_THERMALFEEDBACK
    MyFloat thermal_energy;     /* precomputed: sink_lum_bol(Mdot, Sink_Mass, -1) * Dt */
#endif
};

/* AccumData. Same fields as the legacy SinkFeedOut; adds a per-i scratch
 * field `mass_markedswallow_scratch` that the pair body uses as a running
 * total within one i's neighbor loop (gates further accretion in
 * SINK_SWALLOWGAS && !SINK_GRAVCAPTURE_GAS). Marked "scratch" because it
 * is NOT aggregated across peers in merge_accum — within one i, it is a
 * cumulative per-i quantity; across peers it has no well-defined combine
 * semantics. */
struct SinkFeedOut {
#ifdef SINK_CALC_LOCAL_ANGLEWEIGHTS
    double Sink_angle_weighted_kernel_sum;
#endif
#ifdef SINK_REPOSITION_ON_POTMIN
    double       Sink_PotentialMinimumOfNeighbors;
    Vec3<double> Sink_PotentialMinimumOfNeighborsPos;
#endif
    double mass_markedswallow_scratch;   /* per-i scratch; see comment above */
};

/* Active-particle state passed into the pair body. Same shape as
 * SinkEnv1ActiveState — `pos` and `h_search` are top-level (the runner's
 * Mode B walker reads them directly: see neighbor_loop_runner.cc:748,753).
 * `local` carries the host-fill struct for the rest of the per-active
 * physics; `scalars` carries per-call cosmology+globals; origin_* fields
 * carry the requester identity for Mode B remote replies. Trivially
 * copyable for byte-level MPI transfer. */
struct SinkFeedActiveState {
    Vec3<double>         pos;             /* P[i].Pos — runner reads directly */
    double               h_search;        /* runner-supplied search radius */
    SinkFeedLocalIn      local;
    SinkFeedCallScalars  scalars;
    int                  origin_local_idx;
    int                  origin_rank;
};

/* Phase 4.A.0 DeviceContext extension. Carries a UVM-resident pointer
 * to the per-active host-fill array so device-side load_active can read
 * it. Allocated + populated by populate_device_context (on host, before
 * the runner's stage_active kernel); freed by cleanup_device_context
 * (runs unconditionally via NlrDeviceContextCleanupGuard at runner exit
 * on every dispatch path).
 *
 * Trivially copyable: the runner captures DeviceContext by value into
 * the Kokkos device lambda. Static_assert in runner.cc enforces this. */
struct SinkFeedDeviceContext : NeighborLoopDeviceContextBase {
    const SinkFeedLocalIn *per_active_local;   /* UVM, [num_active] */
    bool oracle_dry_run;                       /* set by SinkFeedSpec::set_oracle_brute_pass
                                                  on the brute oracle pass; pair body
                                                  suppresses j-side atomic writes when true.
                                                  Default-initialised to false in
                                                  populate_device_context. */
#ifdef SINGLE_STAR_MERGE_AWAY_CLOSE_BINARIES
    /* UVM, length ctx.num_total. Same index space as ctx.P[j] — covers
     * local + imported-ghost slots after the Mode-A ghost-import step
     * (which appends ghosts to global P[] and grows NumPart BEFORE
     * populate_device_context fires; see runner.cc Mode-A entry ordering).
     * 1 iff P[j].Type==5 && is_star_eligible_for_binary_merge_away(j).
     * Allocated INDEPENDENTLY of num_active: Mode B remote responder ranks
     * with N==0 still walk this rank's local pool for peer queries (per
     * runner.cc "Every rank participates even if N == 0") and load_neighbor
     * reads dctx.binary_merge_eligible[j] regardless of N. */
    const unsigned char *binary_merge_eligible;
#endif
};

/* ============================================================================
 * Inline pair body — single source of truth for sink_feed per-pair physics.
 * Called from SinkFeedSpec::pair_kernel; used by all three runner paths
 * (Mode A GPU lambda, Mode B local host walker, Mode B remote peer-to-peer).
 *
 * SwallowID semantics (D1/S-SYNC):
 *   - kernel-local (this body): atomic_exchange writes the active sink's ID
 *     into neighbor_particle.SwallowID under predicate. Multiple contending
 *     writers on the same rank resolve to one ID non-deterministically;
 *     this is legacy-equivalent (the CPU tree-walk also doesn't enforce a
 *     tiebreak within a rank).
 *   - cross-rank: the home-rank merge in the ghost-writeback bundle uses
 *     PARTICLE_MAX (largest ID wins), making the cross-rank outcome
 *     deterministic. See ghost_writeback_ops.h::ParticleMaxOp.
 *   - SINK_THERMALFEEDBACK only: neighbor_cell->Injected_Sink_Energy uses
 *     atomic_add at kernel time, GAS_ADD snapshot-diff additive merge at
 *     home rank.
 *
 * Order-dependence note: in the SINK_SWALLOWGAS && !SINK_GRAVCAPTURE_GAS
 * path, the running scratch `accum.mass_markedswallow_scratch` gates
 * further accretion within one i's neighbor loop. Within a single rank
 * the j-loop order is tree-walk-driven (Mode A CSR + Mode B local walker
 * agree). Across Mode B remote splits, each peer accumulates its own
 * scratch independently — Mode B remote can therefore swallow a different
 * total mass than Mode A would on the same step.
 *
 * **This divergence is ACCEPTED PHYSICS.** The sink-swallow draw in this
 * branch is fundamentally stochastic (Poisson-style sampling of which gas
 * particles to consume per step). Per-rank scratch budgets cause the
 * sink to over- or under-shoot its accretion target by O(few × the
 * local stochastic rate) on any one step, and the accretion-deficit
 * book-keeping corrects on subsequent timesteps via P[i].Sink_AccretionDeficit
 * — the same way the legacy CPU export-loop semantics already worked.
 * Phil-confirmed 2026-05-09: not a bug, no Mode-A fallback, no distributed-
 * budget protocol required. Future reviewer encountering this comment:
 * do not attempt to "fix" the divergence; it is the documented model.
 *
 * Validation note: the JSIDE_HASH harness will show small ie_sum / sw_max
 * divergence between Mode A and Mode B remote on this branch — these are
 * stochastic-allowed and do NOT constitute a regression as long as
 * the AccumData oracle (which doesn't include the scratch field — see
 * merge_accum manifest) shows zero mismatch.
 *
 * RNG keys: per feedback_rng_loop_uniqueness.md, all per-pair draws key
 * on (local.ID ^ neighbor_particle.ID) and use scalars.rng_step which
 * carries a sink_feed-unique 0xfeed5117ULL XOR shift on top of
 * NumCurrentTiStep.
 * ========================================================================== */

KOKKOS_INLINE_FUNCTION
static void sink_feed_pair_kernel(const SinkFeedActiveState& active,
                                  struct particle_data& neighbor_particle,
                                  struct gas_cell_data* neighbor_cell,
                                  SinkFeedOut& out,
                                  bool oracle_dry_run
#ifdef SINGLE_STAR_MERGE_AWAY_CLOSE_BINARIES
                                  , unsigned char neighbor_binary_merge_eligible
#endif
                                  )
{
    const SinkFeedLocalIn& local       = active.local;
    const SinkFeedCallScalars& scalars = active.scalars;

    if(neighbor_particle.Mass <= 0) return;

    const double h_i  = (double)local.KernelRadius;
    const double h_i2 = h_i * h_i;
    /* j-side softening: SSOT is P[j].ForceSoftening (compute_all_force_softening
     * keeps it in sync; gravity/gpu_gravtree.cc::gpu_force_softening_kernel_radius
     * is the device-side accessor). Falls back to sink_radius_grav if zero. */
    double soft_j = neighbor_particle.ForceSoftening;
    if(soft_j <= 0) soft_j = scalars.sink_radius_grav;
    double heff_j = DMAX((double)neighbor_particle.KernelRadius, soft_j);

    /* dpos / dvel with periodic wrap + shearbox velocity correction. Sign:
     * j - i. Computed inline (sink_env1 pattern), not received as args. */
    Vec3<double> dpos;
    dpos[0] = (double)neighbor_particle.Pos[0] - (double)local.Pos[0];
    dpos[1] = (double)neighbor_particle.Pos[1] - (double)local.Pos[1];
    dpos[2] = (double)neighbor_particle.Pos[2] - (double)local.Pos[2];
    Vec3<double> dvel;
    dvel[0] = (double)neighbor_particle.Vel[0] - (double)local.Vel[0];
    dvel[1] = (double)neighbor_particle.Vel[1] - (double)local.Vel[1];
    dvel[2] = (double)neighbor_particle.Vel[2] - (double)local.Vel[2];
    nearest_xyz(dpos, -1);
    NGB_SHEARBOX_BOUNDARY_VELCORR_(local.Pos, neighbor_particle.Pos, dvel, -1);

    double r2 = dpos.norm_sq();
    if(r2 <= 0) return;
    if(r2 >= h_i2 && r2 >= heff_j * heff_j) return;
    double r = sqrt(r2);
    if(r <= 0) return;

    double vrel_sq = dvel.norm_sq();
    double vrel = sqrt(vrel_sq) / scalars.common.cf_atime;

    double ags_h_i = scalars.sink_radius_grav;
#if (ADAPTIVE_GRAVSOFT_FORALL & 32)
    ags_h_i = (double)local.AGS_KernelRadius;
#endif

    double sink_radius = scalars.sink_radius_grav;
#ifdef SINK_GRAVCAPTURE_FIXEDSINKRADIUS
    sink_radius = (double)local.SinkRadius;
#endif

    /* sink_vesc_gpu / sink_check_boundedness_gpu / sink_fb_angleweight_*
     * still accept const refs — the const-cast on neighbor_particle works
     * because we need the cell ref pattern (gas-or-fallback). */
    struct gas_cell_data dummy_cell{};
    const struct gas_cell_data& cell_ref = neighbor_cell ? *neighbor_cell : dummy_cell;
    double vesc = sink_vesc_gpu(neighbor_particle, cell_ref,
                                (double)local.Mass, r, ags_h_i);

    MyIDType SwallowID_j = Kokkos::atomic_load(&neighbor_particle.SwallowID);

#ifdef SINK_REPOSITION_ON_POTMIN
    {
        double boundedness_function = neighbor_particle.Potential
                                      + 0.5 * vrel * vrel * scalars.common.cf_atime;
        if(boundedness_function < 0) {
            double wt_rsoft = r / (3. * scalars.sink_radius_grav);
            boundedness_function *= 1. / (1. + wt_rsoft * wt_rsoft);
        }
        double potential_function = boundedness_function;
        if(potential_function < out.Sink_PotentialMinimumOfNeighbors &&
           neighbor_particle.Type != 0 && neighbor_particle.Type != 5) {
            out.Sink_PotentialMinimumOfNeighbors    = potential_function;
            out.Sink_PotentialMinimumOfNeighborsPos = neighbor_particle.Pos;
        }
    }
#endif

    /* ---- sink-sink merger check ---- */
    if(neighbor_particle.Type == 5) {
        if(((local.ID != neighbor_particle.ID) || (r2 > 0)) &&
           (SwallowID_j == 0) && (neighbor_particle.Sink_Mass < local.Sink_Mass)) {
#ifdef SINGLE_STAR_SINK_DYNAMICS
            /* volatile per [[feedback_gpu]] §D.3 (nvc++ device-lambda
             * const-prop bug on int gate vars assigned conditionally —
             * "allow_sink_merger" is a named known-vulnerable case). */
            volatile int allow_sink_merger = 1;
            if(r >= 1.0001 * neighbor_particle.Min_Distance_to_Sink)  allow_sink_merger = 0;
            if(r >= heff_j)                                            allow_sink_merger = 0;
            if(neighbor_particle.Mass > local.Mass)                    allow_sink_merger = 0;
            if((neighbor_particle.Mass == local.Mass) &&
               (neighbor_particle.ID > local.ID))                      allow_sink_merger = 0;
            double max_rmerge = 1.0 * sink_radius;
            double max_mmerge = 10. * neighbor_particle.Sink_Formation_Mass;
#ifdef SINGLE_STAR_MERGE_AWAY_CLOSE_BINARIES
            /* MERGE_AWAY: helper folds the eligibility checks + extends
             * max_rmerge by softening / kernel-radius / orbital-period
             * floors, and overrides max_mmerge to 10 * neighbor.Mass.
             * Replaces the unconditional sink_check_boundedness_gpu gate
             * (the legacy CPU path also drops the boundedness check
             * when MERGE_AWAY is on — see legacy sink_feed.cc:216-227). */
            sink_apply_binary_merge_away_limits(local.Sink_eligible_for_binary_merge_away,
                                                (int)neighbor_binary_merge_eligible,
                                                soft_j,
                                                (double)local.KernelRadius,
                                                (double)neighbor_particle.KernelRadius,
                                                (double)local.Mass,
                                                (double)neighbor_particle.Mass,
                                                sink_radius,
                                                &allow_sink_merger,
                                                &max_rmerge, &max_mmerge,
                                                scalars.common.cf_atime);
#else
            if(!sink_check_boundedness_gpu(neighbor_particle, cell_ref,
                                           vrel, vesc, r, sink_radius)) allow_sink_merger = 0;
#endif
            if(r >= max_rmerge)                  allow_sink_merger = 0;
            if(neighbor_particle.Mass > max_mmerge) allow_sink_merger = 0;
            if(allow_sink_merger == 1)
#endif
            {
                if(vrel < vesc) { SwallowID_j = local.ID; }
            }
        }
    }

    /* ---- grav-capture check (non-Type5) ---- */
#if defined(SINK_GRAVCAPTURE_GAS) || defined(SINK_GRAVCAPTURE_NONGAS)
    if(neighbor_particle.Type != 5 && SwallowID_j < local.ID) {
        volatile int do_gravcap = 1;
#ifdef SINGLE_STAR_SINK_DYNAMICS
        {
            double eps = DMAX(r, DMAX(heff_j, ags_h_i) * KERNEL_FAC_FROM_FORCESOFT_TO_PLUMMER);
            if(eps * eps * eps / ((double)neighbor_particle.Mass + (double)local.Mass)
               > neighbor_particle.SwallowTime) do_gravcap = 0;
        }
#endif
#ifdef SINK_ALPHADISK_ACCRETION
        if((double)local.Sink_Mass_Reservoir
           >= SINK_ALPHADISK_ACCRETION * (double)local.Sink_Mass) do_gravcap = 0;
#endif
        if(do_gravcap && vrel < vesc) {
#ifdef SINK_GRAVCAPTURE_FIXEDSINKRADIUS
            {
                double spec_mom = dot(dvel, dpos);
                spec_mom = r2 * vrel * vrel - spec_mom * spec_mom * scalars.common.cf_a2inv;
                if(spec_mom >= scalars.common.newton_G
                              * ((double)local.Mass + neighbor_particle.Mass) * sink_radius)
                    do_gravcap = 0;
            }
#endif
            if(do_gravcap && sink_check_boundedness_gpu(neighbor_particle, cell_ref,
                                                        vrel, vesc, r, sink_radius)) {
#ifdef SINK_GRAVCAPTURE_NONGAS
                if(neighbor_particle.Type != 0) { SwallowID_j = local.ID; }
#endif
#ifdef SINK_GRAVCAPTURE_GAS
                if(neighbor_particle.Type == 0) {
#if defined(SINK_ENFORCE_EDDINGTON_LIMIT) && !defined(SINK_ALPHADISK_ACCRETION)
                    double p = (double)local.edd_p;
#ifdef SINK_WIND_KICK
                    if(scalars.sink_accreted_fraction > 0)
                        p /= scalars.sink_accreted_fraction;
#endif
                    /* RNG site 1: keys on local.ID XOR neighbor.ID + per-loop-
                     * unique rng_step (0xfeed5117ULL shift in populate_call_scalars). */
                    double w = gizmo_gpu_rand_double((uint64_t)neighbor_particle.ID
                                                     ^ (uint64_t)local.ID,
                                                     scalars.rng_step);
                    if(w < p) { SwallowID_j = local.ID; }
#else
                    SwallowID_j = local.ID;
#endif
                }
#endif
            }
        }
    }
#endif /* SINK_GRAVCAPTURE_GAS || SINK_GRAVCAPTURE_NONGAS */

    /* ---- kernel quantities for gas pairs ---- */
    if(neighbor_particle.Type == 0) {
        double hinv  = 1. / h_i, hinv3 = hinv * hinv * hinv;
        double u = r * hinv, wk = 0, dwk = 0;
        if(u < 1) { kernel_main(u, hinv3, hinv * hinv3, &wk, &dwk, -1); }

#if defined(SINK_SWALLOWGAS) && !defined(SINK_GRAVCAPTURE_GAS)
        if(SwallowID_j < local.ID) {
            double dm_toacc = (double)local.Sink_AccretionDeficit
                              - out.mass_markedswallow_scratch;
            double f_accreted = 1.;
#ifdef SINK_WIND_KICK
            if(scalars.sink_accreted_fraction > 0) f_accreted = scalars.sink_accreted_fraction;
#endif
            double p = 0;
            if(dm_toacc > 0 && local.Density > 0) {
                p = dm_toacc * wk / (double)local.Density;
            }
#ifdef SINK_WIND_KICK
            if(f_accreted > 0) {
                p /= f_accreted;
                double sink_mass_withdisk = (double)local.Sink_Mass;
#ifdef SINK_ALPHADISK_ACCRETION
                sink_mass_withdisk += (double)local.Sink_Mass_Reservoir;
#endif
                if(sink_mass_withdisk < (double)local.Mass && (double)local.Density > 0) {
                    p = ((1. - f_accreted) / f_accreted) * (double)local.Mdot
                        * (double)local.Dt * wk / (double)local.Density;
                }
            }
#endif
            /* RNG site 2: same pairwise key + +1 counter offset to separate from site 1. */
            double w = gizmo_gpu_rand_double((uint64_t)neighbor_particle.ID
                                             ^ (uint64_t)local.ID,
                                             scalars.rng_step + 1);
            if(w < p) {
                SwallowID_j = local.ID;
                out.mass_markedswallow_scratch += (double)neighbor_particle.Mass * f_accreted;
            }
        }
#endif /* SINK_SWALLOWGAS && !SINK_GRAVCAPTURE_GAS */

#ifdef SINK_CALC_LOCAL_ANGLEWEIGHTS
        if((double)local.Dt > 0 && r > 0 && SwallowID_j == 0 && neighbor_particle.Mass > 0) {
            Vec3<double> J_dir = (Vec3<double>)local.Jgas_in_Kernel;
            double norm_j2 = J_dir.norm_sq();
            if(norm_j2 > 0) { J_dir *= 1. / sqrt(norm_j2); } else { J_dir = {0, 0, 1}; }
            double cos_theta = dot(dpos / r, J_dir);
            out.Sink_angle_weighted_kernel_sum +=
                sink_fb_angleweight_localcoupling_gpu(neighbor_particle, cell_ref,
                                                       cos_theta, r, h_i);
        }
#endif

#ifdef SINK_THERMALFEEDBACK
        if((double)local.Density > 0 && wk > 0 && neighbor_cell) {
            double dE = (wk / (double)local.Density)
                        * (double)local.thermal_energy
                        * (double)neighbor_particle.Mass;
            if(!oracle_dry_run) {
                Kokkos::atomic_add(&neighbor_cell->Injected_Sink_Energy, dE);
            }
        }
#endif
    }

    /* ---- commit SwallowID if set ---- */
    if(SwallowID_j > 0 && !oracle_dry_run) {
        Kokkos::atomic_exchange(&neighbor_particle.SwallowID, SwallowID_j);
    }
}

/* ============================================================================
 * SinkFeedSpec — the NeighborLoopSpec contract for sink_feed.
 * NeighborData lifetime: same as sink_env1.
 * ========================================================================== */

struct SinkFeedSpec {
    /* ====================================================================
     * PHYSICS BLOCK
     * ==================================================================== */

    static constexpr const char *loop_name = "sink_feed";

    static constexpr int                     search_mode        = MODE_B_SEARCH_SYMMETRIC;
    static constexpr unsigned int            neighbor_type_mask = (unsigned int)SINK_NEIGHBOR_BITFLAG;
    static constexpr mode_b_radius_policy_t  radius_policy      = MODE_B_RADIUS_DEFAULT;

    /* WritePattern describes runner-managed AccumData reduction;
     * uses_ghost_writeback is the orthogonal axis governing direct j-side
     * writes to P[j]/CellP[j]. See runner.h. */
    static constexpr WritePattern   write_pattern   = WritePattern::ActiveReduceOnly;
    static constexpr SidxCacheKind  sidx_cache_kind = SidxCacheKind::AllTypes;

    static constexpr double accum_tolerance = 1e-10;

    static bool is_active(int particle_index) { return sink_feed_is_active(particle_index) != 0; }

    using CallScalars   = SinkFeedCallScalars;
    using ActiveData    = SinkFeedActiveState;
    using AccumData     = SinkFeedOut;
    using DeviceContext = SinkFeedDeviceContext;     /* Phase 4.A.0 extension */

    /* NeighborData carries NON-CONST pointers — sink_feed's pair_kernel
     * does atomic_exchange + atomic_add into neighbor_particle.SwallowID
     * and neighbor_cell->Injected_Sink_Energy. sink_env1's NeighborData
     * uses const pointers because its pair body is read-only. */
    struct NeighborData {
        struct particle_data *neighbor_particle;
        struct gas_cell_data *neighbor_cell;     /* nullptr for non-gas / when no CellP */
        bool oracle_dry_run;                     /* propagated from ctx by load_neighbor;
                                                    pair body suppresses j-side atomic
                                                    writes when true (oracle brute pass). */
#ifdef SINGLE_STAR_MERGE_AWAY_CLOSE_BINARIES
        unsigned char binary_merge_eligible;    /* propagated from ctx.binary_merge_eligible[j]
                                                   by load_neighbor; passed through to the
                                                   merge-away helper in the pair body. */
#endif
    };

    /* Per-active aux passed by caller through args.aux. The Aux struct
     * stays host-only (apply_active_writeback uses it on host, post-kernel).
     * For device-visible per-active inputs use SinkFeedDeviceContext above. */
    struct Aux {
        SinkFeedOut *per_active_accum;      /* [num_active] — host buffer */
        const SinkFeedLocalIn *host_locals; /* [num_active] — host-side mirror,
                                              used by populate_device_context to
                                              fill the UVM ctx.per_active_local */
    };

    static constexpr bool uses_ghost_write_detector = true;
    static constexpr bool uses_ghost_writeback      = true;

    /* ghost_write_detector_begin/end: runner default (calls
     * ::ghost_write_detector_begin(loop_name) / ::ghost_write_detector_end()). */
    static void ghost_writeback_begin      (const struct neighbor_loop_args&,
                                             const struct NeighborLoopPlan&);
    static void ghost_writeback_end        (const struct neighbor_loop_args&,
                                             const struct NeighborLoopPlan&);

    /* Per-active host hooks. */
    static double      search_radius(const neighbor_loop_args& args,
                                      int active_slot, int i);
    static CallScalars populate_call_scalars(const neighbor_loop_args& args);

    /* Phase 4.A.0 device-context lifecycle. populate allocates the
     * ctx.per_active_local UVM array and copies in from args.aux->host_locals.
     * cleanup frees the UVM. Both run on host (pre/post device dispatch). */
    static void populate_device_context(const neighbor_loop_args& args, DeviceContext& ctx);
    static void cleanup_device_context (const neighbor_loop_args& args, DeviceContext& ctx);

    /* Build ActiveData[slot] on device. Reads the host-staged per-active
     * SinkFeedLocalIn from the UVM array attached to dctx in
     * populate_device_context. */
    KOKKOS_INLINE_FUNCTION
    static ActiveData load_active(const DeviceContext& dctx,
                                   int active_slot, int /*i*/,
                                   double h_search,
                                   const CallScalars& scalars)
    {
        ActiveData active;
        active.local            = dctx.per_active_local[active_slot];
        /* Runner-facing top-level fields (Mode B walker reads pos/h_search
         * directly). Copy from the host-staged local so post-drift Mode B
         * walks use the same per-active position the host-fill captured. */
        active.pos[0] = (double)active.local.Pos[0];
        active.pos[1] = (double)active.local.Pos[1];
        active.pos[2] = (double)active.local.Pos[2];
        active.h_search         = h_search;
        active.scalars          = scalars;
        active.origin_local_idx = active_slot;
        active.origin_rank      = -1;
        return active;
    }

    /* Zero accumulator. Byte-zero of the POD AccumData, plus the
     * SINK_REPOSITION_ON_POTMIN sentinel matching the legacy
     * (sink_feed_gpu.cc:217). */
    KOKKOS_INLINE_FUNCTION
    static void zero_accum(AccumData& accum)
    {
        for(size_t b = 0; b < sizeof(accum); b++) ((char*)&accum)[b] = 0;
#ifdef SINK_REPOSITION_ON_POTMIN
        accum.Sink_PotentialMinimumOfNeighbors = (double)SINK_MINPOTVALUE_INIT;
#endif
        accum.mass_markedswallow_scratch = 0.;
    }

    /* Build NeighborData for j. Uses the base ctx fields (P, CellP) which
     * are present in DeviceContext-derived structs via inheritance. */
    KOKKOS_INLINE_FUNCTION
    static NeighborData load_neighbor(const DeviceContext& dctx,
                                       int j,
                                       const IdentitySidecar& /*id*/,
                                       const ActiveData& /*active*/)
    {
        NeighborData neighbor;
        neighbor.neighbor_particle = &dctx.P[j];
        neighbor.neighbor_cell     = (dctx.CellP && dctx.P[j].Type == 0) ? &dctx.CellP[j] : nullptr;
        neighbor.oracle_dry_run    = dctx.oracle_dry_run;
#ifdef SINGLE_STAR_MERGE_AWAY_CLOSE_BINARIES
        neighbor.binary_merge_eligible = dctx.binary_merge_eligible[j];
#endif
        return neighbor;
    }

    /* Oracle brute-pass hook (Phase 4.A.0 contract): runner copies ctx and
     * calls this before the brute evaluate pass to suppress j-side writes.
     * Without it, oracle would atomic_exchange / atomic_add into P[j] /
     * CellP[j] twice (tree + brute) and corrupt additive fields. */
    static void set_oracle_brute_pass(DeviceContext& ctx, bool on)
    {
        ctx.oracle_dry_run = on;
    }

    /* The physics — forwards to the inline pair body above. The pair body
     * does direct j-side atomic writes through neighbor.neighbor_particle
     * and neighbor.neighbor_cell (non-const). Ghost writeback bundle
     * handles reverse-comm. */
    KOKKOS_INLINE_FUNCTION
    static void pair_kernel(const ActiveData& active,
                             const NeighborData& neighbor,
                             AccumData& accum,
                             NoScatter& /*scatter*/)
    {
        sink_feed_pair_kernel(active, *neighbor.neighbor_particle,
                              neighbor.neighbor_cell, accum,
                              neighbor.oracle_dry_run
#ifdef SINGLE_STAR_MERGE_AWAY_CLOSE_BINARIES
                              , neighbor.binary_merge_eligible
#endif
                              );
    }

    static void apply_active_writeback(const neighbor_loop_args& args,
                                        int active_slot, int i,
                                        const AccumData& accum);

    /* Per-field merge — manifest in sink_feed_loop.cc. mass_markedswallow_scratch
     * is intentionally NOT in the manifest (per-i scratch, not aggregable). */
    static void merge_accum(AccumData& local_accum, const AccumData& peer_accum);

    /* ====================================================================
     * ENGINE APPARATUS
     * ==================================================================== */
    using ScatterData    = NoScatter;
    using IdentityFields = NoIdentity;
    using IterControl    = NotIterative;

    /* ====================================================================
     * DIAGNOSTICS — env-gated.
     * ==================================================================== */
    static double compare_accum(const AccumData& local, const AccumData& oracle);
};

#endif /* SINK_PARTICLES */

#endif /* SINK_FEED_LOOP_H */
