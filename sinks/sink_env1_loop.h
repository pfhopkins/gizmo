/* sinks/sink_env1_loop.h — sink "first-pass environment" neighbor loop module.
 *
 * Defines SinkEnv1Spec for the neighbor-loop runner (see
 * mesh/neighbor_loop_runner.h). This is the canonical worked example of
 * a NeighborLoopSpec — copy this file's structure when porting a new
 * physics loop onto the runner.
 *
 * Default per-loop layout is two files: this header carries the Spec
 * contract and any inline (host+device) physics that the runner must
 * inline into the device kernel; sinks/sink_env1_loop.cc carries
 * host-only hooks and diagnostics. Unusual loops may need a reason to
 * split further, but the two-file default keeps the directory readable
 * once 5+ loops are migrated.
 *
 * Layout convention inside SinkEnv1Spec:
 *   PHYSICS BLOCK    — edit when changing the loop's physics
 *   ENGINE APPARATUS — touch only when changing the runner contract
 *   DIAGNOSTICS      — env-gated, declared here, defined in the .cc
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) and Claude for GIZMO.
 */
#ifndef SINK_ENV1_LOOP_H
#define SINK_ENV1_LOOP_H

#include "../declarations/allvars.h"

#ifdef SINK_PARTICLES

/* ----------------------------------------------------------------------------
 * Compile-time guard: SINGLE_STAR_SINK_DYNAMICS + SINK_GRAVCAPTURE_GAS j-side
 * SwallowTime atomic write is not yet wired through the runner's
 * SymmetricPairScatter writeback channel. This is the sole guard for that
 * combination across the sink_env1 stack — do not narrow it.
 * --------------------------------------------------------------------------*/
#if defined(SINK_GRAVCAPTURE_GAS) && defined(SINGLE_STAR_SINK_DYNAMICS)
#error "SinkEnv1Spec: SINGLE_STAR_SINK_DYNAMICS + SINK_GRAVCAPTURE_GAS j-side SwallowTime atomic write is not yet wired through the runner's SymmetricPairScatter channel. Do not enable SINGLE_STAR_SINK_DYNAMICS with SINK_GRAVCAPTURE_GAS until the writeback hook lands."
#endif

#include "../mesh/neighbor_loop_runner.h"
#include "../mesh/mode_b_local_walker.h"      /* MODE_B_SEARCH_*, MODE_B_RADIUS_* */
#include "sinks_gpu_decls.h"                  /* struct sink_env_gpu_out */
/* NOTE: caller translation units must include "../mesh/kernel.h" BEFORE
 * this header. kernel.h has no include guards (it defines static inline
 * kernel_main, used by the pair body below). The runner and the sink
 * loop's .cc both include kernel.h first; new callers must do the same. */

/* Make sure KOKKOS_INLINE_FUNCTION is defined; if Kokkos is unavailable
 * (host-only TUs), fall back to plain inline. */
#ifndef KOKKOS_INLINE_FUNCTION
#define KOKKOS_INLINE_FUNCTION inline
#endif

/* Forward-decl from sinks/sink.h (active-particle predicate). */
int sink_isactive(int i);

/* ============================================================================
 * Per-pair physics types (file scope; PascalCase).
 *
 * SinkEnv1Spec re-exports these via `using CallScalars = SinkEnv1CallScalars;`
 * and `using ActiveData = SinkEnv1ActiveState;`. File-scope so the inline
 * pair body below can reference them without a circular dependency on
 * SinkEnv1Spec.
 * ========================================================================== */

struct SinkEnv1CallScalars {
    NlrCommonScalars common;            /* cf_atime, cf_a2inv, cf_a3inv, G, ... */
    double           sink_radius_grav;  /* SinkParticle_GravityKernelRadius */
};

/* Active-particle state passed into the pair body. Trivially copyable for
 * byte-level MPI transfer in the Mode B remote path. Conditional fields
 * mirror the same #ifdefs as the pair body so payload size matches per
 * build config.
 *
 * h_search MUST come from the runner's per-active radii staging (driven
 * by SinkEnv1Spec::search_radius), NOT from P[i].KernelRadius — they
 * happen to be equal in the current code but the pair body is
 * contractually keyed off the caller-supplied radius. */
struct SinkEnv1ActiveState {
    Vec3<double>  pos;                  /* P[i].Pos */
    Vec3<double>  vel;                  /* P[i].Vel */
    MyIDType      id;                   /* P[i].ID — for self-skip predicate */
    double        h_search;             /* per-active radius */
    double        ags_h;                /* AGS_KernelRadius if defined, else sink_radius_grav */
#if defined(SINK_GRAVCAPTURE_GAS) || (SINK_GRAVACCRETION == 8)
    double        mass;                 /* active sink mass (boundedness/Bondi paths) */
#endif
#if defined(SINK_GRAVCAPTURE_FIXEDSINKRADIUS)
    double        sink_radius;
#endif
#if defined(SINK_RETURN_ANGMOM_TO_GAS)
    Vec3<double>  sink_angmom;          /* Sink_Specific_AngMom */
#endif
    int           origin_local_idx;     /* index into requester's per_active_accum */
    int           origin_rank;          /* requester's MPI rank (Mode B remote) */
    SinkEnv1CallScalars scalars;        /* per-call cosmology + gravity */
};

/* ============================================================================
 * Inline pair body — single source of truth for sink_env1 per-pair physics.
 * Called from SinkEnv1Spec::pair_kernel, used by all three runner paths
 * (Mode A GPU neighbor list, Mode B local host walker, Mode B remote
 * peer-to-peer).
 *
 * Caller is responsible for the per-active early-return (Mass<=0 || h<=0).
 *
 * J-side writes:
 *   - SINGLE_STAR_SINK_DYNAMICS: neighbor_particle.SwallowTime is atomically min'd. The
 *     SINGLE_STAR_SINK_DYNAMICS + SINK_GRAVCAPTURE_GAS combination is currently #error-guarded
 *     above (the runner's SymmetricPairScatter channel does not yet carry
 *     this through).
 *   - SINK_GRAVCAPTURE_GAS: accum.mass_to_swallow_edd accumulates; no j-write.
 * ========================================================================== */

KOKKOS_INLINE_FUNCTION
static void sink_env1_pair_kernel(const SinkEnv1ActiveState& active,
                                  const struct particle_data& neighbor_particle,
                                  const struct gas_cell_data* neighbor_cell,
                                  struct sink_env_gpu_out& accum)
{
    /* Self-skip / mass / type-5 filter. */
    if(neighbor_particle.Mass <= 0 || neighbor_particle.Type == 5 || neighbor_particle.ID == active.id) return;

    const double h_i      = active.h_search;
    const double hinv     = 1.0 / h_i;
    const double hinv3    = hinv * hinv * hinv;
    const double ags_h_i  = active.ags_h;

    /* dP, dv with periodic wrap. Sign convention: j - i. */
    Vec3<double> dP;
    dP[0] = (double)neighbor_particle.Pos[0] - active.pos[0];
    dP[1] = (double)neighbor_particle.Pos[1] - active.pos[1];
    dP[2] = (double)neighbor_particle.Pos[2] - active.pos[2];
    Vec3<double> dv;
    dv[0] = (double)neighbor_particle.Vel[0] - active.vel[0];
    dv[1] = (double)neighbor_particle.Vel[1] - active.vel[1];
    dv[2] = (double)neighbor_particle.Vel[2] - active.vel[2];
    nearest_xyz(dP, -1);
    NGB_SHEARBOX_BOUNDARY_VELCORR_(active.pos, neighbor_particle.Pos, dv, -1);

    const double wt = (double)neighbor_particle.Mass;

#ifdef SINK_REPOSITION_ON_POTMIN
    if(neighbor_particle.Type != 0 && neighbor_particle.Type != 5) {
        double rfac  = dP.norm_sq() * (10.0 / (h_i * h_i)
                                     + 0.1 / (active.scalars.sink_radius_grav * active.scalars.sink_radius_grav));
        double wtfac = wt / (1.0 + rfac);
        if((MyFloat)neighbor_particle.Mass > accum.DF_mmax_particles) accum.DF_mmax_particles = (MyFloat)neighbor_particle.Mass;
        for(int kv = 0; kv < 3; kv++) accum.DF_mean_vel[kv] += wtfac * dv[kv];
        accum.DF_rms_vel += wtfac;
        accum.DF_rms_vel += wtfac;
        accum.DF_rms_vel += wtfac;
    }
#endif

    if(neighbor_particle.Type == 0) {
        accum.Mgas_in_Kernel += wt;
        if(neighbor_cell) { accum.Sink_SurroudingGasInternalEnergy += wt * neighbor_cell->InternalEnergy; }
        Vec3<double> J_gas = cross(dP, dv);
        for(int kv = 0; kv < 3; kv++) accum.Jgas_in_Kernel[kv] += wt * J_gas[kv];
#if defined(SINK_OUTPUT_MOREINFO)
        if(neighbor_cell) { accum.Sfr_in_Kernel += neighbor_cell->Sfr; }
#endif
#if (SINK_GRAVACCRETION >= 5) || defined(SINGLE_STAR_SINK_DYNAMICS) || defined(SINGLE_STAR_TIMESTEPPING)
        for(int kv = 0; kv < 3; kv++) accum.Sink_SurroundingGasVel[kv] += wt * dv[kv];
#endif
#ifdef JET_DIRECTION_FROM_KERNEL_AND_SINK
        for(int kv = 0; kv < 3; kv++) accum.Sink_SurroundingGasCOM[kv] += wt * dP[kv];
#endif
#if defined(SINK_RETURN_ANGMOM_TO_GAS) || defined(SINK_RETURN_BFLUX)
        {
            double u_wb = dP.norm() / DMAX(h_i, (double)neighbor_particle.KernelRadius);
            double wk_wb = 0, dwk_wb = 0;
            if(u_wb < 1) { kernel_main(u_wb, 1.0, 1.0, &wk_wb, &dwk_wb, -1); }
#if defined(SINK_RETURN_ANGMOM_TO_GAS)
            double r2j = dP.norm_sq();
            double Lrj = dot(active.sink_angmom, dP);
            Vec3<double> Ang_pass = active.sink_angmom * r2j - dP * Lrj;
            for(int kv = 0; kv < 3; kv++)
                accum.angmom_prepass_sum_for_passback[kv] += wk_wb * wt * Ang_pass[kv];
#endif
#if defined(SINK_RETURN_BFLUX)
            accum.kernel_norm_topass_in_swallowloop += wk_wb;
#endif
        }
#endif
#if (SINK_GRAVACCRETION == 8)
        if(neighbor_cell) {
            double u_h = dP.norm() / h_i;
            double wk_h = 0, dwk_h = 0;
            if(u_h < 1) { kernel_main(u_h, hinv3, hinv3 * hinv, &wk_h, &dwk_h, -1); }
            double rj = u_h * h_i * active.scalars.common.cf_atime;
            double csj = neighbor_cell->effective_soundspeed();
            double vdotrj = -dot(dP, dv);
            double vr_mdot = 4 * M_PI * wt * (wk_h * active.scalars.common.cf_a3inv) * rj * vdotrj;
            if(rj < active.scalars.sink_radius_grav * active.scalars.common.cf_atime) {
                double bondi_mdot = 4 * M_PI * active.scalars.common.newton_G * active.scalars.common.newton_G * active.mass * active.mass
                    / pow(csj * csj + dv.norm_sq() * active.scalars.common.cf_a2inv, 1.5)
                    * wt * (wk_h * active.scalars.common.cf_a3inv);
                vr_mdot = DMAX(vr_mdot, bondi_mdot);
                accum.hubber_mdot_bondi_limiter += bondi_mdot;
            }
            accum.hubber_mdot_vr_estimator    += vr_mdot;
            accum.hubber_mdot_disk_estimator  += wt * wk_h * sqrt(rj) / (neighbor_cell->Density * csj * csj);
        }
#endif
    } else if(neighbor_particle.Type == 4 ||
              ((neighbor_particle.Type == 2 || neighbor_particle.Type == 3) && !active.scalars.common.comoving_integration_on)) {
        accum.Mstar_in_Kernel += wt;
        Vec3<double> J_star = cross(dP, dv);
        for(int kv = 0; kv < 3; kv++) accum.Jstar_in_Kernel[kv] += wt * J_star[kv];
    } else {
        accum.Malt_in_Kernel += wt;
        Vec3<double> J_alt = cross(dP, dv);
        for(int kv = 0; kv < 3; kv++) accum.Jalt_in_Kernel[kv] += wt * J_alt[kv];
    }

    /* SINK_GRAVCAPTURE_GAS path — boundedness check + SwallowID-based
     * mass-marked-swallow accumulation. SINGLE_STAR_SINK_DYNAMICS is
     * compile-guarded out at top of header (j-side SwallowTime atomic write
     * not yet carried through SymmetricPairScatter channel). */
#ifdef SINK_GRAVCAPTURE_GAS
#ifdef GRAIN_FLUID
    if(neighbor_particle.Mass > 0 && (neighbor_particle.Type == 0 || ((1<<neighbor_particle.Type) & GRAIN_PTYPES)))
#else
    if(neighbor_particle.Mass > 0 && neighbor_particle.Type == 0)
#endif
    {
        double dr_code = dP.norm();
        double vrel = dv.norm() / active.scalars.common.cf_atime;
#if defined(MAGNETIC) && defined(GRAIN_LORENTZFORCE)
        if((1<<neighbor_particle.Type) & GRAIN_PTYPES) {
            Vec3<double> B_vec; for(int kv = 0; kv < 3; kv++) B_vec[kv] = neighbor_particle.Gas_B[kv];
            double vrel_dot = dot(dv, B_vec), bmag2 = B_vec.norm_sq();
            vrel = (fabs(vrel_dot) / sqrt(bmag2)) / active.scalars.common.cf_atime;
        }
#endif
        struct gas_cell_data neighbor_cell_local;
        if(neighbor_particle.Type == 0 && neighbor_cell) { neighbor_cell_local = *neighbor_cell; }
        else { /* zero-init */
            for(size_t b = 0; b < sizeof(neighbor_cell_local); b++) {
                ((char*)&neighbor_cell_local)[b] = 0;
            }
        }
        double vbound = sink_vesc_gpu(neighbor_particle, neighbor_cell_local, active.mass, dr_code, ags_h_i);
        if(vrel < vbound) {
            double local_sink_radius = active.scalars.sink_radius_grav;
#ifdef SINK_GRAVCAPTURE_FIXEDSINKRADIUS
            local_sink_radius = active.sink_radius;
            double spec_mom = dot(dv, dP);
            double r2 = dP.norm_sq();
            spec_mom = r2*vrel*vrel - spec_mom*spec_mom*active.scalars.common.cf_a2inv;
            if(spec_mom >= active.scalars.common.newton_G * (active.mass + (double)neighbor_particle.Mass) * local_sink_radius) { return; }
#endif
            if(sink_check_boundedness_gpu(neighbor_particle, neighbor_cell_local, vrel, vbound, dr_code, local_sink_radius) == 1) {
                if(neighbor_particle.SwallowID < active.id) { accum.mass_to_swallow_edd += (MyFloat)neighbor_particle.Mass; }
            }
        }
    }
#endif  /* SINK_GRAVCAPTURE_GAS */
}

/* ============================================================================
 * SinkEnv1Spec — the NeighborLoopSpec contract for sink_env1.
 *
 * NeighborData lifetime contract:
 *   neighbor.neighbor_particle MUST point into ctx.P (= P_gpu UVM array
 *     under Mode A; = P_host directly in Mode B walker).
 *   neighbor.neighbor_cell MUST be either nullptr OR point into ctx.CellP
 *     (UVM under Mode A).
 *   Pointers are valid only for the duration of one pair_kernel call.
 *   DO NOT store; DO NOT cross arena scope.
 * ========================================================================== */

struct SinkEnv1Spec {
    /* ====================================================================
     * PHYSICS BLOCK — edit when changing the loop's physics
     * ==================================================================== */

    /* (1) Loop identity. Drives env-var prefixes (GIZMO_SINK_ENV1_*) and
     *     PHASE0_NLR `loop=` field. */
    static constexpr const char *loop_name = "sink_env1";

    /* (2) Search policy. */
    static constexpr int                     search_mode        = MODE_B_SEARCH_SYMMETRIC;
    static constexpr unsigned int            neighbor_type_mask = (unsigned int)SINK_NEIGHBOR_BITFLAG;
    static constexpr mode_b_radius_policy_t  radius_policy      = MODE_B_RADIUS_DEFAULT;

    /* (3) Writeback policy. */
    static constexpr WritePattern   write_pattern   = WritePattern::ActiveReduceOnly;
    static constexpr SidxCacheKind  sidx_cache_kind = SidxCacheKind::AllTypes;

    /* (4) Tolerances.
     *
     * Mode B vs Brute oracle: same pair_kernel, same candidate SET, but
     * DIFFERENT iteration order (tree-walk returns walk-order; brute returns
     * ascending P[] index order). Per-active accumulators sum the same MyFloat
     * contributions in different orders -> double-precision floor of
     * ~N*eps_machine relative residual, plus possible cancellation
     * amplification on small-magnitude fields. 1e-10 is sharp enough to catch
     * real bugs, loose enough to not trip on summation-reorder noise. */
    static constexpr double accum_tolerance = 1e-10;

    /* (5) Active-particle predicate, passed by the caller to
     *     nlr_build_active_list. */
    static bool is_active(int particle_index) { return sink_isactive(particle_index) != 0; }

    /* (6) Per-pair physics types — file-scope structs above. */
    using CallScalars   = SinkEnv1CallScalars;
    using ActiveData    = SinkEnv1ActiveState;
    using AccumData     = struct sink_env_gpu_out;

    struct NeighborData {
        const struct particle_data *neighbor_particle;
        const struct gas_cell_data *neighbor_cell;     /* nullptr for non-gas / when no CellP */
    };

    /* (7) Per-active aux passed by caller through neighbor_loop_args::aux.
     *     Recover with nlr_aux<SinkEnv1Spec>(args) inside hooks. */
    struct Aux {
        struct sink_env_gpu_out *per_active_accum;   /* [num_active] — host buffer */
    };

    /* (8) Imported-ghost lifecycle traits.
     *
     * Three independent traits, each gating a hook-method pair below. The
     * runner calls each pair only when (a) the trait is true AND (b) the
     * chosen path imports ghosts (nlr_path_uses_imported_ghosts(plan.path)).
     *
     * For SinkEnv1Spec specifically:
     *   - detector: ON (catches illegal kernel writes to imported ghosts;
     *     compiles to no-op outside GIZMO_GPU_ARENA_DEBUG)
     *   - ghost_writeback: OFF (ActiveReduceOnly with no j-side writes
     *     outside SINGLE_STAR_SINK_DYNAMICS; the regular writeback channel
     *     has nothing to do)
     *   - sidechannel_writeback: ON only under SINGLE_STAR_SINK_DYNAMICS
     *     (wraps ghost_writeback_zero_swallowtime / _swallowtime).
     *
     * The three-channel split is a transitional shape; consolidation into
     * a single ghost-writeback scaffold is queued for the runner-template-
     * hardening pass. */
    static constexpr bool uses_ghost_write_detector  = true;
    static constexpr bool uses_ghost_writeback       = false;
#ifdef SINGLE_STAR_SINK_DYNAMICS
    static constexpr bool uses_sidechannel_writeback = true;
#else
    static constexpr bool uses_sidechannel_writeback = false;
#endif

    /* Hook method declarations. The runner invokes each only when the
     * corresponding `uses_*` trait above is true AND the chosen path
     * imports ghosts. On Mode A path the runner preserves this exact order:
     *
     *   gizmo_request_filtered_ghost_import (in runner)
     *   ghost_write_detector_begin
     *   ghost_writeback_begin                (no-op for SinkEnv1Spec)
     *   sidechannel_writeback_begin          (SINGLE_STAR_SINK_DYNAMICS-gated)
     *   <run_mode_a kernel>
     *   sidechannel_writeback_end            (reverse order)
     *   ghost_writeback_end
     *   ghost_write_detector_end
     *   ghost_exchange_cleanup               (in runner; NTask>1 only)
     */
    static void ghost_write_detector_begin (const struct neighbor_loop_args&,
                                             const struct NeighborLoopPlan&);
    static void ghost_write_detector_end   (const struct neighbor_loop_args&,
                                             const struct NeighborLoopPlan&);
    static void ghost_writeback_begin      (const struct neighbor_loop_args&,
                                             const struct NeighborLoopPlan&);
    static void ghost_writeback_end        (const struct neighbor_loop_args&,
                                             const struct NeighborLoopPlan&);
    static void sidechannel_writeback_begin(const struct neighbor_loop_args&,
                                             const struct NeighborLoopPlan&);
    static void sidechannel_writeback_end  (const struct neighbor_loop_args&,
                                             const struct NeighborLoopPlan&);

    /* (9) Per-active host hooks (pre-arena epoch). */

    /* Per-active search radius from P[i].KernelRadius. Pre-arena, pre-drift. */
    static double      search_radius(const neighbor_loop_args& args,
                                      int active_slot, int i);

    /* Per-call scalars captured into a POD. */
    static CallScalars populate_call_scalars(const neighbor_loop_args& args);

    /* (10) Per-active and per-pair device hooks. KOKKOS_INLINE_FUNCTION =
     *      callable from Mode A device kernel and Mode B/Brute host walker. */

    /* Build ActiveData[slot] from device-visible state (post-neighbor-list
     * build). Byte-exact reconstruction of the legacy GPU lambda's q-packing
     * block. */
    KOKKOS_INLINE_FUNCTION
    static ActiveData load_active(const NeighborLoopDeviceContextBase& ctx,
                                   int active_slot, int i,
                                   double h_search,
                                   const CallScalars& scalars)
    {
        ActiveData active;
        active.pos      = ctx.P[i].Pos;
        active.vel      = ctx.P[i].Vel;
        active.id       = ctx.P[i].ID;
        active.h_search = h_search;
#if (ADAPTIVE_GRAVSOFT_FORALL & 32)
        active.ags_h    = (double)ctx.P[i].AGS_KernelRadius;
#else
        active.ags_h    = scalars.sink_radius_grav;
#endif
#if defined(SINK_GRAVCAPTURE_GAS) || (SINK_GRAVACCRETION == 8)
        active.mass     = (double)ctx.P[i].Mass;
#endif
#if defined(SINK_GRAVCAPTURE_FIXEDSINKRADIUS)
        active.sink_radius = (double)ctx.P[i].SinkRadius;
#endif
#if defined(SINK_RETURN_ANGMOM_TO_GAS)
        for(int kv = 0; kv < 3; kv++) active.sink_angmom[kv] = ctx.P[i].Sink_Specific_AngMom[kv];
#endif
        active.origin_local_idx = active_slot;
        active.origin_rank      = -1;          /* device — rank N/A in lambda */
        active.scalars          = scalars;
        return active;
    }

    /* Zero accumulator. Byte-zero of the POD AccumData. */
    KOKKOS_INLINE_FUNCTION
    static void zero_accum(AccumData& accum)
    {
        for(size_t b = 0; b < sizeof(accum); b++) ((char*)&accum)[b] = 0;
    }

    /* Build NeighborData for j. ctx.P/ctx.CellP are UVM under Mode A; in
     * Mode B walker they are P_host/CellP_host — same shape. */
    KOKKOS_INLINE_FUNCTION
    static NeighborData load_neighbor(const NeighborLoopDeviceContextBase& ctx,
                                       int j,
                                       const IdentitySidecar& /*id*/,
                                       const ActiveData& /*active*/)
    {
        NeighborData neighbor;
        neighbor.neighbor_particle = &ctx.P[j];
        neighbor.neighbor_cell     = (ctx.CellP && ctx.P[j].Type == 0) ? &ctx.CellP[j] : nullptr;
        return neighbor;
    }

    /* The physics — forwards to the inline pair body above. */
    KOKKOS_INLINE_FUNCTION
    static void pair_kernel(const ActiveData& active,
                             const NeighborData& neighbor,
                             AccumData& accum,
                             NoScatter& /*scatter*/)
    {
        sink_env1_pair_kernel(active, *neighbor.neighbor_particle, neighbor.neighbor_cell, accum);
    }

    /* (11) Host writebacks (post-dispatch). */

    /* Copy AccumData into args.aux->per_active_accum[active_slot]. The
     * caller's scatter loop reads per_active_accum and applies its own
     * physics-specific reductions into SinkTempInfo. */
    static void apply_active_writeback(const neighbor_loop_args& args,
                                        int active_slot, int i,
                                        const AccumData& accum);

    /* Per-field merge of a peer-rank reply (peer_accum) into a local
     * accumulator (local_accum). Used by run_mode_b_remote at the cross-rank
     * boundary; per-field reduction op MUST match the pair_kernel writes
     * (sum for additive fields, MAX for DF_mmax_particles). Includes all
     * #ifdef-gated optional fields. */
    static void merge_accum(AccumData& local_accum, const AccumData& peer_accum);

    /* ====================================================================
     * ENGINE APPARATUS — touch only when changing the runner contract
     * ==================================================================== */

    using ScatterData    = NoScatter;
    using IdentityFields = NoIdentity;
    using IterControl    = NotIterative;

    /* ====================================================================
     * DIAGNOSTICS — env-gated; safe to ignore for physics edits.
     *   PERMANENT_DIAGNOSTIC: compare_accum (oracle gate)
     *   SPIKE_DIAGNOSTIC:     diagnostic_dump_active, diagnostic_dump_neighbor_list
     *                          (cross-validation probes; scheduled retire
     *                          after 3d ports)
     * Definitions live in sink_env1_loop.cc.
     * ==================================================================== */

    static double compare_accum(const AccumData& local, const AccumData& oracle);
    static void   diagnostic_dump_active(const ActiveDumpView<SinkEnv1Spec>& v);
    static void   diagnostic_dump_neighbor_list(const NeighborListDumpView<SinkEnv1Spec>& v);
};

#endif /* SINK_PARTICLES */

#endif /* SINK_ENV1_LOOP_H */
