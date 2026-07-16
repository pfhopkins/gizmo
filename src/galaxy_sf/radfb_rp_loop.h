/* galaxy_sf/radfb_rp_loop.h — RadFBRPSpec for the runner-template port of
 * radiation_pressure_winds_gpu (Phase 4 / Wave 3 / radfb_local).
 *
 * Iterative two-pass scatter loop: active Type-4 (stellar) particles —
 * cosmo-aware (+ Types 2/3 non-cosmo) — scatter UV / IR / jet radiation-
 * pressure momentum kicks into surrounding gas neighbors.
 *
 *   iter 0  — accumulate wt_sum = Σ h_j² over valid neighbors into
 *             AccumData; no j-side writes; ghost-writeback hooks no-op
 *             (gated by Aux::iter_index).
 *   iter 1  — apply kicks using staged wt_sum (copied from IterScratch
 *             into RadFBRPLocalIn.wt_sum by after_iter_global, post-iter
 *             staging hook); atomic_add to P[j].Vel, CellP[j].VelPred,
 *             P[j].dp; accumulate source-side jet_momentum_used into
 *             AccumData; ghost-writeback bundle fires normally.
 *
 * SSOT for radfb_rp physics types and per-pair kernel — supersedes the
 * pre-port `radfb_local_functions.h` and the legacy `radfb_local_gpu.cc`,
 * both retired in the radfb_local cleanup commit.
 *
 * RNG stream is intentionally NOT byte-identical to legacy. Legacy keyed
 * on pair-ordinal `nn` (gone with the runner API). New stream keys on
 * (loc.ID ^ Pj.ID) plus a per-loop XOR shift RADFBRP_RNG_SHIFT (collision-
 * audited against feedback_rng_loop_uniqueness.md). Order-independent +
 * oracle-safe; population statistics identical. See
 * OPEN_3d_radfb_local_design.md "RNG stream" section.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) and Claude for GIZMO. */
#ifndef RADFB_RP_LOOP_H
#define RADFB_RP_LOOP_H

/* Kokkos_Core must precede allvars.h (its macros may conflict with stdlib
 * names). */
#include <Kokkos_Core.hpp>

#include "../declarations/allvars.h"
#include "../declarations/multifluid_helpers.h"

#ifdef GALSF_FB_FIRE_RT_LOCALRP

/* NOTE: caller TUs must `#include "../mesh/kernel.h"` BEFORE this header
 * (no include guards on kernel.h). Same pattern as thermal_fb_loop.h. */
#include "../mesh/neighbor_loop_runner.h"
#include "../mesh/mode_b_local_walker.h"        /* MODE_B_SEARCH_*, MODE_B_RADIUS_* */
#include "../declarations/gpu_rng.h"            /* gizmo_gpu_rand_double */
#include "../radiation/rt_functions.h"          /* rt_kappa, RT_FREQ_BIN_FIRE_* */

/* No KOKKOS_INLINE_FUNCTION fallback here — Kokkos_Core.hpp is included
 * unconditionally above. This Spec carries device-callable pair-kernel
 * accessors; misordered includes must compile-fail loudly, not silently
 * resolve to host-only `inline`. Same convention as neighbor_loop_runner.h. */

/* Per-loop RNG XOR shift — STREAM INTENTIONALLY CHANGED from legacy.
 *
 * Legacy radfb_local_functions.h keyed RNG on pair-ordinal `nn`:
 *     gizmo_gpu_rand_double(loc.ID ^ kp[j].ID, All.Ti_Current + 3 + nn).
 * The runner pair_kernel API does NOT pass `nn`, and Mode-A tree vs
 * Mode-B brute candidate orderings differ — so a `nn`-keyed RNG would
 * (a) require new runner plumbing AND (b) break the oracle (Mode A vs
 * Mode B would draw different per-pair randoms).
 *
 * New stream is order-independent + oracle-safe:
 *     gizmo_gpu_rand_double(loc.ID ^ Pj.ID,
 *                            scalars.rng_ti_counter ^ RADFBRP_RNG_SHIFT).
 * Population statistics identical to legacy (per-pair stochastic
 * acceptance/rejection rate unchanged; only the per-pair seed function
 * changes). Phil 2026-05-16 approved.
 *
 * Collision audit (2026-05-16): the only other 64-bit per-loop RNG
 * shift in any ported _loop file is sinks/sink_feed_loop.cc:77 =
 * 0xfeed5117ULL. Picked 0xfadfb1234567890dULL — distinct from
 * sink_feed and from every legacy small-integer shift surveyed in
 * feedback_rng_loop_uniqueness.md (49531, 121, 17, 11, 3, 7). */
static constexpr uint64_t RADFBRP_RNG_SHIFT = 0xfadfb1234567890dULL;

/* Age threshold above which a star contributes nothing — matches the
 * legacy `radfb_rp_age_threshold` in radfb_local_gpu.cc. */
static inline double radfb_rp_age_threshold_value(void)
{
#if defined(SINGLE_STAR_SINK_DYNAMICS) || (GALSF_FB_FIRE_STELLAREVOLUTION > 2)
    return 1.0e10;
#else
    return 0.15;
#endif
}

/* Host-side helpers (defined in radfb_rp_loop.cc).
 *
 * radfb_rp_local_fill — runs the CPU stochastic gate per active source;
 *   returns 1 (fires this step, fills *loc + *src_radius_out) or 0
 *   (skip). Called by the TOPLEVEL caller — NOT by Spec::is_active and
 *   NOT by nlr_build_active_list. The toplevel hands the runner only
 *   firing sources; Spec::is_active is a belt-and-suspenders defensive
 *   predicate (cosmo-aware Type filter), never the stochastic gate.
 *
 * radfb_rp_build_call_scalars — routes All.* reads through
 *   nlr_host_all_ptr() (feedback_all_dev_trap_host_side rule). */
struct RadFBRPLocalIn;
struct RadFBRPCallScalars;
int  radfb_rp_local_fill(int i,
                          struct particle_data *P_host,
                          struct gas_cell_data *CellP_host,
                          struct RadFBRPLocalIn *loc,
                          double *src_radius_out);
RadFBRPCallScalars radfb_rp_build_call_scalars(void);

/* ============================================================================
 * Per-pair physics types.
 *
 * RadFBRPCallScalars carries the cosmology + run-invariant unit conversion
 * factors needed by the inline pair kernel. Direct local-RP scalar reads
 * route through scalars; canonical rt_kappa retains legacy transitive All.*
 * reach (matches legacy radfb_local_gpu.cc; cleanup tracked in
 * OPEN_rt_kappa_per_cell_cache.md).
 * ========================================================================== */
struct RadFBRPCallScalars {
    NlrCommonScalars common;             /* cf_atime, cf_a2inv, cf_a3inv, ... */

    double           rp_renorm;          /* All.RP_Local_Momentum_Renormalization */
    double           dv_cap_codeunits;   /* 1.0e4 / UNIT_VEL_IN_KMS */
    uint64_t         rng_ti_counter;     /* (uint64_t)All.Ti_Current — XORed
                                          * with RADFBRP_RNG_SHIFT at the
                                          * gpu_rand_double call site */
#ifdef SINK_WIND_SPAWN
    MyIDType         spawned_wind_cell_id;
#endif
};

/* Per-source input pack — host-filled by radfb_rp_local_fill in the toplevel
 * pre-runner build, UVM-staged in populate_device_context, read on device by
 * load_active. Trivially copyable.
 *
 * `wt_sum` is overwritten between iters: zero at iter 0 entry (carried
 * through populate_device_context); refreshed by after_iter_global (the
 * post-iter staging hook) from IterScratch.wt_sum after iter 0's after_iter
 * stashes it there. iter 1's device kernel reads it as the kick-
 * normalization denominator. */
struct RadFBRPLocalIn {
    Vec3<MyDouble> Pos;
    MyFloat        KernelRadius;          /* RtauMax — neighbor-search radius */
    MyFloat        dE_over_c;             /* total photon momentum budget this step */
    MyFloat        f_lum_ion;             /* ionizing luminosity fraction */
    MyIDType       ID;                    /* source particle ID, for RNG */
    /* Staged Σ h_j². Iter 0 = 0 at populate; written by after_iter_global
     * from drv.scratch_uvm[sg][slot].wt_sum before iter 1 dispatch. DOUBLE,
     * not MyFloat — legacy computes the denominator in double and narrowing
     * here would be a silent precision downgrade (directive 5). */
    double         wt_sum;
#if (GALSF_FB_FIRE_STELLAREVOLUTION <= 2)
    MyFloat        delta_v_imparted_rp;   /* per-kick target velocity (stochastic threshold) */
#endif
#if (GALSF_FB_FIRE_STELLAREVOLUTION > 2)
    MyFloat        jet_momentum_tocouple; /* extra jet momentum budget (0 if none) */
#endif
};

/* AccumData — iter-0 and iter-1 share the same struct, but only one slot
 * is meaningful per iter:
 *   iter 0: wt_sum used (Σ h_j²); jet_momentum_used = 0.
 *   iter 1: jet_momentum_used used; wt_sum field is incidentally zero
 *           since iter 1's pair_kernel does not touch it.
 * Zeroed per outer iter by zero_accum. */
struct RadFBRPAccum {
    double   wt_sum;             /* iter 0 output */
    MyDouble jet_momentum_used;  /* iter 1 output (jet branch only) */
};

/* IterScratch — host-only per-active state, carries iter-0's accumulated
 * wt_sum into the iter-0→iter-1 staging bridge. Flow:
 *   iter 0 device kernel:   accum.wt_sum += h_j² (per pair).
 *   iter 0 after_iter (per active, runs for BOTH production AND oracle):
 *                            ctx.scratch.wt_sum = accum.wt_sum  (status-only
 *                            otherwise — no P/CellP writes; oracle-safe).
 *   iter 0 after_iter_global (post-iter staging hook; mutates per_active_local
 *                              for both ctx and ctx_oracle; NO physics writes):
 *                            drv.scratch_uvm[sg][slot].wt_sum →
 *                              drv.ctx.per_active_local[slot].wt_sum
 *                            drv.scratch_oracle_uvm[sg][slot].wt_sum →
 *                              drv.ctx_oracle.per_active_local[slot].wt_sum
 *                            (+ Kokkos::fence() before iter-1 device dispatch).
 *   iter 1 device kernel:    reads loc.wt_sum (staged); applies kicks.
 * Carries through IterScratch (the runner's intended per-active iter-state)
 * — not through AccumData directly, which is less stable if the runner
 * zeros/repurposes accum between hooks. */
struct RadFBRPIterScratch {
    double wt_sum;
};

/* DeviceContext extension. Holds the UVM pointer to per-active RadFBRPLocalIn
 * (toplevel-owned host buffer; runner copies into UVM in populate). Carries
 * the oracle-suppression flag and a per-iter snapshot of iter_index (mirrored
 * from Aux::iter_index by reset_per_iter_device_context so device lambdas
 * have access — Aux is host-only). Trivially copyable; runner captures by
 * value into Kokkos device lambdas. */
struct RadFBRPDeviceContext : NeighborLoopDeviceContextBase {
    RadFBRPLocalIn *per_active_local;     /* UVM, [num_active]; wt_sum field
                                           * rewritten between iters by
                                           * after_iter_global (post-iter
                                           * staging hook) from IterScratch */
    bool            oracle_dry_run;
    int             iter_index_snapshot;  /* mirror of Aux::iter_index, refreshed
                                           * per iter; read by load_active on device */
};

/* Active-particle state passed into the pair body. `pos` and `h_search` are
 * top-level so the runner's Mode B walker reads them directly. */
struct RadFBRPActiveState {
    Vec3<double>         pos;
    double               h_search;
    int                  iter_index;     /* set in load_active from ctx.iter_index */
    RadFBRPLocalIn       local;
    RadFBRPCallScalars   scalars;
};

/* ============================================================================
 * Inline pair body — iter 1 kick application.
 *
 * Ported from the legacy `radfb_rp_pair_kick` in radfb_local_functions.h,
 * with All.* reads scrubbed (route through scalars) and RNG stream changed
 * to the order-independent (loc.ID ^ Pj.ID, rng_ti_counter ^ shift) form
 * documented above.
 *
 * Called by RadFBRPSpec::pair_kernel ONLY on iter 1 (iter 0 path accumulates
 * wt_sum without invoking this function). `oracle_dry_run` short-circuits
 * j-side atomic writes (oracle brute pass mustn't double-deposit).
 * ========================================================================== */
KOKKOS_INLINE_FUNCTION
static void radfb_rp_pair_kick(
    const RadFBRPLocalIn& loc,
    const RadFBRPCallScalars& scalars,
    struct particle_data& Pj,
    struct gas_cell_data& Cj,
    double r2,
    const Vec3<double>& dp_ij,
    bool oracle_dry_run,
    RadFBRPAccum& out)
{
    if (Pj.Type != 0) return;
#ifdef HYDRO_MULTIFLUID_DM
    if (Pj.FluidType == FLUID_DM) return; /* skip dark-fluid neighbors */
#endif
    double Mass_j = (double)Pj.Mass;
    if (Mass_j <= 0 || r2 <= 0) return;
    if (loc.wt_sum <= 0) return;

    double h_j = Pj.Get_Particle_Size();
    double wk  = (h_j * h_j) / (double)loc.wt_sum;
    if (wk <= 0) return;

    double dE = (double)loc.dE_over_c;

    /* --- single-scattering kick (UV/optical absorbed fraction) --- */
    double dv_ss = wk * dE / Mass_j;

#if (GALSF_FB_FIRE_STELLAREVOLUTION > 2)
    /* estimate absorbed fraction per cell.
     *
     * rt_kappa is the SSOT for opacity; we pass a one-element view of the
     * neighbor (`&Pj`, `&Cj` with index 0) so the canonical helper can be
     * reused from the runner pair body. This preserves legacy opacity
     * behavior including the transitive All.* / cooling-stack reach.
     * Longer-term, replace hot pair calls with an rt_kappa-populated
     * per-cell cache — see OPEN_rt_kappa_per_cell_cache.md. */
    double cf_a   = scalars.common.cf_atime;
    double h_phys = h_j * cf_a;
    double sigma_cell = ((double)loc.wt_sum / (h_j * h_j))
                      * (Mass_j / (h_phys * h_phys));
    double tau_uv = rt_kappa(0, RT_FREQ_BIN_FIRE_UV, &Pj, &Cj) * sigma_cell;
    double tau_op = rt_kappa(0, RT_FREQ_BIN_FIRE_OPT, &Pj, &Cj) * sigma_cell;
    double frac_abs = (double)loc.f_lum_ion
                    + (1.0 - (double)loc.f_lum_ion)
                      * (1.0 - 0.5*(exp(-tau_uv) + exp(-tau_op)));
    dv_ss *= frac_abs;

    /* jet contribution */
    double jet_kick = 0.0;
    if ((double)loc.jet_momentum_tocouple > 0) {
        jet_kick = wk * (double)loc.jet_momentum_tocouple / Mass_j;
        /* i-side accumulation always runs (oracle compares this for parity). */
        Kokkos::atomic_add(&out.jet_momentum_used,
                            (MyDouble)(wk * (double)loc.jet_momentum_tocouple));
    }
    dv_ss += jet_kick;
#endif

    /* --- multiple-scattering (IR re-radiation) kick.
     * Canonical rt_kappa via the same one-element view pattern as above. --- */
    double kappa_ir = rt_kappa(0, RT_FREQ_BIN_FIRE_IR, &Pj, &Cj);
    double cf_a_for_ms = scalars.common.cf_atime;
    double dv_ms = scalars.rp_renorm
                 * dE / Mass_j * kappa_ir
                 * (Mass_j / (4.0 * M_PI * r2 * cf_a_for_ms * cf_a_for_ms));

#if (GALSF_FB_FIRE_STELLAREVOLUTION <= 2)
    /* stochastic gate per neighbor — order-independent RNG stream */
    double dv_imparted = (double)loc.delta_v_imparted_rp;
    double prob = (dv_ms + dv_ss) / dv_imparted;
    if (prob > 1.0) { dv_imparted *= prob; prob = 1.0; }
    double p_rand = gizmo_gpu_rand_double((uint64_t)loc.ID ^ (uint64_t)Pj.ID,
                                           scalars.rng_ti_counter ^ RADFBRP_RNG_SHIFT);
    if (p_rand >= prob) return;

    /* cap total dv */
    if (dv_imparted > scalars.dv_cap_codeunits) dv_imparted = scalars.dv_cap_codeunits;

    /* direction: follow the stronger kick */
    Vec3<double> dir;
    if (dv_ss > dv_ms) {
        dir = dp_ij;                                    /* UV: radially outward from star */
    } else {
        dir[0] = -(double)Pj.GradRho[0];
        dir[1] = -(double)Pj.GradRho[1];
        dir[2] = -(double)Pj.GradRho[2];                /* IR: along opacity gradient */
    }
    double norm = dir.norm_sq();
    if (norm > 0) {
        norm = sqrt(norm);
        dir[0] /= norm; dir[1] /= norm; dir[2] /= norm;
    } else {
        dir[0] = 0; dir[1] = 0; dir[2] = 1;
    }

    Vec3<double> dv_kick;
    {
        double scale = dv_imparted * scalars.common.cf_atime;
        dv_kick[0] = scale * dir[0];
        dv_kick[1] = scale * dir[1];
        dv_kick[2] = scale * dir[2];
    }
#else /* STELLAREVOLUTION > 2: always apply both components, no stochastic gate */

    /* IR kick: along opacity gradient */
    Vec3<double> dir_ir;
    dir_ir[0] = -(double)Pj.GradRho[0];
    dir_ir[1] = -(double)Pj.GradRho[1];
    dir_ir[2] = -(double)Pj.GradRho[2];
    double norm_ir = dir_ir.norm_sq();
    if (norm_ir > 0) {
        norm_ir = sqrt(norm_ir);
        dir_ir[0] /= norm_ir; dir_ir[1] /= norm_ir; dir_ir[2] /= norm_ir;
    } else {
        dir_ir[0] = 0; dir_ir[1] = 0; dir_ir[2] = 1;
    }

    /* UV/jet kick: radially outward from star */
    Vec3<double> dir_uv = dp_ij;
    double norm_uv = dir_uv.norm_sq();
    if (norm_uv > 0) {
        norm_uv = sqrt(norm_uv);
        dir_uv[0] /= norm_uv; dir_uv[1] /= norm_uv; dir_uv[2] /= norm_uv;
    } else {
        dir_uv[0] = 0; dir_uv[1] = 0; dir_uv[2] = 1;
    }

    /* cap each component */
    if (dv_ms > scalars.dv_cap_codeunits) dv_ms = scalars.dv_cap_codeunits;
    if (dv_ss > scalars.dv_cap_codeunits) dv_ss = scalars.dv_cap_codeunits;

    /* combined kick (split into two directions) */
    Vec3<double> dv_kick;
    {
        double sir = dv_ms * scalars.common.cf_atime;
        double suv = dv_ss * scalars.common.cf_atime;
        dv_kick[0] = sir * dir_ir[0] + suv * dir_uv[0];
        dv_kick[1] = sir * dir_ir[1] + suv * dir_uv[1];
        dv_kick[2] = sir * dir_ir[2] + suv * dir_uv[2];
    }
#endif

    /* j-side atomic writes — suppressed under oracle dry-run. */
    if (oracle_dry_run) return;

    for (int k = 0; k < 3; k++) {
        Kokkos::atomic_add(&Pj.Vel[k],     (MyDouble)dv_kick[k]);
        Kokkos::atomic_add(&Cj.VelPred[k], (MyDouble)dv_kick[k]);
        Kokkos::atomic_add(&Pj.dp[k],      (MyDouble)(dv_kick[k] * Mass_j));
    }
}

/* ============================================================================
 * RadFBRPSpec — iterative 2-pass scatter Spec (sink_feed + iterative pattern).
 * ========================================================================== */
struct RadFBRPSpec {
    static constexpr const char *loop_name = "radfbrp";
    static constexpr ModeBEvalOMP modeb_eval_omp = ModeBEvalOMP::EpsilonAtomic; /* EpsilonAtomic: iter-1 kicks atomic_add Pj.Vel/Cj.VelPred/Pj.dp, gated !oracle_dry_run, ID^ID RNG + staged wt_sum order-indep, never read back -> ulp */

    /* Search policy. Legacy radfb_local_gpu.cc:259 used NGB_SEARCH_ONEWAY +
     * j_type_bitmask=1 (gas only). */
    static constexpr int                     search_mode        = MODE_B_SEARCH_ONEWAY;
    static constexpr unsigned int            neighbor_type_mask = (1u << 0);   /* gas only */
    static constexpr mode_b_radius_policy_t  radius_policy      = MODE_B_RADIUS_DEFAULT;

    /* Write policy. */
    static constexpr WritePattern   write_pattern   = WritePattern::ActiveReduceOnly;
    static constexpr SidxCacheKind  sidx_cache_kind = SidxCacheKind::GasOnly;   /* tbm = 1 */
    static constexpr bool mode_a_active_sources_in_sidx_pool = false; /* non-pool active sources (sink/star/grain) -> runner stages explicit P[].Pos */
    static constexpr bool           uses_ghost_writeback      = true;
    static constexpr bool           uses_ghost_write_detector = true;

    /* Iterative metadata. 2 iters strict (iter 0 = wt_sum aggregation;
     * iter 1 = kicks). after_iter returns Converged at iter 1 always.
     * IterScratch carries iter-0 wt_sum through after_iter →
     * after_iter_global → per_active_local. */
    using IterControl  = Iterative;
    using IterScratch  = RadFBRPIterScratch;
    static constexpr int    max_iters                    = 2;
    static constexpr bool   mode_a_rebuild_csr_every_iter = false;
    /* Single j-mask subgroup (gas-only); SupportsSubgroups omitted →
     * default false_type. Runner runtime-asserts single subgroup at
     * entry. NO actives_partition_by_subgroup (that's only for
     * multi-subgroup Specs like ags_density). Mirrors mechfb. */
    static constexpr double mode_a_csr_buffer_factor      = 1.3;
    /* radius_tolerance unused — we don't AdjustRadius; convergence is by
     * iter-1 always returning Converged. Provide a non-zero value to keep
     * runner contract happy (mirrors AgsDensitySpec's choice). */
    static constexpr double radius_tolerance              = 1e-9;

    /* Oracle compare tolerance for AccumData. Default 1e-10 — the iter-0
     * wt_sum aggregation reads only Pj.Get_Particle_Size() which is not
     * mutated by any radfb_rp pair; iter-1 kicks atomic_add into independent
     * Vel/VelPred/dp fields. No thermal_fb-style order-dependence expected. */
    static constexpr double accum_tolerance = 1e-10;

    /* Type aliases. */
    using CallScalars    = RadFBRPCallScalars;
    using ActiveData     = RadFBRPActiveState;
    using AccumData      = RadFBRPAccum;
    using DeviceContext  = RadFBRPDeviceContext;
    using ScatterData    = NoScatter;
    using IdentityFields = NoIdentity;

    /* NeighborData carries non-const pointers (kernel writes to j-side
     * fields on iter 1). Oracle flag propagated per-call from ctx. */
    struct NeighborData {
        struct particle_data *neighbor_particle;
        struct gas_cell_data *neighbor_cell;
        bool                  oracle_dry_run;
    };

    /* Aux — host-only per-call state. Two responsibilities:
     *   (1) iter_index: THE single source of truth for iter-gated writeback.
     *       Set by reset_per_iter_device_context BEFORE any rank can enter
     *       a writeback hook. All four ghost_write_detector / ghost_writeback
     *       hooks read this via nlr_aux<RadFBRPSpec>(args) and early-return
     *       unless iter_index == 1.
     *   (2) host_locals: non-owning pointer back to the toplevel-owned
     *       RadFBRPLocalIn buffer (so apply_active_writeback can subtract
     *       jet_momentum_used from NewStar_Momentum_For_JetFeedback without
     *       re-running the stochastic gate to recover loc state). */
    struct Aux {
        int                   iter_index;   /* runner-set per outer iter */
        const RadFBRPLocalIn *host_locals;  /* [num_active]; toplevel-owned */
        const int            *host_active_src; /* [num_active]; particle indices */
    };

    /* ====================================================================
     * Host hooks (bodies in radfb_rp_loop.cc).
     * ==================================================================== */

    /* Defensive predicate. NEVER called by nlr_build_active_list (toplevel
     * builds active list directly via radfb_rp_local_fill); this is only
     * a belt-and-suspenders cosmo-aware Type re-check. Returning true here
     * for a non-firing source would NOT include it (toplevel already gated). */
    static bool        is_active(int particle_index);

    static double      search_radius(const neighbor_loop_args& args,
                                      int active_slot, int i);
    static CallScalars populate_call_scalars(const neighbor_loop_args& args);

    static void populate_device_context(const neighbor_loop_args& args, DeviceContext& ctx);
    static void cleanup_device_context (const neighbor_loop_args& args, DeviceContext& ctx);

    /* Per-iter reset — CRITICAL ordering. Sets Aux::iter_index = iter_index
     * BEFORE any rank can enter a detector/writeback hook. Also handles the
     * iter-0 → iter-1 transition by copying IterScratch.wt_sum into per-active
     * UVM RadFBRPLocalIn.wt_sum (so iter-1 device kernel reads the staged
     * value). */
    static void reset_per_iter_device_context(const neighbor_loop_args_iterative& args,
                                               DeviceContext& ctx,
                                               int iter_index);

    /* Source-side host write — applied per-active post-runner. Subtracts
     * accum.jet_momentum_used from P_host[i].NewStar_Momentum_For_JetFeedback
     * under GALSF_FB_FIRE_STELLAREVOLUTION > 2. NO other source-side writes —
     * the legacy CPU code's only post-kernel source mutation is this jet
     * budget decrement. */
    static void apply_active_writeback(const neighbor_loop_args& args,
                                        int active_slot, int i,
                                        const AccumData& accum);

    /* Per-field merge — manifest in radfb_rp_loop.cc.
     *   wt_sum            : additive across peer accumulators (iter 0 reduce).
     *   jet_momentum_used : additive (iter 1 reduce). */
    static void merge_accum(AccumData& local_accum, const AccumData& peer_accum);

    /* Oracle suppression flag. */
    static void set_oracle_brute_pass(DeviceContext& ctx, bool on) {
        ctx.oracle_dry_run = on;
    }

    /* Ghost-writeback + write-detector bookkeeping. All four gated by
     * Aux::iter_index — iter 0 returns immediately on EVERY rank (no
     * half-paired begin without end). */
    static void ghost_write_detector_begin(const neighbor_loop_args&, const NeighborLoopPlan&);
    static void ghost_write_detector_end  (const neighbor_loop_args&, const NeighborLoopPlan&);
    static void ghost_writeback_begin     (const neighbor_loop_args&, const NeighborLoopPlan&);
    static void ghost_writeback_end       (const neighbor_loop_args&, const NeighborLoopPlan&);

    /* Iterative hooks.
     * after_iter is STATUS-ONLY (no writes, oracle-safe per mechfb r5+r6+r7).
     * after_iter_global stages iter-0 wt_sum into both per_active_locals
     * (production + oracle) so iter-1's device kernel reads the staged value. */
    static IterResult after_iter(const AfterIterContext<RadFBRPSpec>& ctx,
                                  const AccumData& accum);
    static void       after_iter_global(const neighbor_loop_args& args,
                                         const struct NlrIterDriver<RadFBRPSpec>& drv);

    /* Diagnostics — env-gated. */
    static double compare_accum(const AccumData& local, const AccumData& oracle);

    /* ====================================================================
     * Device hooks (header-inline).
     * ==================================================================== */
    KOKKOS_INLINE_FUNCTION
    static void zero_accum(AccumData& accum) {
        accum.wt_sum            = 0;
        accum.jet_momentum_used = 0;
    }

    KOKKOS_INLINE_FUNCTION
    static ActiveData load_active(const DeviceContext& dctx,
                                  int active_slot, int i,
                                  double h_search,
                                  const CallScalars& scalars) {
        ActiveData a{};
        if (dctx.per_active_local != nullptr) {
            a.local = dctx.per_active_local[active_slot];
            a.pos[0] = (double)a.local.Pos[0];
            a.pos[1] = (double)a.local.Pos[1];
            a.pos[2] = (double)a.local.Pos[2];
            a.h_search = (double)a.local.KernelRadius;
        } else {
            /* Fallback — shouldn't fire in production. */
            a.pos[0] = (double)dctx.P[i].Pos[0];
            a.pos[1] = (double)dctx.P[i].Pos[1];
            a.pos[2] = (double)dctx.P[i].Pos[2];
            a.h_search = h_search;
        }
        /* iter_index: snapshot from runner-tracked iter (see ActiveData
         * comment in pair_kernel below — used to branch iter 0 / iter 1). */
        a.iter_index = dctx.iter_index_snapshot;   /* dctx field, mirrored from Aux per iter */
        a.scalars = scalars;
        return a;
    }

    KOKKOS_INLINE_FUNCTION
    static NeighborData load_neighbor(const DeviceContext& dctx, int j,
                                       const IdentitySidecar& /*id*/,
                                       const ActiveData& /*active*/) {
        NeighborData n{};
        n.neighbor_particle = &dctx.P[j];
        n.neighbor_cell     = (dctx.CellP != nullptr && dctx.P[j].Type == 0)
                              ? &dctx.CellP[j] : nullptr;
        n.oracle_dry_run    = dctx.oracle_dry_run;
        return n;
    }

    KOKKOS_INLINE_FUNCTION
    static void pair_kernel(const ActiveData& active,
                             const NeighborData& neighbor,
                             AccumData& accum,
                             NoScatter& /*scatter*/) {
        if (neighbor.neighbor_particle == nullptr) return;
        if (neighbor.neighbor_cell     == nullptr) return;  /* gas-only safety */
        struct particle_data &Pj = *neighbor.neighbor_particle;
        struct gas_cell_data &Cj = *neighbor.neighbor_cell;

        if (Pj.Type != 0) return;
#ifdef HYDRO_MULTIFLUID_DM
        if (Pj.FluidType == FLUID_DM) return; /* skip dark-fluid neighbors */
#endif
        if (Pj.Mass <= 0) return;
#ifdef SINK_WIND_SPAWN
        if (Pj.ID == active.scalars.spawned_wind_cell_id) return;
#endif

        if (active.local.KernelRadius <= 0) return;
        const double h2 = (double)active.local.KernelRadius
                        * (double)active.local.KernelRadius;

        Vec3<double> dp;
        dp[0] = (double)active.local.Pos[0] - (double)Pj.Pos[0];
        dp[1] = (double)active.local.Pos[1] - (double)Pj.Pos[1];
        dp[2] = (double)active.local.Pos[2] - (double)Pj.Pos[2];
        nearest_xyz(dp);
        const double r2 = dp.norm_sq();
        if (r2 >= h2 || r2 <= 0) return;

        if (active.iter_index == 0) {
            /* iter 0 : accumulate wt_sum = Σ h_j² */
            double h_j = Pj.Get_Particle_Size();
            Kokkos::atomic_add(&accum.wt_sum, h_j * h_j);
        } else {
            /* iter 1 : apply kicks using staged active.local.wt_sum */
            radfb_rp_pair_kick(active.local, active.scalars, Pj, Cj,
                                r2, dp, neighbor.oracle_dry_run, accum);
        }
    }
};

#endif /* GALSF_FB_FIRE_RT_LOCALRP */

#endif /* RADFB_RP_LOOP_H */
