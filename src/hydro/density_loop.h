/* hydro/density_loop.h — hydro density runner Spec.
 *
 * Defines DensitySpec for the iterative neighbor-loop runner
 * (mesh/neighbor_loop_runner.h). This replaces the legacy density()
 * driver and retired density_evaluate_gpu path.
 *
 * Contract notes:
 *   - pair_kernel signature matches runner contract verbatim: 5 args
 *     (active, neighbor, accum, scatter, call scalars). NO `int j` — runner
 *     doesn't pass it; if a body needs j it lives in NeighborData.
 *   - AccumData fields LITERALLY match density_functions.h:50-104
 *     density_evaluate_data_out_ (literal field-by-field port):
 *     NV_T_face_weights/ParticleVel/GradH_numer = Vec3<MyDouble>;
 *     NV_D/NV_A = MyFloat[3][3]; GradRho/Gas_B = Vec3<MyFloat>/Vec3<MyDouble>;
 *     Sink_dr_to_NearestGasNeighbor nested under BH_ACCRETE_NEARESTFIRST
 *     || SINGLE_STAR_TIMESTEPPING; GasVel under TURB_DRIVING || GRAIN_FLUID
 *     (NOT GRAIN_FLUID alone); no invented Gas_Density / SmoothedVel.
 *   - ActiveData mirrors density_functions.h:33-46 density_evaluate_data_in_:
 *     Pos=Vec3<MyDouble>, Vel=Vec3<MyFloat>, Type=int, Accel=Vec3<MyFloat>
 *     under SPHAV_CD10 only, DelayTime under GALSF_SUBGRID_WINDS. No
 *     TimeBin (j-side via NeighborData); no origin_* (runner-side via
 *     envelopes, not Spec-side).
 *   - CallScalars contains only the fields NOT already in
 *     NlrCommonScalars (cf_atime/cf_a2inv/comoving_integration_on come
 *     via common.*; pair-kernel body reads cs.common.cf_a2inv). 10 fields.
 *   - zero_accum is per-field, NOT memset: Sink_TimeBinGasNeighbor =
 *     TIMEBINS sentinel; Sink_dr_to_NearestGasNeighbor = MAX_REAL_NUMBER;
 *     all others = 0.
 *   - uses_ghost_writeback = false: density is pure i-side accumulation.
 *     Compile-time static_assert holds the invariant.
 *   - apply_active_writeback is FINAL-ONLY for iterative Specs (runner
 *     header lines 1272-1282).
 *   - mode_a_csr_buffer_factor = 1.3 == legacy DENSITY_H_BUFFER_FACTOR.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */
#ifndef DENSITY_LOOP_H
#define DENSITY_LOOP_H

/* Kokkos_Core must precede allvars.h (its macros may conflict with stdlib names). */
#include <Kokkos_Core.hpp>

#include "../declarations/allvars.h"
#include "../declarations/multifluid_helpers.h"
#include "../mesh/neighbor_loop_runner.h"
#include "../mesh/mode_b_local_walker.h"   /* MODE_B_SEARCH_*, MODE_B_RADIUS_* */
#if defined(AGS_KERNELRADIUS_CALCULATION_IS_ACTIVE)
#include "../gravity/ags_functions.h"      /* ags_density_isactive_P */
#endif
/* NOTE: caller TUs must include "../mesh/kernel.h" before this header.
 * kernel.h has no include guards; provides kernel_main / kernel_hinv used
 * by the inline pair body. */

/* No KOKKOS_INLINE_FUNCTION fallback here — Kokkos_Core.hpp is included
 * unconditionally above. This Spec carries device-callable pair-kernel
 * accessors; misordered includes must compile-fail loudly, not silently
 * resolve to host-only `inline`. Same convention as neighbor_loop_runner.h. */

#include <vector>

/* Forward decl — defined in hydro/density_loop.cc. */
int density_isactive(int i);

/* ============================================================================
 * (1) CallScalars — deduplicated against NlrCommonScalars.
 *
 * NlrCommonScalars (mesh/neighbor_loop_runner.h:350) already carries:
 *   cf_atime, cf_a2inv, cf_a3inv, cf_hubble_a, newton_G, hubble,
 *   comoving_integration_on.
 *
 * Inside iter loop / device hooks: `cs.common.cf_a2inv` (SPHAV_CD10
 * pair-kernel reads), `cs.common.comoving_integration_on` (active
 * predicate star-type gates) come from `common`. This struct adds only
 * the fields NOT in NlrCommonScalars: time-counters, bisection
 * targets/clamps, sink overrides.
 *
 * Post-runner host-only finalization (density_finalize_post_runner) does
 * NOT read CallScalars — it reads host_all->cf_atime, host_all->cf_a2inv,
 * host_all->Time, etc. directly via nlr_host_all_ptr().
 * CallScalars is the device-facing / per-iter-hot-path channel; host_all
 * is the host-only-finalize channel. Keep the split — that's the rule
 * the runner contract uses to separate device-hot-path scalars from
 * host-only finalization.
 *
 * Field types are exact legacy types from global_data_all_struct.h. */
struct DensityCallScalars {
    NlrCommonScalars common;

    /* Time counters (not in NlrCommonScalars) */
    integertime  ti_current;         /* All.Ti_Current (sink active predicate) */
    double       time_now;           /* All.Time */
    double       time_begin;         /* All.TimeBegin */
    double       time_step;          /* All.TimeStep (extra-physics-gas gate) */

    /* Bisection target + clamps */
    double       des_num_ngb;        /* All.DesNumNgb */
    double       max_num_ngb_dev;    /* All.MaxNumNgbDeviation */
    double       min_kernel_radius;  /* All.MinKernelRadius */
    double       max_kernel_radius;  /* All.MaxKernelRadius */

    /* Sink-specific bisection overrides (only present when sinks compiled). */
#ifdef SINK_PARTICLES
    double       sink_ngb_factor;            /* All.SinkNgbFactor */
    double       sink_max_accretion_radius;  /* All.SinkMaxAccretionRadius */
#endif
};

/* ============================================================================
 * (2) AccumData — literal port of density_evaluate_data_out_
 *     (density_functions.h:50-104). EVERY `#ifdef` branch present in the
 *     legacy struct ports here verbatim. Field order matches legacy.
 *     Types match legacy (MyFloat vs MyDouble; Vec3<...> vs array).
 *
 * Merge semantics:
 *   - all fields ADD-reduced EXCEPT
 *   - Sink_TimeBinGasNeighbor: MIN, init = TIMEBINS
 *   - Sink_dr_to_NearestGasNeighbor: MIN, init = MAX_REAL_NUMBER
 * zero_accum (below) is per-field, NOT memset. ========================== */
struct DensityAccumData {
    MyDouble                   Ngb;
    MyDouble                   Rho;
    MyDouble                   DrkernNgb;
    MyDouble                   Particle_DivVel;
    SymmetricTensor2<MyDouble> NV_T;
    Vec3<MyDouble>             NV_T_face_weights;

#if defined(HYDRO_MESHLESS_FINITE_VOLUME) && ((HYDRO_FIX_MESH_MOTION==5)||(HYDRO_FIX_MESH_MOTION==6))
    Vec3<MyDouble>             ParticleVel;
#endif

#ifdef HYDRO_SPH
    MyDouble                   DrkernHydroSumFactor;
#endif

#ifdef RT_SOURCE_INJECTION
    MyDouble                   KernelSum_Around_RT_Source;
#endif

#ifdef HYDRO_PRESSURE_SPH
    MyDouble                   EgyRho;
#endif

#if defined(SPHAV_CD10_VISCOSITY_SWITCH)
    MyFloat                    NV_D[3][3];
    MyFloat                    NV_A[3][3];
#endif

#ifdef DO_DENSITY_AROUND_NONGAS_PARTICLES
    Vec3<MyFloat>              GradRho;
#endif

#if defined(SINK_PARTICLES)
    int                        Sink_TimeBinGasNeighbor;       /* MIN, init = TIMEBINS */
  #if defined(BH_ACCRETE_NEARESTFIRST) || defined(SINGLE_STAR_TIMESTEPPING)
    MyDouble                   Sink_dr_to_NearestGasNeighbor; /* MIN, init = MAX_REAL_NUMBER */
  #endif
#endif

#if defined(TURB_DRIVING) || defined(DO_FLUID_ALTSPECIES_DRAG_CALCULATION)
    Vec3<MyDouble>             GasVel;
#endif

#if defined(DO_FLUID_ALTSPECIES_DRAG_CALCULATION)
    MyDouble                   AmbientGasRho;
    MyDouble                   Gas_Temperature;
    MyDouble                   Gas_fion;
  #if defined(DO_FLUID_DRAG_CALCULATION_WITHBFIELDS)
    Vec3<MyDouble>             Gas_B;
  #endif
  #if defined(GRAIN_EVOLUTION) && (GRAIN_EVOLUTION & (32|64))
    MyDouble                   Gas_VolatileSpecies[GRAIN_NUM_VOLATILE_SPECIES];  /* precompiler_logic.h:1247 #define = 4 */
  #endif
#endif

#ifdef HYDRO_PARTITION_UNITY_IMPROVE_FD
    Vec3<MyDouble>             GradH_numer;
    MyDouble                   GradH_denom;
#endif
};

/* ============================================================================
 * (3) IterScratch — per-active state living across iterations.
 *
 * Legacy bisection brackets (Left[i], Right[i] at density.cc:149-150) +
 * a converged latch (replaces the legacy P[i].TimeBin negation pattern
 * at density.cc:479/591/617, keeping convergence state out of P[i]) + iter-count
 * tracking for the iter>10 underflow warning at density.cc:441-452.
 *
 * Mutated host-side inside after_iter only. Not visible inside pair_kernel. */
struct DensityIterScratch {
    MyFloat  left;
    MyFloat  right;
    bool     converged;
    int      iter_count;
    bool     underflow_warned;
    /* CN tracking: legacy density.cc reads CellP[i].ConditionNumber as
     * the iter-loop's "effective" CN, and at iter==0 may update CellP[i]
     * to THIS iter's value (lines 313-322). Our design forbids CellP
     * writes inside after_iter; we mirror the iter-effective CN in
     * IterScratch instead. density_finalize_post_runner writes
     * CellP[i].ConditionNumber at the end, from the converged
     * accum.NV_T re-inversion. */
    double   condition_number_current;
    bool     cn_initialized;
};

/* ============================================================================
 * (4) ActiveData — literal port of density_evaluate_data_in_
 *     (density_functions.h:33-46).
 *
 * Types match legacy verbatim where the kernel needs legacy data. Fields
 * owned by the runner, such as origin rank/index, stay out of ActiveData.
 *
 * The per-call CallScalars snapshot is NOT carried here. It is one value for
 * the whole call, identical on every rank at a sync point, so it is passed to
 * pair_kernel as its own argument rather than replicated once per active.
 * ========================================================================= */
struct DensityActiveState {
    /* pos is lowercase to match the runner's walker which reads
     * `active.pos[k]` (mesh/neighbor_loop_runner.cc:793). All other
     * fields use legacy density_evaluate_data_in_ capitalization. */
    Vec3<MyDouble>     pos;
    Vec3<MyFloat>      Vel;
    MyFloat            h_search;       /* legacy KernelRadius; runner manages via radii_uvm */
#if defined(SPHAV_CD10_VISCOSITY_SWITCH)
    Vec3<MyFloat>      Accel;          /* P[i].GravAccel + CellP[i].HydroAccel snapshot */
#endif
#ifdef GALSF_SUBGRID_WINDS
    MyFloat            DelayTime;
#endif
    int                Type;
#ifdef HYDRO_MULTIFLUID
    unsigned char      FluidType;  /* packed P[i].FluidType — for same_lagrangian_fluid_id() */
#endif
};

/* NeighborData — j-side per-pair sidecar. Density has NO j-side writes
 * (verified §E inventory + design §1f static_assert), so NeighborData
 * is read-only pointers to the j-side P / CellP. neighbor_index is carried
 * only for diagnostics/assertions, not as an extra pair_kernel argument. */
struct DensityNeighborData {
    struct particle_data *neighbor_particle;
    struct gas_cell_data *neighbor_cell;     /* nullptr for non-gas / when no CellP */
    int                   neighbor_index;    /* j; diagnostic / assertions */
};

/* ============================================================================
 * (5) DensitySpec — the runner-template Spec.
 *
 * Declares the density loop traits, data structs, and hook surface. ===== */
struct DensitySpec {
    /* Identity */
    static constexpr const char *loop_name = "density";
    static constexpr ModeBEvalOMP modeb_eval_omp = ModeBEvalOMP::BitwiseReadonly; /* i-side AccumData only; no j-write, no atomics -> threaded eval bit-identical */

    /* Search policy: legacy density CSR builder uses NGB_SEARCH_ONEWAY
     * with j-bitmask = 1 (gas only) per density_gpu.cc:256. Pair predicate
     * is h_i only (no h_j check). Symmetric search would inflate NumNgb. */
    static constexpr int                     search_mode        = MODE_B_SEARCH_ONEWAY;
    static constexpr unsigned int            neighbor_type_mask = (1u << 0);  /* gas-only */
    static constexpr mode_b_radius_policy_t  radius_policy      = MODE_B_RADIUS_DEFAULT;

    /* Write policy */
    static constexpr WritePattern   write_pattern   = WritePattern::ActiveReduceOnly;
    /* SIDX cache: GAS-ONLY. An earlier draft used
     * AllTypes here, which violated the cache-mask invariant — the
     * all-types cache's compact_xyzh / pool includes DM (Type=1) which
     * a gas-only neighbor_type_mask walk should never see. Walker would
     * return DM neighbors, lazy-drift on those DM particles, drift_particle
     * abort 'no prediction into past allowed' at SP4 hydro. gpu_ngb_list_build
     * now hard-aborts on cache_tbm vs caller tbm mismatch. */
    static constexpr SidxCacheKind  sidx_cache_kind = SidxCacheKind::GasOnly;
    static constexpr bool           uses_ghost_writeback      = false;
    static constexpr bool           uses_ghost_write_detector = false;

    /* Convergence + iter policy */
    using                    IterControl = Iterative;
    using                    IterScratch = DensityIterScratch;
    static constexpr int     max_iters   = MAXITER;  /* legacy density.cc bisection bound */

    /* Mode A CSR policy — exact legacy constant (density_gpu.cc:103
     * DENSITY_H_BUFFER_FACTOR 1.3). */
    static constexpr double mode_a_csr_buffer_factor      = 1.3;
    /* Fallback rule per design v0.4 §1c: flip mode_a_rebuild_csr_every_iter
     * to true ONLY if PA-A vs PB-vs-legacy validation diverges and the
     * cause is the CSR reuse. Don't tune performance before correctness. */
    static constexpr bool   mode_a_rebuild_csr_every_iter = false;

    /* Subgroup policy: single j-mask subgroup, gas-only. SupportsSubgroups
     * is OMITTED (default false_type — runner runtime-asserts single
     * subgroup at entry, TRAP 9). Per-i type branching for non-gas actives
     * lives inside the pair_kernel body. */

    /* Type aliases */
    using CallScalars   = DensityCallScalars;
    using ActiveData    = DensityActiveState;
    using AccumData     = DensityAccumData;
    using NeighborData  = DensityNeighborData;
    /* DeviceContext: density needs no extension beyond the base.
     * AGS extends with need_wakeup_uvm; density's pure i-side accumulation
     * means
     * no per-call device-side scratch is needed. */
    using DeviceContext = NeighborLoopDeviceContextBase;
    /* ScatterData: required by runner contract. Density has no j-side
     * scatter (uses_ghost_writeback=false, pure i-side accumulation),
     * so NoScatter — the runner's empty marker type. */
    using ScatterData   = NoScatter;

    /* Aux — host-owned final-accum storage.
     * apply_active_writeback (final-only for iterative Specs per runner
     * header lines 1272-1282) copies the converged AccumData into
     * per_active_final_accum[active_slot]. density_finalize_post_runner
     * consumes it for final P[]/CellP[] work. Host-only — std::vector OK. */
    struct Aux {
        /* Converged AccumData per active, copied at the post-iter-loop
         * apply_active_writeback (FINAL-only). */
        std::vector<AccumData> per_active_final_accum;
        /* Converged radius per active, copied in after_iter at the
         * Converged return paths. Needed by density_finalize_post_runner
         * to write P[i].KernelRadius; the runner does not surface
         * radii_uvm to the caller, and after_iter can be invoked more than
         * once per active so a converged value written there would not be
         * single-valued. */
        std::vector<double>    per_active_final_h;
    };

    /* J-side write invariant. */
    static_assert(uses_ghost_writeback == false,
        "density() must remain pure i-side accumulation; if a j-side "
        "write is added, register a ghost_writeback op explicitly");

    /* ====================================================================
     * Host hooks. Bodies in hydro/density_loop.cc.
     * ==================================================================== */
    static bool        is_active(int particle_index) { return density_isactive(particle_index) != 0; }
    static double      search_radius(const neighbor_loop_args& args, int active_slot, int i);
    static CallScalars populate_call_scalars(const neighbor_loop_args& args);

    static void apply_active_writeback(const neighbor_loop_args& args,
                                       int active_slot, int i,
                                       const AccumData& accum);
    /* Iterative variant of apply_active_writeback. Runner calls this instead of apply_active_writeback in
     * the iterative production-only post-iter-loop writeback, passing
     * the converged radius and IterScratch. That loop runs exactly once per
     * active, so args.aux can be written
     * from here without contamination. Density implementation copies
     * both `accum` and `final_h` into Aux for density_finalize_post_runner. */
    static void apply_active_writeback_iterative(const neighbor_loop_args& args,
                                                  int active_slot, int i,
                                                  const AccumData& accum,
                                                  double final_h,
                                                  const IterScratch& scratch);
    /* ====================================================================
     * merge_accum — per-field combine for Mode B remote.
     *
     * Per-field op MUST match what pair_kernel writes. Sum for additive
     * fields; MIN for the two sink fields. Mismatch = silent multi-rank
     * corruption, surfacing only as a difference between rank counts.
     *
     * Gating mirrors AccumData declaration verbatim:
     * Sink_TimeBinGasNeighbor under SINK_PARTICLES; Sink_dr_to_NearestGasNeighbor
     * under (SINK_PARTICLES && (BH_ACCRETE_NEARESTFIRST||SINGLE_STAR_TIMESTEPPING)).
     * The asymmetry between declaration gate (BH || SINGLE_STAR) and
     * pair-body update gate (SINGLE_STAR only) is the legacy's; the merge
     * follows the DECLARATION gate, so the field is correctly combined
     * regardless of whether the pair body touched it this iter.
     * ==================================================================== */
    KOKKOS_INLINE_FUNCTION
    static void merge_accum(AccumData& local, const AccumData& peer) {
        /* Core (always ADD) */
        local.Ngb              += peer.Ngb;
        local.Rho              += peer.Rho;
        local.DrkernNgb        += peer.DrkernNgb;
        local.Particle_DivVel  += peer.Particle_DivVel;
        local.NV_T             += peer.NV_T;
        local.NV_T_face_weights[0] += peer.NV_T_face_weights[0];
        local.NV_T_face_weights[1] += peer.NV_T_face_weights[1];
        local.NV_T_face_weights[2] += peer.NV_T_face_weights[2];

#if defined(HYDRO_MESHLESS_FINITE_VOLUME) && ((HYDRO_FIX_MESH_MOTION==5)||(HYDRO_FIX_MESH_MOTION==6))
        local.ParticleVel[0] += peer.ParticleVel[0];
        local.ParticleVel[1] += peer.ParticleVel[1];
        local.ParticleVel[2] += peer.ParticleVel[2];
#endif

#ifdef HYDRO_SPH
        local.DrkernHydroSumFactor += peer.DrkernHydroSumFactor;
#endif

#ifdef RT_SOURCE_INJECTION
        local.KernelSum_Around_RT_Source += peer.KernelSum_Around_RT_Source;
#endif

#ifdef HYDRO_PRESSURE_SPH
        local.EgyRho += peer.EgyRho;
#endif

#if defined(SPHAV_CD10_VISCOSITY_SWITCH)
        for (int a = 0; a < 3; ++a) {
            for (int b = 0; b < 3; ++b) {
                local.NV_D[a][b] += peer.NV_D[a][b];
                local.NV_A[a][b] += peer.NV_A[a][b];
            }
        }
#endif

#ifdef DO_DENSITY_AROUND_NONGAS_PARTICLES
        local.GradRho[0] += peer.GradRho[0];
        local.GradRho[1] += peer.GradRho[1];
        local.GradRho[2] += peer.GradRho[2];
#endif

#if defined(SINK_PARTICLES)
        /* MIN reduction: legacy `if(out->T > T_j) out->T = T_j` at
         * density_functions.h:137. Sentinel init = TIMEBINS in zero_accum,
         * so unset slots stay at TIMEBINS and lose to any peer that found
         * a real neighbor. */
        if (peer.Sink_TimeBinGasNeighbor < local.Sink_TimeBinGasNeighbor) {
            local.Sink_TimeBinGasNeighbor = peer.Sink_TimeBinGasNeighbor;
        }
  #if defined(BH_ACCRETE_NEARESTFIRST) || defined(SINGLE_STAR_TIMESTEPPING)
        /* MIN reduction: legacy `if(dr_eff_wtd < out->dr) out->dr = dr_eff_wtd`
         * at density_functions.h:141. Sentinel init = MAX_REAL_NUMBER. */
        if (peer.Sink_dr_to_NearestGasNeighbor < local.Sink_dr_to_NearestGasNeighbor) {
            local.Sink_dr_to_NearestGasNeighbor = peer.Sink_dr_to_NearestGasNeighbor;
        }
  #endif
#endif

#if defined(TURB_DRIVING) || defined(DO_FLUID_ALTSPECIES_DRAG_CALCULATION)
        local.GasVel[0] += peer.GasVel[0];
        local.GasVel[1] += peer.GasVel[1];
        local.GasVel[2] += peer.GasVel[2];
#endif

#if defined(DO_FLUID_ALTSPECIES_DRAG_CALCULATION)
        local.AmbientGasRho += peer.AmbientGasRho;
        local.Gas_Temperature += peer.Gas_Temperature;
        local.Gas_fion += peer.Gas_fion;
  #if defined(DO_FLUID_DRAG_CALCULATION_WITHBFIELDS)
        local.Gas_B[0] += peer.Gas_B[0];
        local.Gas_B[1] += peer.Gas_B[1];
        local.Gas_B[2] += peer.Gas_B[2];
  #endif
  #if defined(GRAIN_EVOLUTION) && (GRAIN_EVOLUTION & (32|64))
        for (int kv = 0; kv < GRAIN_NUM_VOLATILE_SPECIES; ++kv) {
            local.Gas_VolatileSpecies[kv] += peer.Gas_VolatileSpecies[kv];
        }
  #endif
#endif


#ifdef HYDRO_PARTITION_UNITY_IMPROVE_FD
        local.GradH_numer[0] += peer.GradH_numer[0];
        local.GradH_numer[1] += peer.GradH_numer[1];
        local.GradH_numer[2] += peer.GradH_numer[2];
        local.GradH_denom    += peer.GradH_denom;
#endif
    }



    static IterResult after_iter(const AfterIterContext<DensitySpec>& ctx,
                                 const AccumData& accum);

    /* after_iter_body -- bisection convergence + radius update for one active (the
     * literal port of hydro/density.cc:244-590; full contract at DensitySpec::after_iter
     * in density_loop.cc). Host and device both run this one body: the host hook passes
     * its argument view, a device iteration passes the same pointers. Writes scratch, and under
     * AGS_KERNELRADIUS_CALCULATION_IS_ACTIVE the active's own AGS fields in P[i] (through
     * ags_density_isactive_P), so P must be writable. */
    KOKKOS_INLINE_FUNCTION
    static IterResult after_iter_body(struct particle_data *P, const struct gas_cell_data *CellP,
                                      int i, int iter, double h_search_current,
                                      const CallScalars& cs, IterScratch& scratch,
                                      const AccumData& accum, int this_task)
    {

        /* ---- (a) local normalization (mirrors density.cc:247-268). Computes
         * effective NumNgb / DrkernNgbFactor / Particle_DivVel from accum
         * without touching P[i]. ---- */
        double desnumngb    = cs.des_num_ngb;
        double desnumngbdev = cs.max_num_ngb_dev;
        if (cs.time_now == cs.time_begin) {
            if (cs.max_num_ngb_dev > 0.05) desnumngbdev = 0.05;
        }
        const double desnumngbdev_0 = desnumngbdev;

        double NumNgb_eff = (double)accum.Ngb;
        double DrkernNgbFactor_eff = (double)accum.DrkernNgb;
        double Particle_DivVel_eff = (double)accum.Particle_DivVel;
        const double h = h_search_current;

        if (NumNgb_eff > 0) {
            DrkernNgbFactor_eff *= h / (NUMDIMS * NumNgb_eff);
            Particle_DivVel_eff /= NumNgb_eff;
            NumNgb_eff *= VOLUME_NORM_COEFF_FOR_NDIMS * pow(h, (double)NUMDIMS);
        } else {
            NumNgb_eff = 0;
            DrkernNgbFactor_eff = 0;
            Particle_DivVel_eff = 0;
        }
    #if defined(AGS_KERNELRADIUS_CALCULATION_IS_ACTIVE)
        if (ags_density_isactive_P(i, P) && (P[i].Type > 0)) {
            Particle_DivVel_eff = 0;
        }
    #endif
        if (DrkernNgbFactor_eff > -0.9) {
            DrkernNgbFactor_eff = 1.0 / (1.0 + DrkernNgbFactor_eff);
        } else {
            DrkernNgbFactor_eff = 1.0;
        }
        Particle_DivVel_eff *= DrkernNgbFactor_eff;

        /* ---- (b) NV_T inversion + FaceClosureError computation (gas only,
         * host-side LOCAL — no write back to CellP). Mirrors density.cc:271-303.
         * Produces ConditionNumber AND face_closure_error_local — both used
         * below for ncorr_ngb. The inverted matrix and FaceClosureError are
         * discarded here; finalize re-inverts from Aux for the persistent
         * CellP writes.
         *
         * FaceClosureError nudge at legacy line 326 modifies ncorr_ngb, which
         * changes desnumngb and therefore the convergence trajectory. It
         * cannot be deferred to finalize; the nudge must fire mid-iter. Tinv
         * is kept in scope specifically so the face-leak computation can run. */
        double ConditionNumber = 0.0;
        double face_closure_error_local = 0.0;  /* gas-only; nudge gated below */
        if (P[i].Type == 0) {
            const double V_i = VOLUME_NORM_COEFF_FOR_NDIMS * pow(h, (double)NUMDIMS) / (NumNgb_eff > 0 ? NumNgb_eff : 1.0);
            const double dimensional_NV_T_normalizer = pow(h, (double)(2 - NUMDIMS));
            double NV_T_local[3][3];
            for (int k1 = 0; k1 < 3; ++k1) {
                for (int k2 = 0; k2 < 3; ++k2) {
                    NV_T_local[k1][k2] = accum.NV_T[k1][k2] / dimensional_NV_T_normalizer;
                }
            }
            const double ConditionNumber_threshold = 10.0 * CONDITION_NUMBER_DANGER;
            const double trace_initial = NV_T_local[0][0] + NV_T_local[1][1] + NV_T_local[2][2];
            double conditioning_term_to_add = 1.05 * (trace_initial / NUMDIMS) / ConditionNumber_threshold;
            double Tinv[3][3];
            while (true) {
                ConditionNumber = matrix_invert_ndims(NV_T_local, Tinv);
                if (ConditionNumber < ConditionNumber_threshold) break;
                for (int k1 = 0; k1 < NUMDIMS; ++k1) {
                    NV_T_local[k1][k1] += conditioning_term_to_add;
                }
                conditioning_term_to_add *= 1.2;
            }

            /* Face-leak / FaceClosureError, mirrors density.cc:273-303 verbatim
             * but reading accum.NV_T_face_weights (this iter's accumulator)
             * instead of CellP[i].NV_T_face_weights (which legacy had already
             * scattered into CellP). The legacy NV_T used in the face_out
             * sum is the INVERTED matrix; we use Tinv-with-normalizer-undone
             * to match.
             *
             * Note: legacy line 275 overwrites dx_i with sqrt(V_i * trace).
             * Trace here is the trace of the RAW (un-inverted) NV_T per
             * legacy semantics (CellP[i].NV_T at that program point is the
             * raw accumulated NV_T scattered back from accum, not yet
             * inverted). */
            const double trace_raw_NV_T = (double)accum.NV_T[0][0] + (double)accum.NV_T[1][1] + (double)accum.NV_T[2][2];
            const double dx_i = sqrt(V_i * trace_raw_NV_T);

            double face_in[3];
            face_in[0] = (double)accum.NV_T_face_weights[0];
            face_in[1] = (double)accum.NV_T_face_weights[1];
            face_in[2] = (double)accum.NV_T_face_weights[2];

            /* Full 3x3 inverted (Tinv stored upper triangle in legacy line
             * 292; legacy then reads CellP[i].NV_T as a full symmetric
             * matrix in the face_out sum, with implicit lower-triangle
             * symmetry — matrix_invert_ndims returns full Tinv). */
            double NV_T_inverted[3][3];
            for (int k1 = 0; k1 < 3; ++k1) {
                for (int k2 = 0; k2 < 3; ++k2) {
                    NV_T_inverted[k1][k2] = Tinv[k1][k2] / dimensional_NV_T_normalizer;
                }
            }

            double face_out[3] = {0, 0, 0};
            for (int k1 = 0; k1 < 3; ++k1) {
                for (int k2 = 0; k2 < 3; ++k2) {
                    face_out[k1] += 2.0 * V_i * NV_T_inverted[k1][k2] * face_in[k2];
                }
            }
            double dimless_face_leak = 0;
            for (int k1 = 0; k1 < 3; ++k1) dimless_face_leak += fabs(face_out[k1]) / (double)NUMDIMS;

    #ifdef HYDRO_KERNEL_SURFACE_VOLCORR
            double closure_asymm = 0;
            for (int k1 = 0; k1 < 3; ++k1) closure_asymm += face_in[k1] * face_in[k1];
            const double particle_inverse_volume = NumNgb_eff / (VOLUME_NORM_COEFF_FOR_NDIMS * pow(h, (double)NUMDIMS));
            closure_asymm = sqrt(closure_asymm) / (h * particle_inverse_volume);
            face_closure_error_local = DMIN(DMAX(1.0259 - 2.52444 * closure_asymm, 0.344301), 1.0);
    #else
            face_closure_error_local = dimless_face_leak / (2.0 * (double)NUMDIMS * pow(dx_i, (double)(NUMDIMS - 1)));
    #endif
        }

        /* ---- (c) iter==0 ConditionNumber update logic (mirrors
         * density.cc:313-322). Reads previous-call stored CellP[i].ConditionNumber;
         * decides whether to "promote" this iter's CN. Result lives in
         * scratch.condition_number_current; legacy's CellP[i] write is
         * deferred to finalize. ---- */
        if (P[i].Type == 0) {
            const double c0 = 0.1 * (double)CONDITION_NUMBER_DANGER;
            if ((iter == 0) && (ConditionNumber > CellP[i].ConditionNumber)
                            && (CellP[i].ConditionNumber > 0)) {
                double ncorr_a = 1.0, cn_a = CellP[i].ConditionNumber;
                if (cn_a > c0) ncorr_a = sqrt(1.0 + (cn_a - c0) / (double)CONDITION_NUMBER_DANGER);
                if (ncorr_a > 2.0) ncorr_a = 2.0;
                const double dn_ngb     = fabs(NumNgb_eff - cs.des_num_ngb * ncorr_a) / (desnumngbdev_0 * ncorr_a);

                double ncorr_b = 1.0, cn_b = ConditionNumber;
                if (cn_b > c0) ncorr_b = sqrt(1.0 + (cn_b - c0) / (double)CONDITION_NUMBER_DANGER);
                if (ncorr_b > 2.0) ncorr_b = 2.0;
                const double dn_ngb_alt = fabs(NumNgb_eff - cs.des_num_ngb * ncorr_b) / (desnumngbdev_0 * ncorr_b);

                const double dn_ngb_min = DMIN(dn_ngb, dn_ngb_alt);
                if (dn_ngb_min < 10.0) {
                    scratch.condition_number_current = ConditionNumber;  /* legacy line 322's write — but to scratch, NOT CellP */
                    scratch.cn_initialized = true;
                }
            }
            if (!scratch.cn_initialized) {
                /* No iter==0 promotion fired: use stored prev-call value. */
                scratch.condition_number_current = CellP[i].ConditionNumber;
                scratch.cn_initialized = true;
            }
        }

        /* ---- (d) ncorr_ngb from effective CN + FaceClosureError nudge
         * (mirrors density.cc:324-327). FaceClosureError nudge fires INSIDE
         * the iter loop, modifies ncorr_ngb, changes convergence trajectory.
         * Cannot be deferred. ---- */
        double ncorr_ngb = 1.0;
        if (P[i].Type == 0) {
            const double c0 = 0.1 * (double)CONDITION_NUMBER_DANGER;
            const double cn = scratch.condition_number_current;
            if (cn > c0) ncorr_ngb = sqrt(1.0 + (cn - c0) / (double)CONDITION_NUMBER_DANGER);
            if (ncorr_ngb > 2.0) ncorr_ngb = 2.0;
    #if !defined(HYDRO_KERNEL_SURFACE_VOLCORR)
            const double d00 = 0.35;
            if (face_closure_error_local > d00) {
                ncorr_ngb = DMAX(ncorr_ngb, DMIN(face_closure_error_local / d00, 2.0));
            }
    #endif
        }
        desnumngb    = cs.des_num_ngb * ncorr_ngb;
        desnumngbdev = desnumngbdev_0 * ncorr_ngb;
    #if !defined(EOS_ELASTIC)
        if (iter > 1) {
            desnumngbdev = DMIN(0.25 * desnumngb,
                                desnumngbdev * exp(0.1 * log(desnumngb / (16.0 * desnumngbdev)) * (double)iter));
        }
    #endif

    #ifdef SINK_PARTICLES
        if (P[i].Type == 5) {
            desnumngb = cs.des_num_ngb * cs.sink_ngb_factor;
      #ifdef SINGLE_STAR_SINK_DYNAMICS
            desnumngbdev = (cs.sink_ngb_factor + 1.0);
      #else
            desnumngbdev = 4.0 * (cs.sink_ngb_factor + 1.0);
      #endif
        }
    #endif

    #ifdef GRAIN_FLUID
        if ((1 << P[i].Type) & (GRAIN_PTYPES)) {
            desnumngb    = cs.des_num_ngb;
            desnumngbdev = cs.des_num_ngb / 4.0;
      #if defined(GRAIN_BACKREACTION)
            desnumngbdev = desnumngbdev_0;
      #endif
        }
    #endif

        double minsoft = cs.min_kernel_radius;
        double maxsoft = cs.max_kernel_radius;

    #ifdef DO_DENSITY_AROUND_NONGAS_PARTICLES
        /* legacy density.cc:153-159 constants — Config-resolved at compile time */
        int valid_stellar_types = 2+4+8+16, invalid_stellar_types = 1+32;
      #if (defined(GRAIN_FLUID) || defined(RADTRANSFER)) && (!defined(GALSF) && !(defined(GALSF_FB_MECHANICAL) || defined(GALSF_FB_THERMAL)))
        valid_stellar_types = 16; invalid_stellar_types = 1+2+4+8+32;
        #ifdef RADTRANSFER
        invalid_stellar_types = 64; valid_stellar_types = RT_SOURCES;
        #endif
      #endif
        if (((1 << P[i].Type) & (valid_stellar_types)) && !((1 << P[i].Type) & (invalid_stellar_types))) {
            desnumngb = cs.des_num_ngb;
      #if defined(RT_SOURCE_INJECTION)
            if (desnumngb < 64.0) desnumngb = 64.0;
      #endif
      #ifdef GRAIN_RDI_TESTPROBLEM_LIVE_RADIATION_INJECTION
            desnumngb = DMAX(desnumngb, 128.0);
            if (KERNEL_FUNCTION > 3) desnumngb = DMAX(desnumngb, 256.0);
      #endif
      #ifdef GALSF
            if (desnumngb < 64.0) desnumngb = 64.0;
            const double unitlength_in_kpc = UNIT_LENGTH_IN_KPC * cs.common.cf_atime;
            maxsoft = 2.0 / unitlength_in_kpc;
        #if defined(GALSF_FB_FIRE_STELLAREVOLUTION) && defined(SINK_PARTICLES) && (defined(GALSF_FB_MECHANICAL) || defined(GALSF_FB_THERMAL))
            if ((P[i].SNe_ThisTimeStep > 0) || (P[i].MassReturn_ThisTimeStep > 0)
                    || (cs.time_now == cs.time_begin)) {
                maxsoft = 2.0 / unitlength_in_kpc;
            } else {
                maxsoft = 0.1 / unitlength_in_kpc;
            }
        #endif
      #endif
            desnumngbdev = desnumngb / 2.0;
        }
    #endif

    #ifdef SINK_PARTICLES
        if (P[i].Type == 5) {
            maxsoft = cs.sink_max_accretion_radius / cs.common.cf_atime;
      #ifdef SINGLE_STAR_SINK_DYNAMICS
            minsoft = SinkParticle_GravityKernelRadius;
        #ifdef SINK_GRAVCAPTURE_FIXEDSINKRADIUS
            minsoft = DMAX(minsoft, DMIN(P[i].SinkRadius, 0.1 * SinkParticle_GravityKernelRadius));
        #endif
      #endif
        }
    #endif

        /* ---- (e) Convergence check + bracket / radius update (mirrors
         * density.cc:398-589). The legacy mutates P[i].KernelRadius
         * directly; we compute new_h_search locally and return via
         * IterResult. min/max kernel flags are per-iter scratch (not
         * IterScratch fields). ---- */
        int redo_particle = 0;
        int particle_set_to_minrkern_flag = 0;
        int particle_set_to_maxrkern_flag = 0;
        double new_h = h;  /* candidate next radius; legacy mutates P[i].KernelRadius */

        /* normal range check (line 401-403) */
        if ((NumNgb_eff < (desnumngb - desnumngbdev) && new_h < 0.999 * maxsoft) ||
            (NumNgb_eff > (desnumngb + desnumngbdev) && new_h > 1.001 * minsoft)) {
            redo_particle = 1;
        }

        /* max kernel size (line 406-420) */
        if ((new_h >= 0.999 * maxsoft) && (NumNgb_eff < (desnumngb - desnumngbdev))) {
            redo_particle = 0;
            if (new_h == maxsoft) {
                particle_set_to_maxrkern_flag = 0;
            } else {
                redo_particle = 1;
                new_h = maxsoft;
                particle_set_to_maxrkern_flag = 1;
            }
        }

        /* min kernel size (line 423-438) */
        if ((new_h <= 1.001 * minsoft) && (NumNgb_eff > (desnumngb + desnumngbdev))) {
            redo_particle = 0;
            if (new_h == minsoft) {
                particle_set_to_minrkern_flag = 0;
            } else {
                redo_particle = 1;
                new_h = minsoft;
                particle_set_to_minrkern_flag = 1;
            }
        }

    #ifdef GALSF
        if (cs.common.comoving_integration_on && cs.time_now > cs.time_begin) {
            if ((P[i].Type == 4) && (iter > 1) && (NumNgb_eff > 4) && (NumNgb_eff < 100) && (redo_particle == 1)) {
                redo_particle = 0;
            }
        }
    #endif

        if ((redo_particle == 0) && (P[i].Type == 0)) {
            /* legacy line 447-459: condition-number diagnostic + persist CN
             * to CellP[i].ConditionNumber. Persist is deferred to finalize;
             * here we just stash ConditionNumber into scratch for finalize. */
            scratch.condition_number_current = ConditionNumber;
            /* legacy PRINT_WARNING for ConditionNumber > 1e6 * threshold is a
             * diagnostic only — preserved here for fidelity. */
            if (ConditionNumber > 1e6 * (double)CONDITION_NUMBER_DANGER) {
                PRINT_WARNING("Condition number=%g Cnum_prev=%g threshold=%g iter=%d NumNgb=%g desnumngb=%g KernelRadius=%g i=%d task=%d ID=%llu Type=%d Left=%g Right=%g",
                    ConditionNumber, CellP[i].ConditionNumber, (double)CONDITION_NUMBER_DANGER,
                    iter, NumNgb_eff, desnumngb, new_h, i, this_task,
                    (unsigned long long) P[i].ID, P[i].Type,
                    (double)scratch.left, (double)scratch.right);
            }
        }

        if (redo_particle) {
            /* legacy line 463-468: iter>=MAXITER-10 warning */
            if (iter >= MAXITER - 10) {
                PRINT_WARNING("i=%d task=%d ID=%llu iter=%d Type=%d KernelRadius=%g Drkern=%g Left=%g Right=%g NumNgb=%g Right-Left=%g maxh=%d minh=%d minsoft=%g maxsoft=%g desnum=%g desnumdev=%g",
                    i, this_task, (unsigned long long) P[i].ID, iter, P[i].Type,
                    new_h, DrkernNgbFactor_eff,
                    (double)scratch.left, (double)scratch.right,
                    NumNgb_eff, (double)scratch.right - (double)scratch.left,
                    particle_set_to_maxrkern_flag, particle_set_to_minrkern_flag,
                    minsoft, maxsoft, desnumngb, desnumngbdev);
            }

            /* tight-bracket early exit (legacy line 474-482).
             * Returns Converged — caller in this case in legacy did `npleft--;
             * P[i].TimeBin = -P[i].TimeBin - 1; continue;` which marks the
             * particle done. */
            if (scratch.left > 0 && scratch.right > 0) {
                if ((double)(scratch.right - scratch.left) < 1.0e-3 * (double)scratch.left) {
                    scratch.converged = true;
                    scratch.condition_number_current = ConditionNumber;
                    /* Final radius travels to finalize through the single-valued
                     * apply_active_writeback_iterative channel. */
                    return IterResult{IterStatus::Converged, (double)new_h};
                }
            }

            if ((particle_set_to_maxrkern_flag == 0) && (particle_set_to_minrkern_flag == 0)) {
                /* bracket update (line 486-490) */
                if (NumNgb_eff < (desnumngb - desnumngbdev)) {
                    scratch.left = (MyFloat)DMAX(new_h, (double)scratch.left);
                } else {
                    if (scratch.right != 0) {
                        if (new_h < (double)scratch.right) scratch.right = (MyFloat)new_h;
                    } else {
                        scratch.right = (MyFloat)new_h;
                    }
                }

                /* new radius computation (line 493-583) */
                if (scratch.right > 0 && scratch.left > 0) {
                    /* both brackets set: geometric interpolation */
                    double maxjump = 0;
                    if (iter > 1) maxjump = 0.2 * log((double)scratch.right / (double)scratch.left);
                    if (NumNgb_eff > 1) {
                        double jumpvar = DrkernNgbFactor_eff * log(desnumngb / NumNgb_eff) / NUMDIMS;
                        if (iter > 1) {
                            if (fabs(jumpvar) < maxjump) {
                                if (jumpvar < 0) jumpvar = -maxjump; else jumpvar = maxjump;
                            }
                        }
                        new_h *= exp(jumpvar);
                    } else {
                        new_h *= 2.0;
                    }
                    if ((new_h < (double)scratch.right) && (new_h > (double)scratch.left)) {
                        if (iter > 1) {
                            const double hfac = exp(maxjump);
                            if (new_h > (double)scratch.right / hfac) new_h = (double)scratch.right / hfac;
                            if (new_h < (double)scratch.left  * hfac) new_h = (double)scratch.left  * hfac;
                        }
                    } else {
                        if (new_h > (double)scratch.right) new_h = (double)scratch.right;
                        if (new_h < (double)scratch.left)  new_h = (double)scratch.left;
                        new_h = pow(new_h * (double)scratch.left * (double)scratch.right, 1.0 / 3.0);
                    }
                } else {
                    if (scratch.right == 0 && scratch.left == 0) {
                        printf("DensitySpec::after_iter: Right==0 && Left==0 && KernelRadius=%g (task=%d i=%d)\n", new_h, this_task, i); KOKKOS_IF_ON_HOST((fflush(stdout);))
                        endrun(90001009);
                        return IterResult{IterStatus::Converged, (double)new_h};   /* graceful: bad-stop set; stop iterating this active with the last finite radius; drains at runner completion -> phase poll (per-active: NO immediate collective) */
                    }

                    if (scratch.right == 0 && scratch.left > 0) {
                        double fac_lim;
                        if (NumNgb_eff > 1) {
                            fac_lim = log(desnumngb / NumNgb_eff) / NUMDIMS;
                        } else {
                            fac_lim = 1.4;
                        }
                        if ((NumNgb_eff < 2.0 * desnumngb) && (NumNgb_eff > 0.1 * desnumngb)) {
                            double slope = DrkernNgbFactor_eff;
                            if (iter > 2 && slope < 1.0) slope = 0.5 * (slope + 1.0);
                            double fac = fac_lim * slope;
                            if (iter >= 4) { if (DrkernNgbFactor_eff == 1.0) fac *= 10.0; }
                            if (fac < fac_lim + 0.231) {
                                new_h *= exp(fac);
                            } else {
                                new_h *= exp(fac_lim + 0.231);
                            }
                        } else {
                            new_h *= exp(fac_lim);
                        }
                    }

                    if (scratch.right > 0 && scratch.left == 0) {
                        double fac_lim;
                        if (NumNgb_eff > 1) {
                            fac_lim = log(desnumngb / NumNgb_eff) / NUMDIMS;
                        } else {
                            fac_lim = 1.4;
                        }
                        if (fac_lim < -1.535) fac_lim = -1.535;
                        if ((NumNgb_eff < 2.0 * desnumngb) && (NumNgb_eff > 0.1 * desnumngb)) {
                            double slope = DrkernNgbFactor_eff;
                            if (iter > 2 && slope < 1.0) slope = 0.5 * (slope + 1.0);
                            double fac = fac_lim * slope;
                            if (iter >= 4) { if (DrkernNgbFactor_eff == 1.0) fac *= 10.0; }
                            if (fac > fac_lim - 0.231) {
                                new_h *= exp(fac);
                            } else {
                                new_h *= exp(fac_lim - 0.231);
                            }
                        } else {
                            new_h *= exp(fac_lim);
                        }
                    }
                }
            }
            /* min/max clamps (line 586-589) */
            if (new_h < minsoft) new_h = minsoft;
            if (particle_set_to_minrkern_flag == 1) new_h = minsoft;
            if (new_h > maxsoft) new_h = maxsoft;
            if (particle_set_to_maxrkern_flag == 1) new_h = maxsoft;

            scratch.iter_count = iter + 1;
            return IterResult{IterStatus::AdjustRadius, new_h};
        }

        /* redo_particle == 0: converged */
        scratch.converged = true;
        if (P[i].Type == 0) {
            scratch.condition_number_current = ConditionNumber;
        }
        /* Final radius travels to finalize through the single-valued
         * apply_active_writeback_iterative channel. */
        return IterResult{IterStatus::Converged, (double)new_h};
    }

    /* after_iter_global NOT declared — density has no per-rank
     * post-iter side-effect (bisection state is per-active, no MPI
     * inside the iter loop). Runner SFINAE-detects absence. */

    /* ====================================================================
     * Device hooks. Header-inline because the runner instantiates the
     * pair body from GPU translation units.
     * ==================================================================== */

    /* Per-field init with MIN-reduction sentinels for Sink fields. */
    KOKKOS_INLINE_FUNCTION
    static void zero_accum(AccumData& accum) {
        accum.Ngb              = 0;
        accum.Rho              = 0;
        accum.DrkernNgb        = 0;
        accum.Particle_DivVel  = 0;
        for(int a = 0; a < 6; ++a) accum.NV_T.data[a] = 0;
        accum.NV_T_face_weights[0] = 0;
        accum.NV_T_face_weights[1] = 0;
        accum.NV_T_face_weights[2] = 0;
#if defined(HYDRO_MESHLESS_FINITE_VOLUME) && ((HYDRO_FIX_MESH_MOTION==5)||(HYDRO_FIX_MESH_MOTION==6))
        accum.ParticleVel[0] = 0; accum.ParticleVel[1] = 0; accum.ParticleVel[2] = 0;
#endif
#ifdef HYDRO_SPH
        accum.DrkernHydroSumFactor = 0;
#endif
#ifdef RT_SOURCE_INJECTION
        accum.KernelSum_Around_RT_Source = 0;
#endif
#ifdef HYDRO_PRESSURE_SPH
        accum.EgyRho = 0;
#endif
#if defined(SPHAV_CD10_VISCOSITY_SWITCH)
        for(int a = 0; a < 3; ++a) for(int b = 0; b < 3; ++b) {
            accum.NV_D[a][b] = 0; accum.NV_A[a][b] = 0;
        }
#endif
#ifdef DO_DENSITY_AROUND_NONGAS_PARTICLES
        accum.GradRho[0] = 0; accum.GradRho[1] = 0; accum.GradRho[2] = 0;
#endif
#if defined(SINK_PARTICLES)
        accum.Sink_TimeBinGasNeighbor       = TIMEBINS;          /* MIN-reduction sentinel */
  #if defined(BH_ACCRETE_NEARESTFIRST) || defined(SINGLE_STAR_TIMESTEPPING)
        accum.Sink_dr_to_NearestGasNeighbor = MAX_REAL_NUMBER;   /* MIN-reduction sentinel */
  #endif
#endif
#if defined(TURB_DRIVING) || defined(DO_FLUID_ALTSPECIES_DRAG_CALCULATION)
        accum.GasVel[0] = 0; accum.GasVel[1] = 0; accum.GasVel[2] = 0;
#endif
#if defined(DO_FLUID_ALTSPECIES_DRAG_CALCULATION)
        accum.AmbientGasRho = 0;
        accum.Gas_Temperature = 0;
        accum.Gas_fion = 0;
  #if defined(DO_FLUID_DRAG_CALCULATION_WITHBFIELDS)
        accum.Gas_B[0] = 0; accum.Gas_B[1] = 0; accum.Gas_B[2] = 0;
  #endif
  #if defined(GRAIN_EVOLUTION) && (GRAIN_EVOLUTION & (32|64))
        for(int kv = 0; kv < GRAIN_NUM_VOLATILE_SPECIES; ++kv) accum.Gas_VolatileSpecies[kv] = 0;
  #endif
#endif
#ifdef HYDRO_PARTITION_UNITY_IMPROVE_FD
        accum.GradH_numer[0] = 0; accum.GradH_numer[1] = 0; accum.GradH_numer[2] = 0;
        accum.GradH_denom = 0;
#endif
    }

    /* load_active — snapshot P[i] / CellP[i] fields into ActiveData.
     * Reads through ctx.P / ctx.CellP (path-correct: Mode A arena P_gpu,
     * Mode B request-driven slab).
     *
     * Two legacy-semantics requirements are important because the pair
     * kernel inherits whatever load_active produces:
     *
     *   1. Vel: legacy (density_gpu.cc particle2in) uses CellP[i].VelPred
     *      for gas (Type==0) and P[i].Vel for non-gas. Pair kernel reads
     *      `local->Vel` as the i-side velocity in divergence/curl terms.
     *      Using P[i].Vel uniformly silently corrupts gas density physics.
     *
     *   2. Accel (SPHAV_CD10 only): legacy stores the PRE-SCALED form
     *      `All.cf_a2inv * P[i].GravAccel + CellP[i].HydroAccel`
     *      (density_gpu.cc:310). The cf_a2inv multiply is component-
     *      specific (gravity term only), not applied to the sum.
     *      density_functions.h:153-161 then subtracts the j-side analogue
     *      inline. Storing the raw sum here and "applying cf_a2inv later"
     *      breaks the component separation. Pre-scale at load_active. */
    KOKKOS_INLINE_FUNCTION
    static ActiveData load_active(const NeighborLoopDeviceContextBase& ctx,
                                  int active_slot, int i,
                                  double h_search, const CallScalars& scalars) {
        ActiveData a{};
        a.pos[0]    = (MyDouble)ctx.P[i].Pos[0];
        a.pos[1]    = (MyDouble)ctx.P[i].Pos[1];
        a.pos[2]    = (MyDouble)ctx.P[i].Pos[2];
        a.h_search  = (MyFloat) h_search;
        a.Type      = (int)     ctx.P[i].Type;

        /* Vel: gas uses CellP[i].VelPred; non-gas uses P[i].Vel
         * Mirrors density_gpu.cc particle2in. */
        const bool i_is_gas = (ctx.P[i].Type == 0) && (ctx.CellP != nullptr);
        if (i_is_gas) {
            a.Vel[0] = (MyFloat)ctx.CellP[i].VelPred[0];
            a.Vel[1] = (MyFloat)ctx.CellP[i].VelPred[1];
            a.Vel[2] = (MyFloat)ctx.CellP[i].VelPred[2];
        } else {
            a.Vel[0] = (MyFloat)ctx.P[i].Vel[0];
            a.Vel[1] = (MyFloat)ctx.P[i].Vel[1];
            a.Vel[2] = (MyFloat)ctx.P[i].Vel[2];
        }

#if defined(SPHAV_CD10_VISCOSITY_SWITCH)
        /* Accel: PRE-SCALED legacy form for gas; zero for non-gas
         * (SPHAV_CD10 NV_A accumulator only fires on gas pairs in the
         * legacy kernel; safe zero for non-gas i means no contribution
         * if any branch fires). */
        if (i_is_gas) {
            a.Accel[0] = (MyFloat)(scalars.common.cf_a2inv * ctx.P[i].GravAccel[0] + ctx.CellP[i].HydroAccel[0]);
            a.Accel[1] = (MyFloat)(scalars.common.cf_a2inv * ctx.P[i].GravAccel[1] + ctx.CellP[i].HydroAccel[1]);
            a.Accel[2] = (MyFloat)(scalars.common.cf_a2inv * ctx.P[i].GravAccel[2] + ctx.CellP[i].HydroAccel[2]);
        } else {
            a.Accel[0] = 0; a.Accel[1] = 0; a.Accel[2] = 0;
        }
#endif

#ifdef GALSF_SUBGRID_WINDS
        a.DelayTime = i_is_gas ? (MyFloat)ctx.CellP[i].DelayTime : (MyFloat)0;
#endif

#ifdef HYDRO_MULTIFLUID
        a.FluidType = ctx.P[i].FluidType;
#endif

        (void)active_slot;
        (void)scalars;
        return a;
    }

    /* load_neighbor — route ctx.P[j] / ctx.CellP[j] pointers.
     *
     * Density requests gas-only neighbors. Imported gas ghosts carry valid
     * CellP state in the runner's active P/CellP view, so do not guard by
     * local gas capacity here; doing so silently drops cross-rank gas
     * neighbors. */
    KOKKOS_INLINE_FUNCTION
    static NeighborData load_neighbor(const NeighborLoopDeviceContextBase& ctx,
                                      int j, const IdentitySidecar& id,
                                      const ActiveData& active) {
        NeighborData n{};
        n.neighbor_particle = &ctx.P[j];
        n.neighbor_cell     = (ctx.CellP && ctx.P[j].Type == 0)
                              ? &ctx.CellP[j] : nullptr;
        n.neighbor_index    = j;
        (void)id; (void)active;
        return n;
    }

    /* pair_kernel — literal port of density_accumulate_neighbor +
     * density_evaluate_extra_physics_gas (density_functions.h:104-249).
     * Every legacy `#ifdef` branch ports verbatim per §5 + the no-#error
     * rule (feedback_port_all_branches_no_error_endstate.md).
     *
     * Signature matches runner contract: 4 args (no `int j` — see
     * mesh/neighbor_loop_runner.cc:854/1825/3147). Index j is encoded
     * in neighbor.neighbor_particle/neighbor_cell pointers.
     *
     * j is always gas (neighbor_type_mask = (1u << 0)); the runner's
     * load_neighbor populates neighbor_cell for every j the runner
     * dispatches. Keep the legacy `local->Type == 0` gate around SPHAV_CD10 /
     * TURB_DRIVING / NV_T blocks exactly as-is. */
    KOKKOS_INLINE_FUNCTION
    static void pair_kernel(const ActiveData& i_active,
                            const NeighborData& neighbor,
                            AccumData& accum, NoScatter& /*scatter*/,
                            const CallScalars& cs) {

        struct particle_data       &Pj    = *neighbor.neighbor_particle;
        struct gas_cell_data * const CellPj = neighbor.neighbor_cell;
        /* j is always gas under neighbor_type_mask = (1u<<0); CellPj is
         * non-null for every j the runner walks. The early-return below
         * is a defensive sanity check, valid ONLY while neighbor_type_mask
         * stays gas-only. If a future contributor widens the j-mask, this
         * early return would silently drop non-gas neighbors instead of
         * processing them — rewrite the body to handle non-gas j first,
         * THEN drop the gate. */
        if (CellPj == nullptr) return;

#ifdef GALSF_SUBGRID_WINDS
        if (CellPj->DelayTime > 0) { if (i_active.DelayTime <= 0) return; }
#endif
        if (Pj.Mass <= 0) return;

        /* Position delta with periodic wrap (legacy line 195-197). */
        Vec3<double> dp;
        dp[0] = (double)i_active.pos[0] - (double)Pj.Pos[0];
        dp[1] = (double)i_active.pos[1] - (double)Pj.Pos[1];
        dp[2] = (double)i_active.pos[2] - (double)Pj.Pos[2];
        nearest_xyz(dp);
        const double r2 = dp.norm_sq();
        const double h  = (double)i_active.h_search;
        const double h2 = h * h;
        if (r2 >= h2) return;

        /* Kernel evaluation (legacy line 200-204). */
        const double r = sqrt(r2);
        double hinv, hinv3, hinv4;
        kernel_hinv(h, &hinv, &hinv3, &hinv4);
        const double u = r * hinv;
        double wk, dwk;
        kernel_main(u, hinv3, hinv4, &wk, &dwk, 0);
        const double mass_j = (double)Pj.Mass;
        const double mj_wk  = mass_j * wk;

#ifdef DO_FLUID_ALTSPECIES_DRAG_CALCULATION
        int is_valid_fordrag = IS_PARTICLE_DRAGVALID(i_active.Type, i_active.FluidType);
#ifdef HYDRO_MULTIFLUID
        is_valid_fordrag = (is_valid_fordrag && neighbor.neighbor_particle->FluidType == FLUID_DEFAULT);
#endif
        if (is_valid_fordrag) {
            /* Cross-fluid ambient gas sampling for cross-fluid sources. */
            if (r > 0) {
                accum.AmbientGasRho += mj_wk;
                Vec3<double> dv;
                dv[0] = (double)i_active.Vel[0] - (double)CellPj->VelPred[0];
                dv[1] = (double)i_active.Vel[1] - (double)CellPj->VelPred[1];
                dv[2] = (double)i_active.Vel[2] - (double)CellPj->VelPred[2];
                {
                    Vec3<MyDouble> pos_i_md;
                    pos_i_md[0] = i_active.pos[0]; pos_i_md[1] = i_active.pos[1]; pos_i_md[2] = i_active.pos[2];
                    NGB_SHEARBOX_BOUNDARY_VELCORR_(pos_i_md, Pj.Pos, dv, 1);
                }
                /* the temperature is what the dust modules want; a grain has no cell, so it cannot convert an
                   energy into one itself. the ionized fraction rides along for the same reason. */
                accum.Gas_Temperature += mj_wk * (double)CellPj->gas_temperature_at_u(CellPj->InternalEnergyPred);
#if (defined(COOLING) && !defined(CHIMES)) || (defined(RADTRANSFER) && defined(RT_CHEM_PHOTOION))
                accum.Gas_fion += mj_wk * (double)DMAX(0., DMIN(1., 1. - CellPj->HI));
#endif
                accum.GasVel[0] += mj_wk * ((double)i_active.Vel[0] - dv[0]); /* shares with TURB_DRIVING SmoothedVel; combo unsupported */
                accum.GasVel[1] += mj_wk * ((double)i_active.Vel[1] - dv[1]);
                accum.GasVel[2] += mj_wk * ((double)i_active.Vel[2] - dv[2]);
#if defined(DO_FLUID_DRAG_CALCULATION_WITHBFIELDS)
                accum.Gas_B[0] += wk * (double)CellPj->BPred[0];
                accum.Gas_B[1] += wk * (double)CellPj->BPred[1];
                accum.Gas_B[2] += wk * (double)CellPj->BPred[2];
#endif
#if defined(GRAIN_EVOLUTION) && (GRAIN_EVOLUTION & (32|64))
                for (int kv = 0; kv < GRAIN_NUM_VOLATILE_SPECIES; ++kv) {
                    accum.Gas_VolatileSpecies[kv] += mj_wk * (double)CellPj->VolatileSpecies[kv];
                }
#endif
            }
        }
#endif
#ifdef HYDRO_MULTIFLUID
        if (!same_lagrangian_fluid_id(i_active.FluidType, neighbor.neighbor_particle->FluidType)) return;
#endif

        /* Always-on accumulators (legacy line 206-217). */
        accum.Ngb += wk;
        accum.Rho += mj_wk;
#if defined(HYDRO_MESHLESS_FINITE_VOLUME) && ((HYDRO_FIX_MESH_MOTION==5)||(HYDRO_FIX_MESH_MOTION==6))
        if (i_active.Type == 0 && r == 0) {
            accum.ParticleVel[0] += mj_wk * (double)CellPj->VelPred[0];
            accum.ParticleVel[1] += mj_wk * (double)CellPj->VelPred[1];
            accum.ParticleVel[2] += mj_wk * (double)CellPj->VelPred[2];
        }
#endif
#if defined(RT_SOURCE_INJECTION)
  #if defined(RT_SINK_ANGLEWEIGHT_PHOTON_INJECTION)
        if (cs.time_step == 0)
  #endif
        if ((1 << i_active.Type) & (RT_SOURCES)) {
            accum.KernelSum_Around_RT_Source += 1.0 - u * u;
        }
#endif
        accum.DrkernNgb += -(NUMDIMS * hinv * wk + u * dwk);

#ifdef HYDRO_SPH
        {
            double mass_eff = mass_j;
  #ifdef HYDRO_PRESSURE_SPH
            mass_eff *= (double)CellPj->InternalEnergyPred;
            accum.EgyRho += wk * mass_eff;
  #endif
            accum.DrkernHydroSumFactor += -mass_eff * (NUMDIMS * hinv * wk + u * dwk);
        }
#endif

        /* r > 0 branch (legacy line 228-249). */
        if (r > 0) {
            if (i_active.Type == 0) {
                /* NV_T outer-product accumulation (gas i only). */
                accum.NV_T              += wk * outer_product(dp);
                accum.NV_T_face_weights[0] += wk * dp[0];
                accum.NV_T_face_weights[1] += wk * dp[1];
                accum.NV_T_face_weights[2] += wk * dp[2];
#ifdef HYDRO_PARTITION_UNITY_IMPROVE_FD
                accum.GradH_numer[0] += (dwk / r) * dp[0];
                accum.GradH_numer[1] += (dwk / r) * dp[1];
                accum.GradH_numer[2] += (dwk / r) * dp[2];
                accum.GradH_denom    += u * dwk;
#endif
            }
            /* Velocity delta (legacy line 240). j is always gas; CellP[j].VelPred valid. */
            Vec3<double> dv;
            dv[0] = (double)i_active.Vel[0] - (double)CellPj->VelPred[0];
            dv[1] = (double)i_active.Vel[1] - (double)CellPj->VelPred[1];
            dv[2] = (double)i_active.Vel[2] - (double)CellPj->VelPred[2];
            /* SHEARBOX boundary velocity correction (no-op when not defined). */
            {
                Vec3<MyDouble> pos_i_md;
                pos_i_md[0] = i_active.pos[0]; pos_i_md[1] = i_active.pos[1]; pos_i_md[2] = i_active.pos[2];
                NGB_SHEARBOX_BOUNDARY_VELCORR_(pos_i_md, Pj.Pos, dv, 1);
            }
#if defined(HYDRO_MESHLESS_FINITE_VOLUME) && ((HYDRO_FIX_MESH_MOTION==5)||(HYDRO_FIX_MESH_MOTION==6))
            accum.ParticleVel[0] += mj_wk * ((double)i_active.Vel[0] - dv[0]);
            accum.ParticleVel[1] += mj_wk * ((double)i_active.Vel[1] - dv[1]);
            accum.ParticleVel[2] += mj_wk * ((double)i_active.Vel[2] - dv[2]);
#endif
            accum.Particle_DivVel -= dwk * (dp[0]*dv[0] + dp[1]*dv[1] + dp[2]*dv[2]) / r;

            /* ============ density_evaluate_extra_physics_gas (inlined) ===========
             * Legacy at density_functions.h:103-179. Same `local->Type` gating
             * (Type!=0 vs Type==0) wraps the per-type physics. */
            const double mj_dwk_r = mass_j * dwk / r;

            if (i_active.Type != 0) {
#if defined(SINK_PARTICLES)
                if (i_active.Type == 5) {
                    /* Read-only MIN reductions; the j-side SwallowID/SwallowTime/Sink_Ngb_Flag
                     * writes from the CPU path are sink_swk/sink_feed responsibilities,
                     * NOT density. */
                    short int TimeBin_j = Pj.TimeBin;
                    if (TimeBin_j < 0) { TimeBin_j = -TimeBin_j - 1; }
                    if (accum.Sink_TimeBinGasNeighbor > TimeBin_j) {
                        accum.Sink_TimeBinGasNeighbor = TimeBin_j;
                    }
  #if defined(SINGLE_STAR_TIMESTEPPING)
                    double dr_eff_wtd = Pj.Get_Particle_Size();
                    dr_eff_wtd = sqrt(dr_eff_wtd * dr_eff_wtd + r * r);
                    if ((dr_eff_wtd < accum.Sink_dr_to_NearestGasNeighbor) && (Pj.Mass > 0)) {
                        accum.Sink_dr_to_NearestGasNeighbor = dr_eff_wtd;
                    }
  #endif
                }
#endif
            } else { /* i_active.Type == 0 */
#if defined(TURB_DRIVING)
                accum.GasVel[0] += mj_wk * ((double)i_active.Vel[0] - dv[0]);
                accum.GasVel[1] += mj_wk * ((double)i_active.Vel[1] - dv[1]);
                accum.GasVel[2] += mj_wk * ((double)i_active.Vel[2] - dv[2]);
#endif
#if defined(SPHAV_CD10_VISCOSITY_SWITCH)
                /* i_active.Accel is the PRE-SCALED legacy form
                 * (load_active applied cf_a2inv to GravAccel component only).
                 * Per-pair, subtract the j-side analogue inline. */
                {
                    const double gax_j = cs.common.cf_a2inv * (double)Pj.GravAccel[0] + (double)CellPj->HydroAccel[0];
                    const double gay_j = cs.common.cf_a2inv * (double)Pj.GravAccel[1] + (double)CellPj->HydroAccel[1];
                    const double gaz_j = cs.common.cf_a2inv * (double)Pj.GravAccel[2] + (double)CellPj->HydroAccel[2];
                    const double a0 = (double)i_active.Accel[0] - gax_j;
                    const double a1 = (double)i_active.Accel[1] - gay_j;
                    const double a2 = (double)i_active.Accel[2] - gaz_j;
                    accum.NV_A[0][0] += a0 * dp[0] * wk;
                    accum.NV_A[0][1] += a0 * dp[1] * wk;
                    accum.NV_A[0][2] += a0 * dp[2] * wk;
                    accum.NV_A[1][0] += a1 * dp[0] * wk;
                    accum.NV_A[1][1] += a1 * dp[1] * wk;
                    accum.NV_A[1][2] += a1 * dp[2] * wk;
                    accum.NV_A[2][0] += a2 * dp[0] * wk;
                    accum.NV_A[2][1] += a2 * dp[1] * wk;
                    accum.NV_A[2][2] += a2 * dp[2] * wk;

                    accum.NV_D[0][0] += dv[0] * dp[0] * wk;
                    accum.NV_D[0][1] += dv[0] * dp[1] * wk;
                    accum.NV_D[0][2] += dv[0] * dp[2] * wk;
                    accum.NV_D[1][0] += dv[1] * dp[0] * wk;
                    accum.NV_D[1][1] += dv[1] * dp[1] * wk;
                    accum.NV_D[1][2] += dv[1] * dp[2] * wk;
                    accum.NV_D[2][0] += dv[2] * dp[0] * wk;
                    accum.NV_D[2][1] += dv[2] * dp[1] * wk;
                    accum.NV_D[2][2] += dv[2] * dp[2] * wk;
                }
#endif
            }

#ifdef DO_DENSITY_AROUND_NONGAS_PARTICLES
            accum.GradRho[0] += mj_dwk_r * dp[0];
            accum.GradRho[1] += mj_dwk_r * dp[1];
            accum.GradRho[2] += mj_dwk_r * dp[2];
#endif
        } /* end r > 0 */
    }
};

#endif /* DENSITY_LOOP_H */
