/* gravity/ags_force_loop.h — AGS-force neighbor loop module (Wave 3 close-out).
 *
 * Defines AgsForceSpec for the iterative-shaped runner-template contract
 * (mesh/neighbor_loop_runner.h). Replaces gravity/ags_force_gpu.cc (bespoke
 * GPU evaluator) and mesh/ghost_writeback.cc::ghost_writeback_{zero_,}agsforce
 * (hand-written snapshot-diff).
 *
 * Single-pass physics (max_iters=1, after_iter always Converged) but routed
 * through the iterative runner so the multi-subgroup contract works (one
 * subgroup per j_type_bitmask returned by ags_gravity_kernel_shared_BITFLAG).
 * Same iterative-shaped pattern used by mechfb / radfb_rp / dm_dispersion.
 *
 * Three deliberate corrections vs the retired GPU file:
 *   1. SIDM RNG gets a per-loop salt (AGS_FORCE_RNG_SALT = FNV-1a("ags_force"))
 *      mixed into the counter. Was unsalted in legacy — could correlate with
 *      any other loop sharing (Ti_Current, ID_i^ID_j, tag). Oracle parity is
 *      preserved (both pass through the same salted stream).
 *   2. Ghost wakeup zeroed BEFORE the bundle snapshot (matches legacy
 *      ghost_writeback_zero_agsforce event semantics that generic PARTICLE_MAX
 *      alone wouldn't preserve when the imported ghost wakeup is already
 *      nonzero).
 *   3. j-side atomic writes (Vel / dp / NInteractions / wakeup) gated on
 *      !oracle_dry_run, enabling the Wave 3 PB-O oracle compare. i-side
 *      accumulator writes remain unconditional (oracle compares them
 *      field-for-field after the brute pass).
 *
 * Pair-body physics: verbatim port of gravity/ags_force_gpu.cc:222-360 with
 * the runner's NeighborData adapter. The per-pair flux templates in
 * sidm/{cbe_integrator,dm_fuzzy,sidm_core}_flux_functions.h remain the SSOT
 * for CBE/DM_FUZZY/SIDM scatter physics.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) and Claude for GIZMO.
 */
#ifndef AGS_FORCE_LOOP_H
#define AGS_FORCE_LOOP_H

/* Kokkos_Core must precede allvars.h (its macros may conflict with stdlib
 * names). */
#include <Kokkos_Core.hpp>

#include "../declarations/allvars.h"

#ifdef AGS_KERNELRADIUS_CALCULATION_IS_ACTIVE

#include "../declarations/gpu_rng.h"
#include "../mesh/neighbor_loop_runner.h"
#include "../mesh/mode_b_local_walker.h"   /* MODE_B_SEARCH_*, MODE_B_RADIUS_* */
/* NOTE: caller TUs must include "../mesh/kernel.h" before this header
 * (kernel_main / kernel_hinv used by the inline pair body; no include guard). */
#include "../core/timestep_functions.h"     /* get_particle_timestep_in_physical
                                              * (KOKKOS_INLINE — must be visible in
                                              * neighbor_loop_runner.cc at instantiation) */
#include "ags_functions.h"                  /* get_particle_volume_ags_P,
                                              return_grain_cross_section_per_unit_mass_P */

/* Per-pair flux helper templates the inline pair body calls. The Spec
 * header must own these includes (NOT push them onto caller TUs): the
 * runner.cc explicit instantiation of run_neighbor_loop_iterative<AgsForceSpec>
 * transitively instantiates Spec::pair_kernel → ags_force_pair_kernel_body
 * → cbe/dm_fuzzy/sidm_core_flux_compute_pair, so those template definitions
 * MUST be visible everywhere this header is included, not just in the .cc. */
#include "../sidm/cbe_integrator_flux_functions.h"
#include "../sidm/dm_fuzzy_flux_functions.h"
#include "../sidm/sidm_core_flux_functions.h"

/* No KOKKOS_INLINE_FUNCTION fallback here — Kokkos_Core.hpp is included
 * unconditionally above. This Spec carries device-callable pair-kernel
 * accessors; misordered includes must compile-fail loudly, not silently
 * resolve to host-only `inline`. Same convention as neighbor_loop_runner.h. */

/* Forward decls — defined in gravity/ags_rkern.cc (host SSOT). */
int AGSForce_isactive(int i);
int ags_gravity_kernel_shared_BITFLAG(short int particle_type_primary);

/* Toplevel entry point — implemented in gravity/ags_force_loop.cc. Replaces
 * the legacy body that lived in gravity/ags_rkern.cc. */
void AGSForce_calc(void);


/* ============================================================================
 * Per-loop RNG salt — XOR-mixed into the SIDM scatter counter so this loop's
 * pair RNG stream is independent of any other counter-based RNG site in the
 * codebase. Compile-time FNV-1a hash; see declarations/gpu_rng.h.
 * ========================================================================== */
static constexpr uint64_t AGS_FORCE_RNG_SALT = gizmo_loop_rng_salt("ags_force");


/* ============================================================================
 * Per-pair physics types. PascalCase Spec-level typedefs re-export them.
 * Kept compatible with the (LocalT, KernelT, OutT) templates in
 * sidm/*_flux_functions.h so the SSOT flux bodies plug in unchanged.
 * ========================================================================== */

/* Per-active local fill, mirrors the legacy ags_force_local_t in
 * gravity/ags_force_gpu.cc:69-106 verbatim. The flux templates index into
 * `local.X` directly. */
struct AgsForceLocalIn {
    double Mass;
    double AGS_KernelRadius;       /* un-inflated radius; physics-side */
    Vec3<double> Pos;
    Vec3<double> Vel;
    int Type;
    double dtime;
#if defined(AGS_FACE_CALCULATION_IS_ACTIVE)
    MyDouble NV_T[3][3];
    double V_i;
#endif
#if defined(DM_FUZZY)
    Vec3<double> AGS_Gradients_Density;
    double AGS_Gradients2_Density[3][3];
    double AGS_Numerical_QuantumPotential;
#if (DM_FUZZY > 0)
    double AGS_Psi_Re; Vec3<double> AGS_Gradients_Psi_Re; Mat3<double> AGS_Gradients2_Psi_Re;
    double AGS_Psi_Im; Vec3<double> AGS_Gradients_Psi_Im; Mat3<double> AGS_Gradients2_Psi_Im;
#endif
#endif
#if defined(CBE_INTEGRATOR)
    double CBE_basis_moments[CBE_INTEGRATOR_NBASIS][CBE_INTEGRATOR_NMOMENTS];
#endif
#if defined(CBE_INTEGRATOR_WITHGRADIENTS)
    /* By-value snapshot of P[i].Gradients_CBE_basis_moments — populated by
     * load_active so the flux body's face reconstruction (read of i's
     * gradient row) is Mode-B-safe envelope-transit per design invariant
     * I2. The j-side gradient is read directly from P[j] (natural ghost
     * import). Refreshed each step by CBEGrad_gradient_calc(); inactive
     * particles retain prior-step values (hydro semantics). */
    double Gradients_CBE_basis_moments[CBE_INTEGRATOR_NBASIS]
                                      [CBE_INTEGRATOR_NMOMENTS][3];
#endif
#if defined(DM_SIDM)
    double dtime_sidm;
    MyIDType ID;
#ifdef GRAIN_COLLISIONS
    double Grain_CrossSection_PerUnitMass;
#endif
#endif
#if defined(GRAIN_EVOLUTION) && (GRAIN_EVOLUTION & 7)
    double Grain_Size;
    double Composition[GRAIN_NUM_SPECIES];
#endif
};

/* Per-pair kernel-scratch struct (same shape as legacy ags_force_kernel_t in
 * gravity/ags_force_gpu.cc:111-116). The flux templates read kernel.h_i,
 * kernel.h_j, kernel.r, kernel.dp, kernel.dv, kernel.wk_*, kernel.dwk_*. */
struct AgsForceKernel {
    Vec3<double> dp, dv;
    double r, wk_i, wk_j, dwk_i, dwk_j;
    double h_i, hinv_i, hinv3_i, hinv4_i;
    double h_j, hinv_j, hinv3_j, hinv4_j;
};

/* Per-active accumulator. Field set is the legacy ags_force_gpu_out (in the
 * retired gravity/ags_gpu_decls.h) ported here as the SSOT. */
struct AgsForceOut {
#if defined(DM_SIDM)
    double sidm_kick[3];
    double dtime_sidm;            /* MIN-reduced; sentinel = MAX_REAL_NUMBER in zero_accum */
    int    si_count;
#endif
#ifdef DM_FUZZY
    double acc[3];
    double AGS_Dt_Numerical_QuantumPotential;
#if (DM_FUZZY > 0)
    double AGS_Dt_Psi_Re, AGS_Dt_Psi_Im, AGS_Dt_Psi_Mass;
#endif
#endif
#if defined(CBE_INTEGRATOR)
    double AGS_vsig;              /* MAX-reduced */
    double CBE_basis_moments_dt[CBE_INTEGRATOR_NBASIS][CBE_INTEGRATOR_NMOMENTS];
    double CBE_basis_out_rate_dt[CBE_INTEGRATOR_NBASIS][CBE_INTEGRATOR_NMOMENTS];   /* outgoing-only piece (commit 1 infra; SUM-reduced) */
#if defined(OUTPUT_ADDITIONAL_RUNINFO) || defined(CBE_INTEGRATOR_OUTPUT_MOREINFO)
    /* Wave-CBE Commit 3 root-found v_F diagnostic counters. Per-pair
     * updates in sidm/cbe_integrator_flux_functions.h; merged in
     * AgsForceSpec::merge_accum; ingested into the host-side CbeStepAccum
     * via cbe_step_diagnostics_observe() from apply_active_writeback. */
    double    cbe_face_residual_max;       /* MAX over faces of |sum_basis F_m*A| */
    double    cbe_face_residual_sum;       /* SUM over faces of |sum_basis F_m*A| */
    long long cbe_bracket_fail_count;      /* SUM over faces of root-find bracket fails */
    /* Wave-CBE Commit 4 + Commit 5 face-reconstruction clamp counters,
     * populated by cbe_clamp_face_Q. Density-only clamp (Commit 4 Phase 1)
     * zeroes the basis row when Q_face[m][0] <= MIN_REAL_NUMBER; stress
     * SPD repair (Commit 5) projects [Sxx..Syz] to the nearest SPD tensor
     * with eigenvalue floor when NMOMENTS >= 10. */
    long long cbe_recon_rho_clamp_count;   /* SUM over faces+sides of density clamps */
    long long cbe_recon_S_clamp_count;     /* SUM over faces+sides of S SPD repairs */
    /* Wave-CBE Commit 6c: SUM over directional basis rows for which the
     * free-slot fallback transformed a cost-matrix row during flux
     * pairing. Each flux face evaluation builds TWO cost matrices (a->b
     * and b->a); each can fire on up to NBASIS rows, so per-face
     * increment is bounded by 2*NBASIS. Gradient and BJ-limiter matching
     * share the SSOT pair-builder but pass NULL for the counter (pre-pass
     * matching is not a flux-pairing decision). 0 in builds that disable
     * the free-slot transform via CBE_PAIRING_USE_FREE_SLOT=0. */
    long long cbe_pairing_free_slot_count;
#if defined(CBE_INTEGRATOR_WITHGRADIENTS)
    /* Per-pair non-finite grad·dp event count (defense-in-depth). The flux
     * body sanitises gi_dp/gj_dp to 0 if non-finite; this counter surfaces
     * the event so we know if Tikhonov regularisation failed for some
     * (m,k) on degenerate inputs. Nested WITHGRADIENTS gate keeps the
     * field absent (and cbe_diagnostics.txt at 8 columns) when the
     * reconstruction path is compiled out — bit-identical baseline. */
    long long cbe_grad_nonfinite_count;
#endif
#endif
#endif
#if defined(GRAIN_EVOLUTION) && (GRAIN_EVOLUTION & 7)
    /* Phase 17b pairwise outcomes; multiplicative erosion sentinel = 1.0
     * in zero_accum (multiplicative identity). */
    double Grain_DeltaCoagMass;
    double Grain_DeltaCoag_CompositionMass[GRAIN_NUM_SPECIES];
    double Grain_DeltaErosionFrac;
#endif
};

/* Per-call cosmology / globals captured once from All.* on the host by
 * populate_call_scalars. Routed through ActiveData::scalars to the inline
 * pair body so the body never reads All.* directly. */
struct AgsForceCallScalars {
    NlrCommonScalars common;                 /* cf_atime, cf_a2inv, cf_hubble_a, ... */
    int              TimeBinActive[TIMEBINS];/* SIDM/CBE wakeup tests */
    uint64_t         rng_salt;               /* = AGS_FORCE_RNG_SALT (compile-time) */
};

/* Active-particle state. `h_search` is the SIDM-inflated value handed in by
 * the runner walker; `local.AGS_KernelRadius` is the un-inflated physics
 * radius used by the per-pair filter. Subtle but critical separation:
 * legacy gravity/ags_force_gpu.cc:179-181 + :273 made the same distinction.
 *
 * `pos` is top-level (Mode B walker reads it directly); `local` carries the
 * full host-fill struct for the flux templates. */
struct AgsForceActiveState {
    Vec3<double>        pos;
    double              h_search;          /* SIDM 3x-inflated when DM_SIDM */
    AgsForceLocalIn     local;
    short int           TimeBin;           /* for hydro-convention wakeup write */
    AgsForceCallScalars scalars;
    int                 origin_local_idx;
    int                 origin_rank;
};

/* DeviceContext extension. need_wakeup_uvm is sticky across all subgroups of
 * one toplevel call (same lifecycle as ags_density's, codex round-7 lesson).
 * geofactor_uvm mirrors GeoFactorTable on device for the SIDM probability
 * lookup. */
struct AgsForceDeviceContext : NeighborLoopDeviceContextBase {
    int               *need_wakeup_uvm;    /* UVM, single int; sticky across iters/subgroups */
    bool               oracle_dry_run;
#if defined(DM_SIDM)
    MyDouble          *geofactor_uvm;      /* UVM, GEOFACTOR_TABLE_LENGTH entries */
#endif
};

/* IterScratch is unused (max_iters=1, after_iter always Converged) but the
 * iterative-runner contract requires the typedef. */
struct AgsForceIterScratch { };


/* ============================================================================
 * Inline pair body — verbatim translation of the kernel at
 * gravity/ags_force_gpu.cc:222-357, with three corrections noted in the
 * file docblock: (a) j-writes gated on oracle_dry_run; (b) SIDM RNG salt
 * threaded; (c) wakeup pre-zero handled host-side in ghost_writeback_begin.
 * ========================================================================== */
template <typename NeighborT>
KOKKOS_INLINE_FUNCTION
static void ags_force_pair_kernel_body(const AgsForceActiveState& active,
                                        const NeighborT&           neighbor,
                                        AgsForceOut&               accum)
{
    struct particle_data *P_base       = neighbor.P_base;
    const int             j            = neighbor.j;
    struct particle_data &Pj           = *neighbor.neighbor_particle;
    int                  *need_wakeup  = neighbor.need_wakeup;
    const bool            oracle       = neighbor.oracle_dry_run;
    const AgsForceLocalIn& local       = active.local;

    if(!(Pj.Mass > 0) || !(Pj.AGS_KernelRadius > 0)) return;

    AgsForceKernel kernel;
    kernel.h_i = local.AGS_KernelRadius;
    kernel_hinv(kernel.h_i, &kernel.hinv_i, &kernel.hinv3_i, &kernel.hinv4_i);

    /* Periodic-wrapped separation (matches legacy line 293). */
    kernel.dp = local.Pos - Pj.Pos;
    nearest_xyz(kernel.dp);
    double r2 = kernel.dp.norm_sq();
    if(r2 <= 0) return;
    kernel.r = sqrt(r2);
    kernel.h_j = Pj.AGS_KernelRadius;

    /* Pair-overlap filter on UN-inflated radii. Legacy:
     *   gravity/ags_force_gpu.cc:301-305 — SIDM uses r > h_i+h_j; non-SIDM
     *   uses r > h_i AND r > h_j (symmetric). */
#if defined(DM_SIDM)
    if(kernel.r > kernel.h_i + kernel.h_j) return;
#else
    if(kernel.r > kernel.h_i && kernel.r > kernel.h_j) return;
#endif

    kernel_hinv(kernel.h_j, &kernel.hinv_j, &kernel.hinv3_j, &kernel.hinv4_j);
    double u_i = kernel.r * kernel.hinv_i;
    double u_j = kernel.r * kernel.hinv_j;
    if(u_i < 1) kernel_main(u_i, kernel.hinv3_i, kernel.hinv4_i, &kernel.wk_i, &kernel.dwk_i, 0);
    else        { kernel.wk_i = 0; kernel.dwk_i = 0; }
    if(u_j < 1) kernel_main(u_j, kernel.hinv3_j, kernel.hinv4_j, &kernel.wk_j, &kernel.dwk_j, 0);
    else        { kernel.wk_j = 0; kernel.dwk_j = 0; }

    /* Atomic read of P[j].Vel before forming dv — SIDM may concurrently
     * mutate it via another thread's atomic_add. Matches legacy line 316. */
    for(int k = 0; k < 3; k++) {
        double Vel_j_k = Kokkos::atomic_load(&Pj.Vel[k]);
        kernel.dv[k] = local.Vel[k] - Vel_j_k;
        if(active.scalars.common.comoving_integration_on) {
            kernel.dv[k] += active.scalars.common.cf_hubble_a * kernel.dp[k]
                            / active.scalars.common.cf_a2inv;
        }
    }

#if defined(CBE_INTEGRATOR)
    {
        CbeFluxResult cbe_r = cbe_integrator_flux_compute_pair(
            local, j, P_base, kernel, accum, active.scalars.TimeBinActive);
        if(cbe_r.set_wakeup_j && !oracle) {
            /* Hydro-convention wakeup: active.TimeBin+1 (positive), MAX
             * reverse-comm safe (legacy -1 sentinel silently dropped). */
            short int wakeup_val = (short int)(active.TimeBin + 1);
            Kokkos::atomic_max(&Pj.wakeup, wakeup_val);
            Kokkos::atomic_store(need_wakeup, 1);
        }
    }
#endif

#if defined(DM_FUZZY)
    dm_fuzzy_flux_compute_pair(local, j, P_base, kernel, accum);
#endif

#if defined(DM_SIDM)
    {
        SidmScatterResult sidm_r = sidm_core_flux_compute_pair(
            local, j, P_base, kernel, accum,
            neighbor.geofactor, active.scalars.TimeBinActive,
            active.scalars.rng_salt);
        if(sidm_r.scattered && !oracle) {
            if(sidm_r.set_wakeup_j) {
                short int wakeup_val = (short int)(active.TimeBin + 1);
                Kokkos::atomic_max(&Pj.wakeup, wakeup_val);
                Kokkos::atomic_store(need_wakeup, 1);
            }
            for(int kv = 0; kv < 3; kv++) {
                Kokkos::atomic_add(&Pj.Vel[kv], (MyDouble)sidm_r.dv_sidm[kv]);
                Kokkos::atomic_add(&Pj.dp[kv],  (MyDouble)(sidm_r.dv_sidm[kv] * Pj.Mass));
            }
            Kokkos::atomic_add(&Pj.NInteractions, (long unsigned int)1);
        }
    }
#endif

    (void)P_base; (void)j; (void)need_wakeup;  /* may be unused under specific physics combos */
}


/* ============================================================================
 * AgsForceSpec — NeighborLoopSpec contract.
 * ========================================================================== */
struct AgsForceSpec {
    /* ====================================================================
     * PHYSICS BLOCK
     * ==================================================================== */

    static constexpr const char *loop_name = "ags_force";

    /* SIDM pair-filter is r <= h_i+h_j; non-SIDM is r <= max(h_i,h_j). Both
     * are symmetric pair predicates — neighbor pool must include j with
     * h_j > h_i and r > h_i (the symmetric search semantic). */
    static constexpr int                     search_mode        = MODE_B_SEARCH_SYMMETRIC;
    static constexpr unsigned int            neighbor_type_mask = 0xFFFFFFFFu;  /* per-subgroup override */
    /* AGS-force pair physics: h_j reach is P[j].AGS_KernelRadius for every type
     * (gas and non-gas).  Symmetric filter at ags_force_loop.h:296,304 reads
     * kernel.h_j = Pj.AGS_KernelRadius and rejects only r > max(h_i, h_j).
     * See OPEN_radius_policy_runner_design.md audit table for the cite. */
    static constexpr mode_b_radius_policy_t  radius_policy      =
        MODE_B_RADIUS_GAS_AGS | MODE_B_RADIUS_NONGAS_AGS;

    static constexpr WritePattern   write_pattern   = WritePattern::ActiveReduceOnly;
    /* Multi-bm subgroups — same rationale as ags_density: a step-persistent
     * SIDX built for one type mask cannot be shared across subgroups with
     * different masks. Build a local SIDX per CSR each call. */
    static constexpr SidxCacheKind  sidx_cache_kind = SidxCacheKind::None;

    /* Default 1e-10 oracle bound. The pair body's read-then-atomic-add
     * sites (Pj.Vel via atomic_load + atomic_add under DM_SIDM) are
     * race-tolerant by construction — out.sidm_kick depends only on the
     * pre-scatter Vel snapshot, and oracle dry-run suppresses j-writes
     * exactly so the brute pass sees the unmutated state. Multiplicative
     * Grain_DeltaErosionFrac may show tiny order-dependent residuals; if
     * validation flags those, address via field-specific tolerance, NOT
     * a broad loosening (per reference_wave3_thermalfb_lessons.md). */
    static constexpr double accum_tolerance  = 1e-10;
    static constexpr double radius_tolerance = 1e-9;     /* unused; no radius adjust */

    static bool is_active(int particle_index) {
        return AGSForce_isactive(particle_index) != 0;
    }

    /* Iterative-shaped wrapper for the multi-subgroup runner contract;
     * physics is single-pass (max_iters=1, after_iter always Converged).
     * Same wrapper pattern as mechfb / radfb_rp / dm_dispersion. */
    using IterControl       = Iterative;
    using IterScratch       = AgsForceIterScratch;
    using SupportsSubgroups = std::true_type;

    static constexpr int    max_iters                     = 1;
    static constexpr double mode_a_csr_buffer_factor      = 2.0;
    static constexpr bool   mode_a_rebuild_csr_every_iter = false;
    static constexpr bool   actives_partition_by_subgroup = true;

    using CallScalars   = AgsForceCallScalars;
    using ActiveData    = AgsForceActiveState;
    using AccumData     = AgsForceOut;
    using DeviceContext = AgsForceDeviceContext;

    struct NeighborData {
        struct particle_data *neighbor_particle;
        int                   j;            /* for flux helpers' (j, P) signature */
        struct particle_data *P_base;       /* = dctx.P */
        int                  *need_wakeup;
        bool                  oracle_dry_run;
#if defined(DM_SIDM)
        const MyDouble       *geofactor;
#endif
    };

    /* Aux is empty under the v5 corrective architecture — CBE gradients are
     * persistent on P[i].Gradients_CBE_basis_moments and ghost-transported
     * naturally by standard P[] import, so there is no scratch pointer to
     * thread through here. Kept as the conventional empty struct so the
     * Spec contract typedef remains satisfied. */
    struct Aux { };

    static constexpr bool uses_ghost_write_detector = true;
    static constexpr bool uses_ghost_writeback      = true;

    /* ghost_write_detector_begin/end: runner default (loop_name = "ags_force"). */
    static void ghost_writeback_begin      (const neighbor_loop_args&,
                                             const NeighborLoopPlan&);
    static void ghost_writeback_end        (const neighbor_loop_args&,
                                             const NeighborLoopPlan&);

    /* Per-active and per-call hooks (host). */
    static double      search_radius(const neighbor_loop_args& args,
                                      int active_slot, int i);
    static CallScalars populate_call_scalars(const neighbor_loop_args& args);

    static void populate_device_context(const neighbor_loop_args& args, DeviceContext& ctx);
    static void cleanup_device_context (const neighbor_loop_args& args, DeviceContext& ctx);

    KOKKOS_INLINE_FUNCTION
    static int active_subgroup_key(const DeviceContext& ctx, int i,
                                    const CallScalars& scalars)
    {
        /* Partition key matches the caller's bitmask_groups partition. Mirror
         * of ags_gravity_kernel_shared_BITFLAG (ags_rkern.cc) inlined here as
         * device-callable logic — the named function is host-only and reading
         * bare P[] / All.* from inside a __host__ __device__ inline triggers
         * #20094-D + #20011-D. Same pattern + branches as
         * AgsDensitySpec::active_subgroup_key (gravity/ags_density_loop.h).
         * Reads Type via ctx.particle_type(i) and ComovingIntegrationOn via
         * scalars.common.comoving_integration_on (codex round-9 path).
         * Phase D fix 2026-05-21 (config 107 MAGNETIC+GRAIN_*). */
        const short int t = ctx.particle_type(i);

#ifdef ADAPTIVE_GRAVSOFT_FORALL
        if(!((1 << t) & (ADAPTIVE_GRAVSOFT_FORALL))) { return 0; }
#endif

#ifdef ADAPTIVE_GRAVSOFT_FROM_TIDAL_CRITERION
        if(!((1 << t) & (ADAPTIVE_GRAVSOFT_FROM_TIDAL_CRITERION))) {
            return ADAPTIVE_GRAVSOFT_FROM_TIDAL_CRITERION;
        }
#endif

        if(t == 0) { return 1; }

#if (ADAPTIVE_GRAVSOFT_FORALL & 32) && defined(SINK_PARTICLES)
        if(t == 5) { return 1; }
#endif

#if defined(GALSF) && ( (ADAPTIVE_GRAVSOFT_FORALL & 16) || (ADAPTIVE_GRAVSOFT_FORALL & 8) || (ADAPTIVE_GRAVSOFT_FORALL & 4) )
        if(scalars.common.comoving_integration_on) {
            if(t == 4) { return 17; }       /* 2^0 + 2^4 */
        } else {
            if((t == 4) || (t == 2) || (t == 3)) { return 29; }  /* 2^0 + 2^2 + 2^3 + 2^4 */
        }
#endif

#ifdef DM_SIDM
        if((1 << t) & (DM_SIDM)) { return DM_SIDM; }
#endif

#ifdef AGS_KERNELRADIUS_CALCULATION_IS_ACTIVE
        return (1 << t);
#endif

        return 0;
    }

    /* Fill ActiveData[slot] on device. Reads directly from ctx.P[i] — no
     * host-staged per-active local UVM (same simplification ags_density
     * adopted post-codex-r7). */
    KOKKOS_INLINE_FUNCTION
    static ActiveData load_active(const DeviceContext& dctx,
                                   int /*active_slot*/, int i,
                                   double h_search,
                                   const CallScalars& scalars)
    {
        ActiveData a;
        a.pos[0] = (double)dctx.P[i].Pos[0];
        a.pos[1] = (double)dctx.P[i].Pos[1];
        a.pos[2] = (double)dctx.P[i].Pos[2];
        a.h_search = h_search;
        a.TimeBin  = dctx.P[i].TimeBin;
        a.scalars  = scalars;
        a.origin_local_idx = i;
        a.origin_rank      = -1;

        AgsForceLocalIn& L = a.local;
        L.Mass             = dctx.P[i].Mass;
        L.AGS_KernelRadius = (double)dctx.P[i].AGS_KernelRadius;   /* un-inflated */
        L.Pos              = a.pos;
#if defined(CBE_INTEGRATOR)
        /* In a CBE build local.Vel IS the CBE bulk velocity; load the PREDICTED
         * (drifted) value so the flux reconstructs the cell at the substep time
         * rather than its stale begin-of-step state. DM_SIDM — the only other
         * consumer of local.Vel in this loop (kernel.dv, which the CBE flux does
         * not use) — is mutually exclusive with CBE_INTEGRATOR. */
        L.Vel[0] = (double)dctx.P[i].CBE_VelPred[0];
        L.Vel[1] = (double)dctx.P[i].CBE_VelPred[1];
        L.Vel[2] = (double)dctx.P[i].CBE_VelPred[2];
#else
        L.Vel[0] = (double)dctx.P[i].Vel[0];
        L.Vel[1] = (double)dctx.P[i].Vel[1];
        L.Vel[2] = (double)dctx.P[i].Vel[2];
#endif
        L.Type   = dctx.P[i].Type;
        L.dtime  = get_particle_timestep_in_physical(i, dctx.P);
#if defined(AGS_FACE_CALCULATION_IS_ACTIVE)
        L.V_i = get_particle_volume_ags_P(i, dctx.P);
        for(int a1 = 0; a1 < 3; a1++) for(int b = 0; b < 3; b++) L.NV_T[a1][b] = dctx.P[i].NV_T[a1][b];
#endif
#if defined(DM_FUZZY)
        L.AGS_Gradients_Density = dctx.P[i].AGS_Gradients_Density;
        for(int a1 = 0; a1 < 3; a1++) for(int b = 0; b < 3; b++)
            L.AGS_Gradients2_Density[a1][b] = dctx.P[i].AGS_Gradients2_Density[a1][b];
        L.AGS_Numerical_QuantumPotential = dctx.P[i].AGS_Numerical_QuantumPotential;
#if (DM_FUZZY > 0)
        L.AGS_Psi_Re = dctx.P[i].AGS_Psi_Re_Pred * dctx.P[i].AGS_Density / dctx.P[i].Mass;
        L.AGS_Gradients_Psi_Re = dctx.P[i].AGS_Gradients_Psi_Re;
        for(int a1 = 0; a1 < 3; a1++) for(int b = 0; b < 3; b++)
            L.AGS_Gradients2_Psi_Re[a1][b] = dctx.P[i].AGS_Gradients2_Psi_Re[a1][b];
        L.AGS_Psi_Im = dctx.P[i].AGS_Psi_Im_Pred * dctx.P[i].AGS_Density / dctx.P[i].Mass;
        L.AGS_Gradients_Psi_Im = dctx.P[i].AGS_Gradients_Psi_Im;
        for(int a1 = 0; a1 < 3; a1++) for(int b = 0; b < 3; b++)
            L.AGS_Gradients2_Psi_Im[a1][b] = dctx.P[i].AGS_Gradients2_Psi_Im[a1][b];
#endif
#endif
#if defined(CBE_INTEGRATOR)
        /* Load PREDICTED moments (drifted to the substep time); the flux
         * reconstructs i's cell from these instead of the stale conserved row. */
        for(int m = 0; m < CBE_INTEGRATOR_NBASIS; m++)
            for(int k = 0; k < CBE_INTEGRATOR_NMOMENTS; k++)
                L.CBE_basis_moments[m][k] = dctx.P[i].CBE_basis_moments_pred[m][k];
#endif
#if defined(CBE_INTEGRATOR_WITHGRADIENTS)
        /* By-value snapshot of the persistent gradient row for Mode-B-safe
         * envelope transit (invariant I2). The flux body reads
         * local.Gradients_CBE_basis_moments for i's contribution to
         * Q_face; for j it reads P[j].Gradients_CBE_basis_moments directly
         * (natural P[] ghost import). */
        for(int m = 0; m < CBE_INTEGRATOR_NBASIS; m++)
            for(int k = 0; k < CBE_INTEGRATOR_NMOMENTS; k++)
                for(int d = 0; d < 3; d++)
                    L.Gradients_CBE_basis_moments[m][k][d] =
                        dctx.P[i].Gradients_CBE_basis_moments[m][k][d];
#endif
#if defined(DM_SIDM)
        L.dtime_sidm = dctx.P[i].dtime_sidm;
        L.ID         = dctx.P[i].ID;
#ifdef GRAIN_COLLISIONS
        L.Grain_CrossSection_PerUnitMass = return_grain_cross_section_per_unit_mass_P(i, dctx.P);
#endif
#endif
#if defined(GRAIN_EVOLUTION) && (GRAIN_EVOLUTION & 7)
        L.Grain_Size = (double)dctx.P[i].Grain_Size;
        for(int gs = 0; gs < GRAIN_NUM_SPECIES; gs++)
            L.Composition[gs] = (double)dctx.P[i].Composition[gs];
#endif
        return a;
    }

    KOKKOS_INLINE_FUNCTION
    static void zero_accum(AccumData& accum) {
        for(size_t b = 0; b < sizeof(accum); b++) ((char*)&accum)[b] = 0;
#if defined(DM_SIDM)
        /* MIN-merge sentinel: any neighbor producing dtime_sidm < this will
         * pull it down; apply_active_writeback then MIN-applies against the
         * preamble-seeded P[i].dtime_sidm = 10*dtime. */
        accum.dtime_sidm = MAX_REAL_NUMBER;
#endif
#if defined(GRAIN_EVOLUTION) && (GRAIN_EVOLUTION & 7)
        /* Multiplicative identity for erosion-fraction accumulation. */
        accum.Grain_DeltaErosionFrac = 1.0;
#endif
    }

    KOKKOS_INLINE_FUNCTION
    static NeighborData load_neighbor(const DeviceContext& dctx,
                                       int j,
                                       const IdentitySidecar& /*id*/,
                                       const ActiveData& /*active*/)
    {
        NeighborData n;
        n.neighbor_particle = &dctx.P[j];
        n.j                 = j;
        n.P_base            = dctx.P;
        n.need_wakeup       = dctx.need_wakeup_uvm;
        n.oracle_dry_run    = dctx.oracle_dry_run;
#if defined(DM_SIDM)
        n.geofactor         = dctx.geofactor_uvm;
#endif
        return n;
    }

    static void set_oracle_brute_pass(DeviceContext& ctx, bool on) {
        ctx.oracle_dry_run = on;
    }

    KOKKOS_INLINE_FUNCTION
    static void pair_kernel(const ActiveData& active,
                             const NeighborData& neighbor,
                             AccumData& accum,
                             NoScatter& /*scatter*/)
    {
        ags_force_pair_kernel_body(active, neighbor, accum);
    }

    /* after_iter — single-pass shape: always Converged at iter 0. */
    static IterResult after_iter(const AfterIterContext<AgsForceSpec>& ctx,
                                  const AccumData& /*accum*/) {
        return IterResult{IterStatus::Converged, ctx.h_search_current};
    }

    static void apply_active_writeback(const neighbor_loop_args& args,
                                        int active_slot, int i,
                                        const AccumData& accum);

    static void merge_accum(AccumData& local_accum, const AccumData& peer_accum);

    /* ====================================================================
     * ENGINE APPARATUS
     * ==================================================================== */
    using ScatterData    = NoScatter;
    using IdentityFields = NoIdentity;

    /* ====================================================================
     * DIAGNOSTICS — env-gated.
     * ==================================================================== */
    static double compare_accum(const AccumData& local, const AccumData& oracle);
};

/* `extern template` declaration so call sites bind to the explicit instantiation
 * in mesh/neighbor_loop_runner.cc rather than triggering implicit instantiation
 * (which fails because the runner template body is not visible in this header).
 * Same pattern other Specs rely on — paired with the matching
 * `template void run_neighbor_loop_iterative<AgsForceSpec>(...)` in
 * mesh/neighbor_loop_runner.cc. */
extern template void run_neighbor_loop_iterative<AgsForceSpec>(
    const neighbor_loop_args_iterative&);

#endif /* AGS_KERNELRADIUS_CALCULATION_IS_ACTIVE */
#endif /* AGS_FORCE_LOOP_H */
