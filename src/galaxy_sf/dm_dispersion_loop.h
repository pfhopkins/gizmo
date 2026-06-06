/* galaxy_sf/dm_dispersion_loop.h — DM velocity dispersion runner Spec.
 *
 * Defines DMDispersionSpec for run_neighbor_loop_iterative.
 * Replaces the legacy disp_density_evaluate_gpu() GPU path.
 *
 * Active set: gas particles (Type==0, TimeBin>=0, Mass>0).
 * Neighbor set: DM particles (Type==1, Mass>0).
 * Write pattern: ActiveReduceOnly (read-only on j; no ghost writeback).
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) and Claude for GIZMO.
 */
#ifndef DM_DISPERSION_LOOP_H
#define DM_DISPERSION_LOOP_H

/* Kokkos_Core MUST come before allvars.h — same pattern as density_loop.h:
 * allvars.h pulls declarations/macros.h which #defines `terminate(...)`,
 * mangling std::terminate() inside Kokkos_Core. */
#include <Kokkos_Core.hpp>

#include "../declarations/allvars.h"
#include "../mesh/neighbor_loop_runner.h"
#include "../mesh/mode_b_local_walker.h"   /* MODE_B_SEARCH_*, MODE_B_RADIUS_* */

/* No KOKKOS_INLINE_FUNCTION fallback here — Kokkos_Core.hpp is included
 * unconditionally above. This Spec carries device-callable pair-kernel
 * accessors; misordered includes must compile-fail loudly, not silently
 * resolve to host-only `inline`. Same convention as neighbor_loop_runner.h. */

#include <vector>

#ifdef DM_DISPERSION_LOOP_ACTIVE

/* ============================================================================
 * (1) CallScalars
 *
 * NlrCommonScalars (mesh/neighbor_loop_runner.h) already carries cf_atime,
 * cf_a2inv, cf_a3inv, cf_hubble_a, newton_G, hubble, comoving_integration_on.
 * After-iter only needs MaxKernelRadius on top of that.
 * ========================================================================== */
struct DMDispCallScalars {
    NlrCommonScalars common;
    double max_kernel_radius;   /* All.MaxKernelRadius */
};

/* ============================================================================
 * (2) AccumData — per-active accumulator for DM velocity statistics.
 *
 * All fields are additive; merge_accum sums them element-wise.
 * Ngb is an unweighted neighbor count (each j contributing 1.0, not a
 * kernel-weight), matching the legacy GPU kernel exactly.
 * ========================================================================== */
struct DMDispOut {
    double Ngb;         /* unweighted DM neighbor count */
    double DM_Vx;       /* sum of Vel[0] over j */
    double DM_Vy;       /* sum of Vel[1] over j */
    double DM_Vz;       /* sum of Vel[2] over j */
    double DM_VelDisp;  /* sum of Vel·Vel over j */
    double DM_Rho;      /* kernel-weighted DM mass density (comoving units; physical conversion in finalize) */
};

/* ============================================================================
 * (3) IterScratch — per-active bisection brackets across iterations.
 *
 * Mirrors legacy Left[i]/Right[i] host arrays in disp_density().
 * Runner zero-initializes both fields at Aux allocation.
 * ========================================================================== */
struct DMDispIterScratch {
    float Left;    /* lower bound on KernelRadiusDM from prior iterations */
    float Right;   /* upper bound */
};

/* ============================================================================
 * (4) ActiveData — i-side state snapshot for pair_kernel.
 * ========================================================================== */
struct DMDispActiveState {
    Vec3<MyDouble>    pos;
    MyFloat           h_search;
    DMDispCallScalars scalars;
};

/* ============================================================================
 * (5) NeighborData — j-side sidecar (DM neighbors are read-only).
 * ========================================================================== */
struct DMDispNeighborData {
    struct particle_data *neighbor_particle;
};

/* ============================================================================
 * (6) DMDispersionSpec
 * ========================================================================== */
struct DMDispersionSpec {
    /* Identity */
    static constexpr const char *loop_name = "dm_dispersion";

    /* Search policy: one-way (gas i → DM j only; no j-side writes). */
    static constexpr int                    search_mode        = MODE_B_SEARCH_ONEWAY;
    static constexpr unsigned int           neighbor_type_mask = (1u << 1); /* DM (Type 1) */
    static constexpr mode_b_radius_policy_t radius_policy      = MODE_B_RADIUS_DEFAULT;

    /* Write policy */
    static constexpr WritePattern   write_pattern             = WritePattern::ActiveReduceOnly;
    /* SidxCacheKind::None: DM tbm=(1u<<1) matches neither GasOnly (tbm=1)
     * nor AllTypes (tbm=0x3f) cache; cache_tbm hard-abort would fire.
     * Same reasoning as ags_density_loop.h:361. */
    static constexpr SidxCacheKind  sidx_cache_kind           = SidxCacheKind::None;
    static constexpr bool           uses_ghost_writeback      = false;
    static constexpr bool           uses_ghost_write_detector = false;

    /* Convergence + iter policy */
    static constexpr double accum_tolerance  = 1e-10;
    static constexpr double radius_tolerance = 1e-9;
    using                    IterControl = Iterative;
    using                    IterScratch = DMDispIterScratch;
    static constexpr int     max_iters   = MAXITER;

    /* CSR policy: rebuild every iter (physics-identical to legacy;
     * optimize only after PA/PB parity confirmed). */
    static constexpr double mode_a_csr_buffer_factor      = 1.3;
    static constexpr bool   mode_a_rebuild_csr_every_iter = true;

    /* Type aliases */
    using CallScalars   = DMDispCallScalars;
    using ActiveData    = DMDispActiveState;
    using AccumData     = DMDispOut;
    using NeighborData  = DMDispNeighborData;
    using DeviceContext = NeighborLoopDeviceContextBase;
    using ScatterData   = NoScatter;

    /* Aux — host-owned converged output buffers.
     * apply_active_writeback_iterative copies AccumData + final_h here;
     * dm_dispersion_finalize_post_runner reads and scatters to CellP[]. */
    struct Aux {
        std::vector<DMDispOut> per_active_final_accum;
        std::vector<double>    per_active_final_h;
    };

    static_assert(uses_ghost_writeback == false,
        "dm_dispersion is pure i-side accumulation; no j-side writes.");

    /* Bisection targets — compile-time constants; NOT in CallScalars.
     * Legacy disp_density():49: desnumngb=64, desnumngbdev=48. */
    static constexpr double kDesNumNgb    = 64;
    static constexpr double kDesNumNgbDev = 48;

    /* ========================================================================
     * Host hooks — bodies in galaxy_sf/dm_dispersion_loop.cc
     * ====================================================================== */
    static bool        is_active(int i);
    static double      search_radius(const neighbor_loop_args& args, int active_slot, int i);
    static CallScalars populate_call_scalars(const neighbor_loop_args& args);

    /* apply_active_writeback: unreachable for this Spec — runner SFINAE
     * selects apply_active_writeback_iterative when the Spec opts in.
     * Body is endrun(91201) so a SFINAE regression surfaces loudly. */
    static void apply_active_writeback(const neighbor_loop_args& args,
                                       int active_slot, int i,
                                       const AccumData& accum);

    /* apply_active_writeback_iterative: oracle-safe production writeback.
     * Runner calls this INSTEAD of apply_active_writeback. Copies converged
     * accum + final_h into Aux for dm_dispersion_finalize_post_runner. */
    static void apply_active_writeback_iterative(const neighbor_loop_args& args,
                                                  int active_slot, int i,
                                                  const AccumData& accum,
                                                  double final_h,
                                                  const IterScratch& scratch);

    static void   merge_accum(AccumData& local, const AccumData& peer);
    static void   set_oracle_brute_pass(DeviceContext& ctx, bool on);
    static double compare_accum(const AccumData& local, const AccumData& oracle);

    static IterResult after_iter(const AfterIterContext<DMDispersionSpec>& ctx,
                                 const AccumData& accum);

    /* ========================================================================
     * Device hooks — header-inline (runner instantiates pair body from GPU TUs)
     * ====================================================================== */

    KOKKOS_INLINE_FUNCTION
    static void zero_accum(AccumData& accum) {
        accum.Ngb        = 0;
        accum.DM_Vx      = 0;
        accum.DM_Vy      = 0;
        accum.DM_Vz      = 0;
        accum.DM_VelDisp = 0;
        accum.DM_Rho     = 0;
    }

    KOKKOS_INLINE_FUNCTION
    static ActiveData load_active(const NeighborLoopDeviceContextBase& ctx,
                                  int active_slot, int i,
                                  double h_search, const CallScalars& scalars) {
        ActiveData a{};
        a.pos[0]   = (MyDouble)ctx.P[i].Pos[0];
        a.pos[1]   = (MyDouble)ctx.P[i].Pos[1];
        a.pos[2]   = (MyDouble)ctx.P[i].Pos[2];
        a.h_search = (MyFloat)h_search;
        a.scalars  = scalars;
        (void)active_slot;
        return a;
    }

    KOKKOS_INLINE_FUNCTION
    static NeighborData load_neighbor(const NeighborLoopDeviceContextBase& ctx,
                                      int j, const IdentitySidecar& id,
                                      const ActiveData& active) {
        NeighborData n{};
        n.neighbor_particle = &ctx.P[j];
        (void)id; (void)active;
        return n;
    }

    /* pair_kernel: accumulate unweighted DM velocity statistics + kernel-weighted DM density.
     *
     * Legacy dm_dispersion_gpu.cc kernel (velocity statistics, unweighted):
     *   if (kp[j].Mass <= 0) continue;
     *   if (r2 >= h2) continue;
     *   DM_Vx += Vel[0]; DM_Vy += Vel[1]; DM_Vz += Vel[2];
     *   DM_VelDisp += Vel·Vel;
     *   Ngb += 1.0;
     *
     * DM_Rho (added for DM_HEATING): canonical kernel density.
     *   kernel_main(u, hinv3, hinv4, &wk, &dwk, 0) returns wk already normalized
     *   by hinv3 (mode=0), matching hydro/density_loop.h:565-573.
     *   Accumulation is comoving; finalize converts to physical via cf_a3inv. */
    KOKKOS_INLINE_FUNCTION
    static void pair_kernel(const ActiveData& i_active,
                            const NeighborData& neighbor,
                            AccumData& accum, NoScatter& /*scatter*/) {
        struct particle_data& Pj = *neighbor.neighbor_particle;
        if (Pj.Mass <= 0) return;

        Vec3<double> dp;
        dp[0] = (double)i_active.pos[0] - (double)Pj.Pos[0];
        dp[1] = (double)i_active.pos[1] - (double)Pj.Pos[1];
        dp[2] = (double)i_active.pos[2] - (double)Pj.Pos[2];
        nearest_xyz(dp);
        const double h  = (double)i_active.h_search;
        const double h2 = h * h;
        const double r2 = dp.norm_sq();
        if (r2 >= h2) return;

        /* Kernel-weighted DM density (canonical pattern; matches density_loop.h:565-573). */
        const double r = sqrt(r2);
        double hinv, hinv3, hinv4;
        kernel_hinv(h, &hinv, &hinv3, &hinv4);
        double wk, dwk;
        kernel_main(r * hinv, hinv3, hinv4, &wk, &dwk, 0);
        accum.DM_Rho += (double)Pj.Mass * wk;

        /* Unweighted velocity-stat accumulators (legacy behavior, unchanged). */
        const double vx = (double)Pj.Vel[0];
        const double vy = (double)Pj.Vel[1];
        const double vz = (double)Pj.Vel[2];
        accum.DM_Vx      += vx;
        accum.DM_Vy      += vy;
        accum.DM_Vz      += vz;
        accum.DM_VelDisp += vx*vx + vy*vy + vz*vz;
        accum.Ngb        += 1.0;
    }
};

/* Forward decl — defined in galaxy_sf/dm_dispersion_loop.cc. */
void dm_dispersion_finalize_post_runner(const neighbor_loop_args_iterative& args);

#endif /* DM_DISPERSION_LOOP_ACTIVE */
#endif /* DM_DISPERSION_LOOP_H */
