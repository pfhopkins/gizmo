/* sidm/dm_fuzzy_loop.h — DMGradSpec for the runner-template port of the
 * DM_FUZZY higher-order density-gradient estimator (legacy DMGrad_evaluate /
 * dmgrad_evaluate_gpu).
 *
 * Non-iterative, one-way DM->DM search, fixed radius P[i].AGS_KernelRadius,
 * pure i-side reduce (no j-side writes, no ghost writeback). The estimator runs
 * in two external passes (driven by DMGrad_gradient_calc in dm_fuzzy_loop.cc):
 * pass 0 builds the gradient of AGS_Density (+ Psi_Re/Im if DM_FUZZY>0); pass 1
 * builds the gradient-of-gradient tensor. The pass index is carried in Aux ->
 * CallScalars; pair_kernel branches on it. One AccumData fuses both passes'
 * outputs; apply_active_writeback gates the scatter on the pass.
 *
 * SSOT for the DMGrad per-pair physics — supersedes the legacy
 * sidm/dm_fuzzy_gpu.cc (retired in this commit).
 *
 * Caller TUs must `#include "../mesh/kernel.h"` BEFORE this header (kernel.h
 * has no include guards; the inline pair kernel below calls kernel_main /
 * kernel_hinv). Same pattern as difffilter_loop.h / dm_dispersion_loop.h.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) and Claude for GIZMO.
 */
#ifndef DM_FUZZY_LOOP_H
#define DM_FUZZY_LOOP_H

/* Kokkos_Core before allvars.h — allvars pulls macros.h which #defines
 * terminate(...), mangling std::terminate in Kokkos_Core. Same as
 * difffilter_loop.h / dm_dispersion_loop.h. */
#include <Kokkos_Core.hpp>

#include "../declarations/allvars.h"
#include "../mesh/neighbor_loop_runner.h"
#include "../mesh/mode_b_local_walker.h"   /* MODE_B_SEARCH_*, MODE_B_RADIUS_* */

#ifndef KOKKOS_INLINE_FUNCTION
#define KOKKOS_INLINE_FUNCTION inline
#endif

#ifdef DM_FUZZY

/* ============================================================================
 * DMGradSpec — DM_FUZZY higher-order density-gradient estimator.
 * ========================================================================== */

/* CallScalars: cosmology common block + the external gradient-pass index.
 * loop_iteration 0 => gradient pass; 1 => gradient-of-gradient pass. */
struct DMGradCallScalars {
    NlrCommonScalars common;
    int              loop_iteration;
};

/* AccumData — per-active reduction; fuses pass-0 and pass-1 outputs. All fields
 * additive; merge_accum sums them. apply_active_writeback gates which subset is
 * scattered on the current pass. (The legacy DMGradDataPasser Maxima/Minima are
 * NOT carried — they were write-only dead code; see OPEN_3d_dm_fuzzy_design.md
 * §2(c).) */
struct DMGradAccum {
    double grad_rho[3];          /* pass 0: gradient of AGS_Density            */
    double grad2_rho[3][3];      /* pass 1: grad2_rho[k2][k] = d/dx_k2 of       */
                                 /*         component k of AGS_Gradients_Density*/
#if (DM_FUZZY > 0)
    double grad_psi_re[3];       /* pass 0 */
    double grad_psi_im[3];       /* pass 0 */
    double grad2_psi_re[3][3];   /* pass 1 */
    double grad2_psi_im[3][3];   /* pass 1 */
#endif
};

/* ActiveData — i-side snapshot for pair_kernel. pos / h_search are top-level
 * (the Mode B walker reads them directly). `valid` carries the legacy kernel
 * guard (DMGrad_evaluate: AGS_KernelRadius>0 && AGS_Density>0): an invalid
 * source no-ops every pair, leaving accum at zero_accum. Both passes' i-side
 * inputs are loaded unconditionally (cheap); pair_kernel uses the subset for
 * the current pass. */
struct DMGradActiveState {
    Vec3<double> pos;
    double       h_search;          /* runner radius = P[i].AGS_KernelRadius   */
    int          loop_iteration;    /* 0 = gradient pass; 1 = grad-of-grad     */
    int          valid;             /* AGS_KernelRadius>0 && AGS_Density>0      */
    double       ags_density;       /* pass-0 input                            */
    double       grad_density[3];   /* pass-1 input: P[i].AGS_Gradients_Density */
#if (DM_FUZZY > 0)
    double       psi_re;            /* pass-0 input: Psi_Re_Pred*Density/Mass   */
    double       psi_im;            /* pass-0 input                            */
    double       grad_psi_re[3];    /* pass-1 input                            */
    double       grad_psi_im[3];    /* pass-1 input                            */
#endif
};

/* NeighborData — j-side DM particle (read-only). DMGrad neighbors are P-only:
 * AGS_Density / AGS_Psi_*_Pred / AGS_Gradients_* all live on particle_data. */
struct DMGradNeighborData {
    struct particle_data *neighbor_particle;
};

struct DMGradSpec {
    static constexpr const char *loop_name = "dm_fuzzy_dmgrad";

    /* One-way DM->DM (legacy ngb_treefind_variable_threads_targeted). */
    static constexpr int                    search_mode        = MODE_B_SEARCH_ONEWAY;
    /* Nominal default only — the toplevel overrides per call via
     * args.neighbor_type_mask_override = bm (ags_gravity_kernel_shared_BITFLAG).
     * Never load-bearing on its own. */
    static constexpr unsigned int           neighbor_type_mask = (1u << 1);  /* DM */
    static constexpr mode_b_radius_policy_t radius_policy       = MODE_B_RADIUS_DEFAULT;

    static constexpr WritePattern  write_pattern             = WritePattern::ActiveReduceOnly;
    /* SidxCacheKind::None: the per-call DM/SIDM mask matches neither GasOnly
     * (tbm=1) nor AllTypes (tbm=0x3f); cache_tbm hard-abort would fire. Same
     * reasoning as dm_dispersion_loop.h / ags_density_loop.h. */
    static constexpr SidxCacheKind sidx_cache_kind           = SidxCacheKind::None;
    static constexpr bool          uses_ghost_writeback      = false;
    static constexpr bool          uses_ghost_write_detector = false;

    /* Pure-read pair kernel (no j-writes) → order-independent up to FP
     * summation → tight oracle. */
    static constexpr double accum_tolerance = 1e-10;

    using CallScalars    = DMGradCallScalars;
    using ActiveData     = DMGradActiveState;
    using AccumData      = DMGradAccum;
    using NeighborData   = DMGradNeighborData;
    using DeviceContext  = NeighborLoopDeviceContextBase;
    using ScatterData    = NoScatter;
    using IdentityFields = NoIdentity;
    using IterControl    = NotIterative;

    /* Aux — the external gradient-pass index. populate_call_scalars copies it
     * into CallScalars; apply_active_writeback gates the scatter on it. */
    struct Aux {
        int loop_iteration;
    };

    static_assert(uses_ghost_writeback == false,
        "DMGrad is pure i-side accumulation; no j-side writes.");

    /* Host hooks — bodies in dm_fuzzy_loop.cc. */
    static bool        is_active(int i);
    static double      search_radius(const neighbor_loop_args& args, int active_slot, int i);
    static CallScalars populate_call_scalars(const neighbor_loop_args& args);
    static void        apply_active_writeback(const neighbor_loop_args& args,
                                              int active_slot, int i,
                                              const AccumData& accum);
    static void        merge_accum(AccumData& local, const AccumData& peer);
    static double      compare_accum(const AccumData& local, const AccumData& oracle);
    static void        set_oracle_brute_pass(DeviceContext& ctx, bool on);

    /* ---- Device hooks (header-inline; runner instantiates from GPU TUs). ---- */
    KOKKOS_INLINE_FUNCTION
    static void zero_accum(AccumData& accum) {
        for (int k = 0; k < 3; k++) {
            accum.grad_rho[k] = 0;
            for (int k2 = 0; k2 < 3; k2++) accum.grad2_rho[k2][k] = 0;
#if (DM_FUZZY > 0)
            accum.grad_psi_re[k] = 0;
            accum.grad_psi_im[k] = 0;
            for (int k2 = 0; k2 < 3; k2++) {
                accum.grad2_psi_re[k2][k] = 0;
                accum.grad2_psi_im[k2][k] = 0;
            }
#endif
        }
    }

    KOKKOS_INLINE_FUNCTION
    static ActiveData load_active(const DeviceContext& ctx,
                                  int active_slot, int i,
                                  double h_search, const CallScalars& scalars) {
        (void)active_slot;
        ActiveData a{};
        struct particle_data& Pi = ctx.P[i];
        a.pos[0] = (double)Pi.Pos[0];
        a.pos[1] = (double)Pi.Pos[1];
        a.pos[2] = (double)Pi.Pos[2];
        a.h_search       = h_search;
        a.loop_iteration = scalars.loop_iteration;
        a.ags_density    = (double)Pi.AGS_Density;
        a.valid = (h_search > 0 && a.ags_density > 0) ? 1 : 0;
        for (int k = 0; k < 3; k++) a.grad_density[k] = (double)Pi.AGS_Gradients_Density[k];
#if (DM_FUZZY > 0)
        /* legacy particle2in_DMGrad: Psi_*_Pred * AGS_Density / Mass. */
        const double vol_factor = (double)Pi.AGS_Density / (double)Pi.Mass;
        a.psi_re = (double)Pi.AGS_Psi_Re_Pred * vol_factor;
        a.psi_im = (double)Pi.AGS_Psi_Im_Pred * vol_factor;
        for (int k = 0; k < 3; k++) {
            a.grad_psi_re[k] = (double)Pi.AGS_Gradients_Psi_Re[k];
            a.grad_psi_im[k] = (double)Pi.AGS_Gradients_Psi_Im[k];
        }
#endif
        return a;
    }

    KOKKOS_INLINE_FUNCTION
    static NeighborData load_neighbor(const DeviceContext& ctx, int j,
                                      const IdentitySidecar& /*id*/,
                                      const ActiveData& /*active*/) {
        NeighborData n{};
        n.neighbor_particle = &ctx.P[j];
        return n;
    }

    /* pair_kernel — port of dm_fuzzy_gpu.cc:124-184 (per-pair form). */
    KOKKOS_INLINE_FUNCTION
    static void pair_kernel(const ActiveData& a,
                            const NeighborData& nb,
                            AccumData& accum, NoScatter& /*scatter*/) {
        if (!a.valid) return;
        if (nb.neighbor_particle == nullptr) return;
        struct particle_data& Pj = *nb.neighbor_particle;

        /* legacy neighbor validity guard */
        if (Pj.Mass <= 0 || Pj.AGS_Density <= 0) return;

        Vec3<double> dp;
        dp[0] = a.pos[0] - (double)Pj.Pos[0];
        dp[1] = a.pos[1] - (double)Pj.Pos[1];
        dp[2] = a.pos[2] - (double)Pj.Pos[2];
        nearest_xyz(dp);
        const double r2   = dp.norm_sq();
        const double h_i  = a.h_search;
        const double h2_i = h_i * h_i;
        if (!(r2 > 0) || !(r2 < h2_i)) return;

        const double r = sqrt(r2);
        double hinv, hinv3, hinv4, wk_i, dwk_i;
        kernel_hinv(h_i, &hinv, &hinv3, &hinv4);
        const double u = r * hinv;
        kernel_main(u, hinv3, hinv4, &wk_i, &dwk_i, -1);  /* -1 flag: faithful port */

        if (a.loop_iteration <= 0) {
            /* Pass 0: gradient of density-like scalars. */
            const double d_rho = (double)Pj.AGS_Density - a.ags_density;
            for (int k = 0; k < 3; k++) accum.grad_rho[k] += -wk_i * dp[k] * d_rho;
#if (DM_FUZZY > 0)
            const double d_re = (double)Pj.AGS_Psi_Re_Pred * (double)Pj.AGS_Density
                              / (double)Pj.Mass - a.psi_re;
            const double d_im = (double)Pj.AGS_Psi_Im_Pred * (double)Pj.AGS_Density
                              / (double)Pj.Mass - a.psi_im;
            for (int k = 0; k < 3; k++) {
                accum.grad_psi_re[k] += -wk_i * dp[k] * d_re;
                accum.grad_psi_im[k] += -wk_i * dp[k] * d_im;
            }
#endif
        } else {
            /* Pass 1: gradient-of-gradient tensor. */
            for (int k = 0; k < 3; k++) {
                const double d_grad_rho = (double)Pj.AGS_Gradients_Density[k]
                                        - a.grad_density[k];
                for (int k2 = 0; k2 < 3; k2++)
                    accum.grad2_rho[k2][k] += -wk_i * dp[k2] * d_grad_rho;
#if (DM_FUZZY > 0)
                const double d_gre = (double)Pj.AGS_Gradients_Psi_Re[k]
                                   - a.grad_psi_re[k];
                const double d_gim = (double)Pj.AGS_Gradients_Psi_Im[k]
                                   - a.grad_psi_im[k];
                for (int k2 = 0; k2 < 3; k2++) {
                    accum.grad2_psi_re[k2][k] += -wk_i * dp[k2] * d_gre;
                    accum.grad2_psi_im[k2][k] += -wk_i * dp[k2] * d_gim;
                }
#endif
            }
        }
    }
};

/* Toplevel — defined in sidm/dm_fuzzy_loop.cc; declared (guarded) in core/proto.h. */

#endif /* DM_FUZZY */
#endif /* DM_FUZZY_LOOP_H */
