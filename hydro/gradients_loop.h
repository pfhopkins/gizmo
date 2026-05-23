/* hydro/gradients_loop.h — GradientsSpec for the neighbor-loop runner.
 *
 * Ports the legacy GPU walker `gradient_evaluate_gpu` (hydro/density_gpu.cc:
 * 93-279) to the runner Spec contract. Computes per-active gradients
 * (Density / Pressure / Velocity + many conditional fields) via symmetric
 * gas-gas neighbor topology, accumulating into the standard
 * `GasGraddata_out_` struct. Wraps the unchanged inline pair body
 * `gradient_accumulate_neighbor` (hydro/gradient_functions.h:189) — pair
 * physics is byte-for-byte preserved.
 *
 * Hydro corridor commit 7: second corridor consumer (after cellcorrections
 * 5a/5b/5c); ahead of HydroForceSpec (commit 8). Broad active row policy
 * matching the legacy GPU walker (Type==0 && Mass>0); narrow GasGrad_isactive
 * gate stays at the neighbor side inside the pair body. See
 * OPEN_3d_gradientsspec_design.md and OPEN_3d_hydro_corridor_design.md.
 *
 * Written by Philip F. Hopkins (phopkins@caltech.edu) for GIZMO. */

#ifndef GRADIENTS_LOOP_H
#define GRADIENTS_LOOP_H

#include "../declarations/allvars.h"
#include "../declarations/multifluid_helpers.h"  /* same_lagrangian_fluid_id (no-op when HYDRO_MULTIFLUID undef) */
#include "../mesh/neighbor_loop_runner.h"
#include "../mesh/mode_b_local_walker.h"      /* MODE_B_SEARCH_*, MODE_B_RADIUS_* */
#include "gradient_functions.h"               /* Quantities_for_Gradients,
                                                * GasGraddata_in_/out_,
                                                * kernel_GasGrad,
                                                * gradient_accumulate_neighbor,
                                                * GasGrad_isactive_gpu,
                                                * SHOULD_I_USE_SPH_GRADIENTS */
/* NOTE: caller translation units must include "../mesh/kernel.h" BEFORE
 * this header. kernel.h has no include guards (defines static inline
 * kernel_main / kernel_hinv used by the pair body); double-include triggers
 * redefinition errors. Matches the cellcorrections_loop.h / sink_env1_loop.h
 * convention. gradient_functions.h is similarly include-guard-free for the
 * same reason. */

#ifndef KOKKOS_INLINE_FUNCTION
#define KOKKOS_INLINE_FUNCTION inline
#endif

/* Number of MHD-CG outer iterations (legacy: see hydro/gradients.cc top).
 * Used by hydro_gradient_calc()'s outer loop. */
#if defined(MHD_CONSTRAINED_GRADIENT)
#if (MHD_CONSTRAINED_GRADIENT > 1)
#define NUMBER_OF_GRADIENT_ITERATIONS 3
#else
#define NUMBER_OF_GRADIENT_ITERATIONS 2
#endif
#else
#define NUMBER_OF_GRADIENT_ITERATIONS 1
#endif

/* Host-side narrow predicate (defined in gradients_loop.cc; referenced by
 * cellcorrections_loop.h via extern forward-decl). */
int GasGrad_isactive(int i, struct particle_data *pp, struct gas_cell_data *cell);

/* ============================================================================
 * Per-active scratch carried across MHD-CG iterations + into finalization.
 * Migrated from the file-static `temporary_data_topass` in the legacy
 * hydro/gradients.cc (lines 209-240). Named type so GradientsAux can hold a
 * base pointer and apply_active_writeback can scatter into it.
 * ========================================================================== */
struct temporary_data_topass
{
    struct Quantities_for_Gradients Maxima;
    struct Quantities_for_Gradients Minima;
    MyFloat MaxDistance;
#if defined(KERNEL_CRK_FACES)
    MyDouble m0;
    MyDouble m1[3];
    MyDouble m2[6];
    MyDouble dm0[3];
    MyDouble dm1[3][3];
    MyDouble dm2[6][3];
#endif
#if defined(HYDRO_MESHLESS_FINITE_VOLUME) && (HYDRO_FIX_MESH_MOTION==6)
    Vec3<MyFloat> GlassAcc;
#endif
#ifdef MHD_CONSTRAINED_GRADIENT
    MyDouble FaceDotB;
    MyDouble FaceCrossX[3][3];
    MyDouble BGrad[3][3];
#ifdef MHD_CONSTRAINED_GRADIENT_MIDPOINT
    Vec3<MyDouble> PhiGrad;
#endif
#endif
#ifdef RT_COMPGRAD_EDDINGTON_TENSOR
    Vec3<MyFloat> Gradients_Rad_E_gamma[N_RT_FREQ_BINS];
#endif
#ifdef TURB_DIFF_DYNAMIC
    Vec3<MyDouble> GradVelocity_bar[3];
#endif
};

/* ============================================================================
 * Per-pair physics types.
 * ========================================================================== */

/* ActiveData = GasGraddata_in_ `local` plus per-active scratch the pair body
 * needs (computed once in load_active so it isn't recomputed per neighbor:
 * Mass sign trick result + kernel_mode + V_i + hinv triplet). Legacy Mass
 * sign convention from density_gpu.cc:144-208 preserved byte-for-byte. */
struct GradientsActiveData
{
    /* Runner Mode B remote walker requires `pos` (Vec3<double>) + `h_search`
     * (double) on ActiveData — peer ranks need geometry without unpacking
     * the rest of the struct. Same shape as cellcorrections / sink_env1. */
    Vec3<double> pos;
    double       h_search;

    struct GasGraddata_in_ local;
    double hinv;
    double hinv3;
    double hinv4;
    double V_i;
    int    sph_gradients_flag_i;
    int    kernel_mode_i;
#ifdef HYDRO_MULTIFLUID
    unsigned char FluidType;   /* packed P[i].FluidType — for same_lagrangian_fluid_id() */
#endif
};

/* NeighborData carries the integer index j plus the base P/CellP pointers.
 * gradient_accumulate_neighbor indexes ~30 P[j]/CellP[j] fields by j, so the
 * pair_kernel needs both. Different shape from sink_env1's `*particle_data`
 * because the gradient pair body accesses j by integer throughout. */
struct GradientsNeighborData
{
    int                   j;
    struct particle_data *P;
    struct gas_cell_data *CellP;
#ifdef HYDRO_MULTIFLUID
    unsigned char         FluidType;  /* packed P[j].FluidType */
#endif
};

/* Aux carries the legacy `GasGradDataPasser` base pointer + the current
 * MHD-CG iteration index. apply_active_writeback dispatches on grad_iter to
 * replay either out2particle_GasGrad (iter==0) or out2particle_GasGrad_iter
 * (iter>0). load_active reads grad_iter through DeviceContext below (no
 * access to args from device-side load_active). */
struct GradientsAux
{
    struct temporary_data_topass *passer;    /* sized N_gas (host) */
    int                            grad_iter;
};

/* DeviceContext extension: ferries grad_iter from host (args.aux->grad_iter)
 * to device-side load_active. Trivially copyable — runner captures by value
 * into the Kokkos device lambda. populate_device_context body in
 * gradients_loop.cc. No UVM allocation -> no cleanup_device_context. */
struct GradientsDeviceContext : NeighborLoopDeviceContextBase
{
    int grad_iter;
};

/* ============================================================================
 * GradientsSpec — NeighborLoopSpec contract.
 * ========================================================================== */
struct GradientsSpec
{
    static constexpr const char *loop_name = "gradients";

    /* Symmetric gas-gas topology (matches gizmo_sym_neighbor_list + the
     * legacy gradient_evaluate_gpu CSR consumer). */
    static constexpr int                     search_mode        = MODE_B_SEARCH_SYMMETRIC;
    static constexpr unsigned int            neighbor_type_mask = (1u << 0);
    static constexpr mode_b_radius_policy_t  radius_policy      = MODE_B_RADIUS_DEFAULT;

    /* Pure i-side accumulate (writes via apply_active_writeback into host
     * CellP[i].Gradients / GasGradDataPasser[i]). No ghost-writeback. */
    static constexpr WritePattern   write_pattern              = WritePattern::ActiveReduceOnly;
    static constexpr SidxCacheKind  sidx_cache_kind            = SidxCacheKind::GasOnly;
    static constexpr double         accum_tolerance            = 1e-10;
    static constexpr bool           uses_ghost_writeback       = false;
    static constexpr bool           uses_ghost_write_detector  = false;

    /* Broad active predicate matching legacy GPU walker's row source
     * `gizmo_sym_active_indices` (Type==0 && Mass>0). The narrow filter
     * (GasGrad_isactive: KernelRadius>0, Density>0, DelayTime>0 gates) is
     * applied at the neighbor side inside gradient_accumulate_neighbor via
     * GasGrad_isactive_gpu(j) — same as today. Adding the narrow gate here
     * would silently filter active-i rows the legacy GPU walker processed
     * (directive #5 violation). */
    static bool is_active(int i) {
        if(P[i].Type != 0) return false;
        if(P[i].Mass <= 0) return false;
        return true;
    }

    /* Per-pair physics types */
    using CallScalars   = NlrCommonScalars;
    using ActiveData    = GradientsActiveData;
    using AccumData     = struct GasGraddata_out_;
    using NeighborData  = GradientsNeighborData;
    using Aux           = GradientsAux;

    using ScatterData    = NoScatter;
    using IdentityFields = NoIdentity;
    using IterControl    = NotIterative;
    using DeviceContext  = GradientsDeviceContext;

    /* Per-active host hook: pre-arena search radius from P[i].KernelRadius. */
    static double search_radius(const neighbor_loop_args& /*args*/,
                                 int /*active_slot*/, int i)
    {
        return (double)P[i].KernelRadius;
    }

    /* Per-call scalars: only the common cosmology block. The pair body reads
     * All.* directly (bare `All.*` is fine under the All-mirror refactor —
     * see feedback_bare_all_in_pair_body_ok.md), so scalars is unused. */
    static CallScalars populate_call_scalars(const neighbor_loop_args& /*args*/)
    {
        return nlr_common_scalars_from_all();
    }

    /* DeviceContext extension hook: copy grad_iter from Aux into ctx.
     * Body in gradients_loop.cc. */
    static void populate_device_context(const neighbor_loop_args& args,
                                         DeviceContext& ctx);

    /* ----- Device hooks ----- */

    KOKKOS_INLINE_FUNCTION
    static void zero_accum(AccumData& accum)
    {
        /* GasGraddata_out_ is POD — byte-zero matches legacy
         * `memset(&out, 0, sizeof(out))` from density_gpu.cc:212. */
        for(size_t b = 0; b < sizeof(accum); b++) ((char*)&accum)[b] = 0;
    }

    /* Build ActiveData per active i. Byte-exact reconstruction of the
     * particle2in-equivalent block at density_gpu.cc:138-227, including the
     * Mass sign trick (negate as flag for SPH gradients), MHD-CG iter>0
     * Mass=0 sentinel (read from ctx.grad_iter), and per-active
     * kernel/mode/V_i precompute. No defensive safety early-outs added —
     * preserves current GPU walker behavior exactly (codex round-7 guard). */
    KOKKOS_INLINE_FUNCTION
    static ActiveData load_active(const DeviceContext& ctx,
                                   int /*active_slot*/, int i,
                                   double h_search,
                                   const CallScalars& /*scalars*/)
    {
        ActiveData active;
        for(size_t b = 0; b < sizeof(active); b++) ((char*)&active)[b] = 0;

        struct GasGraddata_in_ &local = active.local;
        struct particle_data   *kp    = ctx.P;
        struct gas_cell_data   *kc    = ctx.CellP;

        active.pos      = kp[i].Pos;
        active.h_search = h_search;

        local.Pos          = kp[i].Pos;
        local.KernelRadius = kp[i].KernelRadius;
        local.Mass         = kp[i].Mass;
        if(local.Mass < 0) { local.Mass = 0; }

        active.sph_gradients_flag_i = SHOULD_I_USE_SPH_GRADIENTS(kc[i].ConditionNumber);
        if(active.sph_gradients_flag_i) { local.Mass *= -1; }

#ifdef MHD_CONSTRAINED_GRADIENT
        if(ctx.grad_iter > 0) {
            if(kc[i].FlagForConstrainedGradients <= 0) { local.Mass = 0; }
        }
#endif

        local.GQuant.Density  = kc[i].Density;
        local.GQuant.Pressure = kc[i].Pressure;
        local.GQuant.Velocity = kc[i].VelPred;
#ifdef MAGNETIC
        local.GQuant.B = kc[i].BPred * (kc[i].Density / kp[i].Mass);
#ifdef DIVBCLEANING_DEDNER
        local.GQuant.Phi = kc[i].PhiPred / kp[i].Mass;
#endif
#endif
#ifdef DOGRAD_INTERNAL_ENERGY
        local.GQuant.InternalEnergy = kc[i].InternalEnergyPred;
#endif
#if defined(MHD_BATTERY_MECHANISMS) && (MHD_BATTERY_MECHANISMS & 1)
        local.GQuant.ElectronNumberDensity = kc[i].n_e();
        local.GQuant.ElectronTemperature   = kc[i].T_e();
#endif
#if defined(MHD_BATTERY_MECHANISMS) && (MHD_BATTERY_MECHANISMS & (2|4|8))
        local.GQuant.E_battery_T2 = kc[i].E_battery_T2_cell;
#endif
#ifdef DOGRAD_SOUNDSPEED
        local.GQuant.SoundSpeed = kc[i].effective_soundspeed();
#endif
#ifdef COSMIC_RAY_FLUID
        for(int k = 0; k < N_CR_PARTICLE_BINS; k++) {
            local.GQuant.CosmicRayPressure[k] = Get_Gas_CosmicRayPressure(i, k, kc);
        }
#endif
#if defined(TURB_DIFF_METALS) && !defined(TURB_DIFF_METALS_LOWORDER)
        for(int k = 0; k < NUM_METAL_SPECIES; k++) {
            local.GQuant.Metallicity[k] = kp[i].Metallicity[k];
        }
#endif
#if defined(RT_COMPGRAD_EDDINGTON_TENSOR) && (N_RT_FREQ_BINS > 0)
        for(int k = 0; k < N_RT_FREQ_BINS; k++) {
            local.GQuant.Rad_E_gamma[k]    = kc[i].Rad_E_gamma_Pred[k];
            local.GQuant.Rad_E_gamma_ET[k] = kc[i].ET[k];
#if defined(RT_M1_SECONDORDER) && defined(RT_EVOLVE_FLUX)
            for(int k2 = 0; k2 < 3; k2++) {
                local.GQuant.Rad_Flux[k][k2] = kc[i].Rad_Flux_Pred[k][k2];
            }
#endif
        }
#endif
#ifdef TURB_DIFF_DYNAMIC
        local.GQuant.Velocity_bar = kc[i].Velocity_bar;
        local.Norm_hat            = kc[i].Norm_hat;
#ifdef GALSF_SUBGRID_WINDS
        local.DelayTime = kc[i].DelayTime;
#endif
#endif
#ifdef MHD_CONSTRAINED_GRADIENT
        local.ConditionNumber = kc[i].ConditionNumber;
        local.NV_T            = kc[i].NV_T;
        for(int k = 0; k < 3; k++) {
            for(int k2 = 0; k2 < 3; k2++) {
                local.BGrad[k][k2] = kc[i].Gradients.B[k][k2];
            }
        }
#ifdef MHD_MODIFIED_GRADIENT
        local.MG_cgcoeff = kc[i].MG_cgcoeff;
#endif
#ifdef MHD_CONSTRAINED_GRADIENT_FAC_MEDDEV
        local.PhiGrad = kc[i].Gradients.Phi;
#endif
#endif

        if(active.sph_gradients_flag_i) { local.Mass *= -1; }    /* negate as flag */

        /* Per-active kernel triple + V_i + kernel_mode (legacy lines 215-226). */
        double h_i = local.KernelRadius;
        kernel_hinv(h_i, &active.hinv, &active.hinv3, &active.hinv4);
        if(local.Mass < 0) { local.Mass *= -1; }                 /* restore for V_i */
        active.V_i = local.Mass / local.GQuant.Density;
        if(active.sph_gradients_flag_i) { local.Mass *= -1; }    /* re-negate for kernel */

        active.kernel_mode_i = -1;
        if(active.sph_gradients_flag_i) active.kernel_mode_i = 0;
#if defined(HYDRO_SPH) || defined(KERNEL_CRK_FACES)
        active.kernel_mode_i = 0;
#endif
#ifdef HYDRO_MULTIFLUID
        active.FluidType = ctx.P[i].FluidType;
#endif
        return active;
    }

    KOKKOS_INLINE_FUNCTION
    static NeighborData load_neighbor(const DeviceContext& ctx,
                                       int j,
                                       const IdentitySidecar& /*id*/,
                                       const ActiveData& /*active*/)
    {
        NeighborData nb{j, ctx.P, ctx.CellP
#ifdef HYDRO_MULTIFLUID
                        , ctx.P[j].FluidType
#endif
        };
        return nb;
    }

    /* The physics — forwards to the unchanged inline pair body. const_cast
     * on &active.local because gradient_accumulate_neighbor takes
     * GasGraddata_in_* non-const (legacy signature) but only reads. */
    KOKKOS_INLINE_FUNCTION
    static void pair_kernel(const ActiveData& active,
                             const NeighborData& neighbor,
                             AccumData& accum,
                             NoScatter& /*scatter*/)
    {
#if defined(HYDRO_MULTIFLUID) && !defined(HYDRO_MULTIFLUID_NOOP_TEST)
        /* Corridor invariant (see hydro/hydro_corridor.h): cross-fluid skip
         * placed FIRST, before kernel construction or accumulator mutation.
         * HYDRO_MULTIFLUID_NOOP_TEST disables the predicate for strict-bit-
         * identity Test 2 validation builds only. */
        if (!same_lagrangian_fluid_id(active.FluidType, neighbor.FluidType)) return;
#endif
        struct kernel_GasGrad kernel;
        kernel.h_i = active.local.KernelRadius;
        gradient_accumulate_neighbor(
            const_cast<struct GasGraddata_in_*>(&active.local),
            &accum, &kernel, neighbor.j,
            active.sph_gradients_flag_i, active.V_i,
            active.hinv, active.hinv3, active.hinv4, active.kernel_mode_i,
            neighbor.P, neighbor.CellP);
    }

    /* Host writebacks (post-dispatch). Body in gradients_loop.cc: replays
     * out2particle_GasGrad (grad_iter==0) or out2particle_GasGrad_iter
     * (grad_iter>0), keyed on nlr_aux<GradientsSpec>(args)->grad_iter. */
    static void apply_active_writeback(const neighbor_loop_args& args,
                                        int active_slot, int i,
                                        const AccumData& accum);

    /* Peer-rank accum merge (Mode B remote). Body in gradients_loop.cc:
     * additive for Gradients[k].*, MAX/MIN for Maxima/Minima, MAX for
     * MaxDistance, additive for everything else. */
    static void merge_accum(AccumData& local_accum, const AccumData& peer_accum);

    /* Oracle comparison. Body in gradients_loop.cc. */
    static double compare_accum(const AccumData& local, const AccumData& oracle);
};

#endif /* GRADIENTS_LOOP_H */
