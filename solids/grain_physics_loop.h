/* solids/grain_physics_loop.h — Spec definitions for the runner-template port
 * of the grain_physics neighbor loops (B7).
 *
 * Three independent Specs:
 *   GrainBackrxSpec    — grain→gas momentum backreaction (GRAIN_BACKREACTION).
 *                        ONEWAY grain sources, j-side atomics + ghost writeback.
 *   GrainRTGasSpec     — gas→grain RT opacity coupling, direction 1
 *                        (RT_OPACITY_FROM_EXPLICIT_GRAINS). SYMMETRIC, pure i-side.
 *   GrainRTGrainSpec   — grain→gas radiation acceleration, direction 2
 *                        (RT_OPACITY_FROM_EXPLICIT_GRAINS). ONEWAY, pure i-side.
 *
 * SSOT for per-pair physics: grain_physics_functions.h (included below).
 * No physics formulas live in this header; pair_kernel bodies call into it.
 *
 * Caller TUs must `#include "../mesh/kernel.h"` BEFORE this header (kernel.h
 * has no include guards; the KOKKOS_INLINE_FUNCTION pair kernels below call
 * kernel_main / kernel_hinv). Same pattern as difffilter_loop.h et al.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */
#ifndef GRAIN_PHYSICS_LOOP_H
#define GRAIN_PHYSICS_LOOP_H

/* Kokkos_Core before allvars.h — allvars pulls macros.h which #defines
 * terminate(...), mangling std::terminate in Kokkos_Core. */
#include <Kokkos_Core.hpp>

#include "../declarations/allvars.h"
#include "../mesh/neighbor_loop_runner.h"
#include "../mesh/mode_b_local_walker.h"   /* MODE_B_SEARCH_*, MODE_B_RADIUS_* */

#ifndef KOKKOS_INLINE_FUNCTION
#define KOKKOS_INLINE_FUNCTION inline
#endif

#ifdef GRAIN_FLUID
#include "grain_physics_functions.h"   /* SSOT: pair-kernel bodies + LocalIn types */


/* ============================================================================
 * GrainBackrxSpec — grain→gas momentum backreaction.
 * (GRAIN_BACKREACTION)
 * ============================================================================
 * Active sources: grain particles (GRAIN_PTYPES).
 * Neighbors: gas particles (Type==0).
 * Writes: atomic j-side updates to P[j].Vel, P[j].dp, P[j].Grain_AccelTimeMin,
 *         CellP[j].VelPred (+ GRAIN_EVOLUTION fields). Ghost writeback required.
 * AccumData: empty (no i-side reduction). Oracle compare is vacuous — validation
 * relies on MA-N cross-rank ghost-writeback records count and PA-vs-PB field
 * comparison (see OPEN_3d_grain_physics_design.md §E).
 * ========================================================================== */
#if defined(GRAIN_BACKREACTION)

/* ActiveData for GrainBackrxSpec — wraps GrainBackrxLocalIn for the runner.
 * pos/h_search carry the walker-facing fields; local carries all pair inputs. */
struct GrainBackrxActiveState {
    GrainBackrxLocalIn local;
    Vec3<double>        pos;
    double              h_search;
};

/* AccumData: empty — GrainBackrxSpec does no per-source reduction. */
struct GrainBackrxAccum {
    char _unused;
};

/* DeviceContext: extends base with oracle_dry_run so set_oracle_brute_pass can
 * suppress j-writes during the brute oracle pass (blocker fix: without this,
 * the oracle pass would fire grain_backrx_pair_kernel atomics before production,
 * applying backreaction twice). Same pattern as ThermalFBDeviceContext. */
struct GrainBackrxDeviceContext : NeighborLoopDeviceContextBase {
    bool oracle_dry_run = false;
};

/* NeighborData: carries the array-level P/CellP pointers + neighbor index j
 * so pair_kernel can forward to grain_backrx_pair_kernel(local, j, P, CellP,
 * dp, r2) without physics duplication. oracle_dry_run is propagated from ctx
 * to suppress j-writes during the brute oracle pass. */
struct GrainBackrxNeighborData {
    struct particle_data *P_arr;
    struct gas_cell_data *CellP_arr;
    int  j;
    bool oracle_dry_run;
};

struct GrainBackrxSpec {
    static constexpr const char *loop_name = "grain_backrx";

    static constexpr int                    search_mode        = MODE_B_SEARCH_ONEWAY;
    static constexpr unsigned int           neighbor_type_mask = 1u << 0;  /* gas */
    static constexpr mode_b_radius_policy_t radius_policy      = MODE_B_RADIUS_DEFAULT;

    static constexpr WritePattern  write_pattern             = WritePattern::ActiveReduceOnly;
    /* GasOnly: all j-neighbors are Type==0 gas. */
    static constexpr SidxCacheKind sidx_cache_kind           = SidxCacheKind::GasOnly;
    static constexpr bool          uses_ghost_writeback      = true;
    static constexpr bool          uses_ghost_write_detector = false;

    /* AccumData is empty; oracle compare is vacuous. */
    static constexpr double accum_tolerance = 1e-10;

    using CallScalars    = NlrCommonScalars;
    using ActiveData     = GrainBackrxActiveState;
    using AccumData      = GrainBackrxAccum;
    using NeighborData   = GrainBackrxNeighborData;
    using DeviceContext  = GrainBackrxDeviceContext;
    using ScatterData    = NoScatter;
    using IdentityFields = NoIdentity;
    using IterControl    = NotIterative;

    struct Aux {};

    static_assert(uses_ghost_writeback == true,
        "GrainBackrxSpec: j-writes to gas require ghost writeback for cross-rank deltas.");

    /* Host hooks — bodies in grain_physics_loop.cc. */
    static bool        is_active(int i);
    static double      search_radius(const neighbor_loop_args& args, int active_slot, int i);
    static CallScalars populate_call_scalars(const neighbor_loop_args& args);
    static void        apply_active_writeback(const neighbor_loop_args& args,
                                              int active_slot, int i,
                                              const AccumData& accum);
    static void        merge_accum(AccumData& local_accum, const AccumData& peer_accum);
    static double      compare_accum(const AccumData& local, const AccumData& oracle);
    static void        set_oracle_brute_pass(DeviceContext& ctx, bool on);
    static void        ghost_writeback_begin(const neighbor_loop_args& args,
                                             const NeighborLoopPlan& plan);
    static void        ghost_writeback_end  (const neighbor_loop_args& args,
                                             const NeighborLoopPlan& plan);

    /* Device hooks — header-inline. */

    KOKKOS_INLINE_FUNCTION
    static void zero_accum(AccumData& /*accum*/) {}  /* nothing to zero */

    KOKKOS_INLINE_FUNCTION
    static ActiveData load_active(const DeviceContext& ctx,
                                  int /*active_slot*/, int i,
                                  double h_search, const CallScalars& /*scalars*/) {
        ActiveData a{};
        struct particle_data& Pi = ctx.P[i];
        a.local.Pos                 = Pi.Pos;
        a.local.KernelRadius        = Pi.KernelRadius;
        a.local.Grain_DeltaMomentum = Pi.Grain_DeltaMomentum;
        a.local.Gas_Density         = Pi.Gas_Density;
        a.local.Grain_AccelTimeMin  = Pi.Grain_AccelTimeMin;
#if defined(GRAIN_EVOLUTION) && (GRAIN_EVOLUTION & (32|64))
        for(int kv = 0; kv < GRAIN_NUM_VOLATILE_SPECIES; kv++) {
            a.local.Grain_DeltaVolatileMass[kv] = Pi.Grain_DeltaVolatileMass[kv];
        }
        a.local.Grain_DeltaInternalEnergyHeating = Pi.Grain_DeltaInternalEnergyHeating;
#endif
        a.pos[0]   = (double)Pi.Pos[0];
        a.pos[1]   = (double)Pi.Pos[1];
        a.pos[2]   = (double)Pi.Pos[2];
        a.h_search = h_search;
        return a;
    }

    KOKKOS_INLINE_FUNCTION
    static NeighborData load_neighbor(const DeviceContext& ctx, int j,
                                      const IdentitySidecar& /*id*/,
                                      const ActiveData& /*active*/) {
        NeighborData n{};
        n.P_arr         = ctx.P;
        n.CellP_arr     = ctx.CellP;
        n.j             = j;
        n.oracle_dry_run = ctx.oracle_dry_run;
        return n;
    }

    KOKKOS_INLINE_FUNCTION
    static void pair_kernel(const ActiveData& active,
                            const NeighborData& nb,
                            AccumData& /*accum*/,
                            NoScatter& /*scatter*/) {
        /* Suppress j-writes during the oracle brute pass so the pass does not
         * apply backreaction before production (double-application bug). The
         * oracle pass validates search parity only; AccumData is empty so the
         * compare is vacuous regardless. */
        if(nb.oracle_dry_run) return;
        if(active.h_search <= 0) return;
        if(nb.P_arr == nullptr || nb.CellP_arr == nullptr) return;
        Vec3<double> dp;
        dp[0] = active.pos[0] - (double)nb.P_arr[nb.j].Pos[0];
        dp[1] = active.pos[1] - (double)nb.P_arr[nb.j].Pos[1];
        dp[2] = active.pos[2] - (double)nb.P_arr[nb.j].Pos[2];
        nearest_xyz(dp);
        double r2 = dp.norm_sq();
        grain_backrx_pair_kernel(active.local, nb.j, nb.P_arr, nb.CellP_arr, dp, r2);
    }
};

#endif /* GRAIN_BACKREACTION */


/* ============================================================================
 * GrainRTGasSpec — gas→grain RT opacity coupling, direction 1.
 * (RT_OPACITY_FROM_EXPLICIT_GRAINS)
 * ============================================================================
 * Active sources: gas particles (Type==0).
 * Neighbors: grain particles (GRAIN_PTYPES). SYMMETRIC at P[j].KernelRadius.
 * Writes: i-side only. Accumulates InterpolatedGeometricDustCrossSection,
 *         Interpolated_Opacity[], and (MHD_BATTERY_MECHANISMS & 8) J_dust_contribution.
 * ========================================================================== */
#if defined(RT_OPACITY_FROM_EXPLICIT_GRAINS)

/* Shared ActiveData for both RT directions — GasGrainRTLocalIn carries all
 * pair-kernel inputs; pos/h_search are the walker-facing fields. */
struct GasGrainRTActiveState {
    GasGrainRTLocalIn local;
    Vec3<double>      pos;
    double            h_search;
};

/* AccumData for GrainRTGasSpec — gas-direction fields only. */
struct GrainRTGasAccum {
    MyDouble InterpolatedGeometricDustCrossSection;
    MyDouble Interpolated_Opacity[N_RT_FREQ_BINS];
#if defined(MHD_BATTERY_MECHANISMS) && (MHD_BATTERY_MECHANISMS & 8)
    Vec3<MyDouble> J_dust_contribution;
#endif
};

/* AccumData for GrainRTGrainSpec — grain-direction field only. */
struct GrainRTGrainAccum {
    Vec3<MyDouble> Interpolated_Radiation_Acceleration;
};

/* Shared NeighborData for both RT directions. */
struct GasGrainRTNeighborData {
    struct particle_data *P_arr;
    struct gas_cell_data *CellP_arr;
    int j;
};

struct GrainRTGasSpec {
    static constexpr const char *loop_name = "grain_rt_gas";

    /* SYMMETRIC: grain kernel radius at j filters pairs (h_to_use = P[j].KernelRadius
     * inside gasgrain_rt_gas_search_pair_kernel). SYMMETRIC mode ensures pairs
     * with h_j > h_i are not missed. */
    static constexpr int                    search_mode        = MODE_B_SEARCH_SYMMETRIC;
    /* GRAIN_PTYPES: neither GasOnly (tbm=1) nor AllTypes (tbm=0x3f);
     * do not use a typed cache. */
    static constexpr unsigned int           neighbor_type_mask = GRAIN_PTYPES;
    static constexpr mode_b_radius_policy_t radius_policy      = MODE_B_RADIUS_DEFAULT;

    static constexpr WritePattern  write_pattern             = WritePattern::ActiveReduceOnly;
    static constexpr SidxCacheKind sidx_cache_kind           = SidxCacheKind::None;
    static constexpr bool          uses_ghost_writeback      = false;
    static constexpr bool          uses_ghost_write_detector = false;

    static constexpr double accum_tolerance = 1e-10;

    using CallScalars    = NlrCommonScalars;
    using ActiveData     = GasGrainRTActiveState;
    using AccumData      = GrainRTGasAccum;
    using NeighborData   = GasGrainRTNeighborData;
    using DeviceContext  = NeighborLoopDeviceContextBase;
    using ScatterData    = NoScatter;
    using IdentityFields = NoIdentity;
    using IterControl    = NotIterative;

    struct Aux {};

    /* Host hooks — bodies in grain_physics_loop.cc. */
    static bool        is_active(int i);
    static double      search_radius(const neighbor_loop_args& args, int active_slot, int i);
    static CallScalars populate_call_scalars(const neighbor_loop_args& args);
    static void        apply_active_writeback(const neighbor_loop_args& args,
                                              int active_slot, int i,
                                              const AccumData& accum);
    static void        merge_accum(AccumData& local_accum, const AccumData& peer_accum);
    static double      compare_accum(const AccumData& local, const AccumData& oracle);
    static void        set_oracle_brute_pass(DeviceContext& ctx, bool on);

    /* Device hooks — header-inline. */

    KOKKOS_INLINE_FUNCTION
    static void zero_accum(AccumData& accum) {
        accum.InterpolatedGeometricDustCrossSection = 0;
        for(int k = 0; k < N_RT_FREQ_BINS; k++) accum.Interpolated_Opacity[k] = 0;
#if defined(MHD_BATTERY_MECHANISMS) && (MHD_BATTERY_MECHANISMS & 8)
        accum.J_dust_contribution = {};
#endif
    }

    KOKKOS_INLINE_FUNCTION
    static ActiveData load_active(const DeviceContext& ctx,
                                  int /*active_slot*/, int i,
                                  double h_search, const CallScalars& /*scalars*/) {
        ActiveData a{};
        struct particle_data& Pi = ctx.P[i];
        a.local.Type        = Pi.Type;
        a.local.Pos         = Pi.Pos;
        a.local.Vel         = Pi.Vel;
        a.local.KernelRadius = Pi.KernelRadius;
        a.local.Mass        = Pi.Mass;
        a.local.Grain_Size  = 0;
        for(int k = 0; k < N_RT_FREQ_BINS; k++) a.local.Grain_Abs_Coeff[k] = 0;
        /* Gas source: no grain-direction fields needed. */
#if defined(MHD_BATTERY_MECHANISMS) && (MHD_BATTERY_MECHANISMS & 8)
        /* Gas source: pack electron fraction + density for J_dust Z_grain estimate. */
        a.local.Ne_gas      = (Pi.Type == 0 && ctx.CellP != nullptr)
                              ? ctx.CellP[i].Ne : 0.0f;
        a.local.Density_gas = (Pi.Type == 0 && ctx.CellP != nullptr)
                              ? ctx.CellP[i].Density : 0.0f;
#endif
        a.pos[0]   = (double)Pi.Pos[0];
        a.pos[1]   = (double)Pi.Pos[1];
        a.pos[2]   = (double)Pi.Pos[2];
        a.h_search = h_search;
        return a;
    }

    KOKKOS_INLINE_FUNCTION
    static NeighborData load_neighbor(const DeviceContext& ctx, int j,
                                      const IdentitySidecar& /*id*/,
                                      const ActiveData& /*active*/) {
        GasGrainRTNeighborData n{};
        n.P_arr    = ctx.P;
        n.CellP_arr = ctx.CellP;
        n.j        = j;
        return n;
    }

    KOKKOS_INLINE_FUNCTION
    static void pair_kernel(const ActiveData& active,
                            const NeighborData& nb,
                            AccumData& accum,
                            NoScatter& /*scatter*/) {
        if(active.h_search <= 0) return;
        if(nb.P_arr == nullptr) return;
        Vec3<double> dp;
        dp[0] = active.pos[0] - (double)nb.P_arr[nb.j].Pos[0];
        dp[1] = active.pos[1] - (double)nb.P_arr[nb.j].Pos[1];
        dp[2] = active.pos[2] - (double)nb.P_arr[nb.j].Pos[2];
        nearest_xyz(dp);
        double r2 = dp.norm_sq();

        /* Construct a local GasGrainRTOut, call SSOT body, then merge into accum. */
        GasGrainRTOut pair_out{};
        gasgrain_rt_gas_search_pair_kernel(active.local, nb.j, nb.P_arr, nb.CellP_arr,
                                           dp, r2, pair_out);
        accum.InterpolatedGeometricDustCrossSection += pair_out.InterpolatedGeometricDustCrossSection;
        for(int k = 0; k < N_RT_FREQ_BINS; k++) {
            accum.Interpolated_Opacity[k] += pair_out.Interpolated_Opacity[k];
        }
#if defined(MHD_BATTERY_MECHANISMS) && (MHD_BATTERY_MECHANISMS & 8)
        for(int k = 0; k < 3; k++) {
            accum.J_dust_contribution[k] += pair_out.J_dust_contribution[k];
        }
#endif
    }
};


/* ============================================================================
 * GrainRTGrainSpec — grain→gas radiation acceleration, direction 2.
 * (RT_OPACITY_FROM_EXPLICIT_GRAINS)
 * ============================================================================
 * Active sources: grain particles (GRAIN_PTYPES).
 * Neighbors: gas particles (Type==0). ONEWAY at local.KernelRadius.
 * Writes: i-side only. Accumulates Interpolated_Radiation_Acceleration.
 * ========================================================================== */

struct GrainRTGrainSpec {
    static constexpr const char *loop_name = "grain_rt_grain";

    static constexpr int                    search_mode        = MODE_B_SEARCH_ONEWAY;
    static constexpr unsigned int           neighbor_type_mask = 1u << 0;  /* gas */
    static constexpr mode_b_radius_policy_t radius_policy      = MODE_B_RADIUS_DEFAULT;

    static constexpr WritePattern  write_pattern             = WritePattern::ActiveReduceOnly;
    /* GasOnly: all j-neighbors are Type==0 gas. */
    static constexpr SidxCacheKind sidx_cache_kind           = SidxCacheKind::GasOnly;
    static constexpr bool          uses_ghost_writeback      = false;
    static constexpr bool          uses_ghost_write_detector = false;

    static constexpr double accum_tolerance = 1e-10;

    using CallScalars    = NlrCommonScalars;
    using ActiveData     = GasGrainRTActiveState;
    using AccumData      = GrainRTGrainAccum;
    using NeighborData   = GasGrainRTNeighborData;
    using DeviceContext  = NeighborLoopDeviceContextBase;
    using ScatterData    = NoScatter;
    using IdentityFields = NoIdentity;
    using IterControl    = NotIterative;

    struct Aux {};

    /* Host hooks — bodies in grain_physics_loop.cc. */
    static bool        is_active(int i);
    static double      search_radius(const neighbor_loop_args& args, int active_slot, int i);
    static CallScalars populate_call_scalars(const neighbor_loop_args& args);
    static void        apply_active_writeback(const neighbor_loop_args& args,
                                              int active_slot, int i,
                                              const AccumData& accum);
    static void        merge_accum(AccumData& local_accum, const AccumData& peer_accum);
    static double      compare_accum(const AccumData& local, const AccumData& oracle);
    static void        set_oracle_brute_pass(DeviceContext& ctx, bool on);

    /* Device hooks — header-inline. */

    KOKKOS_INLINE_FUNCTION
    static void zero_accum(AccumData& accum) {
        accum.Interpolated_Radiation_Acceleration = {};
    }

    KOKKOS_INLINE_FUNCTION
    static ActiveData load_active(const DeviceContext& ctx,
                                  int /*active_slot*/, int i,
                                  double h_search, const CallScalars& /*scalars*/) {
        ActiveData a{};
        struct particle_data& Pi = ctx.P[i];
        a.local.Type         = Pi.Type;
        a.local.Pos          = Pi.Pos;
        a.local.Vel          = Pi.Vel;
        a.local.KernelRadius = Pi.KernelRadius;
        a.local.Mass         = Pi.Mass;
        a.local.Grain_Size   = 0;
        for(int k = 0; k < N_RT_FREQ_BINS; k++) a.local.Grain_Abs_Coeff[k] = 0;
        /* Grain source: fill Grain_Size and Grain_Abs_Coeff for grain→gas direction. */
        if(((1 << Pi.Type) & GRAIN_PTYPES) && Pi.Gas_Density > 0) {
            a.local.Grain_Size = Pi.Grain_Size;
            double R_grain_code   = (double)Pi.Grain_Size / UNIT_LENGTH_IN_CGS;
            double rho_grain_code = All.Grain_Internal_Density / UNIT_DENSITY_IN_CGS;
            double rho_gas_code   = (double)Pi.Gas_Density * All.cf_a3inv;
            for(int k_freq = 0; k_freq < N_RT_FREQ_BINS; k_freq++) {
                double Q_abs = grain_extinction_Q_inline(i, k_freq, ctx.P);
                a.local.Grain_Abs_Coeff[k_freq] = (MyFloat)(Q_abs * 3.0 /
                    (4.0 * C_LIGHT_CODE_REDUCED * rho_grain_code * R_grain_code * rho_gas_code));
            }
        }
#if defined(MHD_BATTERY_MECHANISMS) && (MHD_BATTERY_MECHANISMS & 8)
        a.local.Ne_gas      = 0.0f;
        a.local.Density_gas = 0.0f;
#endif
        a.pos[0]   = (double)Pi.Pos[0];
        a.pos[1]   = (double)Pi.Pos[1];
        a.pos[2]   = (double)Pi.Pos[2];
        a.h_search = h_search;
        return a;
    }

    KOKKOS_INLINE_FUNCTION
    static NeighborData load_neighbor(const DeviceContext& ctx, int j,
                                      const IdentitySidecar& /*id*/,
                                      const ActiveData& /*active*/) {
        GasGrainRTNeighborData n{};
        n.P_arr    = ctx.P;
        n.CellP_arr = ctx.CellP;
        n.j        = j;
        return n;
    }

    KOKKOS_INLINE_FUNCTION
    static void pair_kernel(const ActiveData& active,
                            const NeighborData& nb,
                            AccumData& accum,
                            NoScatter& /*scatter*/) {
        if(active.h_search <= 0) return;
        if(nb.P_arr == nullptr) return;
        Vec3<double> dp;
        dp[0] = active.pos[0] - (double)nb.P_arr[nb.j].Pos[0];
        dp[1] = active.pos[1] - (double)nb.P_arr[nb.j].Pos[1];
        dp[2] = active.pos[2] - (double)nb.P_arr[nb.j].Pos[2];
        nearest_xyz(dp);
        double r2 = dp.norm_sq();

        GasGrainRTOut pair_out{};
        gasgrain_rt_grain_search_pair_kernel(active.local, nb.j, nb.P_arr, nb.CellP_arr,
                                             dp, r2, pair_out);
        for(int k = 0; k < 3; k++) {
            accum.Interpolated_Radiation_Acceleration[k] +=
                pair_out.Interpolated_Radiation_Acceleration[k];
        }
    }
};

#endif /* RT_OPACITY_FROM_EXPLICIT_GRAINS */

#endif /* GRAIN_FLUID */

#endif /* GRAIN_PHYSICS_LOOP_H */
