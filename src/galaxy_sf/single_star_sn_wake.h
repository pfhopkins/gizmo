/* Copyright (c) 2026 Ulrich Steinwandel
 * Author: Ulrich Steinwandel (uli@mpa-garching.mpg.de) */
/* galaxy_sf/single_star_sn_wake.h -- causal wake of single-star supernovae (SNE_BROADCAST_TIMESTEP), after starforge_dev #107 by Mike
 * Grudic: gas that a supernova of this step can reach before its own step ends caps its next step at 0.5 CourantFac of the arrival
 * time and, if inactive, is woken. One-way search from each supernova, radius = how far its fastest signal gets in the longest
 * occupied step; run only on steps with a supernova, evaluated where the cell lives. */
#ifndef SINGLE_STAR_SN_WAKE_H
#define SINGLE_STAR_SN_WAKE_H
#include <Kokkos_Core.hpp>
#include "../declarations/allvars.h"
#ifdef SNE_BROADCAST_TIMESTEP
#include "../mesh/neighbor_loop_runner.h"
#include "../mesh/mode_b_local_walker.h"

struct SnWakeIn {Vec3<double> Pos; double R, v;};   /* position, search radius (comoving), signal speed */
struct SnWakeScalars {
    NlrCommonScalars common;
    double unit_ti;               /* physical time per integer time step */
    double half_courant;          /* 0.5 CourantFac */
    integertime ti_current;
    int *woke_n, *woke_list, woke_cap; /* cells woken on this rank, by local index */
};
struct SnWakeActive {Vec3<double> pos; double h_search; SnWakeIn src;};
struct SnWakeDeviceContext : NeighborLoopDeviceContextBase {const SnWakeIn *per_active_local;};
struct SnWakeNeighbor {struct particle_data *p; int j;};
struct SnWakeOut {long long n;};

struct SnWakeSpec {
    static constexpr const char *loop_name = "single_star_sn_wake";
    static constexpr int                    search_mode        = MODE_B_SEARCH_ONEWAY;
    static constexpr unsigned int           neighbor_type_mask = (1u << 0);
    static constexpr mode_b_radius_policy_t radius_policy      = MODE_B_RADIUS_DEFAULT;
    static constexpr WritePattern  write_pattern   = WritePattern::ActiveReduceOnly;
    static constexpr SidxCacheKind sidx_cache_kind = SidxCacheKind::GasOnly;
    static constexpr bool needs_live_neighbours     = true;  /* writes the live cell */
    static constexpr bool uses_ghost_writeback      = false; /* evaluated where the cell lives */
    static constexpr bool uses_ghost_write_detector = false;
    static constexpr bool keeps_neighbour_mass_positive = true;
    static constexpr ModeBEvalOMP modeb_eval_omp = ModeBEvalOMP::EpsilonAtomic; /* minima and a compare-exchange: order-free */
    using CallScalars    = SnWakeScalars;
    using ActiveData     = SnWakeActive;
    using NeighborData   = SnWakeNeighbor;
    using DeviceContext  = SnWakeDeviceContext;
    using ScatterData    = NoScatter;
    using IdentityFields = NoIdentity;
    using IterControl    = NotIterative;
    using AccumData      = SnWakeOut;
    struct Aux {const SnWakeIn *host_src; int *woke_n, *woke_list, woke_cap;};
    static bool        is_active(int i);
    static double      search_radius(const neighbor_loop_args& args, int active_slot, int i);
    static CallScalars populate_call_scalars(const neighbor_loop_args& args);
    static void populate_device_context(const neighbor_loop_args& args, DeviceContext& ctx);
    static void cleanup_device_context(const neighbor_loop_args& args, DeviceContext& ctx);
    static void apply_active_writeback(const neighbor_loop_args&, int, int, const AccumData&) {}
    KOKKOS_INLINE_FUNCTION static void zero_accum(AccumData& a) {a.n = 0;}
    KOKKOS_INLINE_FUNCTION static void merge_accum(AccumData& l, const AccumData& p) {l.n += p.n;}
    KOKKOS_INLINE_FUNCTION
    static ActiveData load_active(const DeviceContext& dctx, int active_slot, int i, double h_search, const CallScalars&) {
        ActiveData a{};
        if(dctx.per_active_local != nullptr) {a.src = dctx.per_active_local[active_slot];}
        else {for(int k = 0; k < 3; k++) {a.src.Pos[k] = (double)dctx.P[i].Pos[k];} a.src.R = h_search; a.src.v = 0;}
        a.pos = a.src.Pos; a.h_search = a.src.R;
        return a;
    }
    KOKKOS_INLINE_FUNCTION
    static NeighborData load_neighbor(const DeviceContext& dctx, int j, const IdentitySidecar&, const ActiveData&) {
        NeighborData n{}; n.p = &dctx.P[j]; n.j = j; return n;
    }
    KOKKOS_INLINE_FUNCTION
    static void pair_kernel(const ActiveData& a, const NeighborData& nb, AccumData& out, NoScatter&, const CallScalars& cs) {
        const struct particle_data& Pj = *nb.p;
        if(Pj.Type != 0 || Pj.Mass <= 0 || !(a.src.v > 0)) {return;}
        Vec3<double> d;
        for(int k = 0; k < 3; k++) {d[k] = (double)Pj.Pos[k] - a.src.Pos[k];}
        nearest_xyz(d);
        double r = sqrt(d.norm_sq());
        if(r >= a.src.R) {return;}
        double t_arr = ((r > (double)Pj.KernelRadius) ? r : (double)Pj.KernelRadius) * cs.common.cf_atime / a.src.v;
        Kokkos::atomic_min(&nb.p->SNe_ArrivalTime, (MyFloat)t_arr);
        integertime ti_end = Pj.Ti_begstep + Pj.integertime_step();
        if(ti_end <= cs.ti_current) {return;} /* active: chooses its next step under the cap */
        if((double)(ti_end - cs.ti_current) * cs.unit_ti <= cs.half_courant * t_arr) {return;}
        int old = Kokkos::atomic_load(&nb.p->wakeup);
        if(old > 0 || old == -4 || Kokkos::atomic_compare_exchange(&nb.p->wakeup, old, -4) != old) {return;} /* a hydro wake-up stands */
        int k = Kokkos::atomic_fetch_add(cs.woke_n, 1);
        if(k < cs.woke_cap) {cs.woke_list[k] = nb.j;}
        out.n++;
    }
};

#endif
#endif
