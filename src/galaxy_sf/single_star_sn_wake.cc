/* Copyright (c) 2026 Ulrich Steinwandel
 * Author: Ulrich Steinwandel (uli@mpa-garching.mpg.de) */
/* galaxy_sf/single_star_sn_wake.cc -- host side of the causal wake of single-star supernovae (single_star_sn_wake.h). The
 * supernovae of a step are recorded where they couple (single_star_feedback_finalize); whether any rank has one comes with the
 * source-count reduction of the mechanical feedback (mechanical_fb_calc_toplevel, slot 2), so steps without one pay nothing. */
#include <mpi.h>
#include <cstring>
#include <vector>
#include <Kokkos_Core.hpp>
#include "../declarations/gpu_all_mirror.h"  /* before allvars.h */
#include "single_star_sn_wake.h"
#include "../core/proto.h"
#include "../core/wakeup_sidecar.h"
#ifdef SNE_BROADCAST_TIMESTEP

static std::vector<int> sn_wake_list;      /* the stars that exploded this step, local index */
static std::vector<SnWakeIn> sn_wake_in;
static int sn_wake_any = 0;                /* the same on every rank: a supernova somewhere this step */

void sne_broadcast_record(int i, double v)
{
    SnWakeIn e; for(int k = 0; k < 3; k++) {e.Pos[k] = P[i].Pos[k];} e.R = 0; e.v = v;
    sn_wake_list.push_back(i); sn_wake_in.push_back(e);
}

void single_star_sn_this_step(int any) {sn_wake_any = any;}
int single_star_sn_wake_pending_local(void) {return sn_wake_list.empty() ? 0 : 1;} /* recorded outside the feedback loop (ejecta cells) */

bool SnWakeSpec::is_active(int) {return true;}

double SnWakeSpec::search_radius(const neighbor_loop_args& args, int active_slot, int) {return nlr_aux<SnWakeSpec>(args)->host_src[active_slot].R;}

SnWakeScalars SnWakeSpec::populate_call_scalars(const neighbor_loop_args& args)
{
    SnWakeScalars s; Aux *aux = nlr_aux<SnWakeSpec>(args);
    s.common = nlr_common_scalars_from_all();
    s.unit_ti = All.Timebase_interval / All.cf_hubble_a; s.half_courant = 0.5 * All.CourantFac; s.ti_current = All.Ti_Current;
    s.woke_n = aux->woke_n; s.woke_list = aux->woke_list; s.woke_cap = aux->woke_cap;
    return s;
}

void SnWakeSpec::populate_device_context(const neighbor_loop_args& args, DeviceContext& ctx)
{
    Aux *aux = nlr_aux<SnWakeSpec>(args);
    const int N = args.num_active;
    ctx.per_active_local = nullptr;
    if(N <= 0 || aux == nullptr || aux->host_src == nullptr) {return;}
    SnWakeIn *uvm = (SnWakeIn *) gizmo_gpu_alloc_shared((size_t) N * sizeof(SnWakeIn), NULL);
    if(!uvm) {ctx.populate_failed = 1; gizmo_request_controlled_stop(7903, "supernova causal wake: could not stage the sources", __FILE__, __LINE__, __FUNCTION__); return;}
    std::memcpy(uvm, aux->host_src, (size_t) N * sizeof(SnWakeIn));
    ctx.per_active_local = uvm;
}

void SnWakeSpec::cleanup_device_context(const neighbor_loop_args&, DeviceContext& ctx)
{
    if(ctx.per_active_local) {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(const_cast<SnWakeIn *>(ctx.per_active_local)); ctx.per_active_local = nullptr;}
}

/* after the mechanical feedback (all ranks): on a step with a supernova, the gas its signal can reach within the longest occupied
   step is capped and, if its step would outlast the arrival, woken; otherwise nothing */
void sne_causal_wake(void)
{
    int any = sn_wake_any; sn_wake_any = 0;
    if(!any) {sn_wake_list.clear(); sn_wake_in.clear(); return;}
    double dt_max = (double) GET_INTEGERTIME_FROM_TIMEBIN(All.HighestOccupiedTimeBin) * All.Timebase_interval / All.cf_hubble_a;
    for(SnWakeIn &e : sn_wake_in) {e.R = 2. * e.v * dt_max / (All.CourantFac * All.cf_atime);} /* beyond it the arrival outlasts any step */
    int cap = (N_gas > 0) ? N_gas : 1;
    int *buf = (int *) gizmo_gpu_alloc_shared((size_t)(cap + 1) * sizeof(int), NULL);
    if(!buf) {gizmo_request_controlled_stop(7904, "supernova causal wake: no buffer for the woken cells", __FILE__, __LINE__, __FUNCTION__); return;}
    buf[cap] = 0;
    SnWakeSpec::Aux aux = {sn_wake_in.empty() ? nullptr : sn_wake_in.data(), &buf[cap], buf, cap};
    neighbor_loop_args args = nlr_default_args();
    args.active_list = sn_wake_list.empty() ? nullptr : sn_wake_list.data(); args.num_active = (int) sn_wake_list.size(); args.aux = &aux;
    run_neighbor_loop<SnWakeSpec>(args);
    int n = DMIN(buf[cap], cap);
    for(int k = 0; k < n; k++) {wakeup_sidecar_mark(buf[k]);}
    if(n > 0) {NeedToWakeupParticles_local = 1;}
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(buf);
    sn_wake_list.clear(); sn_wake_in.clear();
}
#endif
