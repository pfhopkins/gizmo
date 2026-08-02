/* sinks/sink_swk_loop.cc — host-only hooks + compound ghost-writeback
 * callback for SinkSwkSpec.
 *
 * KOKKOS_INLINE_FUNCTION hooks (load_active, load_neighbor, pair_kernel,
 * zero_accum) and the inline pair body (sink_swk_pair_kernel) live in
 * sink_swk_loop.h so they inline from device kernels (Mode A) and host
 * walkers (Mode B / Brute oracle). This translation unit holds host-only
 * hooks: per-active radius, per-call scalars capture, populate/cleanup
 * device-context (UVM staging), apply_active_writeback,
 * merge_accum (ACCUM_ADD / ACCUM_ADD_VEC3 / ACCUM_MIN / ACCUM_ADD_ARRAY
 * Spec-local manifest), the ghost-writeback compound callback (snapshot
 * + CR-aware delta predicate + pack + clamped apply + cleanup),
 * lifecycle hooks, and env-gated diagnostics.
 *
 * Compound callback (vs sink_feed's per-field manifest): the legacy
 * sinkswallow writeback predicate gates the FULL delta record on
 * (Mass/Sink_Mass/Sink_Mdot/Vel changed). Per-field expansion would
 * change behavior for secondary-only writes (B/BPred under
 * SINK_RETURN_BFLUX paths). Compound callback preserves the legacy
 * predicate exactly while adding CR-field changes to the trip set
 * (the deliberate CR fix vs legacy Mode A — see sink_swk_loop.h pair
 * body comment).
 *
 * Replaces sinks/sink_swallow_and_kick_gpu.cc and
 * mesh/ghost_writeback.cc::ghost_writeback_{zero_,}sinkswallow.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <Kokkos_Core.hpp>

#include "../declarations/gpu_all_mirror.h"  /* MUST precede allvars.h: installs device-pass `#define All AllDeviceMirror` redirect before cell_data.h is parsed */
#include "../declarations/allvars.h"
#include "../declarations/gpu_numeric_macros.h"
#include "../core/proto.h"
#include "../mesh/kernel.h"               /* MUST precede sink_swk_loop.h */
#include "../mesh/ghost_writeback.h"
#include "sink_swk_loop.h"

#ifdef SINK_PARTICLES

/* ============================================================================
 * PHYSICS HOOKS
 * ========================================================================== */

double SinkSwkSpec::search_radius(const neighbor_loop_args& args,
                                   int /*active_slot*/, int i)
{
    return (double)args.P[i].KernelRadius;
}

SinkSwkSpec::CallScalars
SinkSwkSpec::populate_call_scalars(const neighbor_loop_args& /*args*/)
{
    /* Uses nlr_host_all_ptr() for explicit host-snapshot intent. The
     * All-mirror redirect is device-pass-only, so bare All.* in this host
     * hook would also be correct; the accessor documents that these reads
     * run on the host pass. */
    const struct global_data_all_processes *h = nlr_host_all_ptr();
    CallScalars scalars;
    scalars.common                   = nlr_common_scalars_from_all();
    scalars.sink_radius_grav         = SinkParticle_GravityKernelRadius;
    scalars.sink_feedback_factor     = h->SinkFeedbackFactor;
    scalars.sink_radiative_efficiency= h->SinkRadiativeEfficiency;
#if defined(SINK_WIND_KICK) || defined(SINK_WIND_SPAWN)
    scalars.sink_accreted_fraction   = h->Sink_accreted_fraction;
    scalars.sink_outflow_velocity    = h->Sink_outflow_velocity;
#else
    scalars.sink_accreted_fraction   = 0.0;
    scalars.sink_outflow_velocity    = 0.0;
#endif
#ifdef SINK_PHOTONMOMENTUM
    scalars.sink_rad_momentum_factor = h->Sink_Rad_MomentumFactor;
#else
    scalars.sink_rad_momentum_factor = 0.0;
#endif
    /* Fallback zeros above are never observed: every pair-body reader of
     * these scalars is itself under the same #ifdef gate. */
#if defined(COSMIC_RAY_FLUID) && defined(SINK_COSMIC_RAYS)
    scalars.sink_cr_injection_efficiency = h->Sink_CosmicRay_Injection_Efficiency;
#endif
#ifdef GALSF_SUBGRID_WINDS
    scalars.wind_free_travel_max_time_factor = h->WindFreeTravelMaxTimeFactor;
#endif
    return scalars;
}

void SinkSwkSpec::apply_active_writeback(const neighbor_loop_args& args,
                                          int active_slot, int /*i*/,
                                          const AccumData& accum)
{
    Aux *aux = nlr_aux<SinkSwkSpec>(args);
    aux->per_active_accum[active_slot] = accum;
}

/* Per-field merge of a peer rank's contribution. Per-field op MUST match
 * the pair_kernel writes; Accreted_Age uses MIN-merge with the
 * MAX_REAL_NUMBER sentinel from zero_accum. The oracle catches drift
 * between this manifest and pair_kernel writes.
 *
 * Adding a new accumulator field for this loop = ONE LINE under the
 * appropriate physics flag's #ifdef. */
void SinkSwkSpec::merge_accum(AccumData& local_accum, const AccumData& peer_accum)
{
#define ACCUM_ADD(field)        local_accum.field += peer_accum.field;
#define ACCUM_ADD_VEC3(field)   for(int k = 0; k < 3; k++) local_accum.field[k] += peer_accum.field[k];
#define ACCUM_MIN(field)        if(peer_accum.field < local_accum.field) local_accum.field = peer_accum.field;
#define ACCUM_ADD_ARRAY(field, len)                                       \
    for(int k = 0; k < (len); k++) local_accum.field[k] += peer_accum.field[k];

    ACCUM_ADD(accreted_Mass)
    ACCUM_ADD(accreted_Sink_Mass)
    ACCUM_ADD(accreted_Sink_Mass_reservoir)
#if defined(SINK_SWALLOWGAS) && !defined(SINK_GRAVCAPTURE_GAS)
    ACCUM_ADD(Sink_AccretionDeficit)
#endif
#ifdef GRAIN_FLUID
    ACCUM_ADD(accreted_dust_Mass)
#endif
#ifdef RT_REINJECT_ACCRETED_PHOTONS
    ACCUM_ADD(accreted_photon_energy)
#endif
#if defined(SINK_FOLLOW_ACCRETED_MOMENTUM)
    ACCUM_ADD_VEC3(accreted_momentum)
#endif
#if defined(SINK_FOLLOW_ACCRETED_COM)
    ACCUM_ADD_VEC3(accreted_centerofmass)
#endif
#if defined(SINK_RETURN_BFLUX)
    ACCUM_ADD_VEC3(accreted_B)
#endif
#if defined(SINK_FOLLOW_ACCRETED_ANGMOM)
    ACCUM_ADD_VEC3(accreted_J)
#endif
#ifdef SINK_COUNTPROGS
    ACCUM_ADD(Sink_CountProgs)
#endif
#ifdef GALSF
    ACCUM_MIN(Accreted_Age)
#endif
    ACCUM_ADD(n_gas_swallowed)
    ACCUM_ADD(n_sink_swallowed)
    ACCUM_ADD(n_star_swallowed)
    ACCUM_ADD(n_dm_swallowed)
    ACCUM_ADD_ARRAY(delta_TimeBin_Sink_mass,          TIMEBINS)
    ACCUM_ADD_ARRAY(delta_TimeBin_Sink_dynamicalmass, TIMEBINS)
    ACCUM_ADD_ARRAY(delta_TimeBin_Sink_Mdot,          TIMEBINS)
    ACCUM_ADD_ARRAY(delta_TimeBin_Sink_Medd,          TIMEBINS)

#undef ACCUM_ADD
#undef ACCUM_ADD_VEC3
#undef ACCUM_MIN
#undef ACCUM_ADD_ARRAY
}

/* ============================================================================
 * DEVICE CONTEXT LIFECYCLE
 * ========================================================================== */

void SinkSwkSpec::populate_device_context(const neighbor_loop_args& args,
                                           DeviceContext& ctx)
{
    ctx.oracle_dry_run = false;

    Aux *aux = nlr_aux<SinkSwkSpec>(args);
    const int N = args.num_active;
    if(N <= 0) {
        ctx.per_active_local = nullptr;
        return;
    }
    SinkSwallowLocalIn *uvm = (SinkSwallowLocalIn *)
        Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(N * sizeof(SinkSwallowLocalIn));
    std::memcpy(uvm, aux->host_locals, N * sizeof(SinkSwallowLocalIn));
    ctx.per_active_local = uvm;
}

void SinkSwkSpec::cleanup_device_context(const neighbor_loop_args& /*args*/,
                                          DeviceContext& ctx)
{
    if(ctx.per_active_local) {
        Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(
            const_cast<SinkSwallowLocalIn *>(ctx.per_active_local));
        ctx.per_active_local = nullptr;
    }
}

/* ============================================================================
 * COMPOUND GHOST-WRITEBACK CALLBACK
 *
 * One callback ships the full sinkswallow delta record (mirrors the legacy
 * ghost_delta_sinkswallow_t in mesh/ghost_writeback.cc:1260; PLUS CR fields
 * under #ifdef COSMIC_RAY_FLUID && SINK_COSMIC_RAYS — the deliberate CR fix).
 *
 * Predicate matches legacy primary gate (Mass / Sink_Mass / Sink_Mdot / Vel
 * changed) and adds CR-field changes when CR is active so a kernel that
 * writes CR without touching Vel/Mass would still trip the gate. Apply
 * mirrors legacy ghost_writeback_sinkswallow:1447-1478 with clamps on
 * Mass / Sink_Mass / Sink_Mass_Reservoir / MassTrue.
 * ========================================================================== */

namespace sink_swk_writeback_detail {

struct SinkSwkSnap {
    MyDouble Mass;
    MyDouble Vel[3];
    MyFloat  dp[3];
    MyFloat  Sink_Mass;
    MyFloat  Sink_Mdot;
#ifdef SINK_ALPHADISK_ACCRETION
    MyFloat  Sink_Mass_Reservoir;
#endif
#ifdef HYDRO_MESHLESS_FINITE_VOLUME
    MyDouble MassTrue;
#endif
    MyDouble VelPred[3];
    MyDouble InternalEnergy;
    MyDouble InternalEnergyPred;
#ifdef MAGNETIC
    MyDouble B[3];
    MyDouble BPred[3];
#endif
#ifdef GALSF_SUBGRID_WINDS
    MyFloat  DelayTime;
#endif
#if defined(COSMIC_RAY_FLUID) && defined(SINK_COSMIC_RAYS)
    MyFloat  CosmicRayEnergy    [N_CR_PARTICLE_BINS];
    MyFloat  CosmicRayEnergyPred[N_CR_PARTICLE_BINS];
    MyFloat  CosmicRayFlux      [N_CR_PARTICLE_BINS][3];
    MyFloat  CosmicRayFluxPred  [N_CR_PARTICLE_BINS][3];
#endif
};

struct SinkSwkDelta {
    int      home_index;
    MyDouble dMass;
    MyDouble dVel[3];
    MyFloat  ddp[3];
    MyFloat  dSink_Mass;
    MyFloat  dSink_Mdot;
#ifdef SINK_ALPHADISK_ACCRETION
    MyFloat  dSink_Mass_Reservoir;
#endif
#ifdef HYDRO_MESHLESS_FINITE_VOLUME
    MyDouble dMassTrue;
#endif
    MyDouble dVelPred[3];
    MyDouble dInternalEnergy;
    MyDouble dInternalEnergyPred;
#ifdef MAGNETIC
    MyDouble dB[3];
    MyDouble dBPred[3];
#endif
#ifdef GALSF_SUBGRID_WINDS
    MyFloat  dDelayTime;
#endif
#if defined(COSMIC_RAY_FLUID) && defined(SINK_COSMIC_RAYS)
    MyFloat  dCosmicRayEnergy    [N_CR_PARTICLE_BINS];
    MyFloat  dCosmicRayEnergyPred[N_CR_PARTICLE_BINS];
    MyFloat  dCosmicRayFlux      [N_CR_PARTICLE_BINS][3];
    MyFloat  dCosmicRayFluxPred  [N_CR_PARTICLE_BINS][3];
#endif
};

struct Ctx {
    SinkSwkSnap *snap;
    int          num_ghosts;
};

static Ctx s_ctx{nullptr, 0};

static void snapshot_fn(void *vctx, int num_ghosts, int num_local)
{
    Ctx *c = static_cast<Ctx*>(vctx);
    c->num_ghosts = num_ghosts;
    if(num_ghosts <= 0) { c->snap = nullptr; return; }
    c->snap = (SinkSwkSnap*) malloc(num_ghosts * sizeof(SinkSwkSnap));
    for(int g = 0; g < num_ghosts; g++) {
        int j = num_local + g;
        SinkSwkSnap& s = c->snap[g];
        s.Mass = P[j].Mass;
        for(int k = 0; k < 3; k++) { s.Vel[k] = P[j].Vel[k]; s.dp[k] = P[j].dp[k]; }
        s.Sink_Mass = P[j].Sink_Mass;
        s.Sink_Mdot = P[j].Sink_Mdot;
#ifdef SINK_ALPHADISK_ACCRETION
        s.Sink_Mass_Reservoir = P[j].Sink_Mass_Reservoir;
#endif
#ifdef HYDRO_MESHLESS_FINITE_VOLUME
        s.MassTrue = CellP[j].MassTrue;
#endif
        for(int k = 0; k < 3; k++) s.VelPred[k] = CellP[j].VelPred[k];
        s.InternalEnergy     = CellP[j].InternalEnergy;
        s.InternalEnergyPred = CellP[j].InternalEnergyPred;
#ifdef MAGNETIC
        for(int k = 0; k < 3; k++) { s.B[k] = CellP[j].B[k]; s.BPred[k] = CellP[j].BPred[k]; }
#endif
#ifdef GALSF_SUBGRID_WINDS
        s.DelayTime = CellP[j].DelayTime;
#endif
#if defined(COSMIC_RAY_FLUID) && defined(SINK_COSMIC_RAYS)
        for(int kc = 0; kc < N_CR_PARTICLE_BINS; kc++) {
            s.CosmicRayEnergy[kc]     = CellP[j].CosmicRayEnergy[kc];
            s.CosmicRayEnergyPred[kc] = CellP[j].CosmicRayEnergyPred[kc];
            for(int kk = 0; kk < 3; kk++) {
                s.CosmicRayFlux[kc][kk]     = CellP[j].CosmicRayFlux[kc][kk];
                s.CosmicRayFluxPred[kc][kk] = CellP[j].CosmicRayFluxPred[kc][kk];
            }
        }
#endif
    }
}

static int delta_for_ghost_fn(void *vctx, int g, int num_local)
{
    Ctx *c = static_cast<Ctx*>(vctx);
    if(!c->snap) return 0;
    int j = num_local + g;
    const SinkSwkSnap& s = c->snap[g];
    int modified = (P[j].Mass != s.Mass)
                || (P[j].Sink_Mass != s.Sink_Mass)
                || (P[j].Sink_Mdot != s.Sink_Mdot)
                || (P[j].Vel[0] != s.Vel[0] || P[j].Vel[1] != s.Vel[1] || P[j].Vel[2] != s.Vel[2]);
#if defined(SINK_RETURN_BFLUX) && defined(MAGNETIC)
    /* B/BPred-only changes (the SINK_RETURN_BFLUX block in the pair body
     * writes them BEFORE the SwallowID-match branch via atomic_add — Mode B
     * applies them locally, so Mode A's reverse-comm must catch them too).
     * Same Mode A/B asymmetry pattern as CR, below. */
    if(!modified) {
        for(int k = 0; k < 3 && !modified; k++) {
            if(CellP[j].B[k]     != s.B[k])     modified = 1;
            if(CellP[j].BPred[k] != s.BPred[k]) modified = 1;
        }
    }
#endif
#if defined(COSMIC_RAY_FLUID) && defined(SINK_COSMIC_RAYS)
    /* CR-aware: trip the predicate on CR-only changes too. Without this,
     * the CR-fix in the pair body would still drop CR deltas on Mode A
     * imported ghosts when no Vel/Mass change accompanied the CR injection. */
    if(!modified) {
        for(int kc = 0; kc < N_CR_PARTICLE_BINS && !modified; kc++) {
            if(CellP[j].CosmicRayEnergy[kc]     != s.CosmicRayEnergy[kc])     modified = 1;
            if(CellP[j].CosmicRayEnergyPred[kc] != s.CosmicRayEnergyPred[kc]) modified = 1;
            for(int kk = 0; kk < 3 && !modified; kk++) {
                if(CellP[j].CosmicRayFlux[kc][kk]     != s.CosmicRayFlux[kc][kk])     modified = 1;
                if(CellP[j].CosmicRayFluxPred[kc][kk] != s.CosmicRayFluxPred[kc][kk]) modified = 1;
            }
        }
    }
#endif
    return modified ? 1 : 0;
}

static void pack_fn(void *vctx, int g, int num_local, void *out_delta)
{
    Ctx *c = static_cast<Ctx*>(vctx);
    SinkSwkDelta *d = static_cast<SinkSwkDelta*>(out_delta);
    int *home_index = ghost_get_home_index();
    int j = num_local + g;
    const SinkSwkSnap& s = c->snap[g];
    d->home_index = home_index[g];
    d->dMass = P[j].Mass - s.Mass;
    for(int k = 0; k < 3; k++) {
        d->dVel[k] = P[j].Vel[k] - s.Vel[k];
        d->ddp[k]  = P[j].dp[k]  - s.dp[k];
    }
    d->dSink_Mass = P[j].Sink_Mass - s.Sink_Mass;
    d->dSink_Mdot = P[j].Sink_Mdot - s.Sink_Mdot;
#ifdef SINK_ALPHADISK_ACCRETION
    d->dSink_Mass_Reservoir = P[j].Sink_Mass_Reservoir - s.Sink_Mass_Reservoir;
#endif
#ifdef HYDRO_MESHLESS_FINITE_VOLUME
    d->dMassTrue = CellP[j].MassTrue - s.MassTrue;
#endif
    for(int k = 0; k < 3; k++) d->dVelPred[k] = CellP[j].VelPred[k] - s.VelPred[k];
    d->dInternalEnergy     = CellP[j].InternalEnergy     - s.InternalEnergy;
    d->dInternalEnergyPred = CellP[j].InternalEnergyPred - s.InternalEnergyPred;
#ifdef MAGNETIC
    for(int k = 0; k < 3; k++) {
        d->dB[k]     = CellP[j].B[k]     - s.B[k];
        d->dBPred[k] = CellP[j].BPred[k] - s.BPred[k];
    }
#endif
#ifdef GALSF_SUBGRID_WINDS
    d->dDelayTime = CellP[j].DelayTime - s.DelayTime;
#endif
#if defined(COSMIC_RAY_FLUID) && defined(SINK_COSMIC_RAYS)
    for(int kc = 0; kc < N_CR_PARTICLE_BINS; kc++) {
        d->dCosmicRayEnergy[kc]     = CellP[j].CosmicRayEnergy[kc]     - s.CosmicRayEnergy[kc];
        d->dCosmicRayEnergyPred[kc] = CellP[j].CosmicRayEnergyPred[kc] - s.CosmicRayEnergyPred[kc];
        for(int kk = 0; kk < 3; kk++) {
            d->dCosmicRayFlux[kc][kk]     = CellP[j].CosmicRayFlux[kc][kk]     - s.CosmicRayFlux[kc][kk];
            d->dCosmicRayFluxPred[kc][kk] = CellP[j].CosmicRayFluxPred[kc][kk] - s.CosmicRayFluxPred[kc][kk];
        }
    }
#endif
}

static void apply_fn(void *vctx, const void *in_delta)
{
    (void)vctx;
    const SinkSwkDelta *d = static_cast<const SinkSwkDelta*>(in_delta);
    int idx = d->home_index;
    P[idx].Mass += d->dMass;
    if(P[idx].Mass < 0 || P[idx].Mass != P[idx].Mass) P[idx].Mass = 0;
    for(int k = 0; k < 3; k++) {
        P[idx].Vel[k] += d->dVel[k];
        P[idx].dp[k]  += d->ddp[k];
    }
    P[idx].Sink_Mass += d->dSink_Mass;
    if(P[idx].Sink_Mass < 0 || P[idx].Sink_Mass != P[idx].Sink_Mass) P[idx].Sink_Mass = 0;
    P[idx].Sink_Mdot += d->dSink_Mdot;
#ifdef SINK_ALPHADISK_ACCRETION
    P[idx].Sink_Mass_Reservoir += d->dSink_Mass_Reservoir;
    if(P[idx].Sink_Mass_Reservoir < 0 || P[idx].Sink_Mass_Reservoir != P[idx].Sink_Mass_Reservoir)
        P[idx].Sink_Mass_Reservoir = 0;
#endif
#ifdef HYDRO_MESHLESS_FINITE_VOLUME
    CellP[idx].MassTrue += d->dMassTrue;
    if(CellP[idx].MassTrue < 0 || CellP[idx].MassTrue != CellP[idx].MassTrue) CellP[idx].MassTrue = 0;
#endif
    for(int k = 0; k < 3; k++) CellP[idx].VelPred[k] += d->dVelPred[k];
    CellP[idx].InternalEnergy     += d->dInternalEnergy;
    CellP[idx].InternalEnergyPred += d->dInternalEnergyPred;
#ifdef MAGNETIC
    for(int k = 0; k < 3; k++) {
        CellP[idx].B[k]     += d->dB[k];
        CellP[idx].BPred[k] += d->dBPred[k];
    }
#endif
#ifdef GALSF_SUBGRID_WINDS
    CellP[idx].DelayTime += d->dDelayTime;
#endif
#if defined(COSMIC_RAY_FLUID) && defined(SINK_COSMIC_RAYS)
    for(int kc = 0; kc < N_CR_PARTICLE_BINS; kc++) {
        CellP[idx].CosmicRayEnergy[kc]     += d->dCosmicRayEnergy[kc];
        CellP[idx].CosmicRayEnergyPred[kc] += d->dCosmicRayEnergyPred[kc];
        for(int kk = 0; kk < 3; kk++) {
            CellP[idx].CosmicRayFlux[kc][kk]     += d->dCosmicRayFlux[kc][kk];
            CellP[idx].CosmicRayFluxPred[kc][kk] += d->dCosmicRayFluxPred[kc][kk];
        }
    }
#endif
}

static void cleanup_fn(void *vctx)
{
    Ctx *c = static_cast<Ctx*>(vctx);
    if(c->snap) { free(c->snap); c->snap = nullptr; }
    c->num_ghosts = 0;
}

static const ghost_writeback_callback callback = {
    sizeof(SinkSwkDelta),
    snapshot_fn,
    delta_for_ghost_fn,
    pack_fn,
    apply_fn,
    cleanup_fn,
    & s_ctx,
};

static const ghost_writeback_callback *const raw_cbs[] = { & callback, nullptr };
static const ghost_writeback_bundle bundle = { raw_cbs, 1 };

}  /* namespace sink_swk_writeback_detail */

static inline const ghost_writeback_bundle *sink_swk_ghost_writeback_bundle_ptr(void)
{
    return & sink_swk_writeback_detail::bundle;
}

/* ============================================================================
 * IMPORTED-GHOST LIFECYCLE HOOKS
 * ========================================================================== */

/* ghost_write_detector_begin/end: runner default (loop_name = "sink_swk"). */

void SinkSwkSpec::ghost_writeback_begin(const neighbor_loop_args& /*args*/,
                                         const NeighborLoopPlan& /*plan*/)
{
    ghost_writeback_begin_bundle(sink_swk_ghost_writeback_bundle_ptr());
}

void SinkSwkSpec::ghost_writeback_end(const neighbor_loop_args& /*args*/,
                                       const NeighborLoopPlan& /*plan*/)
{
    ghost_writeback_end_bundle(sink_swk_ghost_writeback_bundle_ptr());
}

/* ============================================================================
 * DIAGNOSTICS — env-gated.
 * ========================================================================== */

#endif /* SINK_PARTICLES */
