/* transport_subcycle.cc - implementation of transport (RT + CR) subcycling
 *
 * Flux exchange uses the standard code_block_xchange pattern with a transport-only
 * core function. This correctly handles cross-domain interactions via MPI.
 * The tree walk is repeated each sub-step (future optimization: cache neighbor lists).
 *
 * Written by Mike Grudic for GIZMO.
 */

#include <mpi.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../mesh/kernel.h"
#include "../mesh/neighbor_list.h"
#include "../hydro/hydro_structs.h"
#include "../hydro/compute_finitevol_faces_functions.h"
extern void hydro_evaluate_gpu(struct particle_data *, struct gas_cell_data *,
                               int, int *, int, int64_t *, int *, int64_t, void *);

#ifdef TRANSPORT_SUBCYCLE

#include "transport_subcycle.h"
#include "../mesh/ghost_symlist_lifecycle.h"   /* gizmo_gradients_prep_symlist (topology build) */
#include "../radiation/rt_direct_ray_transport_functions.h"
#include "../radiation/rt_diffusion_explicit_functions.h"

/* The sub-steps run on the topology the hydro pass already built -- its active-gas list, its
 * gas ghost pool and its shared symmetric list -- whenever nothing between the end of
 * hydro_force and the sub-step loop could have changed it.
 *
 * What runs in that window, and what each may change:
 *   second half-kick, DM heating ............ values only (positions and h do not move)
 *   sink accretion .......................... values; a swallow zeroes gas mass in place
 *                                             (ghost_exchange_gas_eligibility_changed)
 *   wind spawning, rearrange_particle_sequence  particle membership/order (supply identity
 *                                             epoch, and NumPart)
 *   FoF ..................................... reorders particles: never kept on a FoF step
 *   RT source injection, RT chemistry ....... values; may import and clean up a pool of
 *                                             its own, which the parked one is immune to
 *   special (reflecting/outflow) boundaries and excision move or remove particles in the
 *   window, so those builds never keep.
 * A new routine placed in that window must be added here and signalled, or the kept topology
 * is unsafe.
 *
 * The list lives on the plain heap (it must outlive the working pool's last-in-first-out
 * scope), admitted only when every node has the headroom; otherwise everything behaves as
 * before: freed after hydro_force, rebuilt here. */
static struct {
    int kept, parked, num_local, num_gas_local;
    long long identity_epoch, eligibility_epoch;
} g_kept_topology = {0, 0, 0, 0, 0, 0};

int transport_subcycle_keep_wanted(void)
{
#if defined(BOX_DEFINED_SPECIAL_XYZ_BOUNDARY_CONDITIONS_ARE_ACTIVE) || defined(PARTICLE_EXCISION)
    return 0;
#endif
#if (defined(SINK_PARTICLES) || defined(GALSF_SUBGRID_WINDS)) && defined(FOF)
    if(All.Time >= All.TimeNextOnTheFlyFoF) {return 0;}
#endif
    return 1;
}

/* End of hydro_force: take over the shared list and the pool instead of freeing them.
 * Collective (every rank reaches the hydro epilogue). Returns 1 if kept. */
int transport_subcycle_keep_topology(void)
{
    if(!gizmo_sym_neighbor_list.on_heap) {return 0;}   /* admission is collective, so this is too */
    /* Ghosts are expected wherever the corridor imported (NTask > 1 with active gas anywhere);
       then every rank must hold a live pool to park, or none keeps. The active list was built in
       the working pool; it moves to the heap with the list. One collective decides both. */
    const int need_pool = (NTask > 1 && gizmo_sym_any_active_global) ? 1 : 0;
    const size_t active_bytes = (size_t)(gizmo_sym_num_active > 0 ? gizmo_sym_num_active : 1) * sizeof(int);
    int *active_heap = (int *) malloc(active_bytes);
    int got = (active_heap && (!need_pool || ghost_pool_is_live())) ? 1 : 0, all_got = 0;
    MPI_Allreduce(&got, &all_got, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if(!all_got) {
        free(active_heap);
        gizmo_kept_list_note_refused((long long)((size_t)(gizmo_sym_neighbor_list.num_active + 1) * sizeof(int64_t)
            + (size_t)(gizmo_sym_neighbor_list.total_pairs > 0 ? gizmo_sym_neighbor_list.total_pairs : 1) * sizeof(int)) + (long long)active_bytes);
        return 0;
    }
    if(gizmo_sym_num_active > 0) {memcpy(active_heap, gizmo_sym_active_indices, (size_t)gizmo_sym_num_active * sizeof(int));}
    if(gizmo_sym_active_indices_on_heap) {free(gizmo_sym_active_indices);} else {myfree(gizmo_sym_active_indices);}
    gizmo_sym_active_indices = active_heap; gizmo_sym_active_indices_on_heap = 1;
    gizmo_kept_list_grow((long long)active_bytes);
    if(need_pool) {ghost_pool_park();}   /* every rank has a live pool here, so every rank parks */
    g_kept_topology.parked = need_pool;
    g_kept_topology.num_local = NumPart; g_kept_topology.num_gas_local = N_gas;
    ghost_exchange_topology_stamp(&g_kept_topology.identity_epoch, &g_kept_topology.eligibility_epoch);
    g_kept_topology.kept = 1;
    return 1;
}

/* Collective. 1 = the kept topology is live again with current ghost values; 0 = it was
 * dropped (or there was none) and the caller builds a fresh one. */
static int transport_subcycle_reuse_kept_topology(void)
{
    if(!g_kept_topology.kept) {return 0;}   /* same on every rank: keeping was collective */
    g_kept_topology.kept = 0;
    long long identity = 0, eligibility = 0;
    ghost_exchange_topology_stamp(&identity, &eligibility);
    int ok = (identity == g_kept_topology.identity_epoch) && (eligibility == g_kept_topology.eligibility_epoch)
          && (NumPart == g_kept_topology.num_local) && (N_gas == g_kept_topology.num_gas_local) && !ghost_pool_is_live();
    if(ok) {   /* the same active rows, in the same order */
        int n = 0;
        for(int ii : ActiveParticleList) {
            if(P[ii].Type != 0 || P[ii].Mass <= 0) {continue;}
            if(n >= gizmo_sym_num_active || gizmo_sym_active_indices[n] != ii) {ok = 0; break;}
            n++;
        }
        if(n != gizmo_sym_num_active) {ok = 0;}
    }
    int restored = 0;
    if(ok && g_kept_topology.parked) {restored = ghost_pool_restore(); ok = restored;}
    int all_ok = 0;
    MPI_Allreduce(&ok, &all_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if(all_ok && g_kept_topology.parked && ghost_refresh_values() != GHOST_REFRESH_OK) {all_ok = 0;}   /* collective outcome */
    if(!all_ok) {
        if(restored) {ghost_exchange_cleanup();} else {ghost_pool_discard_parked();}
        gizmo_sym_neighbor_list_free();
        return 0;
    }
    return 1;
}

/* The topology the sub-step loop consumes (active-index list + gas ghost pool + shared
 * symmetric gas list, gizmo_sym_*): the hydro pass's, when it was kept and is still valid,
 * else built here, after every particle-set mutation of the window above. Freed after the
 * loop (run.cc: gizmo_sym_neighbor_list_free + ghost_exchange_cleanup). */
void transport_subcycle_prepare_topology(void)
{
    if(transport_subcycle_reuse_kept_topology()) {return;}
    const double gsl_safety = gizmo_ghost_safety_factor();
    gizmo_gradients_prep_symlist(gsl_safety, gsl_safety, false, 0, NULL);   /* the sub-steps read only the host list */
}

/* ======================================================================================== */
/* GPU dispatcher (transport_subcycle_exchange_fluxes below) consumes hydro_data_out
   directly. The legacy CPU-tree scaffolding (particle2in_transport,
   out2particle_transport, transport_flux_evaluate, and the old code_block_xchange
   headers) is retired. */


/* ===== INITIAL OPERATIONS (zero rate arrays) ===== */
void transport_flux_initial_operations(void)
{
    int k;
    for(int i : ActiveParticleList) {
        if(P[i].Type != 0 || P[i].Mass <= 0) continue;
#if defined(RT_SOLVER_EXPLICIT) && defined(RT_EVOLVE_ENERGY)
        for(k=0; k<N_RT_FREQ_BINS; k++) CellP[i].Dt_Rad_E_gamma[k] = 0;
#endif
#if defined(RT_SOLVER_EXPLICIT) && defined(RT_EVOLVE_FLUX)
        for(k=0; k<N_RT_FREQ_BINS; k++) CellP[i].Dt_Rad_Flux[k] = {};
#endif
#if defined(RT_SOLVER_EXPLICIT) && defined(RT_INFRARED)
        CellP[i].Dt_Rad_E_gamma_T_weighted_IR = 0;
#endif
#if defined(RT_SOLVER_EXPLICIT) && defined(RT_EVOLVE_INTENSITIES)
        for(k=0; k<N_RT_FREQ_BINS; k++) {for(int k_dir=0; k_dir<N_RT_INTENSITY_BINS; k_dir++) {CellP[i].Dt_Rad_Intensity[k][k_dir] = 0;}}
#endif
#ifdef COSMIC_RAY_FLUID
        for(k=0; k<N_CR_PARTICLE_BINS; k++) {
            CellP[i].DtCosmicRayEnergy[k] = 0;
#if defined(CRFLUID_EVOLVE_SPECTRUM)
            CellP[i].DtCosmicRay_Number_in_Bin[k] = 0;
#endif
#ifdef CRFLUID_EVOLVE_SCATTERINGWAVES
            for(int kAlf=0; kAlf<2; kAlf++) {CellP[i].DtCosmicRayAlfvenEnergy[k][kAlf] = 0;}
#endif
        }
#endif
    }
}


/* ===== PUBLIC ENTRY POINT ===== */
void transport_subcycle_exchange_fluxes(int transport_sub)
{
    CPU_Step[CPU_MISC] += measure_time();
    /* The previous sub-step's kick moved the owners' transported fields; the
       topology is unchanged (nothing drifts inside the loop), so bring the ghost
       copies current in place. If the pool cannot be refreshed, rebuild it and
       the shared CSR collectively rather than flux against stale values. */
    if(transport_sub > 0 && NTask > 1 && ghost_refresh_values() != GHOST_REFRESH_OK) {
        const double gsl_safety = gizmo_ghost_safety_factor();
        gizmo_gradients_refresh_symlist(gsl_safety, gsl_safety, false, 0, NULL);
    }
    transport_flux_initial_operations();

    /* GPU/neighbor-list path: reuse the hydro GPU kernel with the symmetric CSR list.
       The hydro kernel computes face geometry + RT fluxes (via rt_diffusion_explicit.h);
       we scatter only the RT output fields and discard the hydro force outputs.
       This eliminates the tree-walk + MPI exchange per subcycle step. */
    {
        struct hydro_data_out *hydro_out = (struct hydro_data_out *) mymalloc("transport_hydro_out",
            (gizmo_sym_num_active > 0 ? gizmo_sym_num_active : 1) * sizeof(struct hydro_data_out));

        hydro_evaluate_gpu(P, CellP, NumPart,
                           gizmo_sym_active_indices, gizmo_sym_num_active,
                           gizmo_sym_neighbor_list.offsets,
                           gizmo_sym_neighbor_list.neighbors,
                           gizmo_sym_neighbor_list.total_pairs,
                           (void *)hydro_out);

        /* Scatter only the transport (RT and cosmic-ray) rate outputs; discard the hydro forces */
        for(int aa = 0; aa < gizmo_sym_num_active; aa++) {
            int ii = gizmo_sym_active_indices[aa];
#if defined(RT_SOLVER_EXPLICIT) && defined(RT_EVOLVE_ENERGY)
            for(int kf=0; kf<N_RT_FREQ_BINS; kf++)
                CellP[ii].Dt_Rad_E_gamma[kf] += hydro_out[aa].Dt_Rad_E_gamma[kf];
#endif
#if defined(RT_SOLVER_EXPLICIT) && defined(RT_EVOLVE_FLUX)
            for(int kf=0; kf<N_RT_FREQ_BINS; kf++)
                CellP[ii].Dt_Rad_Flux[kf] += hydro_out[aa].Dt_Rad_Flux[kf];
#endif
#if defined(RT_SOLVER_EXPLICIT) && defined(RT_INFRARED)
            CellP[ii].Dt_Rad_E_gamma_T_weighted_IR += hydro_out[aa].Dt_Rad_E_gamma_T_weighted_IR;
#endif
#if defined(RT_SOLVER_EXPLICIT) && defined(RT_EVOLVE_INTENSITIES)
            for(int kf=0; kf<N_RT_FREQ_BINS; kf++) {for(int k_dir=0; k_dir<N_RT_INTENSITY_BINS; k_dir++) {CellP[ii].Dt_Rad_Intensity[kf][k_dir] += hydro_out[aa].Dt_Rad_Intensity[kf][k_dir];}}
#endif
#ifdef COSMIC_RAY_FLUID
            for(int kb=0; kb<N_CR_PARTICLE_BINS; kb++) {
                CellP[ii].DtCosmicRayEnergy[kb] += hydro_out[aa].DtCosmicRayEnergy[kb];
#if defined(CRFLUID_EVOLVE_SPECTRUM)
                CellP[ii].DtCosmicRay_Number_in_Bin[kb] += hydro_out[aa].DtCosmicRay_Number_in_Bin[kb];
#endif
#ifdef CRFLUID_EVOLVE_SCATTERINGWAVES
                for(int kAlf=0; kAlf<2; kAlf++) {CellP[ii].DtCosmicRayAlfvenEnergy[kb][kAlf] += hydro_out[aa].DtCosmicRayAlfvenEnergy[kb][kAlf];}
#endif
#if !defined(CRFLUID_EVOLVE_SCATTERINGWAVES) && !defined(CRFLUID_EVOLVE_SPECTRUM)
                CellP[ii].DtCosmicRayEnergy[kb] += CellP[ii].DtCosmicRayEnergy_StreamingLoss[kb]; /* hydro-pass streaming loss, held over the step like its gas-heating counterpart */
#endif
            }
#endif
        }

        myfree(hydro_out);
    }

    /* add back the radiation pressure work terms saved during the hydro pass.
       The flux exchange zeroed Dt_Rad_E_gamma and recomputed only transport fluxes;
       the work terms must be re-added so each sub-step's kick includes them. */
#if defined(RT_RAD_PRESSURE_FORCES) && defined(RT_EVOLVE_ENERGY)
    for(int i : ActiveParticleList) {
        if(P[i].Type != 0 || P[i].Mass <= 0) continue;
        for(int kf=0; kf<N_RT_FREQ_BINS; kf++)
            CellP[i].Dt_Rad_E_gamma[kf] += CellP[i].Dt_Rad_E_gamma_Work[kf];
    }
#endif

    CPU_Step[CPU_RTNONFLUXOPS] += measure_time();
}


/* ======================================================================================== */
/* ===== KICK ============================================================================= */
/* ======================================================================================== */

void transport_subcycle_kick(void)
{
    double sub_frac = All.Transport_Subcycle_dt_fraction;
    /* Per-i pure-local kernel: body writes only to P[i]/CellP[i]; the
     * mutating drift-kick helpers (rt_update_driftkick,
     * CosmicRay_Update_DriftKick) already have OMP exposure via
     * do_kick_for_extra_physics (core/kicks.cc), and the auxiliary
     * per-cell calls here (rt_eddington_update_calculation, rt_kappa)
     * read/write only per-i state. Guard by n_act >= 16 to avoid OMP
     * overhead on tiny active sets (matches sink_properties_loop, F2). */
    const int n_act = (int)ActiveParticleList.size();
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) if(n_act >= 16)
#endif
    for(int _a = 0; _a < n_act; _a++) {
        int idx = ActiveParticleList[_a];
        if(P[idx].Type != 0 || P[idx].Mass <= 0) continue;
        int i = idx;
        double sub_dt = get_particle_timestep_in_physical(i, P) * sub_frac;
        if(sub_dt <= 0) continue;

#ifdef RADTRANSFER
        /* rt_update_driftkick adds IR gas heating to DtInternalEnergy as a rate.
           Since we call it N times, the rate would accumulate N-fold. Save/restore
           DtInternalEnergy each sub-step to prevent accumulation, but keep the final
           sub-step's contribution so the cooling function sees the correct IR heating rate. */
        rt_update_driftkick(i, sub_dt, 0, P, CellP); /* kick: advance conserved variables */
        rt_eddington_update_calculation(i, P, CellP);
        /* update opacities — Rad_Kappa (especially IR band) depends on Dust_Temperature and
           Radiation_Temperature which were just modified by rt_update_driftkick. Stale opacities
           cause the transport flux computation in subsequent sub-steps to use the wrong diffusion rate. */
        for(int kf = 0; kf < N_RT_FREQ_BINS; kf++) {CellP[i].Rad_Kappa[kf] = rt_kappa(i, kf, P, CellP);}
        /* sync Pred = conserved for the next sub-step's flux exchange */
        for(int kf = 0; kf < N_RT_FREQ_BINS; kf++) {
            CellP[i].Rad_E_gamma_Pred[kf] = CellP[i].Rad_E_gamma[kf];
#ifdef RT_EVOLVE_FLUX
            CellP[i].Rad_Flux_Pred[kf] = CellP[i].Rad_Flux[kf];
#endif
#ifdef RT_EVOLVE_INTENSITIES
            for(int ka = 0; ka < N_RT_INTENSITY_BINS; ka++)
                CellP[i].Rad_Intensity_Pred[kf][ka] = CellP[i].Rad_Intensity[kf][ka];
#endif
        }
#endif

#ifdef COSMIC_RAY_FLUID
        CosmicRay_Update_DriftKick(i, sub_dt, 0, P, CellP);
        for(int kb = 0; kb < N_CR_PARTICLE_BINS; kb++) {
            CellP[i].CosmicRayEnergyPred[kb] = CellP[i].CosmicRayEnergy[kb];
            CellP[i].CosmicRayFluxPred[kb] = CellP[i].CosmicRayFlux[kb];
#ifdef CRFLUID_EVOLVE_SCATTERINGWAVES
            for(int kAlf = 0; kAlf < 2; kAlf++) {CellP[i].CosmicRayAlfvenEnergyPred[kb][kAlf] = CellP[i].CosmicRayAlfvenEnergy[kb][kAlf];}
#endif
        }
#endif
    }
}


#endif // TRANSPORT_SUBCYCLE
