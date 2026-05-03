/* sink_swallow_and_kick_gpu.cc — GPU neighbor-loop port for
 * sink_swallow_and_kick_evaluate (D1).
 *
 * Ports sinks/sink_swallow_and_kick.cc:174-513 to a Kokkos parallel_for kernel.
 * i-particles are active sinks (sink_isactive && SwallowID==0); j-particles are
 * all types (SINK_NEIGHBOR_BITFLAG).  Atomic j-writes to Mass / Vel / VelPred /
 * dp / B / BPred / MassTrue / InternalEnergy / InternalEnergyPred (gas and
 * sink-merger paths) are reconciled across ranks via ghost_writeback_sinkswallow.
 * Per-source output struct is scattered into SinkTempInfo on the host; per-source
 * swallow counters are MPI_Reduced for the summary log.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>
#include <Kokkos_Core.hpp>

#include "../declarations/gpu_all_mirror.h"
#include "../declarations/allvars.h"
#include "../system/gpu_particles_arena.h"
#include "../core/proto.h"
#include "../mesh/kernel.h"
#include "../mesh/gpu_neighbor_list.h"
#include "../mesh/neighbor_list.h"
#include "../mesh/ghost_writeback.h"
#include "../mesh/ghost_symlist_lifecycle.h"

#include "../declarations/gpu_numeric_macros.h"
#include "../declarations/gpu_error_check.h"
#include "../declarations/gpu_dispatch_templates.h"

#include "sinks_gpu_decls.h"

#if defined(SINK_PARTICLES)

#include "sink_functions.h"
#include "sink_swallow_and_kick_functions.h"

/* Fill SinkSwallowLocalIn for source particle i — mirrors INPUTFUNCTION_NAME
 * in sink_swallow_and_kick.cc:60-93.  Host-only; runs once per active sink
 * before the kernel launch. */
static void sink_swallow_local_fill(int i, struct SinkSwallowLocalIn *loc)
{
    int j_tempinfo = P[i].IndexMapToTempStruc;
    loc->Pos          = P[i].Pos;
    loc->Vel          = P[i].Vel;
    loc->KernelRadius = P[i].KernelRadius;
    loc->Mass         = P[i].Mass;
    loc->Sink_Mass    = P[i].Sink_Mass;
    loc->ID           = P[i].ID;
    loc->ID_child_number = P[i].ID_child_number;
    loc->ID_generation   = P[i].ID_generation;
    loc->Mdot         = P[i].Sink_Mdot;
#if defined(SINK_CALC_LOCAL_ANGLEWEIGHTS) || defined(SINK_WIND_KICK)
#if defined(SINK_FOLLOW_ACCRETED_ANGMOM)
    loc->Jgas_in_Kernel = P[i].Sink_Specific_AngMom;
#else
    loc->Jgas_in_Kernel = SinkTempInfo[j_tempinfo].Jgas_in_Kernel;
#endif
#endif
#ifdef SINK_ALPHADISK_ACCRETION
    loc->Sink_Mass_Reservoir = P[i].Sink_Mass_Reservoir;
#endif
#if defined(SINK_CALC_LOCAL_ANGLEWEIGHTS)
    loc->Sink_angle_weighted_kernel_sum = SinkTempInfo[j_tempinfo].Sink_angle_weighted_kernel_sum;
#endif
    loc->Dt = (MyFloat)get_particle_feedback_timestep_in_physical(i);
#ifdef SINK_INTERACT_ON_GAS_TIMESTEP
    loc->Dt = P[i].dt_since_last_gas_search;
#endif
#if defined(SINK_RETURN_ANGMOM_TO_GAS)
    loc->Sink_Specific_AngMom = P[i].Sink_Specific_AngMom;
    loc->angmom_norm_topass_in_swallowloop = SinkTempInfo[j_tempinfo].angmom_norm_topass_in_swallowloop;
#endif
#if defined(SINK_RETURN_BFLUX)
    loc->B = P[i].B;
    loc->kernel_norm_topass_in_swallowloop = SinkTempInfo[j_tempinfo].kernel_norm_topass_in_swallowloop;
#endif
#ifdef SINGLE_STAR_FB_LOCAL_RP
    loc->Luminosity = (MyFloat)sink_lum_bol(loc->Mdot, loc->Sink_Mass, i);
#endif
}


/* Host scatter of SinkSwallowOut into SinkTempInfo — mirrors OUTPUTFUNCTION_NAME
 * at sink_swallow_and_kick.cc:135-168 (mode=0, sum accumulation). */
static void sink_swallow_apply_out(const struct SinkSwallowOut *out, int i)
{
    int target = P[i].IndexMapToTempStruc;
    SinkTempInfo[target].accreted_Mass               += out->accreted_Mass;
    SinkTempInfo[target].accreted_Sink_Mass          += out->accreted_Sink_Mass;
    SinkTempInfo[target].accreted_Sink_Mass_reservoir += out->accreted_Sink_Mass_reservoir;
#if defined(SINK_SWALLOWGAS) && !defined(SINK_GRAVCAPTURE_GAS)
    SinkTempInfo[target].Sink_AccretionDeficit       += out->Sink_AccretionDeficit;
#endif
#ifdef GRAIN_FLUID
    SinkTempInfo[target].accreted_dust_Mass          += out->accreted_dust_Mass;
#endif
#ifdef RT_REINJECT_ACCRETED_PHOTONS
    SinkTempInfo[target].accreted_photon_energy      += out->accreted_photon_energy;
#endif
#if defined(SINK_FOLLOW_ACCRETED_MOMENTUM)
    for(int k = 0; k < 3; k++) SinkTempInfo[target].accreted_momentum[k] += out->accreted_momentum[k];
#endif
#if defined(SINK_FOLLOW_ACCRETED_COM)
    for(int k = 0; k < 3; k++) SinkTempInfo[target].accreted_centerofmass[k] += out->accreted_centerofmass[k];
#endif
#if defined(SINK_RETURN_BFLUX)
    for(int k = 0; k < 3; k++) SinkTempInfo[target].accreted_B[k] += out->accreted_B[k];
#endif
#if defined(SINK_FOLLOW_ACCRETED_ANGMOM)
    for(int k = 0; k < 3; k++) SinkTempInfo[target].accreted_J[k] += out->accreted_J[k];
#endif
#ifdef SINK_COUNTPROGS
    P[i].Sink_CountProgs += out->Sink_CountProgs;
#endif
#ifdef GALSF
    if(P[i].StellarAge > out->Accreted_Age) P[i].StellarAge = out->Accreted_Age;
#endif
    /* Apply per-bin TimeBin_Sink_* deltas accumulated during sink-sink mergers. */
    for(int b = 0; b < TIMEBINS; b++) {
        TimeBin_Sink_mass[b]          += out->delta_TimeBin_Sink_mass[b];
        TimeBin_Sink_dynamicalmass[b] += out->delta_TimeBin_Sink_dynamicalmass[b];
        TimeBin_Sink_Mdot[b]          += out->delta_TimeBin_Sink_Mdot[b];
        TimeBin_Sink_Medd[b]          += out->delta_TimeBin_Sink_Medd[b];
    }
}


void sink_swallow_and_kick_evaluate_gpu(struct particle_data *P_host,
                                         struct gas_cell_data *CellP_host,
                                         int num_total,
                                         int *i_active_host, int num_active,
                                         const double *i_radii_host,
                                         int j_type_bitmask)
{
    GIZMO_GPU_ENSURE_ALL_FRESH(sinkswallow);

    int imported_ghosts = 0;
    if(ghost_get_num_ghosts() <= 0) {
        gizmo_density_prep_ghosts(gizmo_ghost_safety_factor());
        imported_ghosts = 1;
    }

    int num_all = ghost_get_num_local() + ghost_get_num_ghosts();
    if(num_all <= 0) num_all = num_total;

    /* Wrapper fast-path: no active sinks → skip arena_acquire, big SharedSpace
     * allocs, kernel, scatter.  Preserve ghost_writeback collectives for
     * multi-rank consistency (each self-guards on NTask<=1 || num_ghosts<=0). */
    if(num_active == 0) {
        ghost_write_detector_begin("sink_swallow_and_kick");
        ghost_writeback_zero_sinkswallow();
        ghost_writeback_sinkswallow();
        ghost_write_detector_end();
        if(imported_ghosts) ghost_exchange_cleanup();
        return;
    }

    /* Step 13 Phase 1 arena. */
    gpu_particles_arena_acquire(num_all, P_host, CellP_host);
    struct particle_data *P_gpu = gpu_particles_arena_P();
    struct gas_cell_data *CellP_gpu = gpu_particles_arena_CellP();

    int alloc_n = (num_active > 0) ? num_active : 1;
    struct SinkSwallowLocalIn *d_local = (struct SinkSwallowLocalIn *)
        Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(alloc_n * sizeof(struct SinkSwallowLocalIn));
    struct SinkSwallowOut *d_out = (struct SinkSwallowOut *)
        Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(alloc_n * sizeof(struct SinkSwallowOut));
    int *d_active = (int *)
        Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(alloc_n * sizeof(int));
    double *d_radii = (double *)
        Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(alloc_n * sizeof(double));
    if(num_active > 0) {
        for(int a = 0; a < num_active; a++) sink_swallow_local_fill(i_active_host[a], &d_local[a]);
        memcpy(d_active, i_active_host, num_active * sizeof(int));
        memcpy(d_radii,  i_radii_host,  num_active * sizeof(double));
    }
    memset(d_out, 0, alloc_n * sizeof(struct SinkSwallowOut));
#ifdef GALSF
    for(int a = 0; a < num_active; a++) d_out[a].Accreted_Age = MAX_REAL_NUMBER;
#endif

    gpu_neighbor_list_t gnl;
    gpu_ngb_list_build(P_gpu, num_all,
                       i_active_host, num_active,
                       NGB_SEARCH_ONEWAY, j_type_bitmask,
                       &gnl, NULL, 1.0, i_radii_host, NULL, "sink_swk");

    PRINT_STATUS("  GPU sink_swallow: %d active, %d pairs", num_active, gnl.total_pairs);

    /* Snapshot ghost-side fields for the sinkswallow writeback. */
    ghost_write_detector_begin("sink_swallow_and_kick");
    ghost_writeback_zero_sinkswallow();

    {
        int  *offsets   = gnl.offsets;
        int  *neighbors = gnl.neighbors;
        struct SinkSwallowLocalIn *local_arr = d_local;
        struct SinkSwallowOut     *out_arr   = d_out;
        double *radii_arr = d_radii;
        struct particle_data *kp = P_gpu;
        struct gas_cell_data *kc = CellP_gpu;

        gizmo_gpu_kernel_launch("sink_swallow_kernel", num_active, KOKKOS_LAMBDA(int aa) {
            const struct SinkSwallowLocalIn& loc = local_arr[aa];
            double h_i = radii_arr[aa];
            if(h_i <= 0 || loc.Mass <= 0) return;

            /* Per-source precomputation */
            double mom_budget = 0;
#if defined(SINK_CALC_LOCAL_ANGLEWEIGHTS)
#if defined(SINGLE_STAR_FB_LOCAL_RP)
            mom_budget = (double)loc.Luminosity * (double)loc.Dt / C_LIGHT_CODE;
#else
            mom_budget = sink_lum_bol((double)loc.Mdot, (double)loc.Sink_Mass, -1) * (double)loc.Dt / C_LIGHT_CODE;
#endif
#endif
            Vec3<double> J_dir{0, 0, 1};
#if defined(SINK_CALC_LOCAL_ANGLEWEIGHTS) || defined(SINK_WIND_KICK)
            {
                Vec3<double> tmp{(double)loc.Jgas_in_Kernel[0], (double)loc.Jgas_in_Kernel[1], (double)loc.Jgas_in_Kernel[2]};
                double n2 = tmp.norm_sq();
                if(n2 > 0) { double n = sqrt(n2); J_dir = tmp / n; }
            }
#endif
            double sink_mass_withdisk = (double)loc.Sink_Mass;
#if defined(SINK_WIND_KICK) && defined(SINK_ALPHADISK_ACCRETION)
            sink_mass_withdisk += (double)loc.Sink_Mass_Reservoir;
#endif

            int start = offsets[aa], end = offsets[aa + 1];
            for(int nn = start; nn < end; nn++) {
                int j = neighbors[nn];
                if(kp[j].Mass <= 0) continue;
                Vec3<double> dpos = kp[j].Pos - loc.Pos;
                nearest_xyz(dpos, -1);
                double r2 = dpos.norm_sq();
                if(r2 >= h_i * h_i) continue;
                sink_swallow_pair_kernel(loc, j, kp, kc, out_arr[aa],
                                         dpos, r2, h_i,
                                         mom_budget, J_dir,
                                         sink_mass_withdisk);
            }
        });
    }

    gizmo_gpu_check_last_error("sink_swallow", num_active);

    /* Scatter P/CellP back to host. */
    memcpy(P_host,     P_gpu,     num_all * sizeof(struct particle_data));
    memcpy(CellP_host, CellP_gpu, num_all * sizeof(struct gas_cell_data));

    /* Reduce ghost-side j-side deltas to home. */
    ghost_writeback_sinkswallow();
    ghost_write_detector_end();

    /* Host scatter of per-source outputs into SinkTempInfo + P[i] (mass loss,
     * CountProgs, Accreted_Age).  Also accumulate per-rank swallow counters. */
    int N_gas_sw = 0, N_sink_sw = 0, N_star_sw = 0, N_dm_sw = 0;
    for(int a = 0; a < num_active; a++) {
        int i = i_active_host[a];
        sink_swallow_apply_out(&d_out[a], i);
        N_gas_sw  += d_out[a].n_gas_swallowed;
        N_sink_sw += d_out[a].n_sink_swallowed;
        N_star_sw += d_out[a].n_star_swallowed;
        N_dm_sw   += d_out[a].n_dm_swallowed;
    }
    /* MPI_Reduce for summary log, matching sink_swallow_and_kick.cc:525-528. */
    int Ntot_gas = 0, Ntot_sink = 0, Ntot_star = 0, Ntot_dm = 0;
    MPI_Reduce(&N_gas_sw,  &Ntot_gas,  1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&N_sink_sw, &Ntot_sink, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&N_star_sw, &Ntot_star, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&N_dm_sw,   &Ntot_dm,   1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if(ThisTask == 0 && (Ntot_gas || Ntot_sink || Ntot_star || Ntot_dm)) {
        printf("Accretion done: swallowed %d gas, %d star, %d dm, and %d sink particles\n",
               Ntot_gas, Ntot_star, Ntot_dm, Ntot_sink);
        fflush(stdout);
    }

    gpu_ngb_list_free(&gnl, NULL);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_radii);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_active);
    /* Full P+CellP scatter syncs arena→host; ghost_writeback_sinkswallow above
     * already invalidates for any host changes it applied. */
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_out);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_local);

    if(imported_ghosts) ghost_exchange_cleanup();
}

GPU_ALL_SYNC_FUNC(sinkswallow)

#else

void sink_swallow_and_kick_evaluate_gpu(struct particle_data *, struct gas_cell_data *,
                                         int, int *, int, const double *, int) {}
GPU_ALL_SYNC_FUNC_STUB(sinkswallow)

#endif /* SINK_PARTICLES */
