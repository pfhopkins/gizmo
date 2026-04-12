/* density_gpu.cc — GPU-accelerated density evaluation via Kokkos.
 *
 * This file is a GPU translation unit (TU): compiled by nvcc/hipcc when
 * OPENMP_GPU_OFFLOAD is enabled. It provides:
 *   1. GPU neighbor list construction (BVH traversal via Kokkos::parallel_for)
 *   2. GPU density kernel (CSR neighbor iteration via Kokkos::parallel_for)
 *
 * Entry point: density_evaluate_gpu() — called from density.cc during
 * the h-iteration loop, replacing the CPU CSR path for each iteration.
 *
 * Architecture follows the cooling.cc pattern:
 *   - __managed__ All_dev with #define All All_dev for GPU global access
 *   - Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE> for device-accessible data
 *   - CPU gather → GPU kernel → CPU scatter
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

/* Standard and Kokkos headers BEFORE global_data_all_struct.h
 * (macros.h #define terminate(x) conflicts with std::terminate in <exception>) */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifdef OPENMP_GPU_OFFLOAD
#include <Kokkos_Core.hpp>
#endif

/* GPU All mirror: same pattern as cooling.cc */
#ifdef OPENMP_GPU_OFFLOAD
#include "../declarations/global_data_all_struct.h"
#endif
#if defined(OPENMP_GPU_OFFLOAD) && defined(GIZMO_GPU_COMPILER)
static __managed__ struct global_data_all_processes All_dev;
#define All All_dev
#endif

#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../core/timestep_functions.h"
#include "../mesh/kernel.h"
#include "../mesh/neighbor_list.h"
#include "../mesh/sfc_tiles.h"
#include "../mesh/sfc_tiles_functions.h"
#include "density_functions.h"
#include "gradient_functions.h"
#include "hydro_structs.h"
#ifndef HYDRO_SPH
#include "reimann.h"
#endif
/* Define xchange macro names so hydro_functions.h can reference the structs */
#define INPUT_STRUCT_NAME hydro_data_in
#define OUTPUT_STRUCT_NAME hydro_data_out
#include "hydro_functions.h"
#undef INPUT_STRUCT_NAME
#undef OUTPUT_STRUCT_NAME

#if defined(OPENMP_GPU_OFFLOAD) && defined(GIZMO_USE_NEIGHBOR_LIST_FOR_DENSITY)

/* Axis periodicity flags (matching sfc_tiles.cc) */
#if defined(BOX_PERIODIC) && !defined(BOX_REFLECT_X) && !defined(BOX_OUTFLOW_X)
#define TILE_PERIODIC_X 1
#else
#define TILE_PERIODIC_X 0
#endif
#if defined(BOX_PERIODIC) && !defined(BOX_REFLECT_Y) && !defined(BOX_OUTFLOW_Y)
#define TILE_PERIODIC_Y 1
#else
#define TILE_PERIODIC_Y 0
#endif
#if defined(BOX_PERIODIC) && !defined(BOX_REFLECT_Z) && !defined(BOX_OUTFLOW_Z)
#define TILE_PERIODIC_Z 1
#else
#define TILE_PERIODIC_Z 0
#endif

/* Sync All_dev for this TU (called from gizmo_gpu_sync_all in cooling.cc) */
void gizmo_gpu_sync_all_density(void) {
#if defined(GIZMO_GPU_COMPILER)
#pragma push_macro("All")
#undef All
    extern struct global_data_all_processes All;
    All_dev = All;
#pragma pop_macro("All")
#endif
}


/* ================================================================
   GPU neighbor list construction (B1)
   ================================================================
   Strategy:
   - Build tiles + BVH on CPU (recursive, serial — fine for ~100s of tiles)
   - Copy tiles, BVH, pool, active_indices to SharedSpace
   - Pass 1: Kokkos::parallel_for counting neighbors per particle
   - Exclusive prefix scan → CSR offsets
   - Pass 2: Kokkos::parallel_for filling neighbor indices
   ================================================================ */

/* GPU-resident neighbor list: CSR arrays in SharedSpace */
struct gpu_neighbor_list_t {
    int *offsets;       /* [num_active+1] in SharedSpace */
    int *neighbors;     /* [total_pairs] in SharedSpace */
    int num_active;
    int total_pairs;

    /* Device-resident copies of spatial index data */
    sfc_tile_t *d_tiles;
    tile_bvh_node_t *d_bvh;
    int *d_pool;
    int *d_active;
    int ntiles;
    int bvh_root;

    /* Periodicity parameters (copied to avoid global access in kernels) */
    int periodic_flags[3];
    double box_sizes[3];
    double box_halves[3];
};


static void gpu_ngb_list_build(struct particle_data *P_shared, int num_total,
                               int *active_indices_host, int num_active,
                               int search_mode, int type_bitmask,
                               gpu_neighbor_list_t *gnl)
{
    gnl->num_active = num_active;

    /* Capture periodicity parameters from globals (on host) */
    gnl->periodic_flags[0] = TILE_PERIODIC_X;
    gnl->periodic_flags[1] = TILE_PERIODIC_Y;
    gnl->periodic_flags[2] = TILE_PERIODIC_Z;
    gnl->box_sizes[0] = boxSize_X; gnl->box_sizes[1] = boxSize_Y; gnl->box_sizes[2] = boxSize_Z;
    gnl->box_halves[0] = boxHalf_X; gnl->box_halves[1] = boxHalf_Y; gnl->box_halves[2] = boxHalf_Z;

    /* Build SFC tiles + BVH on CPU (serial, recursive) */
    sfc_tile_t *h_tiles;
    int *h_pool;
    int num_pool;
    int ntiles = build_sfc_tiles(P_shared, num_total, type_bitmask, TILE_TARGET_SIZE,
                                 &h_tiles, &h_pool, &num_pool);
    gnl->ntiles = ntiles;

    tile_bvh_node_t *h_bvh;
    int bvh_nnodes = build_tile_bvh(h_tiles, ntiles, &h_bvh);
    gnl->bvh_root = bvh_nnodes - 1;

    /* Copy spatial index to SharedSpace */
    int bvh_size = (2 * ntiles - 1);
    if(bvh_size < 1) bvh_size = 1;
    gnl->d_tiles = (sfc_tile_t *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(ntiles * sizeof(sfc_tile_t));
    gnl->d_bvh = (tile_bvh_node_t *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(bvh_size * sizeof(tile_bvh_node_t));
    gnl->d_pool = (int *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(((num_pool > 0) ? num_pool : 1) * sizeof(int));
    gnl->d_active = (int *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(((num_active > 0) ? num_active : 1) * sizeof(int));

    memcpy(gnl->d_tiles, h_tiles, ntiles * sizeof(sfc_tile_t));
    memcpy(gnl->d_bvh, h_bvh, bvh_nnodes * sizeof(tile_bvh_node_t));
    memcpy(gnl->d_pool, h_pool, num_pool * sizeof(int));
    memcpy(gnl->d_active, active_indices_host, num_active * sizeof(int));

    /* Free CPU temporaries (reverse mymalloc order) */
    myfree(h_bvh);
    myfree(h_tiles);
    myfree(h_pool);

    /* Allocate CSR offsets in SharedSpace */
    gnl->offsets = (int *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>((num_active + 1) * sizeof(int));

    /* Pass 1: count neighbors per active particle */
    {
        sfc_tile_t *tiles = gnl->d_tiles;
        tile_bvh_node_t *bvh = gnl->d_bvh;
        int *pool = gnl->d_pool;
        int *active = gnl->d_active;
        int *offsets = gnl->offsets;
        int bvh_root = gnl->bvh_root;
        int smode = search_mode;
        int pf0 = gnl->periodic_flags[0], pf1 = gnl->periodic_flags[1], pf2 = gnl->periodic_flags[2];
        double bs0 = gnl->box_sizes[0], bs1 = gnl->box_sizes[1], bs2 = gnl->box_sizes[2];
        double bh0 = gnl->box_halves[0], bh1 = gnl->box_halves[1], bh2 = gnl->box_halves[2];

        Kokkos::parallel_for("ngb_count", num_active, KOKKOS_LAMBDA(int aa) {
            int pf[3] = {pf0, pf1, pf2};
            double bs[3] = {bs0, bs1, bs2};
            double bh[3] = {bh0, bh1, bh2};
            int i = active[aa];
            int cnt = search_neighbors_sfc_gpu(P_shared, i, P_shared[i].KernelRadius,
                                               tiles, ntiles, pool, smode,
                                               bvh, bvh_root, NULL, pf, bs, bh);
            offsets[aa] = cnt;  /* store count temporarily; will become offset after scan */
        });
        Kokkos::fence();
    }

    /* Exclusive prefix scan on host (offsets is UVM-accessible) */
    int total = 0;
    for(int aa = 0; aa < num_active; aa++) {
        int cnt = gnl->offsets[aa];
        gnl->offsets[aa] = total;
        total += cnt;
    }
    gnl->offsets[num_active] = total;
    gnl->total_pairs = total;

    /* Allocate CSR neighbors array */
    gnl->neighbors = (int *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(((total > 0) ? total : 1) * sizeof(int));

    /* Pass 2: fill neighbor indices */
    {
        sfc_tile_t *tiles = gnl->d_tiles;
        tile_bvh_node_t *bvh = gnl->d_bvh;
        int *pool = gnl->d_pool;
        int *active = gnl->d_active;
        int *offsets = gnl->offsets;
        int *neighbors = gnl->neighbors;
        int bvh_root = gnl->bvh_root;
        int smode = search_mode;
        int pf0 = gnl->periodic_flags[0], pf1 = gnl->periodic_flags[1], pf2 = gnl->periodic_flags[2];
        double bs0 = gnl->box_sizes[0], bs1 = gnl->box_sizes[1], bs2 = gnl->box_sizes[2];
        double bh0 = gnl->box_halves[0], bh1 = gnl->box_halves[1], bh2 = gnl->box_halves[2];

        Kokkos::parallel_for("ngb_fill", num_active, KOKKOS_LAMBDA(int aa) {
            int pf[3] = {pf0, pf1, pf2};
            double bs[3] = {bs0, bs1, bs2};
            double bh[3] = {bh0, bh1, bh2};
            int i = active[aa];
            search_neighbors_sfc_gpu(P_shared, i, P_shared[i].KernelRadius,
                                     tiles, ntiles, pool, smode,
                                     bvh, bvh_root, &neighbors[offsets[aa]],
                                     pf, bs, bh);
        });
        Kokkos::fence();
    }
}


static void gpu_ngb_list_free(gpu_neighbor_list_t *gnl)
{
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(gnl->neighbors);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(gnl->offsets);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(gnl->d_active);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(gnl->d_pool);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(gnl->d_bvh);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(gnl->d_tiles);
}


/* ================================================================
   GPU density kernel (B2)
   ================================================================
   For each active particle: load data, iterate CSR neighbors,
   accumulate density via density_functions.h, write results back.
   ================================================================ */

/* Entry point called from density.cc during h-iteration.
 * P_host/CellP_host are the regular (host-malloc'd) particle arrays.
 * This function:
 *   1. Copies P/CellP to SharedSpace (UVM)
 *   2. Builds GPU neighbor list (BVH + 2-pass CSR)
 *   3. Runs GPU density accumulation kernel
 *   4. Scatters results back to host P/CellP for active particles
 */
void density_evaluate_gpu(struct particle_data *P_host, struct gas_cell_data *CellP_host,
                          int num_total, int *active_indices_host, int num_active)
{
    /* Allocate SharedSpace copies of full particle arrays (neighbors indexed by global j) */
    struct particle_data *P_gpu = (struct particle_data *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(num_total * sizeof(struct particle_data));
    struct gas_cell_data *CellP_gpu = (struct gas_cell_data *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(num_total * sizeof(struct gas_cell_data));
    memcpy(P_gpu, P_host, num_total * sizeof(struct particle_data));
    memcpy(CellP_gpu, CellP_host, num_total * sizeof(struct gas_cell_data));

    /* Build GPU neighbor list */
    gpu_neighbor_list_t gnl;
    gpu_ngb_list_build(P_gpu, num_total, active_indices_host, num_active,
                       NGB_SEARCH_ONEWAY, 1 /* gas only */, &gnl);

    if(ThisTask == 0) {
        printf("  GPU density: %d active particles, %d neighbor pairs (%.1f avg)\n",
               num_active, gnl.total_pairs,
               num_active > 0 ? (double)gnl.total_pairs / num_active : 0.0);
        fflush(stdout);
    }

    /* GPU density accumulation kernel */
    {
        int *offsets = gnl.offsets;
        int *neighbors = gnl.neighbors;
        int *active = gnl.d_active;
        struct particle_data *kp = P_gpu;
        struct gas_cell_data *kc = CellP_gpu;

        Kokkos::parallel_for("density_kernel", num_active, KOKKOS_LAMBDA(int aa) {
            int ii = active[aa];

            /* particle2in: load searching particle data */
            struct density_evaluate_data_in_ local;
            local.Type = kp[ii].Type;
            local.KernelRadius = kp[ii].KernelRadius;
            local.Pos = kp[ii].Pos;
            if(kp[ii].Type == 0) { local.Vel = kc[ii].VelPred; } else { local.Vel = kp[ii].Vel; }
#if defined(SPHAV_CD10_VISCOSITY_SWITCH)
            if(kp[ii].Type == 0) { local.Accel = All.cf_a2inv * kp[ii].GravAccel + kc[ii].HydroAccel; }
#endif
#ifdef GALSF_SUBGRID_WINDS
            if(kp[ii].Type == 0) { local.DelayTime = kc[ii].DelayTime; } else { local.DelayTime = 0; }
#endif

            /* Initialize output */
            struct density_evaluate_data_out_ out;
            memset(&out, 0, sizeof(out));
#if defined(SINK_PARTICLES)
            out.Sink_TimeBinGasNeighbor = TIMEBINS;
#endif

            /* Kernel setup */
            struct kernel_density kernel;
            double h2 = local.KernelRadius * local.KernelRadius;
            kernel_hinv(local.KernelRadius, &kernel.hinv, &kernel.hinv3, &kernel.hinv4);

            /* Accumulate over neighbors (density_accumulate_neighbor calls
               density_evaluate_extra_physics_gas internally for r > 0) */
            for(int idx = offsets[aa]; idx < offsets[aa + 1]; idx++)
            {
                int j = neighbors[idx];
                density_accumulate_neighbor(&local, &out, &kernel, j, h2, kp, kc);
            }

            /* out2particle: scatter results into SharedSpace arrays */
            kp[ii].NumNgb = out.Ngb;
            kp[ii].DrkernNgbFactor = out.DrkernNgb;
            kp[ii].Particle_DivVel = out.Particle_DivVel;

            if(kp[ii].Type == 0)
            {
                kc[ii].Density = out.Rho;
#if defined(HYDRO_MESHLESS_FINITE_VOLUME) && ((HYDRO_FIX_MESH_MOTION==5)||(HYDRO_FIX_MESH_MOTION==6))
                kc[ii].ParticleVel = out.ParticleVel;
#endif
                for(int k = 0; k < 6; k++) { kc[ii].NV_T.data[k] = out.NV_T.data[k]; }
                kc[ii].NV_T_face_weights = out.NV_T_face_weights;
#ifdef HYDRO_PARTITION_UNITY_IMPROVE_FD
                kc[ii].GradH_numer = out.GradH_numer;
                kc[ii].GradH_denom = out.GradH_denom;
#endif
#ifdef HYDRO_SPH
                kc[ii].DrkernHydroSumFactor = out.DrkernHydroSumFactor;
#endif
#ifdef HYDRO_PRESSURE_SPH
                kc[ii].EgyWtDensity = out.EgyRho;
#endif
#if defined(TURB_DRIVING)
                kc[ii].SmoothedVel = out.GasVel;
#endif
#if defined(SPHAV_CD10_VISCOSITY_SWITCH)
                for(int k1 = 0; k1 < 3; k1++)
                    for(int k2 = 0; k2 < 3; k2++) {
                        kc[ii].NV_D[k1][k2] = out.NV_D[k1][k2];
                        kc[ii].NV_A[k1][k2] = out.NV_A[k1][k2];
                    }
#endif
            }

#if defined(GRAIN_FLUID)
            if((1 << kp[ii].Type) & (GRAIN_PTYPES)) {
                kc[ii].Density = out.Rho;
                kp[ii].Gas_InternalEnergy = out.Gas_InternalEnergy;
                kp[ii].Gas_Velocity = out.GasVel;
#if defined(GRAIN_LORENTZFORCE)
                kp[ii].Gas_B = out.Gas_B;
#endif
            }
#endif
#ifdef DO_DENSITY_AROUND_NONGAS_PARTICLES
            kp[ii].GradRho = out.GradRho;
#endif
#if defined(SINK_PARTICLES)
            if(kp[ii].Type == 5) {
                kp[ii].Sink_TimeBinGasNeighbor = out.Sink_TimeBinGasNeighbor;
#if defined(BH_ACCRETE_NEARESTFIRST) || defined(SINGLE_STAR_TIMESTEPPING)
                kp[ii].Sink_dr_to_NearestGasNeighbor = out.Sink_dr_to_NearestGasNeighbor;
#endif
            }
#endif
        }); /* end KOKKOS_LAMBDA */
        Kokkos::fence();

#if defined(__CUDACC__)
        {cudaError_t _ce = cudaGetLastError(); if(_ce != cudaSuccess) {printf("[GPU] density kernel error: %s\n", cudaGetErrorString(_ce)); fflush(stdout);}}
#elif defined(__HIPCC__)
        {hipError_t _ce = hipGetLastError(); if(_ce != hipSuccess) {printf("[GPU] density kernel error: %s\n", hipGetErrorString(_ce)); fflush(stdout);}}
#endif
    }

    gpu_ngb_list_free(&gnl);

    /* Scatter results back to host arrays for active particles */
    for(int aa = 0; aa < num_active; aa++) {
        int ii = active_indices_host[aa];
        P_host[ii] = P_gpu[ii];
        CellP_host[ii] = CellP_gpu[ii];
    }

    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(CellP_gpu);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(P_gpu);
}

/* ================================================================
   GPU gradient kernel (B3)
   ================================================================
   For each active particle: load data, iterate symmetric CSR neighbors,
   accumulate gradients via gradient_functions.h, return output structs.
   The caller (gradients.cc) does the scatter into CellP + GasGradDataPasser.
   ================================================================ */

/* Entry point for GPU gradient evaluation (gradient_iteration==0 only).
 * Takes a pre-built symmetric CSR neighbor list (offsets/neighbors).
 * Fills out_host[0..num_active-1] with accumulated gradient outputs.
 * Caller must allocate out_host (num_active * sizeof(GasGraddata_out_)). */
void gradient_evaluate_gpu(struct particle_data *P_host, struct gas_cell_data *CellP_host,
                           int num_total, int *active_indices_host, int num_active,
                           int *csr_offsets_host, int *csr_neighbors_host, int csr_total_pairs,
                           void *out_host_void)
{
    struct GasGraddata_out_ *out_host = (struct GasGraddata_out_ *)out_host_void;

    /* Allocate SharedSpace copies */
    struct particle_data *P_gpu = (struct particle_data *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(num_total * sizeof(struct particle_data));
    struct gas_cell_data *CellP_gpu = (struct gas_cell_data *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(num_total * sizeof(struct gas_cell_data));
    memcpy(P_gpu, P_host, num_total * sizeof(struct particle_data));
    memcpy(CellP_gpu, CellP_host, num_total * sizeof(struct gas_cell_data));

    /* Copy CSR neighbor list to SharedSpace */
    int *d_offsets = (int *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>((num_active + 1) * sizeof(int));
    int *d_neighbors = (int *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(((csr_total_pairs > 0) ? csr_total_pairs : 1) * sizeof(int));
    int *d_active = (int *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(((num_active > 0) ? num_active : 1) * sizeof(int));
    memcpy(d_offsets, csr_offsets_host, (num_active + 1) * sizeof(int));
    memcpy(d_neighbors, csr_neighbors_host, csr_total_pairs * sizeof(int));
    memcpy(d_active, active_indices_host, num_active * sizeof(int));

    /* Allocate output array in SharedSpace */
    struct GasGraddata_out_ *d_out = (struct GasGraddata_out_ *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(((num_active > 0) ? num_active : 1) * sizeof(struct GasGraddata_out_));

    if(ThisTask == 0) {
        printf("  GPU gradient: %d active particles, %d neighbor pairs (%.1f avg)\n",
               num_active, csr_total_pairs,
               num_active > 0 ? (double)csr_total_pairs / num_active : 0.0);
        fflush(stdout);
    }

    /* GPU gradient accumulation kernel */
    {
        int *offsets = d_offsets;
        int *neighbors = d_neighbors;
        int *active = d_active;
        struct particle_data *kp = P_gpu;
        struct gas_cell_data *kc = CellP_gpu;
        struct GasGraddata_out_ *kout = d_out;

        Kokkos::parallel_for("gradient_kernel", num_active, KOKKOS_LAMBDA(int aa) {
            int ii = active[aa];

            /* particle2in equivalent: load searching particle data */
            struct GasGraddata_in_ local;
            memset(&local, 0, sizeof(local));
            local.Pos = kp[ii].Pos;
            local.KernelRadius = kp[ii].KernelRadius;
            local.Mass = kp[ii].Mass;
            if(local.Mass < 0) {local.Mass = 0;}
            int sph_gradients_flag_i = SHOULD_I_USE_SPH_GRADIENTS(kc[ii].ConditionNumber);
            if(sph_gradients_flag_i) {local.Mass *= -1;}

            /* Load gradient quantities */
            local.GQuant.Density = kc[ii].Density;
            local.GQuant.Pressure = kc[ii].Pressure;
            local.GQuant.Velocity = kc[ii].VelPred;
#ifdef MAGNETIC
            local.GQuant.B = kc[ii].BPred * (kc[ii].Density / kp[ii].Mass);
#ifdef DIVBCLEANING_DEDNER
            local.GQuant.Phi = kc[ii].PhiPred / kp[ii].Mass;
#endif
#endif
#ifdef DOGRAD_INTERNAL_ENERGY
            local.GQuant.InternalEnergy = kc[ii].InternalEnergyPred;
#endif
#ifdef DOGRAD_SOUNDSPEED
            local.GQuant.SoundSpeed = kc[ii].effective_soundspeed();
#endif
#ifdef COSMIC_RAY_FLUID
            for(int k=0;k<N_CR_PARTICLE_BINS;k++) {local.GQuant.CosmicRayPressure[k] = Get_Gas_CosmicRayPressure(ii, k, kc);}
#endif
#if defined(TURB_DIFF_METALS) && !defined(TURB_DIFF_METALS_LOWORDER)
            for(int k=0;k<NUM_METAL_SPECIES;k++) {local.GQuant.Metallicity[k] = kp[ii].Metallicity[k];}
#endif
#ifdef RT_COMPGRAD_EDDINGTON_TENSOR
            {double V_i_inv = kc[ii].Density / kp[ii].Mass;
             for(int k=0;k<N_RT_FREQ_BINS;k++) {
                 local.GQuant.Rad_E_gamma[k] = kc[ii].Rad_E_gamma_Pred[k] * V_i_inv;
                 local.GQuant.Rad_E_gamma_ET[k] = local.GQuant.Rad_E_gamma[k] * kc[ii].ET[k];
#if defined(RT_M1_SECONDORDER) && defined(RT_EVOLVE_FLUX)
                 for(int k2=0;k2<3;k2++) {local.GQuant.Rad_Flux[k][k2] = kc[ii].Rad_Flux_Pred[k][k2] * V_i_inv;}
#endif
             }}
#endif
#ifdef TURB_DIFF_DYNAMIC
            local.GQuant.Velocity_bar = kc[ii].Velocity_bar;
            local.Norm_hat = kc[ii].Norm_hat;
#ifdef GALSF_SUBGRID_WINDS
            local.DelayTime = kc[ii].DelayTime;
#endif
#endif
#ifdef MHD_CONSTRAINED_GRADIENT
            local.ConditionNumber = kc[ii].ConditionNumber;
            local.NV_T = kc[ii].NV_T;
            for(int k=0;k<3;k++) for(int k2=0;k2<3;k2++) {local.BGrad[k][k2] = kc[ii].Gradients.B[k][k2];}
#ifdef MHD_MODIFIED_GRADIENT
            local.MG_cgcoeff = kc[ii].MG_cgcoeff;
#endif
#ifdef MHD_CONSTRAINED_GRADIENT_FAC_MEDDEV
            local.PhiGrad = kc[ii].Gradients.Phi;
#endif
#endif

            if(sph_gradients_flag_i) {local.Mass *= -1;} /* negate Mass as flag for SPH gradients */

            /* Initialize output */
            struct GasGraddata_out_ out;
            memset(&out, 0, sizeof(out));

            /* Pre-compute kernel quantities */
            double h_i = local.KernelRadius;
            double hinv, hinv3, hinv4;
            kernel_hinv(h_i, &hinv, &hinv3, &hinv4);
            if(local.Mass < 0) {local.Mass *= -1;} /* restore for V_i computation */
            double V_i = local.Mass / local.GQuant.Density;
            if(sph_gradients_flag_i) {local.Mass *= -1;} /* re-negate for kernel */

            int kernel_mode_i = -1;
            if(sph_gradients_flag_i) kernel_mode_i = 0;
#if defined(HYDRO_SPH) || defined(KERNEL_CRK_FACES)
            kernel_mode_i = 0;
#endif

            struct kernel_GasGrad kernel;
            kernel.h_i = h_i;

            /* Accumulate over symmetric neighbors */
            for(int idx = offsets[aa]; idx < offsets[aa + 1]; idx++)
            {
                int j = neighbors[idx];
                gradient_accumulate_neighbor(&local, &out, &kernel, j,
                                             sph_gradients_flag_i, V_i,
                                             hinv, hinv3, hinv4, kernel_mode_i,
                                             kp, kc);
            }

            /* Store output for this particle */
            kout[aa] = out;
        }); /* end KOKKOS_LAMBDA */
        Kokkos::fence();

#if defined(__CUDACC__)
        {cudaError_t _ce = cudaGetLastError(); if(_ce != cudaSuccess) {printf("[GPU] gradient kernel error: %s\n", cudaGetErrorString(_ce)); fflush(stdout);}}
#elif defined(__HIPCC__)
        {hipError_t _ce = hipGetLastError(); if(_ce != hipSuccess) {printf("[GPU] gradient kernel error: %s\n", hipGetErrorString(_ce)); fflush(stdout);}}
#endif
    }

    /* Copy output back to host */
    memcpy(out_host, d_out, num_active * sizeof(struct GasGraddata_out_));

    /* Cleanup SharedSpace */
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_out);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_active);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_neighbors);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_offsets);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(CellP_gpu);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(P_gpu);
}


/* ================================================================
   GPU hydro force kernel (B3b)
   ================================================================ */

void hydro_evaluate_gpu(struct particle_data *P_host, struct gas_cell_data *CellP_host,
                        int num_total, int *active_indices_host, int num_active,
                        int *csr_offsets_host, int *csr_neighbors_host, int csr_total_pairs,
                        void *out_host_void)
{
    struct hydro_data_out *out_host = (struct hydro_data_out *)out_host_void;

    /* Allocate SharedSpace copies */
    struct particle_data *P_gpu = (struct particle_data *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(num_total * sizeof(struct particle_data));
    struct gas_cell_data *CellP_gpu = (struct gas_cell_data *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(num_total * sizeof(struct gas_cell_data));
    memcpy(P_gpu, P_host, num_total * sizeof(struct particle_data));
    memcpy(CellP_gpu, CellP_host, num_total * sizeof(struct gas_cell_data));

    /* Copy CSR neighbor list to SharedSpace */
    int *d_offsets = (int *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>((num_active + 1) * sizeof(int));
    int *d_neighbors = (int *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(((csr_total_pairs > 0) ? csr_total_pairs : 1) * sizeof(int));
    int *d_active = (int *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(((num_active > 0) ? num_active : 1) * sizeof(int));
    memcpy(d_offsets, csr_offsets_host, (num_active + 1) * sizeof(int));
    memcpy(d_neighbors, csr_neighbors_host, csr_total_pairs * sizeof(int));
    memcpy(d_active, active_indices_host, num_active * sizeof(int));

    /* Copy TimeBinActive to SharedSpace */
    int *d_TimeBinActive = (int *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(TIMEBINS * sizeof(int));
    memcpy(d_TimeBinActive, TimeBinActive, TIMEBINS * sizeof(int));

    /* Wakeup flag in SharedSpace */
    int *d_NeedToWakeup = (int *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(sizeof(int));
    *d_NeedToWakeup = 0;

    /* Output array in SharedSpace */
    struct hydro_data_out *d_out = (struct hydro_data_out *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(((num_active > 0) ? num_active : 1) * sizeof(struct hydro_data_out));

    if(ThisTask == 0) {
        printf("  GPU hydro: %d active particles, %d neighbor pairs (%.1f avg)\n",
               num_active, csr_total_pairs,
               num_active > 0 ? (double)csr_total_pairs / num_active : 0.0);
        fflush(stdout);
    }

    /* GPU hydro force kernel */
    {
        int *offsets = d_offsets;
        int *neighbors = d_neighbors;
        int *active = d_active;
        struct particle_data *kp = P_gpu;
        struct gas_cell_data *kc = CellP_gpu;
        struct hydro_data_out *kout = d_out;
        int *kTimeBinActive = d_TimeBinActive;
        int *kNeedWakeup = d_NeedToWakeup;

        Kokkos::parallel_for("hydro_kernel", num_active, KOKKOS_LAMBDA(int aa) {
            int ii = active[aa];

            /* particle2in equivalent: load searching particle data */
            struct hydro_data_in local;
            memset(&local, 0, sizeof(local));
            local.Pos = kp[ii].Pos;
            local.Vel = kc[ii].VelPred;
#ifdef HYDRO_MESHLESS_FINITE_VOLUME
            local.ParticleVel = kc[ii].VelPred;
#endif
            local.KernelRadius = kp[ii].KernelRadius;
            local.Mass = kp[ii].Mass;
            local.Density = kc[ii].Density;
            local.Pressure = kc[ii].Pressure;
            local.ConditionNumber = kc[ii].ConditionNumber;
            local.FaceClosureError = kc[ii].FaceClosureError;
            local.InternalEnergyPred = kc[ii].InternalEnergyPred;
            local.SoundSpeed = kc[ii].effective_soundspeed();
            local.dt_hydrostep_i = get_particle_timestep_in_physical(ii, kp);
            local.DrkernNgbFactor = kp[ii].DrkernNgbFactor;
            local.Gradients.Density = kc[ii].Gradients.Density;
            local.Gradients.Pressure = kc[ii].Gradients.Pressure;
            local.Gradients.Velocity = kc[ii].Gradients.Velocity;
            local.NV_T = kc[ii].NV_T;
            local.TimeBin = kp[ii].TimeBin;
#ifdef MAGNETIC
            local.BPred = kc[ii].BPred;
            local.Gradients.B = kc[ii].Gradients.B;
#ifdef DIVBCLEANING_DEDNER
            local.PhiPred = kc[ii].PhiPred / kp[ii].Mass;
            local.Gradients.Phi = kc[ii].Gradients.Phi;
#endif
#ifdef MHD_MODIFIED_GRADIENT
            local.MG_cgcoeff = kc[ii].MG_cgcoeff;
#endif
#endif
#ifdef DOGRAD_INTERNAL_ENERGY
            local.Gradients.InternalEnergy = kc[ii].Gradients.InternalEnergy;
#endif
#ifdef DOGRAD_SOUNDSPEED
            local.Gradients.SoundSpeed = kc[ii].Gradients.SoundSpeed;
#endif
#if defined(TURB_DIFF_METALS) || (defined(METALS) && defined(HYDRO_MESHLESS_FINITE_VOLUME))
            for(int k=0;k<NUM_METAL_SPECIES;k++) {local.Metallicity[k] = kp[ii].Metallicity[k];}
#endif
#ifdef COSMIC_RAY_FLUID
            for(int k=0;k<N_CR_PARTICLE_BINS;k++) {local.CosmicRayPressure[k] = Get_Gas_CosmicRayPressure(ii, k, kc);}
#endif
#ifdef CONDUCTION
            local.Kappa_Conduction = kc[ii].Kappa_Conduction;
#endif
#ifdef VISCOSITY
            local.Eta_ShearViscosity = kc[ii].Eta_ShearViscosity;
            local.Zeta_BulkViscosity = kc[ii].Zeta_BulkViscosity;
#endif
#ifdef TURB_DIFFUSION
            local.TD_DiffCoeff = kc[ii].TD_DiffCoeff;
#endif
#if defined(KERNEL_CRK_FACES)
            for(int k=0;k<16;k++) {local.Tensor_CRK_Face_Corrections[k] = kc[ii].Tensor_CRK_Face_Corrections[k];}
#endif

            /* Initialize output and workspace */
            struct hydro_data_out out;
            memset(&out, 0, sizeof(out));
            struct kernel_hydra kernel;
            memset(&kernel, 0, sizeof(kernel));
            struct Conserved_var_Riemann Fluxes;

            kernel.h_i = local.KernelRadius;
            kernel.sound_i = local.SoundSpeed;
            kernel.spec_egy_u_i = local.InternalEnergyPred;
#ifdef MAGNETIC
            {
                double fac_magnetic_pressure_loc = 1.0 / All.cf_atime; /* B*B*fac = pressure units */
                kernel.b2_i = local.BPred.norm_sq(); /* raw B^2, without fac_magnetic_pressure */
                kernel.alfven2_i = kernel.b2_i * fac_magnetic_pressure_loc / local.Density;
                kernel.alfven2_i = DMIN(kernel.alfven2_i, 1000. * kernel.sound_i * kernel.sound_i);
            }
#endif

            /* Accumulate over symmetric neighbors */
            for(int idx = offsets[aa]; idx < offsets[aa + 1]; idx++)
            {
                int j = neighbors[idx];
                memset(&Fluxes, 0, sizeof(Fluxes));
                hydro_accumulate_neighbor(&local, &out, &kernel, &Fluxes, j,
                                          local.dt_hydrostep_i, kp, kc,
                                          kTimeBinActive, kNeedWakeup);
            }

            /* Store output for this particle */
            kout[aa] = out;
        }); /* end KOKKOS_LAMBDA */
        Kokkos::fence();

#if defined(__CUDACC__)
        {cudaError_t _ce = cudaGetLastError(); if(_ce != cudaSuccess) {printf("[GPU] hydro kernel error: %s\n", cudaGetErrorString(_ce)); fflush(stdout);}}
#elif defined(__HIPCC__)
        {hipError_t _ce = hipGetLastError(); if(_ce != hipSuccess) {printf("[GPU] hydro kernel error: %s\n", hipGetErrorString(_ce)); fflush(stdout);}}
#endif
    }

    /* Copy output back to host */
    memcpy(out_host, d_out, num_active * sizeof(struct hydro_data_out));

    /* Copy wakeup flag back */
    if(*d_NeedToWakeup) NeedToWakeupParticles_local = 1;

    /* Scatter j-particle modifications (dMass, wakeup) back to host P/CellP.
       Ghost writeback handles the MPI communication — here we just copy the
       SharedSpace arrays back so local+ghost modifications are visible on host. */
    for(int j = 0; j < num_total; j++) {
        P_host[j].wakeup = P_gpu[j].wakeup;
#ifdef HYDRO_MESHLESS_FINITE_VOLUME
        CellP_host[j].dMass = CellP_gpu[j].dMass;
#endif
    }

    /* Cleanup SharedSpace */
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_out);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_NeedToWakeup);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_TimeBinActive);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_active);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_neighbors);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_offsets);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(CellP_gpu);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(P_gpu);
}


#else /* !OPENMP_GPU_OFFLOAD || !GIZMO_USE_NEIGHBOR_LIST_FOR_DENSITY */

/* Stub: GPU density/gradient/hydro not available */
void density_evaluate_gpu(struct particle_data *, struct gas_cell_data *,
                          int, int *, int) {}
void gradient_evaluate_gpu(struct particle_data *, struct gas_cell_data *,
                           int, int *, int, int *, int *, int, void *) {}
void hydro_evaluate_gpu(struct particle_data *, struct gas_cell_data *,
                        int, int *, int, int *, int *, int, void *) {}
void gizmo_gpu_sync_all_density(void) {}

#endif /* OPENMP_GPU_OFFLOAD && GIZMO_USE_NEIGHBOR_LIST_FOR_DENSITY */
