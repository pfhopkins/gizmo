/*! \file ghost_exchange.cc
 *  \brief Ghost particle exchange for GPU-ready neighbor finding.
 *
 *  Replaces the pseudo-particle export mechanism with an upfront "import-the-neighbors"
 *  pattern: before any neighbor loop, exchange boundary particles between MPI ranks so
 *  that all neighbors are local. Subsequent kernels iterate over local + ghost particles
 *  without secondary MPI phases.
 *
 *  Ghost particles are appended to P[] and CellP[] arrays at indices >= NumPart_before_ghost.
 *  After all neighbor operations complete, ghost_exchange_cleanup() resets NumPart/N_gas.
 *
 *  This file was written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 *
 *  KNOWN LIMITATIONS (Phase 0 — to be optimized):
 *  - Sends full P[i]/CellP[i] structs per ghost (~2-5 KB/particle). Should use compact
 *    struct with only fields needed by the active kernel set (~200 bytes).
 *  - O(NTopleaves^2) overlap check between local and remote leaves. Should use spatial
 *    sorting or tree-based pruning for simulations with many top-level leaves.
 *  - Global MPI_Allreduce on need_leaf (NTopleaves ints). Could use point-to-point for
 *    sparse communication patterns.
 *  - Per-task routing is approximate: uses MPI_Allreduce(MPI_MAX) on need_leaf, so a leaf
 *    requested by ANY task is sent to ALL requesting tasks. Per-task leaf request lists
 *    would reduce traffic.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../declarations/allvars.h"
#include "../core/proto.h"

/*
 * ============================================================================
 * COMPACT GHOST STRUCT FIELD REQUIREMENTS (for future optimization)
 *
 * Phase 0 sends full P[i] + CellP[i] structs. When optimizing, the minimum
 * fields needed per kernel are:
 *
 * ALL kernels need from P: Pos[3], Mass, Type, KernelRadius, NumNgb, TimeBin
 *
 * density_evaluate additionally needs:
 *   P: Vel[3]
 *   CellP: VelPred[3], InternalEnergyPred, Density
 *   + #ifdef MAGNETIC: CellP.BPred[3]
 *   + #ifdef COSMIC_RAY_FLUID: CellP.CosmicRayEnergyPred[N_CR_PARTICLE_BINS]
 *   + #ifdef RADTRANSFER: CellP.Rad_E_gamma[N_RT_FREQ_BINS], Rad_E_gamma_Pred
 *
 * hydro_gradient_calc additionally needs:
 *   All of density fields, plus:
 *   CellP: Pressure, MaxSignalVel
 *
 * hydro_force_evaluate additionally needs:
 *   All of gradient fields, plus:
 *   CellP: DtInternalEnergy, HydroAccel[3], SoundSpeed, Gradients
 *   P: GravAccel[3]
 *   + #ifdef DIVBCLEANING_DEDNER: CellP.PhiPred
 *   + all MHD/RT/CR evolved quantities
 * ============================================================================
 */

/* saved state for cleanup */
static int NumPart_before_ghost = -1;
static int N_gas_before_ghost = -1;
static int NumGhostParticles = 0;

/* Ghost provenance map: for each ghost particle, the home MPI rank and index.
   Used by ghost_writeback to reverse-communicate j-particle modifications.
   Allocated with malloc (not mymalloc) to avoid stack ordering issues. */
static int *ghost_home_rank_map = NULL;     /* [NumGhostParticles] home MPI rank */
static int *ghost_home_index_map = NULL;    /* [NumGhostParticles] home P[]/CellP[] index */
static int *ghost_wb_recv_count = NULL;     /* [NTask] ghosts received from each rank */
static int *ghost_wb_recv_disp = NULL;      /* [NTask] displacement by source rank */
static int *ghost_wb_send_count = NULL;     /* [NTask] ghosts we sent to each rank */
static int *ghost_wb_send_disp = NULL;      /* [NTask] displacement for what each rank got from us */

/* saved per-leaf hmax at time of ghost exchange, for h-growth detection */
static double *saved_leaf_hmax = NULL;
static int saved_leaf_hmax_n = 0;


/* ---- Utility: walk TopNodes to find which leaf a particle belongs to ---- */
static inline int ghost_toptree_leaf(peanokey key)
{
    int no = 0;
    peanokey mask = ((peanokey)7) << (3 * (BITS_PER_DIMENSION - 1));
    int shift = 3 * (BITS_PER_DIMENSION - 1);
    while(TopNodes[no].Daughter >= 0)
    {
        no = TopNodes[no].Daughter + (int)((key & mask) >> shift);
        mask >>= 3;
        shift -= 3;
    }
    return TopNodes[no].Leaf;
}


/*!
 * \brief Main ghost exchange routine. Call before neighbor operations.
 *
 * For each remote top-level leaf whose bounding region overlaps any local
 * particle's search sphere, imports all particles from that leaf.
 * Ghost particles are appended to P[]/CellP[] starting at NumPart.
 *
 * After all neighbor operations, call ghost_exchange_cleanup() to remove ghosts.
 *
 * safety_factor: multiplier on search_radius for the overlap criterion.
 *   1.0 = normal (previous-step hmax is accurate).
 *   >1.0 = inflate search radius to account for h-growth during density iteration
 *          (e.g. 2.0 on first timestep when densities are just guesses).
 */
void ghost_exchange(double safety_factor)
{
    if(NTask <= 1) return;
    double t_ghost_start = my_second();

    /* save current state for cleanup */
    NumPart_before_ghost = NumPart;
    N_gas_before_ghost = N_gas;
    NumGhostParticles = 0;

    int i, k, task;
    int tile_target = 64;

    /* ================================================================
       Step 1: Build SFC tiles from local particles.
       Particles are already Peano-Hilbert sorted. Group into tiles of
       ~tile_target particles, computing bbox and hmax per tile.
       Uses malloc (not mymalloc) for tile metadata to avoid stack issues.
       ================================================================ */
    int local_ntiles = 0, num_pool = 0;

    /* Count pool particles (all types, positive mass) */
    for(i = 0; i < NumPart; i++) { if(P[i].Mass > 0) num_pool++; }

    /* Build pool index array */
    int *pool = (int *) malloc((num_pool > 0 ? num_pool : 1) * sizeof(int));
    int p = 0;
    for(i = 0; i < NumPart; i++) { if(P[i].Mass > 0) pool[p++] = i; }

    local_ntiles = (num_pool + tile_target - 1) / tile_target;
    if(local_ntiles < 1) local_ntiles = 1;

    /* Compact tile metadata for exchange: bbox (6 doubles) + hmax (1 double) + count (1 int) = 60 bytes */
    struct tile_meta_t {
        double lo[3], hi[3], hmax;
        int count;
    };

    tile_meta_t *local_meta = (tile_meta_t *) malloc(local_ntiles * sizeof(tile_meta_t));
    int *tile_first = (int *) malloc(local_ntiles * sizeof(int)); /* index into pool[] */

    for(int t = 0; t < local_ntiles; t++)
    {
        int start = t * tile_target;
        int count = tile_target;
        if(start + count > num_pool) count = num_pool - start;
        tile_first[t] = start;
        local_meta[t].count = count;
        local_meta[t].hmax = 0;

        int j0 = pool[start];
        for(k = 0; k < 3; k++) local_meta[t].lo[k] = local_meta[t].hi[k] = P[j0].Pos[k];
        if(P[j0].KernelRadius > local_meta[t].hmax) local_meta[t].hmax = P[j0].KernelRadius;

        for(int s = 1; s < count; s++) {
            int j = pool[start + s];
            for(k = 0; k < 3; k++) {
                if(P[j].Pos[k] < local_meta[t].lo[k]) local_meta[t].lo[k] = P[j].Pos[k];
                if(P[j].Pos[k] > local_meta[t].hi[k]) local_meta[t].hi[k] = P[j].Pos[k];
            }
            if(P[j].KernelRadius > local_meta[t].hmax) local_meta[t].hmax = P[j].KernelRadius;
        }
    }

    /* Save per-tile hmax for h-growth detection */
    if(saved_leaf_hmax) {free(saved_leaf_hmax); saved_leaf_hmax = NULL;}
    saved_leaf_hmax_n = local_ntiles;
    saved_leaf_hmax = (double *) malloc(local_ntiles * sizeof(double));
    for(int t = 0; t < local_ntiles; t++) saved_leaf_hmax[t] = local_meta[t].hmax;

    /* ================================================================
       Step 2: Gather tile metadata from all ranks.
       Each rank sends its tile count and metadata to all ranks.
       ================================================================ */
    int *all_ntiles = (int *) malloc(NTask * sizeof(int));
    MPI_Allgather(&local_ntiles, 1, MPI_INT, all_ntiles, 1, MPI_INT, MPI_COMM_WORLD);

    int *tile_disp = (int *) malloc(NTask * sizeof(int));
    int total_tiles = 0;
    for(task = 0; task < NTask; task++) {
        tile_disp[task] = total_tiles;
        total_tiles += all_ntiles[task];
    }

    /* Exchange tile metadata using MPI_Allgatherv with MPI_BYTE */
    tile_meta_t *all_meta = (tile_meta_t *) malloc(total_tiles * sizeof(tile_meta_t));
    int *meta_counts = (int *) malloc(NTask * sizeof(int));
    int *meta_disps = (int *) malloc(NTask * sizeof(int));
    for(task = 0; task < NTask; task++) {
        meta_counts[task] = all_ntiles[task] * sizeof(tile_meta_t);
        meta_disps[task] = tile_disp[task] * sizeof(tile_meta_t);
    }
    MPI_Allgatherv(local_meta, local_ntiles * sizeof(tile_meta_t), MPI_BYTE,
                   all_meta, meta_counts, meta_disps, MPI_BYTE, MPI_COMM_WORLD);
    free(meta_counts); free(meta_disps);

    if(ThisTask == 0) {
        double hmax_min = 1e30, hmax_max = 0, hmax_sum = 0;
        for(int t = 0; t < local_ntiles; t++) {
            if(local_meta[t].hmax < hmax_min) hmax_min = local_meta[t].hmax;
            if(local_meta[t].hmax > hmax_max) hmax_max = local_meta[t].hmax;
            hmax_sum += local_meta[t].hmax;
        }
        PRINT_STATUS("Ghost exchange (tile-based): %d local tiles, %d total across %d ranks, tile hmax=[%.4g, %.4g] avg=%.4g",
                     local_ntiles, total_tiles, NTask, hmax_min, hmax_max, hmax_sum / local_ntiles);
    }

    /* ================================================================
       Step 3: Per-task tile overlap check.
       For each remote task, check which of its tiles overlap with any
       of our tiles AND which of our tiles overlap with any of its tiles.
       No global Allreduce — each rank independently computes per-task
       recv and send lists using the symmetric overlap criterion.
       ================================================================ */

    /* Per-tile need flags: need_from[total_tiles] = do WE need this remote tile?
       send_to[local_ntiles * NTask] = does task t need our tile lt? */
    int *need_from = (int *) calloc(total_tiles, sizeof(int));
    int *send_to = (int *) calloc(local_ntiles * NTask, sizeof(int));
    int my_tile_start = tile_disp[ThisTask];

    for(task = 0; task < NTask; task++)
    {
        if(task == ThisTask) continue;
        int t_start = tile_disp[task];
        int t_count = all_ntiles[task];

        for(int rt_idx = 0; rt_idx < t_count; rt_idx++)
        {
            int rt = t_start + rt_idx;
            tile_meta_t *rm = &all_meta[rt];

            for(int lt_idx = 0; lt_idx < local_ntiles; lt_idx++)
            {
                int lt = my_tile_start + lt_idx;
                tile_meta_t *lm = &all_meta[lt];
                double search_r = DMAX(lm->hmax, rm->hmax) * safety_factor;
                if(search_r <= 0) continue;
                double search_r2 = search_r * search_r;

                /* Min distance between two AABBs with periodic wrapping */
                double dist2 = 0;
                int overlaps = 1;
                for(k = 0; k < 3; k++)
                {
#if defined(BOX_PERIODIC)
                    int is_periodic = 1;
                    double bsize = (k==0) ? boxSize_X : ((k==1) ? boxSize_Y : boxSize_Z);
#if defined(BOX_REFLECT_X)
                    if(k==0) is_periodic = 0;
#endif
#if defined(BOX_REFLECT_Y)
                    if(k==1) is_periodic = 0;
#endif
#if defined(BOX_REFLECT_Z)
                    if(k==2) is_periodic = 0;
#endif
#if defined(BOX_OUTFLOW_X)
                    if(k==0) is_periodic = 0;
#endif
#if defined(BOX_OUTFLOW_Y)
                    if(k==1) is_periodic = 0;
#endif
#if defined(BOX_OUTFLOW_Z)
                    if(k==2) is_periodic = 0;
#endif
#else
                    int is_periodic = 0;
                    double bsize = 0;
#endif
                    double c_local = 0.5 * (lm->lo[k] + lm->hi[k]);
                    double c_remote = 0.5 * (rm->lo[k] + rm->hi[k]);
                    double hw_local = 0.5 * (lm->hi[k] - lm->lo[k]);
                    double hw_remote = 0.5 * (rm->hi[k] - rm->lo[k]);

                    double dx = fabs(c_local - c_remote);
                    if(is_periodic && dx > 0.5 * bsize) dx = bsize - dx;

                    double gap = dx - hw_local - hw_remote;
                    if(gap <= 0) continue; /* AABBs overlap on this axis */

                    if(gap > search_r) { overlaps = 0; break; }
                    dist2 += gap * gap;
                }

                if(overlaps && dist2 < search_r2) {
                    need_from[rt] = 1;          /* we need this remote tile */
                    send_to[lt_idx * NTask + task] = 1; /* task t needs our tile lt */
                    /* don't break — need to check all local tiles for send_to */
                }
            }
        }
    }

    /* ================================================================
       Step 4: Compute per-task send/recv counts from overlap results.
       ================================================================ */
    int *recv_count = (int *) mymalloc("ghost_rc", NTask * sizeof(int));
    int *send_count = (int *) mymalloc("ghost_sc", NTask * sizeof(int));
    memset(recv_count, 0, NTask * sizeof(int));
    memset(send_count, 0, NTask * sizeof(int));

    for(task = 0; task < NTask; task++) {
        if(task == ThisTask) continue;
        /* Recv: remote tiles we need from this task */
        for(int rt = tile_disp[task]; rt < tile_disp[task] + all_ntiles[task]; rt++) {
            if(need_from[rt]) recv_count[task] += all_meta[rt].count;
        }
        /* Send: our tiles this task needs */
        for(int lt = 0; lt < local_ntiles; lt++) {
            if(send_to[lt * NTask + task]) send_count[task] += local_meta[lt].count;
        }
    }

    /* Compute totals and displacements */
    int total_recv = 0, total_send = 0;
    int *recv_disp = (int *) mymalloc("ghost_rd", NTask * sizeof(int));
    int *send_disp = (int *) mymalloc("ghost_sd", NTask * sizeof(int));
    recv_disp[0] = 0; send_disp[0] = 0;
    for(task = 0; task < NTask; task++) {
        total_recv += recv_count[task];
        total_send += send_count[task];
        if(task > 0) {
            recv_disp[task] = recv_disp[task-1] + recv_count[task-1];
            send_disp[task] = send_disp[task-1] + send_count[task-1];
        }
    }

    int tiles_needed = 0, tiles_sent = 0;
    for(int rt = 0; rt < total_tiles; rt++) tiles_needed += need_from[rt];
    for(int lt = 0; lt < local_ntiles; lt++) {
        for(task = 0; task < NTask; task++) { if(send_to[lt * NTask + task]) { tiles_sent++; break; } }
    }

    /* Check space: ghost particles are appended to P[]/CellP[] arrays, which are
       allocated to All.MaxPart = PartAllocFactor * (TotNumPart / NTask).
       If there isn't enough room, we MUST exit — silently skipping would produce
       wrong results (incomplete neighbor data near domain boundaries). */
    if(NumPart + total_recv > All.MaxPart) {
        double needed_factor = (double)(NumPart + total_recv) / ((double)All.TotNumPart / NTask);
        printf("\n=======================================================================\n");
        printf("ERROR: Ghost exchange requires %d ghost particles on task %d,\n", total_recv, ThisTask);
        printf("  but only %d free slots available (NumPart=%d, MaxPart=%d).\n",
               All.MaxPart - NumPart, NumPart, All.MaxPart);
        printf("  Current PartAllocFactor = %.2f\n", All.PartAllocFactor);
        printf("  Minimum PartAllocFactor needed = %.2f (recommend %.2f for safety)\n",
               needed_factor, needed_factor * 1.2);
        printf("  Fix: increase PartAllocFactor in your parameterfile to at least %.1f\n",
               needed_factor * 1.2);
        printf("  (or increase the number of MPI ranks to reduce particles per rank)\n");
        printf("=======================================================================\n");
        fflush(stdout);
        endrun(7701);
    }

    /* ================================================================
       Step 5: Pack particles from tiles needed by each task.
       ================================================================ */
    struct particle_data *send_P = (struct particle_data *) mymalloc("ghost_sP",
        (total_send > 0 ? total_send : 1) * sizeof(struct particle_data));
    struct gas_cell_data *send_CellP = (struct gas_cell_data *) mymalloc("ghost_sC",
        (total_send > 0 ? total_send : 1) * sizeof(struct gas_cell_data));
    /* Record home index of each sent particle for ghost writeback */
    int *send_home_idx = (int *) malloc((total_send > 0 ? total_send : 1) * sizeof(int));

    int *task_offset = (int *) mymalloc("ghost_toff", NTask * sizeof(int));
    memcpy(task_offset, send_disp, NTask * sizeof(int));

    for(task = 0; task < NTask; task++)
    {
        if(send_count[task] <= 0) continue;
        for(int t = 0; t < local_ntiles; t++)
        {
            if(!send_to[t * NTask + task]) continue;
            for(int s = 0; s < local_meta[t].count; s++)
            {
                if(task_offset[task] >= send_disp[task] + send_count[task]) break;
                int j = pool[tile_first[t] + s];
                int off = task_offset[task]++;
                send_P[off] = P[j];
                send_home_idx[off] = j; /* record home index for writeback provenance */
                if(P[j].Type == 0 && j < N_gas)
                    send_CellP[off] = CellP[j];
                else
                    memset(&send_CellP[off], 0, sizeof(struct gas_cell_data));
            }
        }
    }

    /* ================================================================
       Step 6: Exchange via MPI_Alltoallv.
       ================================================================ */
    int *recv_bytes = (int *) mymalloc("ghost_rb", NTask * sizeof(int));
    int *send_bytes = (int *) mymalloc("ghost_sb", NTask * sizeof(int));
    int *recv_bdisp = (int *) mymalloc("ghost_rbd", NTask * sizeof(int));
    int *send_bdisp = (int *) mymalloc("ghost_sbd", NTask * sizeof(int));
    for(task = 0; task < NTask; task++) {
        recv_bytes[task] = recv_count[task] * sizeof(struct particle_data);
        send_bytes[task] = send_count[task] * sizeof(struct particle_data);
        recv_bdisp[task] = recv_disp[task] * sizeof(struct particle_data);
        send_bdisp[task] = send_disp[task] * sizeof(struct particle_data);
    }

    MPI_Alltoallv(send_P, send_bytes, send_bdisp, MPI_BYTE,
                  &P[NumPart], recv_bytes, recv_bdisp, MPI_BYTE, MPI_COMM_WORLD);

    for(task = 0; task < NTask; task++) {
        recv_bytes[task] = recv_count[task] * sizeof(struct gas_cell_data);
        send_bytes[task] = send_count[task] * sizeof(struct gas_cell_data);
        recv_bdisp[task] = recv_disp[task] * sizeof(struct gas_cell_data);
        send_bdisp[task] = send_disp[task] * sizeof(struct gas_cell_data);
    }

    MPI_Alltoallv(send_CellP, send_bytes, send_bdisp, MPI_BYTE,
                  &CellP[NumPart], recv_bytes, recv_bdisp, MPI_BYTE, MPI_COMM_WORLD);

    /* Update counts */
    NumGhostParticles = total_recv;
    NumPart += total_recv;

    /* ================================================================
       Step 7: Build ghost provenance map for writeback.
       Exchange home indices so each ghost knows its home rank + index.
       ================================================================ */
    {
        /* Exchange home indices: send_home_idx[total_send] → recv_home_idx[total_recv] */
        int *recv_home_idx = (int *) malloc((total_recv > 0 ? total_recv : 1) * sizeof(int));
        for(task = 0; task < NTask; task++) {
            send_bytes[task] = send_count[task] * sizeof(int);
            recv_bytes[task] = recv_count[task] * sizeof(int);
            send_bdisp[task] = send_disp[task] * sizeof(int);
            recv_bdisp[task] = recv_disp[task] * sizeof(int);
        }
        MPI_Alltoallv(send_home_idx, send_bytes, send_bdisp, MPI_BYTE,
                      recv_home_idx, recv_bytes, recv_bdisp, MPI_BYTE, MPI_COMM_WORLD);

        /* Build per-ghost provenance: home_rank and home_index */
        ghost_home_rank_map = (int *) malloc((total_recv > 0 ? total_recv : 1) * sizeof(int));
        ghost_home_index_map = recv_home_idx; /* take ownership — freed in cleanup */
        for(task = 0; task < NTask; task++) {
            for(int g = 0; g < recv_count[task]; g++) {
                ghost_home_rank_map[recv_disp[task] + g] = task;
            }
        }

        /* Preserve comm maps for reverse Alltoallv (malloc copies of mymalloc'd arrays) */
        ghost_wb_recv_count = (int *) malloc(NTask * sizeof(int));
        ghost_wb_recv_disp  = (int *) malloc(NTask * sizeof(int));
        ghost_wb_send_count = (int *) malloc(NTask * sizeof(int));
        ghost_wb_send_disp  = (int *) malloc(NTask * sizeof(int));
        memcpy(ghost_wb_recv_count, recv_count, NTask * sizeof(int));
        memcpy(ghost_wb_recv_disp,  recv_disp,  NTask * sizeof(int));
        memcpy(ghost_wb_send_count, send_count, NTask * sizeof(int));
        memcpy(ghost_wb_send_disp,  send_disp,  NTask * sizeof(int));
    }

    double t_ghost_end = my_second();
    if(ThisTask == 0)
        PRINT_STATUS("Ghost exchange: %d local + %d ghost = %d total (recv %d tiles, sent %d/%d) [%.4f s]",
                     NumPart_before_ghost, NumGhostParticles, NumPart,
                     tiles_needed, tiles_sent, local_ntiles, timediff(t_ghost_start, t_ghost_end));
    /* Warn if ghost particles used >80% of available headroom */
    if(NumPart > 0.8 * All.MaxPart) {
        double usage_frac = (double)NumPart / (double)All.MaxPart;
        PRINT_WARNING("Ghost exchange: particle arrays %.0f%% full (%d/%d). "
                      "Consider increasing PartAllocFactor (currently %.2f) to avoid running out of space.",
                      100.0 * usage_frac, NumPart, All.MaxPart, All.PartAllocFactor);
    }

    /* Cleanup: mymalloc in reverse order, then free malloc'd send metadata */
    myfree(send_bdisp); myfree(recv_bdisp); myfree(send_bytes); myfree(recv_bytes);
    myfree(task_offset);
    myfree(send_CellP); myfree(send_P);
    myfree(send_disp); myfree(recv_disp);
    myfree(send_count); myfree(recv_count);
    free(send_home_idx);
    free(send_to); free(need_from);
    free(all_meta); free(tile_disp); free(all_ntiles);
    free(tile_first); free(local_meta); free(pool);
}


/*!
 * \brief Check whether any leaf's hmax grew since the last ghost exchange.
 *
 * Recomputes per-leaf hmax from current particle KernelRadius values and
 * compares against the values saved during the last ghost_exchange() call.
 * Returns 1 if any leaf's hmax grew by more than 10%, meaning the ghost
 * pool may be incomplete and a re-exchange is needed.
 */
int ghost_exchange_needs_redo(void)
{
    if(NTask <= 1) return 0;
    if(!saved_leaf_hmax || saved_leaf_hmax_n <= 0) return 0;

    /* Rebuild tiles from current local particles to get current per-tile hmax */
    int nlocal = (NumPart_before_ghost >= 0) ? NumPart_before_ghost : NumPart;
    int tile_target = 64;

    /* Count pool and build tile hmax values (same tiling as ghost_exchange) */
    int num_pool = 0;
    int i;
    for(i = 0; i < nlocal; i++) { if(P[i].Mass > 0) num_pool++; }

    int ntiles = (num_pool + tile_target - 1) / tile_target;
    if(ntiles < 1) ntiles = 1;
    if(ntiles != saved_leaf_hmax_n) return 1; /* tile count changed, definitely redo */

    double *current_hmax = (double *) malloc(ntiles * sizeof(double));
    memset(current_hmax, 0, ntiles * sizeof(double));

    int p = 0;
    for(i = 0; i < nlocal; i++) {
        if(P[i].Mass <= 0) continue;
        int t = p / tile_target;
        if(t >= ntiles) t = ntiles - 1;
        if(P[i].KernelRadius > current_hmax[t]) current_hmax[t] = P[i].KernelRadius;
        p++;
    }

    /* Check if any tile's hmax grew by more than 10% */
    int needs_redo_local = 0;
    for(int t = 0; t < ntiles; t++) {
        if(current_hmax[t] > saved_leaf_hmax[t] * 1.1) {
            needs_redo_local = 1;
            break;
        }
    }

    /* Global consensus: if ANY rank needs redo, all redo */
    int needs_redo = 0;
    MPI_Allreduce(&needs_redo_local, &needs_redo, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    if(needs_redo && ThisTask == 0) {
        double max_growth = 0;
        for(int t = 0; t < ntiles; t++) {
            if(saved_leaf_hmax[t] > 0) {
                double growth = (current_hmax[t] - saved_leaf_hmax[t]) / saved_leaf_hmax[t];
                if(growth > max_growth) max_growth = growth;
            }
        }
        PRINT_STATUS("Ghost exchange redo needed: max tile hmax growth = %.1f%%", 100.0 * max_growth);
    }

    free(current_hmax);
    return needs_redo;
}


/*!
 * \brief Remove ghost particles after neighbor operations complete.
 *
 * Resets NumPart and N_gas to pre-exchange values. Must be called after
 * all neighbor loops (density, gradients, hydro force) that use ghosts.
 */
void ghost_exchange_cleanup(void)
{
    if(NumPart_before_ghost < 0) return;
    NumPart = NumPart_before_ghost;
    N_gas = N_gas_before_ghost;
    NumGhostParticles = 0;
    NumPart_before_ghost = -1;
    /* Free ghost provenance map */
    if(ghost_home_rank_map)  { free(ghost_home_rank_map);  ghost_home_rank_map = NULL; }
    if(ghost_home_index_map) { free(ghost_home_index_map); ghost_home_index_map = NULL; }
    if(ghost_wb_recv_count)  { free(ghost_wb_recv_count);  ghost_wb_recv_count = NULL; }
    if(ghost_wb_recv_disp)   { free(ghost_wb_recv_disp);   ghost_wb_recv_disp = NULL; }
    if(ghost_wb_send_count)  { free(ghost_wb_send_count);  ghost_wb_send_count = NULL; }
    if(ghost_wb_send_disp)   { free(ghost_wb_send_disp);   ghost_wb_send_disp = NULL; }
}

/* Accessors for ghost provenance data — used by ghost_writeback.cc */
int ghost_get_num_ghosts(void) { return NumGhostParticles; }
int ghost_get_num_local(void)  { return (NumPart_before_ghost >= 0) ? NumPart_before_ghost : NumPart; }
int *ghost_get_home_rank(void)  { return ghost_home_rank_map; }
int *ghost_get_home_index(void) { return ghost_home_index_map; }
int *ghost_get_wb_recv_count(void) { return ghost_wb_recv_count; }
int *ghost_get_wb_recv_disp(void)  { return ghost_wb_recv_disp; }
int *ghost_get_wb_send_count(void) { return ghost_wb_send_count; }
int *ghost_get_wb_send_disp(void)  { return ghost_wb_send_disp; }
