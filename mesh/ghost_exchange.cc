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
 */
void ghost_exchange(void)
{
    if(NTask <= 1) return;

    /* save current state for cleanup */
    NumPart_before_ghost = NumPart;
    N_gas_before_ghost = N_gas;
    NumGhostParticles = 0;

    /* Step 1: For each local particle, find its top-level leaf and record hmax per leaf.
       Also build a mapping from leaf index to the particles it contains. */

    /* Count particles per local leaf */
    int *local_leaf_count = (int *) mymalloc("ghost_llc", NTopleaves * sizeof(int));
    memset(local_leaf_count, 0, NTopleaves * sizeof(int));

    int i;
    for(i = 0; i < NumPart; i++)
    {
        int leaf = ghost_toptree_leaf(Key[i]);
        local_leaf_count[leaf]++;
    }

    /* Step 2: Use per-leaf hmax from ExtNodes (already computed by force_update_hmax).
       A remote leaf is needed if the gap between its bounding box and any local leaf's
       bounding box is less than max(hmax_local_leaf, hmax_remote_leaf). This correctly
       handles the symmetric search r_ij < max(h_i, h_j) at the leaf level. */

    /* Gather per-leaf hmax values globally — each rank knows its own leaves' hmax,
       need to know remote leaves' hmax for the overlap check. */
    double *leaf_hmax = (double *) mymalloc("ghost_lhmax", NTopleaves * sizeof(double));
    memset(leaf_hmax, 0, NTopleaves * sizeof(double));
    int m;
    for(m = 0; m < MULTIPLEDOMAINS; m++)
    {
        int start = DomainStartList[ThisTask * MULTIPLEDOMAINS + m];
        int end = DomainEndList[ThisTask * MULTIPLEDOMAINS + m];
        if(start < 0 || end < start) continue;
        int ll;
        for(ll = start; ll <= end; ll++)
        {
            int ni = DomainNodeIndex[ll];
            if(ni >= 0) leaf_hmax[ll] = Extnodes[ni].hmax;
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, leaf_hmax, NTopleaves, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    /* Step 3: For each remote leaf, check overlap with local leaves using per-leaf hmax.
       search_radius = max(hmax_local_leaf, hmax_remote_leaf) for each pair. */

    int *need_leaf = (int *) mymalloc("ghost_nl", NTopleaves * sizeof(int));
    memset(need_leaf, 0, NTopleaves * sizeof(int));

    int leaf, local_leaf;
    for(leaf = 0; leaf < NTopleaves; leaf++)
    {
        if(DomainTask[leaf] == ThisTask) continue;
        int remote_node = DomainNodeIndex[leaf];
        if(remote_node < 0) continue;
        double hmax_remote = leaf_hmax[leaf];

        /* check overlap with each local leaf */
        for(m = 0; m < MULTIPLEDOMAINS; m++)
        {
            int start = DomainStartList[ThisTask * MULTIPLEDOMAINS + m];
            int end = DomainEndList[ThisTask * MULTIPLEDOMAINS + m];
            if(start < 0 || end < start) continue;
            for(local_leaf = start; local_leaf <= end; local_leaf++)
            {
                int local_node = DomainNodeIndex[local_leaf];
                if(local_node < 0) continue;

                double search_radius = DMAX(leaf_hmax[local_leaf], hmax_remote);
                if(search_radius <= 0) continue;

                /* min distance between two axis-aligned bounding boxes */
                double dist2 = 0;
                int k;
                for(k = 0; k < 3; k++)
                {
                    double d = fabs(Nodes[remote_node].center[k] - Nodes[local_node].center[k]);
#ifdef BOX_PERIODIC
                    double boxsize_k = (k==0) ? boxSize_X : ((k==1) ? boxSize_Y : boxSize_Z);
                    if(d > 0.5 * boxsize_k) d = boxsize_k - d;
#endif
                    d -= 0.5 * (Nodes[remote_node].len + Nodes[local_node].len);
                    if(d > 0) dist2 += d * d;
                }

                if(dist2 < search_radius * search_radius)
                {
                    need_leaf[leaf] = 1;
                    goto next_remote_leaf;
                }
            }
        }
        next_remote_leaf:;
    }
    /* leaf_hmax freed at end with other allocations (stack allocator requires reverse order) */

    /* Step 4: Exchange leaf particle counts globally so each rank knows how many
       particles are in each leaf (needed to compute recv counts). */
    int *global_leaf_count = (int *) mymalloc("ghost_glc", NTopleaves * sizeof(int));
    MPI_Allreduce(local_leaf_count, global_leaf_count, NTopleaves, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    /* Step 5: Compute send/recv counts per task.
       recv_count[t] = total particles we want from task t (sum over needed leaves owned by t)
       send_count[t] = total particles task t wants from us

       For send_count, we need to know which of OUR leaves each remote task requested.
       Exchange need_leaf arrays: each task broadcasts which leaves it needs, then each task
       can determine what to send. We use MPI_Allreduce(MPI_MAX) on need_leaf as a
       Phase 0 approximation: if ANY task needs a leaf, we send it to ALL requesters.
       This over-sends but guarantees correctness. Per-task routing deferred. */

    int *global_need_leaf = (int *) mymalloc("ghost_gnl", NTopleaves * sizeof(int));
    MPI_Allreduce(need_leaf, global_need_leaf, NTopleaves, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    /* recv_count[t] = sum of global_leaf_count[leaf] for leaves we need owned by task t */
    int *recv_count = (int *) mymalloc("ghost_rc", NTask * sizeof(int));
    memset(recv_count, 0, NTask * sizeof(int));
    for(leaf = 0; leaf < NTopleaves; leaf++)
    {
        if(need_leaf[leaf])
            recv_count[DomainTask[leaf]] += global_leaf_count[leaf];
    }

    /* send_count: how many of our particles are in globally-requested leaves, per requesting task.
       Phase 0: count our particles in any globally-needed leaf. Each requesting task gets the same set. */
    int n_boundary_particles = 0;
    for(i = 0; i < NumPart; i++)
    {
        int leaf_i = ghost_toptree_leaf(Key[i]);
        if(global_need_leaf[leaf_i] && DomainTask[leaf_i] == ThisTask)
            n_boundary_particles++;
    }

    /* Each task that wants particles from us gets n_boundary_particles.
       Use Alltoall to communicate: we send recv_count (what we want from each task),
       each task receives what others want from it. */
    int *send_count = (int *) mymalloc("ghost_sc", NTask * sizeof(int));
    MPI_Alltoall(recv_count, 1, MPI_INT, send_count, 1, MPI_INT, MPI_COMM_WORLD);

    /* Compute totals and displacements */
    int total_recv = 0, total_send = 0;
    int *recv_disp = (int *) mymalloc("ghost_rd", NTask * sizeof(int));
    int *send_disp = (int *) mymalloc("ghost_sd", NTask * sizeof(int));
    recv_disp[0] = 0; send_disp[0] = 0;
    int task;
    for(task = 0; task < NTask; task++)
    {
        total_recv += recv_count[task];
        total_send += send_count[task];
        if(task > 0)
        {
            recv_disp[task] = recv_disp[task-1] + recv_count[task-1];
            send_disp[task] = send_disp[task-1] + send_count[task-1];
        }
    }

    /* Check space */
    if(NumPart + total_recv > All.MaxPart)
    {
        if(ThisTask == 0)
            PRINT_WARNING("Ghost exchange: need %d ghost particles but only %d slots free (MaxPart=%d). Skipping.\n",
                          total_recv, All.MaxPart - NumPart, All.MaxPart);
        myfree(send_disp); myfree(recv_disp); myfree(send_count); myfree(recv_count);
        myfree(global_need_leaf); myfree(global_leaf_count); myfree(need_leaf); myfree(leaf_hmax); myfree(local_leaf_count);
        NumPart_before_ghost = -1;
        return;
    }

    /* Step 6: Pack particles to send.
       global_need_leaf (from Step 5) tells us which of our leaves are requested.
       send_count[t] (from Alltoall) tells us how many particles task t expects.

       Phase 0 approach: the same set of boundary particles goes to every requesting task.
       Pack them into the send buffer, replicated per destination task. */
    struct particle_data *send_P = (struct particle_data *) mymalloc("ghost_sP",
        total_send * sizeof(struct particle_data));
    struct gas_cell_data *send_CellP = (struct gas_cell_data *) mymalloc("ghost_sC",
        total_send * sizeof(struct gas_cell_data));

    int *task_offset = (int *) mymalloc("ghost_toff", NTask * sizeof(int));
    memcpy(task_offset, send_disp, NTask * sizeof(int));

    /* For each requesting task, pack our boundary particles */
    for(task = 0; task < NTask; task++)
    {
        if(send_count[task] <= 0) continue;
        for(i = 0; i < NumPart; i++)
        {
            int leaf_i = ghost_toptree_leaf(Key[i]);
            if(!global_need_leaf[leaf_i] || DomainTask[leaf_i] != ThisTask) continue;
            if(task_offset[task] >= send_disp[task] + send_count[task]) break;

            int idx = task_offset[task]++;
            send_P[idx] = P[i];
            if(P[i].Type == 0 && i < N_gas)
                send_CellP[idx] = CellP[i];
            else
                memset(&send_CellP[idx], 0, sizeof(struct gas_cell_data));
        }
    }

    /* Step 8: Exchange via MPI_Alltoallv.
       Send/recv full particle_data and gas_cell_data structs.
       Counts are in particles; MPI needs bytes. */

    /* Convert counts to bytes for MPI */
    int *recv_bytes = (int *) mymalloc("ghost_rb", NTask * sizeof(int));
    int *send_bytes = (int *) mymalloc("ghost_sb", NTask * sizeof(int));
    int *recv_bdisp = (int *) mymalloc("ghost_rbd", NTask * sizeof(int));
    int *send_bdisp = (int *) mymalloc("ghost_sbd", NTask * sizeof(int));
    for(task = 0; task < NTask; task++)
    {
        recv_bytes[task] = recv_count[task] * sizeof(struct particle_data);
        send_bytes[task] = send_count[task] * sizeof(struct particle_data);
        recv_bdisp[task] = recv_disp[task] * sizeof(struct particle_data);
        send_bdisp[task] = send_disp[task] * sizeof(struct particle_data);
    }

    /* Receive directly into P[] at ghost region */
    MPI_Alltoallv(send_P, send_bytes, send_bdisp, MPI_BYTE,
                  &P[NumPart], recv_bytes, recv_bdisp, MPI_BYTE, MPI_COMM_WORLD);

    /* Exchange CellP similarly */
    for(task = 0; task < NTask; task++)
    {
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
    /* N_gas unchanged — ghost gas is at indices >= NumPart_before_ghost, accessed via CellP[j] */

    if(ThisTask == 0)
        PRINT_STATUS("Ghost exchange: %d local + %d ghost = %d total particles",
                     NumPart_before_ghost, NumGhostParticles, NumPart);

    /* Cleanup temporaries (MUST be in reverse mymalloc order — stack allocator) */
    myfree(send_bdisp); myfree(recv_bdisp); myfree(send_bytes); myfree(recv_bytes);
    myfree(task_offset);
    myfree(send_CellP); myfree(send_P);
    myfree(send_disp); myfree(recv_disp);
    myfree(send_count); myfree(recv_count);
    myfree(global_need_leaf); myfree(global_leaf_count);
    myfree(need_leaf); myfree(leaf_hmax); myfree(local_leaf_count);
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
}
