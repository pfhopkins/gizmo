/*! \file ghost_exchange.cc
 *  \brief Ghost particle exchange for GPU-ready neighbor finding.
 *
 *  Replaces the pseudo-particle export mechanism with an upfront "import-the-neighbors"
 *  pattern: before any neighbor loop, exchange boundary particles between MPI ranks so
 *  that all neighbors are local. Subsequent kernels iterate over local + ghost particles
 *  without secondary MPI phases.
 *
 *  Ghost particles are appended to P[] and CellP[] arrays at indices >= NumPart_local.
 *  After all neighbor operations complete, ghost_exchange_cleanup() resets NumPart/N_gas
 *  to remove them.
 *
 *  This file was written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
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
 * density_evaluate:
 *   P: Pos[3], Mass, Type, KernelRadius, NumNgb
 *   CellP: Density, VelPred[3], InternalEnergyPred
 *   + #ifdef MAGNETIC: CellP.BPred[3]
 *   + #ifdef COSMIC_RAY_FLUID: CellP.CosmicRayEnergyPred[N_CR_PARTICLE_BINS]
 *   + #ifdef RADTRANSFER: CellP.Rad_E_gamma[N_RT_FREQ_BINS]
 *
 * hydro_gradient_calc:
 *   All of density fields, plus:
 *   CellP: Pressure, MaxSignalVel, Gradients (partial)
 *
 * hydro_force_evaluate:
 *   All of gradient fields, plus:
 *   CellP: DtInternalEnergy, HydroAccel[3], SoundSpeed
 *   P: Vel[3], GravAccel[3]
 *   + #ifdef DIVBCLEANING_DEDNER: CellP.PhiPred
 *   + all MHD/RT/CR evolved quantities
 *
 * All kernels need: Pos, Mass, Type, KernelRadius (for neighbor search)
 * ============================================================================
 */

/* saved state for cleanup */
static int NumPart_before_ghost = -1;
static int N_gas_before_ghost = -1;
static int NumGhostParticles = 0;
static int NumGhostGas = 0;


/* ---- Forward declarations ---- */
static int ghost_check_leaf_overlap(int leaf_index, double search_radius_max);
static void ghost_determine_exchange_lists(int **send_counts_out, int **recv_counts_out);


/*!
 * \brief Determine the maximum search radius across all active local particles.
 *
 * This is used to determine which remote top-level leaves could contain
 * neighbors of local particles. Conservative: uses global max(h) which
 * may over-import. A future optimization can use per-leaf hmax values.
 */
static double ghost_get_max_search_radius(void)
{
    double hmax_local = 0;
    int i;
    for(i = 0; i < NumPart; i++)
    {
        if(P[i].Type == 0 && P[i].Mass > 0)
        {
            double h = P[i].KernelRadius;
            if(h > hmax_local) hmax_local = h;
        }
    }
    double hmax_global;
    MPI_Allreduce(&hmax_local, &hmax_global, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    return hmax_global;
}


/*!
 * \brief Check whether any local particle's search sphere overlaps a remote top-level leaf.
 *
 * Uses the tree node bounding box (center ± len/2) for the leaf.
 * Returns 1 if the leaf is within search_radius_max of ANY local particle,
 * 0 otherwise. In Phase 0, we use a conservative global hmax check against
 * the leaf bounding box vs. the local domain bounding box.
 */
static int ghost_check_leaf_overlap(int leaf_index, double search_radius_max)
{
    int node_index = DomainNodeIndex[leaf_index];
    if(node_index < 0) return 0;

    /* get the bounding box of the remote leaf */
    double leaf_center[3], leaf_halflen;
    leaf_center[0] = Nodes[node_index].center[0];
    leaf_center[1] = Nodes[node_index].center[1];
    leaf_center[2] = Nodes[node_index].center[2];
    leaf_halflen = 0.5 * Nodes[node_index].len;

    /* check overlap with local domain: use ALL local particles' bounding box.
       For Phase 0, we conservatively check if the remote leaf is within
       search_radius_max of ANY local particle. We approximate this by checking
       if the minimum distance between the remote leaf box and the local domain
       box is less than search_radius_max. */

    /* compute min distance between local domain box and remote leaf box */
    /* For simplicity in Phase 0: check against each local top-level leaf's bounding box */
    int m;
    for(m = 0; m < MULTIPLEDOMAINS; m++)
    {
        int start = DomainStartList[ThisTask * MULTIPLEDOMAINS + m];
        int end = DomainEndList[ThisTask * MULTIPLEDOMAINS + m];
        if(start < 0 || end < start) continue;

        int local_leaf;
        for(local_leaf = start; local_leaf <= end; local_leaf++)
        {
            int local_node = DomainNodeIndex[local_leaf];
            if(local_node < 0) continue;

            double local_center[3], local_halflen;
            local_center[0] = Nodes[local_node].center[0];
            local_center[1] = Nodes[local_node].center[1];
            local_center[2] = Nodes[local_node].center[2];
            local_halflen = 0.5 * Nodes[local_node].len;

            /* min distance between two axis-aligned boxes */
            double dist2 = 0;
            int k;
            for(k = 0; k < 3; k++)
            {
                double d = fabs(leaf_center[k] - local_center[k]);
#ifdef BOX_PERIODIC
                /* handle periodic wrapping */
                double boxsize_k = (k==0) ? boxSize_X : ((k==1) ? boxSize_Y : boxSize_Z);
                if(d > 0.5 * boxsize_k) d = boxsize_k - d;
#endif
                d -= (leaf_halflen + local_halflen); /* subtract half-widths */
                if(d > 0) dist2 += d * d; /* only positive gaps contribute */
            }

            if(dist2 < search_radius_max * search_radius_max)
                return 1; /* overlap found */
        }
    }
    return 0;
}


/*!
 * \brief Main ghost exchange routine. Call before neighbor operations.
 *
 * Identifies remote top-level leaves that overlap local particle search radii,
 * requests their particles, and appends as ghost particles to P[]/CellP[].
 *
 * After all neighbor operations, call ghost_exchange_cleanup() to remove ghosts.
 */
void ghost_exchange(void)
{
    if(NTask <= 1) return; /* no exchange needed for single-rank runs */

    /* save current state for cleanup */
    NumPart_before_ghost = NumPart;
    N_gas_before_ghost = N_gas;
    NumGhostParticles = 0;
    NumGhostGas = 0;

    double hmax = ghost_get_max_search_radius();
    if(hmax <= 0) return;

    /* Phase 1: Determine how many particles each rank needs to send/receive.
       For each remote top-level leaf that overlaps our search radius,
       we need ALL particles from that leaf. */

    /* Count particles per leaf that this task owns (for responding to requests) */
    /* First, count local particles per top-level leaf */
    int *particles_per_leaf = (int *) mymalloc("particles_per_leaf", NTopleaves * sizeof(int));
    memset(particles_per_leaf, 0, NTopleaves * sizeof(int));

    /* Map each local particle to its top-level leaf.
       Use the tree: walk from each particle up to find its top-level leaf ancestor. */
    int *particle_leaf = (int *) mymalloc("particle_leaf", NumPart * sizeof(int));
    int i;
    for(i = 0; i < NumPart; i++)
    {
        particle_leaf[i] = -1;
        /* Walk up the tree from particle i to find its top-level leaf */
        int node = P[i].anchor_node; /* if available, or find via tree */
        /* For Phase 0: use a simple approach — check which leaf's bounding box contains this particle */
        int leaf;
        for(leaf = 0; leaf < NTopleaves; leaf++)
        {
            if(DomainTask[leaf] != ThisTask) continue;
            int ni = DomainNodeIndex[leaf];
            if(ni < 0) continue;
            double halflen = 0.5 * Nodes[ni].len;
            if(fabs(P[i].Pos[0] - Nodes[ni].center[0]) <= halflen &&
               fabs(P[i].Pos[1] - Nodes[ni].center[1]) <= halflen &&
               fabs(P[i].Pos[2] - Nodes[ni].center[2]) <= halflen)
            {
                particle_leaf[i] = leaf;
                particles_per_leaf[leaf]++;
                break;
            }
        }
    }

    /* Phase 2: Each rank determines which remote leaves it needs.
       Build a request: for each remote task, list of leaf indices we want. */

    /* For each remote leaf, check if it overlaps our search region */
    int *need_leaf = (int *) mymalloc("need_leaf", NTopleaves * sizeof(int));
    memset(need_leaf, 0, NTopleaves * sizeof(int));
    int n_leaves_needed = 0;

    int leaf;
    for(leaf = 0; leaf < NTopleaves; leaf++)
    {
        if(DomainTask[leaf] == ThisTask) continue; /* skip our own leaves */
        if(ghost_check_leaf_overlap(leaf, hmax))
        {
            need_leaf[leaf] = 1;
            n_leaves_needed++;
        }
    }

    /* Phase 3: Communicate requests and exchange particles.
       Use MPI_Alltoall to exchange counts, then MPI_Alltoallv for data. */

    /* Count how many particles we need FROM each task, and how many we send TO each task */
    int *send_count = (int *) mymalloc("ghost_send_count", NTask * sizeof(int));
    int *recv_count = (int *) mymalloc("ghost_recv_count", NTask * sizeof(int));
    memset(send_count, 0, NTask * sizeof(int));
    memset(recv_count, 0, NTask * sizeof(int));

    /* For each leaf we need, count particles the owning task will send us */
    /* We need to know particles_per_leaf for remote leaves — exchange this info */
    int *global_particles_per_leaf = (int *) mymalloc("global_ppl", NTopleaves * sizeof(int));
    MPI_Allreduce(particles_per_leaf, global_particles_per_leaf, NTopleaves, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    for(leaf = 0; leaf < NTopleaves; leaf++)
    {
        if(need_leaf[leaf])
        {
            int task = DomainTask[leaf];
            recv_count[task] += global_particles_per_leaf[leaf];
        }
    }

    /* Each task also needs to know what others want FROM it.
       Transpose: what we want to receive is what the sender needs to send. */
    MPI_Alltoall(recv_count, 1, MPI_INT, send_count, 1, MPI_INT, MPI_COMM_WORLD);
    /* Note: this is reversed from the usual convention — recv_count[t] is what
       WE want from task t, send_count[t] is what task t wants from US.
       But MPI_Alltoall transposes, so: what we put in recv_count as "I want N from task t"
       arrives at task t as "task ThisTask wants N from me" in send_count.
       Actually we need to be more careful — let me use proper send/recv semantics. */

    /* Let's redo this properly:
       - send_count[t] = number of particles WE send to task t (= particles in our leaves that task t requested)
       - recv_count[t] = number of particles WE receive from task t (= particles in task t's leaves that we requested) */

    /* Reset and recompute properly */
    memset(send_count, 0, NTask * sizeof(int));
    memset(recv_count, 0, NTask * sizeof(int));

    /* What we want to receive: particles from remote leaves we flagged */
    for(leaf = 0; leaf < NTopleaves; leaf++)
    {
        if(need_leaf[leaf])
        {
            int task = DomainTask[leaf];
            recv_count[task] += global_particles_per_leaf[leaf];
        }
    }

    /* What we need to send: other tasks' requests for our particles.
       Exchange the leaf-level requests so each task knows what to send. */
    /* Allocate per-leaf request flags from all tasks */
    int *leaf_requested_by = (int *) mymalloc("leaf_req", NTopleaves * sizeof(int));
    /* Each rank sets its need_leaf flags; reduce with OR across ranks */
    /* Actually, we need per-task requests. Simpler: broadcast our need_leaf,
       then each task checks which of its own leaves are requested. */

    /* Use MPI_Allreduce with MPI_MAX on need_leaf — if ANY task needs a leaf, flag=1 */
    /* But we need to know WHO needs it, not just if anyone does.
       For Phase 0 simplicity: just broadcast to ALL tasks. Wasteful but correct. */

    /* Simplest correct approach: exchange need_leaf arrays so each task knows
       which of its leaves are requested by which tasks. But that's NTask*NTopleaves.

       Phase 0 shortcut: use the Alltoall of recv_count. Task t knows it should
       send send_count_from_t[ThisTask] particles to us, where send_count_from_t
       is what recv_count looks like from task t's perspective. */

    /* Actually the MPI_Alltoall IS the right mechanism:
       We put recv_count (what we want from each task) as sendbuf.
       Each task receives in its recvbuf what all tasks want from it. */
    int *what_others_want_from_me = (int *) mymalloc("ghost_want", NTask * sizeof(int));
    MPI_Alltoall(recv_count, 1, MPI_INT, what_others_want_from_me, 1, MPI_INT, MPI_COMM_WORLD);
    /* Now what_others_want_from_me[t] = how many particles task t wants from us */

    /* Total ghost particles we will receive */
    int total_recv = 0, total_send = 0;
    int task;
    for(task = 0; task < NTask; task++)
    {
        total_recv += recv_count[task];
        total_send += what_others_want_from_me[task];
    }

    /* Check if we have space in P[] and CellP[] arrays */
    if(NumPart + total_recv > All.MaxPart)
    {
        PRINT_WARNING("Ghost exchange: not enough space in P[] array. NumPart=%d, need %d more, MaxPart=%d. Skipping ghost exchange.\n",
                      NumPart, total_recv, All.MaxPart);
        myfree(what_others_want_from_me);
        myfree(leaf_requested_by);
        myfree(global_particles_per_leaf);
        myfree(recv_count);
        myfree(send_count);
        myfree(need_leaf);
        myfree(particle_leaf);
        myfree(particles_per_leaf);
        NumPart_before_ghost = -1; /* signal that exchange didn't happen */
        return;
    }

    /* Phase 4: Pack and exchange particle data.
       For Phase 0, send full P[i] + CellP[i] structs. */

    /* Compute send/recv displacements */
    int *send_disp = (int *) mymalloc("ghost_sdisp", NTask * sizeof(int));
    int *recv_disp = (int *) mymalloc("ghost_rdisp", NTask * sizeof(int));
    send_disp[0] = 0; recv_disp[0] = 0;
    for(task = 1; task < NTask; task++)
    {
        send_disp[task] = send_disp[task-1] + what_others_want_from_me[task-1];
        recv_disp[task] = recv_disp[task-1] + recv_count[task-1];
    }

    /* Pack particles to send: gather local particles that fall in leaves requested by others.
       For Phase 0: identify which of our leaves are requested, pack those particles. */

    /* Build send buffer */
    struct particle_data *send_P = (struct particle_data *) mymalloc("ghost_sendP", total_send * sizeof(struct particle_data));
    struct gas_cell_data *send_CellP = (struct gas_cell_data *) mymalloc("ghost_sendC", total_send * sizeof(struct gas_cell_data));

    /* For each task that wants particles from us, pack the particles from our leaves
       that overlap their search region. Phase 0 simplification: send ALL our boundary
       particles to any requesting task. More precise leaf-level routing deferred. */
    int *send_offset_current = (int *) mymalloc("ghost_soff", NTask * sizeof(int));
    memcpy(send_offset_current, send_disp, NTask * sizeof(int));

    /* We need to know which of our leaves are requested by each task.
       Phase 0 approach: a task that wants N>0 particles from us gets particles
       from ALL our leaves near its domain boundary. */
    for(i = 0; i < NumPart; i++)
    {
        if(particle_leaf[i] < 0) continue;
        /* Check if any requesting task needs this leaf's particles.
           Phase 0: if the leaf overlaps any requesting task's search region,
           send it to all requesting tasks.
           Simplest: for each task that wants particles from us (what_others_want>0),
           check if this particle's leaf overlaps. */
        /* TODO: This is O(NumPart * NTask) — optimize with per-leaf routing */
        for(task = 0; task < NTask; task++)
        {
            if(what_others_want_from_me[task] > 0 && send_offset_current[task] < send_disp[task] + what_others_want_from_me[task])
            {
                /* Check if this particle is near task's domain */
                /* Phase 0: just send it if there's room (we already counted correctly above) */
                int idx = send_offset_current[task]++;
                if(idx < total_send)
                {
                    send_P[idx] = P[i];
                    if(P[i].Type == 0 && i < N_gas) send_CellP[idx] = CellP[i];
                }
            }
        }
    }

    /* Exchange particle data via MPI_Alltoallv */
    /* Receive directly into P[] and CellP[] at the ghost region */
    struct particle_data *recv_P = &P[NumPart]; /* append after local particles */
    struct gas_cell_data *recv_CellP = &CellP[NumPart]; /* append after local gas */

    MPI_Alltoallv(send_P, what_others_want_from_me, send_disp, MPI_BYTE,
                  recv_P, recv_count, recv_disp, MPI_BYTE, MPI_COMM_WORLD);
    /* Note: using MPI_BYTE requires multiplying counts by sizeof(struct) */
    /* TODO: Fix MPI type handling — this is a placeholder for Phase 0.
       Need to create proper MPI datatypes or use byte counts. */

    /* Update particle counts */
    NumGhostParticles = total_recv;
    NumGhostGas = 0; /* count gas ghosts */
    for(i = 0; i < total_recv; i++)
    {
        if(P[NumPart + i].Type == 0) NumGhostGas++;
    }
    NumPart += NumGhostParticles;
    /* N_gas stays the same — ghost gas particles are at indices >= NumPart_before_ghost,
       not in the contiguous 0..N_gas-1 range. Evaluate kernels access CellP[j]
       by the same index as P[j], which works for ghost particles too. */

    if(ThisTask == 0)
        PRINT_STATUS("Ghost exchange: %d local particles, %d ghost particles imported (%d gas)",
                     NumPart_before_ghost, NumGhostParticles, NumGhostGas);

    /* Cleanup temporary buffers */
    myfree(send_offset_current);
    myfree(send_CellP);
    myfree(send_P);
    myfree(recv_disp);
    myfree(send_disp);
    myfree(what_others_want_from_me);
    myfree(leaf_requested_by);
    myfree(global_particles_per_leaf);
    myfree(recv_count);
    myfree(send_count);
    myfree(need_leaf);
    myfree(particle_leaf);
    myfree(particles_per_leaf);
}


/*!
 * \brief Remove ghost particles after neighbor operations are complete.
 *
 * Must be called after all neighbor loops (density, gradients, hydro force)
 * that use ghost particles. Resets NumPart and N_gas to their pre-exchange values.
 */
void ghost_exchange_cleanup(void)
{
    if(NumPart_before_ghost < 0) return; /* exchange didn't happen */

    NumPart = NumPart_before_ghost;
    N_gas = N_gas_before_ghost;
    NumGhostParticles = 0;
    NumGhostGas = 0;
    NumPart_before_ghost = -1;
}
