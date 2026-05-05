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
#include "../system/mpi_alltoallv_typed.h"
#include "../core/step_phases.h"   /* gizmo_verbose_diag() */
#include "gpu_neighbor_list.h" /* gpu_compact_xyzh_mark_h_dirty_range */

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
static int PreviousGhostCount = 0; /* ghost count from the most recent completed exchange, for domain headroom */

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
/* Single source of truth: every caller fills a ghost_exchange_spec_t and
 * calls ghost_exchange_impl(spec). Behavior parameterized by:
 *   request_type_mask : bitmask of P[].Type values whose actives drive
 *                       imports. (1<<0)=gas, (1<<5)=sink, etc.
 *                       (1<<6)-1 = all types.
 *   supply_type_mask  : bitmask of P[].Type values that may fill the
 *                       candidate pool (and contribute to tile hmax).
 *   search_mode       : NGB_SEARCH_ONEWAY (r_ij < h_i; correct for density)
 *                       or NGB_SEARCH_SYMMETRIC (r_ij < max(h_i, h_j);
 *                       gradients, hydro_force, sinks).
 *   safety_factor     : multiplier on search radius.
 * One core, branched only where flags actually matter (pool filter, active
 * filter, search_r formula). Adding a new caller = one wrapper line; new
 * logic dimension = one spec field + one branch in core. Matches the legacy
 * ngb_treefind_* design in the old GIZMO. */
struct ghost_exchange_spec_t {
    unsigned int request_type_mask;
    unsigned int supply_type_mask;
    int          search_mode;
    double       safety_factor;
    const char  *caller_name;
};
#define GHOST_TYPE_GAS   (1u << 0)
#define GHOST_TYPE_DM    (1u << 1)
#define GHOST_TYPE_DISK  (1u << 2)
#define GHOST_TYPE_BULGE (1u << 3)
#define GHOST_TYPE_STAR  (1u << 4)
#define GHOST_TYPE_SINK  (1u << 5)
#define GHOST_TYPE_ALL   ((1u << 6) - 1u)

static inline int ghost_type_passes(int ptype, unsigned int mask) { return (mask & (1u << (unsigned)ptype)) != 0u; }

/* Forward decls. */
static void ghost_exchange_request_driven_impl(const struct ghost_exchange_spec_t *spec);
static void ghost_exchange_tile_overlap_impl(const struct ghost_exchange_spec_t *spec);
static void gx_print_waste(const struct ghost_exchange_spec_t *spec, int this_call, int total_recv);
static double gx_eff_h(int j, unsigned int supply_mask);

/* Env gate: GIZMO_GHOST_REQUEST_DRIVEN=1 selects the per-active query-driven
 * exchange (Phase 2 design). Default 0 retains the legacy tile-overlap path
 * during validation. Once Phase 2 validates against R1 baseline waste
 * numbers, this becomes the only path and the gate goes away. */
static int ghost_request_driven_enabled(void)
{
    static int cached = -1;
    if(cached < 0) {
        const char *e = getenv("GIZMO_GHOST_REQUEST_DRIVEN");
        cached = (e && e[0] && e[0] != '0') ? 1 : 0;
    }
    return cached;
}

static void ghost_exchange_impl(const struct ghost_exchange_spec_t *spec)
{
    if(ghost_request_driven_enabled()) {
        ghost_exchange_request_driven_impl(spec);
    } else {
        ghost_exchange_tile_overlap_impl(spec);
    }
}

static void ghost_exchange_tile_overlap_impl(const struct ghost_exchange_spec_t *spec)
{
    const double safety_factor = spec->safety_factor;
    const unsigned int request_mask = spec->request_type_mask;
    const unsigned int supply_mask  = spec->supply_type_mask;
    const int  search_mode = spec->search_mode;
    if(NTask <= 1) return;
    double t_ghost_start = my_second(), t_ghost_phase;

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

    /* Count pool particles. supply_mask gates which Types may join the pool;
     * removes tile-hmax pollution from non-supply types (e.g. DM init-time
     * KernelRadius poisoning hydro tile bboxes). */
    for(i = 0; i < NumPart; i++) {
        if(P[i].Mass <= 0) continue;
        if(!ghost_type_passes((int)P[i].Type, supply_mask)) continue;
        num_pool++;
    }

    /* Build pool index array */
    int *pool = (int *) malloc((num_pool > 0 ? num_pool : 1) * sizeof(int));
    int p = 0;
    for(i = 0; i < NumPart; i++) {
        if(P[i].Mass <= 0) continue;
        if(!ghost_type_passes((int)P[i].Type, supply_mask)) continue;
        pool[p++] = i;
    }

    local_ntiles = (num_pool + tile_target - 1) / tile_target;
    if(local_ntiles < 1) local_ntiles = 1;

    /* Compact tile metadata for exchange. Two parallel bbox/hmax sets:
     *   (lo, hi, hmax, count)         — over ALL particles in tile
     *   (active_lo, active_hi, active_hmax, active_count) — over ACTIVE only
     *
     * The all-particle set governs what this rank can SUPPLY as ghosts (j may
     * be inactive on its home rank but still a neighbor of an active i on a
     * peer rank — WAKEUP-style semantics demand this).
     *
     * The active-only set governs what this rank actually NEEDS: only tiles
     * containing at least one active particle drive remote-tile imports.
     *
     * Old code (pre-tile, tree-based ngb_treefind_variable_threads) iterated
     * FirstActiveParticle and built per-rank exports off active i's search
     * radius — the new tile path silently dropped that gating and was
     * shipping ~all of the global pool on tiny-N steps. */
    struct tile_meta_t {
        double lo[3], hi[3], hmax;
        int count;
        double active_lo[3], active_hi[3], active_hmax;
        int active_count;
    };

    tile_meta_t *local_meta = (tile_meta_t *) malloc(local_ntiles * sizeof(tile_meta_t));
    int *tile_first = (int *) malloc(local_ntiles * sizeof(int)); /* index into pool[] */

    /* Per-particle effective search radius: the largest kernel any enabled loop
       will use when centred on this particle. Gas KernelRadius is always
       included; additional fields are pulled in under their compile flags so
       ghost bboxes cover the widest search any active kernel needs (e.g. DM
       dispersion's KernelRadiusDM, adaptive-gravsoft's AGS_KernelRadius). */
    auto effective_ghost_radius = [supply_mask](int j) -> double {
        /* For typed callers (any non-all-types mask), use the simple
         * KernelRadius — matches the legacy ngb search and avoids conflating
         * AGS / DM / wind kernels into the hydro/sink/feedback radius. The
         * all-types path retains the original conditional fan-in. */
        if(supply_mask != GHOST_TYPE_ALL) {
            if(!ghost_type_passes((int)P[j].Type, supply_mask)) return 0.0;
            return (double)P[j].KernelRadius;
        }
        double h = P[j].KernelRadius;
#if defined(GALSF_SUBGRID_WINDS) && (GALSF_SUBGRID_WIND_SCALING==2)
        if(P[j].Type == 0 && j < N_gas) {
            if((double)CellP[j].KernelRadiusDM > h) h = CellP[j].KernelRadiusDM;
        }
#endif
#ifdef AGS_KERNELRADIUS_CALCULATION_IS_ACTIVE
        if((double)P[j].AGS_KernelRadius > h) h = P[j].AGS_KernelRadius;
#endif
        return h;
    };

    /* Per-particle active flag — ActiveParticleList lookup at O(1). The active
     * stats below restrict the tile's "what do I need from peers?" criterion
     * to particles that will actually walk neighbors this step. Allocate as
     * char rather than bool to keep the address-stable contract. */
    char *is_active = (char *) calloc(NumPart > 0 ? NumPart : 1, sizeof(char));
    for(size_t kk = 0; kk < ActiveParticleList.size(); kk++) {
        int i_act = ActiveParticleList[kk];
        if(i_act < 0 || i_act >= NumPart) continue;
        /* request_mask gates which types' actives drive imports. */
        if(!ghost_type_passes((int)P[i_act].Type, request_mask)) continue;
        is_active[i_act] = 1;
    }

    for(int t = 0; t < local_ntiles; t++)
    {
        int start = t * tile_target;
        int count = tile_target;
        if(start + count > num_pool) count = num_pool - start;
        tile_first[t] = start;
        local_meta[t].count = count;
        local_meta[t].hmax = 0;
        local_meta[t].active_count = 0;
        local_meta[t].active_hmax = 0;
        for(k = 0; k < 3; k++) {
            local_meta[t].active_lo[k] = 0;  /* will be overwritten on first active */
            local_meta[t].active_hi[k] = 0;
        }

        int j0 = pool[start];
        for(k = 0; k < 3; k++) local_meta[t].lo[k] = local_meta[t].hi[k] = P[j0].Pos[k];
        {double h0 = effective_ghost_radius(j0); if(h0 > local_meta[t].hmax) local_meta[t].hmax = h0;}
        if(is_active[j0]) {
            local_meta[t].active_count = 1;
            for(k = 0; k < 3; k++) local_meta[t].active_lo[k] = local_meta[t].active_hi[k] = P[j0].Pos[k];
            double h0 = effective_ghost_radius(j0);
            local_meta[t].active_hmax = h0;
        }

        for(int s = 1; s < count; s++) {
            int j = pool[start + s];
            for(k = 0; k < 3; k++) {
                if(P[j].Pos[k] < local_meta[t].lo[k]) local_meta[t].lo[k] = P[j].Pos[k];
                if(P[j].Pos[k] > local_meta[t].hi[k]) local_meta[t].hi[k] = P[j].Pos[k];
            }
            double hj = effective_ghost_radius(j);
            if(hj > local_meta[t].hmax) local_meta[t].hmax = hj;
            if(is_active[j]) {
                if(local_meta[t].active_count == 0) {
                    for(k = 0; k < 3; k++) local_meta[t].active_lo[k] = local_meta[t].active_hi[k] = P[j].Pos[k];
                    local_meta[t].active_hmax = hj;
                } else {
                    for(k = 0; k < 3; k++) {
                        if(P[j].Pos[k] < local_meta[t].active_lo[k]) local_meta[t].active_lo[k] = P[j].Pos[k];
                        if(P[j].Pos[k] > local_meta[t].active_hi[k]) local_meta[t].active_hi[k] = P[j].Pos[k];
                    }
                    if(hj > local_meta[t].active_hmax) local_meta[t].active_hmax = hj;
                }
                local_meta[t].active_count++;
            }
        }
    }
    free(is_active);

    /* Diagnostic: identify top-N tile-hmax outliers and what's setting them.
     * Gated on GIZMO_VERBOSE_DIAG=1 + first 3 calls to keep output manageable.
     * For each top tile we walk its particles to find the one(s) with the
     * largest effective_ghost_radius and dump Type/Mass/h/Pos/ID/TimeBin.
     * Goal: settle whether 582-scale tile hmax comes from DM init values,
     * sink kernel inflation, stale gas, or a real outlier. */
    {
        static int gx_topn_calls = 0;
        if(gizmo_verbose_diag() && gx_topn_calls < 3 && local_ntiles > 0) {
            const int TOPN = 8;
            int top_idx[TOPN]; double top_h[TOPN];
            for(int q = 0; q < TOPN; q++) { top_idx[q] = -1; top_h[q] = -1.0; }
            for(int t = 0; t < local_ntiles; t++) {
                double h = local_meta[t].hmax;
                int slot = -1;
                for(int q = 0; q < TOPN; q++) { if(h > top_h[q]) { slot = q; break; } }
                if(slot >= 0) {
                    for(int q = TOPN - 1; q > slot; q--) { top_h[q] = top_h[q-1]; top_idx[q] = top_idx[q-1]; }
                    top_h[slot] = h; top_idx[slot] = t;
                }
            }
            for(int q = 0; q < TOPN && top_idx[q] >= 0; q++) {
                int t = top_idx[q];
                int start = tile_first[t];
                int n = local_meta[t].count;
                int worst_j = -1; double worst_h = -1.0;
                for(int s = 0; s < n; s++) {
                    int j = pool[start + s];
                    double hj = effective_ghost_radius(j);
                    if(hj > worst_h) { worst_h = hj; worst_j = j; }
                }
                if(worst_j >= 0) {
                    printf("[GX_TOPHMAX rank=%d call=%d t=%d tile_hmax=%.6g particle: idx=%d ID=%llu Type=%d Mass=%.4g KernelRadius=%.6g Pos=(%.4g,%.4g,%.4g) TimeBin=%d active_count=%d tile_count=%d]\n",
                           ThisTask, gx_topn_calls, t, top_h[q],
                           worst_j, (unsigned long long)P[worst_j].ID, (int)P[worst_j].Type,
                           (double)P[worst_j].Mass, (double)P[worst_j].KernelRadius,
                           (double)P[worst_j].Pos[0], (double)P[worst_j].Pos[1], (double)P[worst_j].Pos[2],
                           (int)P[worst_j].TimeBin, local_meta[t].active_count, local_meta[t].count);
                }
            }
            fflush(stdout);
            gx_topn_calls++;
        }
    }

    /* Save per-tile hmax for h-growth detection */
    if(saved_leaf_hmax) {free(saved_leaf_hmax); saved_leaf_hmax = NULL;}
    saved_leaf_hmax_n = local_ntiles;
    saved_leaf_hmax = (double *) malloc(local_ntiles * sizeof(double));
    for(int t = 0; t < local_ntiles; t++) saved_leaf_hmax[t] = local_meta[t].hmax;

    double t_ghost_tiles = timediff(t_ghost_start, my_second());

    /* ================================================================
       Step 2: Gather tile metadata from all ranks.
       Each rank sends its tile count and metadata to all ranks.
       ================================================================ */
    t_ghost_phase = my_second();
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

    double t_ghost_meta = timediff(t_ghost_phase, my_second());
    double t_phase_overlap_start = my_second();

    /* Diagnostic: number this ghost_exchange call to track progress in multi-call steps */
    static int ghost_call_seq = 0;
    ghost_call_seq++;
    int this_call = ghost_call_seq;
    if(gizmo_verbose_diag()) {
        printf("[GX rank=%d call=%d] after_allgatherv: local_ntiles=%d total_tiles=%d NumPart=%d\n",
               ThisTask, this_call, local_ntiles, total_tiles, NumPart);
        fflush(stdout);
    }

    /* ================================================================
       Step 3: Per-task tile overlap check.
       For each remote task, check which of its tiles overlap with any
       of our tiles AND which of our tiles overlap with any of its tiles.
       No global Allreduce — each rank independently computes per-task
       recv and send lists using the symmetric overlap criterion.
       ================================================================ */

    /* Per-tile need flags: need_from[total_tiles] = do WE need this remote tile?
       send_to[local_ntiles * NTask] = does task t need our tile lt?
     *
     * Asymmetric criterion:
     *   need_from[rt]            = our active tiles' bbox vs rt's all-bbox
     *   send_to[lt][task]        = task's active tiles' bbox vs our lt's all-bbox
     * Active-aware on each rank's REQUEST side, all-particle on the SUPPLY
     * side (a remote inactive j may still be a neighbor of an active i — see
     * WAKEUP). Tile pairs where the request side has zero actives early-exit
     * — that's where the tiny-N wins come from. */
    int *need_from = (int *) calloc(total_tiles, sizeof(int));
    int *send_to = (int *) calloc(local_ntiles * NTask, sizeof(int));
    int my_tile_start = tile_disp[ThisTask];

    /* Per-axis min-AABB-AABB squared distance under periodic wrap. Returns
     * negative gap on this axis if the AABBs overlap. Inlined for hot loop. */
    auto axis_gap = [&](double c_a, double hw_a, double c_b, double hw_b, int kk) -> double {
#if defined(BOX_PERIODIC)
        int is_periodic = 1;
        double bsize = (kk==0) ? boxSize_X : ((kk==1) ? boxSize_Y : boxSize_Z);
#if defined(BOX_REFLECT_X)
        if(kk==0) is_periodic = 0;
#endif
#if defined(BOX_REFLECT_Y)
        if(kk==1) is_periodic = 0;
#endif
#if defined(BOX_REFLECT_Z)
        if(kk==2) is_periodic = 0;
#endif
#if defined(BOX_OUTFLOW_X)
        if(kk==0) is_periodic = 0;
#endif
#if defined(BOX_OUTFLOW_Y)
        if(kk==1) is_periodic = 0;
#endif
#if defined(BOX_OUTFLOW_Z)
        if(kk==2) is_periodic = 0;
#endif
#else
        int is_periodic = 0;
        double bsize = 0;
#endif
        double dx = fabs(c_a - c_b);
        if(is_periodic && dx > 0.5 * bsize) dx = bsize - dx;
        return dx - hw_a - hw_b;
    };

    /* Pass 1: need_from[rt] — driven by OUR active tiles. Outer loop over
     * local tiles with active_count > 0 only (tiny-N: just a handful). */
    for(int lt_idx = 0; lt_idx < local_ntiles; lt_idx++)
    {
        int lt = my_tile_start + lt_idx;
        tile_meta_t *lm = &all_meta[lt];
        if(lm->active_count == 0) continue;
        double c_lo[3], c_hw[3];
        for(k = 0; k < 3; k++) {
            c_lo[k] = 0.5 * (lm->active_lo[k] + lm->active_hi[k]);
            c_hw[k] = 0.5 * (lm->active_hi[k] - lm->active_lo[k]);
        }
        for(task = 0; task < NTask; task++)
        {
            if(task == ThisTask) continue;
            int t_start = tile_disp[task];
            int t_count = all_ntiles[task];
            for(int rt_idx = 0; rt_idx < t_count; rt_idx++)
            {
                int rt = t_start + rt_idx;
                if(need_from[rt]) continue;            /* already flagged by another lt */
                tile_meta_t *rm = &all_meta[rt];
                /* Search radius depends on caller mode:
                 *   ONEWAY (density): r_ij < h_i — only the LOCAL active's h
                 *     matters. Importing a remote tile because of its own
                 *     particles' large h_j is incorrect for density.
                 *   SYMMETRIC (gradients/sinks/etc.): r_ij < max(h_i, h_j) —
                 *     remote h_j matters because j's kernel may reach back
                 *     to i. */
                double search_r = (search_mode == NGB_SEARCH_ONEWAY)
                                    ? lm->active_hmax * safety_factor
                                    : DMAX(lm->active_hmax, rm->hmax) * safety_factor;
                if(search_r <= 0) continue;
                double search_r2 = search_r * search_r;
                double dist2 = 0;
                int overlaps = 1;
                for(k = 0; k < 3; k++) {
                    double c_r = 0.5 * (rm->lo[k] + rm->hi[k]);
                    double hw_r = 0.5 * (rm->hi[k] - rm->lo[k]);
                    double gap = axis_gap(c_lo[k], c_hw[k], c_r, hw_r, k);
                    if(gap <= 0) continue;
                    if(gap > search_r) { overlaps = 0; break; }
                    dist2 += gap * gap;
                }
                if(overlaps && dist2 < search_r2) need_from[rt] = 1;
            }
        }
    }

    /* Pass 2: derive send_to[lt][task] from peers' need_from directly.
     *
     * Earlier versions recomputed an "active-on-the-other-side" overlap test
     * here. That's algebraically the same predicate as peer's pass 1 — but
     * the two ranks running floating-point min-distance arithmetic on the
     * same all_meta blob can disagree on a tile pair that lies right at the
     * search-radius threshold. When they do, A says send_count[B]=N and B
     * says recv_count[A]=N±1, and the downstream Alltoallv truncates with
     * MPI_ERR_TRUNCATE. Killing the recompute entirely makes the two sides
     * bit-identical by construction.
     *
     * Cost of the Allgather: total_tiles*NTask ints (~1.5 MB at 2 ranks ×
     * 200k tiles, scales as O(NTask^2) — switch to Alltoall of per-rank
     * slices if NTask grows past ~100). The pass-1 cost is unchanged: outer
     * loop active-gated, ~handful of tiles on tiny-N. */
    int *all_need_from = (int *) malloc((size_t)NTask * total_tiles * sizeof(int));
    if(gizmo_verbose_diag()) {
        printf("[GX rank=%d call=%d] BEFORE Allgather(need_from): total_tiles=%d NTask=%d\n",
               ThisTask, this_call, total_tiles, NTask);
        fflush(stdout);
    }
    MPI_Allgather(need_from, total_tiles, MPI_INT,
                  all_need_from, total_tiles, MPI_INT, MPI_COMM_WORLD);
    if(gizmo_verbose_diag()) {
        printf("[GX rank=%d call=%d] AFTER  Allgather(need_from) OK\n", ThisTask, this_call);
        fflush(stdout);
    }
    for(task = 0; task < NTask; task++)
    {
        if(task == ThisTask) continue;
        const int *peer_need = all_need_from + (size_t)task * total_tiles;
        for(int lt_idx = 0; lt_idx < local_ntiles; lt_idx++) {
            send_to[lt_idx * NTask + task] = peer_need[my_tile_start + lt_idx];
        }
    }
    free(all_need_from);

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

    double t_ghost_overlap = timediff(t_phase_overlap_start, my_second()); /* steps 3+4 (overlap + schedule) */
    double t_phase_mpi_start = my_second();

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
    /* Per-particle exchange: send_count/recv_count/send_disp/recv_disp are
     * already in element units. gizmo_mpi_alltoallv_typed builds a contiguous
     * MPI_Datatype per call so element-count int*'s drive the wire — dodging
     * the 2.1 GB per-peer int-overflow that bites fire_m11i at >~6M parts/rank. */
    if(gizmo_verbose_diag()) {
        int ts=0, tr=0;
        for(int tt=0; tt<NTask; tt++) { ts += send_count[tt]; tr += recv_count[tt]; }
        printf("[GX rank=%d call=%d] BEFORE Alltoallv(P): total_send=%d total_recv=%d send[0]=%d recv[0]=%d\n",
               ThisTask, this_call, ts, tr, send_count[0], recv_count[0]);
        fflush(stdout);
    }
    gizmo_mpi_alltoallv_typed(send_P, send_count, send_disp,
                              &P[NumPart], recv_count, recv_disp,
                              sizeof(struct particle_data), MPI_COMM_WORLD);
    if(gizmo_verbose_diag()) {
        printf("[GX rank=%d call=%d] AFTER  Alltoallv(P) OK\n", ThisTask, this_call);
        fflush(stdout);
    }

    /* CellP exchange: only meaningful when the simulation has any gas
       particles globally. With TotN_gas==0 (N-body / DM-only runs), CellP
       is allocated to size 0, so writing to &CellP[NumPart] would dereference
       an out-of-bounds pointer. Skip the CellP alltoallv in that case —
       no gas ghosts can exist if no gas exists anywhere. */
    if(All.TotN_gas > 0) {
        if(gizmo_verbose_diag()) {
            printf("[GX rank=%d call=%d] BEFORE Alltoallv(CellP)\n", ThisTask, this_call);
            fflush(stdout);
        }
        gizmo_mpi_alltoallv_typed(send_CellP, send_count, send_disp,
                                  &CellP[NumPart], recv_count, recv_disp,
                                  sizeof(struct gas_cell_data), MPI_COMM_WORLD);
        if(gizmo_verbose_diag()) {
            printf("[GX rank=%d call=%d] AFTER  Alltoallv(CellP) OK\n", ThisTask, this_call);
            fflush(stdout);
        }
    }

    /* Update counts */
    NumGhostParticles = total_recv;
    NumPart += total_recv;

    /* Diagnostic: ghost composition by Type. If a hydro-context exchange is
     * pulling DM/sink/star ghosts back, the type-mask refactor needs to
     * gate them out. Gated on GIZMO_VERBOSE_DIAG=1. */
    if(gizmo_verbose_diag() && total_recv > 0) {
        int by_type[6] = {0,0,0,0,0,0};
        for(int g = 0; g < total_recv; g++) {
            int gi = NumPart_before_ghost + g;
            int tt = (int)P[gi].Type;
            if(tt >= 0 && tt < 6) by_type[tt]++;
        }
        printf("[GX_GHOSTTYPE rank=%d call=%d caller=%s total_recv=%d  T0(gas)=%d T1=%d T2=%d T3=%d T4(star)=%d T5(sink)=%d]\n",
               ThisTask, this_call, (spec->caller_name ? spec->caller_name : "?"),
               total_recv,
               by_type[0], by_type[1], by_type[2], by_type[3], by_type[4], by_type[5]);
        fflush(stdout);
    }

    /* Phase-0 import-waste diagnostic — see gx_print_waste(). */
    gx_print_waste(spec, this_call, total_recv);

    /* Multi-rank correctness: ghost slots [NumPart_before_ghost, NumPart) just
     * received fresh particle_data from remote ranks via MPI_Alltoallv. Their
     * KernelRadius values overwrote whatever was in those local P[] slots
     * (uninitialized or stale from prior step's ghost import). Any subsequent
     * cached gpu_ngb_list_build reading compact_xyzh[j*4+3] for h_j on those
     * slots would see stale values without this dirty-mark, silently missing
     * symmetric neighbor pairs where r<h_ghost.  Single-rank runs hit this
     * branch only when NumGhostParticles>0 (rare for true 1-rank), so this
     * fix is functionally a multi-rank correctness guarantee. */
    if(NumGhostParticles > 0) {
        gpu_compact_xyzh_mark_h_dirty_range(NumPart_before_ghost, NumPart);
    }

    /* ================================================================
       Step 7: Build ghost provenance map for writeback.
       Exchange home indices so each ghost knows its home rank + index.
       ================================================================ */
    {
        /* Exchange home indices: send_home_idx[total_send] → recv_home_idx[total_recv] */
        int *recv_home_idx = (int *) malloc((total_recv > 0 ? total_recv : 1) * sizeof(int));
        if(gizmo_verbose_diag()) {
            printf("[GX rank=%d call=%d] BEFORE Alltoallv(home_idx)\n", ThisTask, this_call);
            fflush(stdout);
        }
        gizmo_mpi_alltoallv_typed(send_home_idx, send_count, send_disp,
                                  recv_home_idx, recv_count, recv_disp,
                                  sizeof(int), MPI_COMM_WORLD);
        if(gizmo_verbose_diag()) {
            printf("[GX rank=%d call=%d] AFTER  Alltoallv(home_idx) OK\n", ThisTask, this_call);
            fflush(stdout);
        }

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

    double t_ghost_mpi = timediff(t_phase_mpi_start, my_second()); /* steps 5+6 (pack + MPI + unpack) */
    double t_ghost_end = my_second();
    double t_ghost_total = timediff(t_ghost_start, t_ghost_end);

    /* Active-count diagnostic (gated on GIZMO_VERBOSE_DIAG; collective so all
     * ranks must call). Lets us correlate ghost-exchange wall with how many
     * particles are actually active on this step. */
    int n_active_global = 0;
    if(gizmo_verbose_diag()) {
        int n_active = (int)ActiveParticleList.size();
        printf("[GX rank=%d call=%d] BEFORE MPI_Reduce(n_active=%d)\n", ThisTask, this_call, n_active);
        fflush(stdout);
        MPI_Reduce(&n_active, &n_active_global, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        printf("[GX rank=%d call=%d] AFTER  MPI_Reduce OK active_global=%d\n", ThisTask, this_call, n_active_global);
        fflush(stdout);
    }
    if(ThisTask == 0) {
        PRINT_STATUS("Ghost exchange: %d local + %d ghost = %d total (recv %d tiles, sent %d/%d) [%.4f s]",
                     NumPart_before_ghost, NumGhostParticles, NumPart,
                     tiles_needed, tiles_sent, local_ntiles, t_ghost_total);
        if(gizmo_verbose_diag()) {
            printf("  ghost_exchange phases: tiles_build=%.4f meta_allgather=%.4f overlap+sched=%.4f pack+mpi=%.4f total=%.4f  active_global=%d\n",
                   t_ghost_tiles, t_ghost_meta, t_ghost_overlap, t_ghost_mpi, t_ghost_total,
                   n_active_global);
            fflush(stdout);
        }
    }
    /* Warn if ghost particles used >80% of available headroom */
    if(NumPart > 0.8 * All.MaxPart) {
        double usage_frac = (double)NumPart / (double)All.MaxPart;
        PRINT_WARNING("Ghost exchange: particle arrays %.0f%% full (%d/%d). "
                      "Consider increasing PartAllocFactor (currently %.2f) to avoid running out of space.",
                      100.0 * usage_frac, NumPart, All.MaxPart, All.PartAllocFactor);
    }

    /* Cleanup: mymalloc in reverse order, then free malloc'd send metadata */
    myfree(task_offset);
    myfree(send_CellP); myfree(send_P);
    myfree(send_disp); myfree(recv_disp);
    myfree(send_count); myfree(recv_count);
    free(send_home_idx);
    free(send_to); free(need_from);
    free(all_meta); free(tile_disp); free(all_ntiles);
    free(tile_first); free(local_meta); free(pool);
}


/* ============================================================================
 * Request-driven ghost exchange (Phase 2).
 *
 * Replaces the tile-overlap candidate generator + whole-tile MPI payload of
 * ghost_exchange_tile_overlap_impl with:
 *   1. Each rank builds a list of compact query records {pos[3], h, _pad}
 *      for each LOCAL active particle matching spec->request_type_mask.
 *   2. Per-rank query counts via Allgather; queries themselves via Allgatherv.
 *   3. For each remote rank's queries, walk the LOCAL pool and apply the
 *      EXACT predicate per particle (r_ij < h_i for ONEWAY,
 *      r_ij < max(h_i, h_j) for SYMMETRIC). Mark matched local pool indices
 *      in a bitmask per peer rank to dedupe (same j matched by N queries
 *      ships once).
 *   4. Per-peer match list is the basis for Alltoallv pack.
 *   5. Alltoallv particle_data + gas_cell_data + home_idx (existing typed
 *      Alltoallv used for the legacy path).
 *   6. Install as ghosts (NumPart += total_recv, mark dirty for compact_xyzh,
 *      build ghost_home_*_map, save ghost_wb_* arrays).
 *
 * Killshot diagnostic on Vista (job 694703) showed the legacy path imports
 * with 99.97-100% waste — millions of particles imported per call, of which
 * <1000 are actual neighbors under the criterion that imported them. This
 * path's match step performs the EXACT predicate before pack, so per-call
 * waste is zero by construction (modulo a small over-supply factor from
 * ghosts that pass for one query but turn out unused after kernel).
 *
 * For now the local-pool walk is a flat O(N_local_tiles * N_remote_queries)
 * scan with bbox-vs-sphere prune at the tile level; per-particle accept at
 * the leaf. A BVH speedup is straightforward later (the all-particle BVH
 * already used by the GPU neighbor list can be reused) but not a Phase-2
 * blocker — even the flat scan is dominated by per-particle work for the
 * tiny-N case that motivates this restructure.
 * ============================================================================ */
struct gx_query_t {
    double pos[3];
    double h;
    int    type;       /* for caller diagnostics; predicate uses h_i directly */
    int    _pad;
};

/* effective_ghost_radius re-implemented as a free function for cross-path reuse
 * (tile_overlap_impl has its own captured-lambda copy; this is for the
 * request-driven path + waste diagnostic). */
static double gx_eff_h(int j, unsigned int supply_mask)
{
    if(supply_mask != GHOST_TYPE_ALL) {
        if(!ghost_type_passes((int)P[j].Type, supply_mask)) return 0.0;
        return (double)P[j].KernelRadius;
    }
    double h = (double)P[j].KernelRadius;
#if defined(GALSF_SUBGRID_WINDS) && (GALSF_SUBGRID_WIND_SCALING==2)
    if(P[j].Type == 0 && j < N_gas) {
        if((double)CellP[j].KernelRadiusDM > h) h = CellP[j].KernelRadiusDM;
    }
#endif
#ifdef AGS_KERNELRADIUS_CALCULATION_IS_ACTIVE
    if((double)P[j].AGS_KernelRadius > h) h = P[j].AGS_KernelRadius;
#endif
    return h;
}

/* Print [GX_WASTE] for any ghost_exchange path. Walks (sampled) local actives
 * × imported ghosts, applies ONEWAY and SYMMETRIC predicates, prints the
 * per-ghost OR-aggregate waste ratio. PAIRS_BUDGET caps cost on global steps. */
static void gx_print_waste(const struct ghost_exchange_spec_t *spec, int this_call, int total_recv)
{
    if(!gizmo_verbose_diag()) return;
    if(total_recv <= 0 || NumPart_before_ghost <= 0) return;
    const unsigned int request_mask = spec->request_type_mask;
    const unsigned int supply_mask  = spec->supply_type_mask;
    const int  search_mode = spec->search_mode;
    const long long PAIRS_BUDGET = 10000000LL;
    int sample_cap = (int)(PAIRS_BUDGET / (long long)total_recv);
    if(sample_cap < 4) sample_cap = 4;
    if(sample_cap > 1024) sample_cap = 1024;
    int n_active_sample = 0;
    int *active_indices = (int *) malloc((size_t)sample_cap * sizeof(int));
    for(size_t kk = 0; kk < ActiveParticleList.size() && n_active_sample < sample_cap; kk++) {
        int i_act = ActiveParticleList[kk];
        if(i_act < 0 || i_act >= NumPart_before_ghost) continue;
        if(!ghost_type_passes((int)P[i_act].Type, request_mask)) continue;
        active_indices[n_active_sample++] = i_act;
    }
    int total_active_full = 0;
    for(size_t kk = 0; kk < ActiveParticleList.size(); kk++) {
        int i_act = ActiveParticleList[kk];
        if(i_act < 0 || i_act >= NumPart_before_ghost) continue;
        if(!ghost_type_passes((int)P[i_act].Type, request_mask)) continue;
        total_active_full++;
    }
    long long pairs_tested = 0;
    long long g_oneway_used = 0, g_symm_used = 0;
    char *used_oneway = (char *) calloc(total_recv, sizeof(char));
    char *used_symm   = (char *) calloc(total_recv, sizeof(char));
    for(int aa = 0; aa < n_active_sample; aa++) {
        int i = active_indices[aa];
        double h_i = gx_eff_h(i, supply_mask);
        double h2_i = h_i * h_i;
        double px = P[i].Pos[0], py = P[i].Pos[1], pz = P[i].Pos[2];
        for(int g = 0; g < total_recv; g++) {
            int gi = NumPart_before_ghost + g;
            double dx_raw = px - P[gi].Pos[0];
            double dy_raw = py - P[gi].Pos[1];
            double dz_raw = pz - P[gi].Pos[2];
            MyDouble xtmp = 0; (void)xtmp;
            double adx = NGB_PERIODIC_BOX_LONG_X(dx_raw, dy_raw, dz_raw, 1);
            double ady = NGB_PERIODIC_BOX_LONG_Y(dx_raw, dy_raw, dz_raw, 1);
            double adz = NGB_PERIODIC_BOX_LONG_Z(dx_raw, dy_raw, dz_raw, 1);
            double r2 = adx*adx + ady*ady + adz*adz;
            pairs_tested++;
            if(!used_oneway[g] && r2 < h2_i) used_oneway[g] = 1;
            if(!used_symm[g]) {
                double h_j = gx_eff_h(gi, supply_mask);
                double h2_max = (h2_i > h_j*h_j) ? h2_i : h_j*h_j;
                if(r2 < h2_max) used_symm[g] = 1;
            }
        }
    }
    for(int g = 0; g < total_recv; g++) {
        g_oneway_used += used_oneway[g];
        g_symm_used   += used_symm[g];
    }
    free(used_oneway); free(used_symm); free(active_indices);
    double waste_o = 100.0 * (1.0 - (double)g_oneway_used / (double)total_recv);
    double waste_s = 100.0 * (1.0 - (double)g_symm_used   / (double)total_recv);
    printf("[GX_WASTE rank=%d call=%d caller=%s mode=%s imported=%d n_active=%d (sampled=%d) used_oneway=%lld used_symm=%lld waste_oneway=%.2f%% waste_symm=%.2f%% pairs_tested=%lld]\n",
           ThisTask, this_call,
           (spec->caller_name ? spec->caller_name : "?"),
           (search_mode == NGB_SEARCH_ONEWAY ? "ONEWAY" : "SYMMETRIC"),
           total_recv, total_active_full, n_active_sample,
           g_oneway_used, g_symm_used, waste_o, waste_s, pairs_tested);
    fflush(stdout);
}

/* Absolute periodic-shortened delta on one axis. */
static inline double gx_abs_dx(double dx, double bsize, double bhalf, int periodic)
{
    double a = fabs(dx);
    if(periodic && a > bhalf) a = bsize - a;
    return a;
}

static void ghost_exchange_request_driven_impl(const struct ghost_exchange_spec_t *spec)
{
    if(NTask <= 1) return;
    const double safety_factor = spec->safety_factor;
    const unsigned int request_mask = spec->request_type_mask;
    const unsigned int supply_mask  = spec->supply_type_mask;
    const int  search_mode = spec->search_mode;
    double t_ghost_start = my_second();

    NumPart_before_ghost = NumPart;
    N_gas_before_ghost = N_gas;
    NumGhostParticles = 0;

    /* Periodic-axis flags for box wrap (cached for the per-pair distance fn). */
    int per_x = 0, per_y = 0, per_z = 0;
    double bsx = 0, bsy = 0, bsz = 0, bhx = 0, bhy = 0, bhz = 0;
#if defined(BOX_PERIODIC)
    bsx = boxSize_X; bsy = boxSize_Y; bsz = boxSize_Z;
    bhx = boxHalf_X; bhy = boxHalf_Y; bhz = boxHalf_Z;
    per_x = per_y = per_z = 1;
#  if defined(BOX_REFLECT_X) || defined(BOX_OUTFLOW_X)
    per_x = 0;
#  endif
#  if defined(BOX_REFLECT_Y) || defined(BOX_OUTFLOW_Y)
    per_y = 0;
#  endif
#  if defined(BOX_REFLECT_Z) || defined(BOX_OUTFLOW_Z)
    per_z = 0;
#  endif
#endif

    static int gx_call_seq_rd = 0;
    gx_call_seq_rd++;
    int this_call = gx_call_seq_rd;

    /* === Step 1: build local query list === */
    int n_local_queries = 0;
    for(size_t kk = 0; kk < ActiveParticleList.size(); kk++) {
        int i = ActiveParticleList[kk];
        if(i < 0 || i >= NumPart) continue;
        if(P[i].Mass <= 0) continue;
        if(!ghost_type_passes((int)P[i].Type, request_mask)) continue;
        n_local_queries++;
    }
    struct gx_query_t *local_queries = (struct gx_query_t *)
        malloc((size_t)(n_local_queries > 0 ? n_local_queries : 1) * sizeof(struct gx_query_t));
    {
        int q = 0;
        for(size_t kk = 0; kk < ActiveParticleList.size(); kk++) {
            int i = ActiveParticleList[kk];
            if(i < 0 || i >= NumPart) continue;
            if(P[i].Mass <= 0) continue;
            if(!ghost_type_passes((int)P[i].Type, request_mask)) continue;
            local_queries[q].pos[0] = P[i].Pos[0];
            local_queries[q].pos[1] = P[i].Pos[1];
            local_queries[q].pos[2] = P[i].Pos[2];
            double h = (double)P[i].KernelRadius;
            local_queries[q].h    = h * safety_factor;
            local_queries[q].type = (int)P[i].Type;
            local_queries[q]._pad = 0;
            q++;
        }
    }

    /* === Step 2: Allgather query counts, Allgatherv query records === */
    int *all_q_counts = (int *) malloc(NTask * sizeof(int));
    MPI_Allgather(&n_local_queries, 1, MPI_INT, all_q_counts, 1, MPI_INT, MPI_COMM_WORLD);
    int *q_disps = (int *) malloc(NTask * sizeof(int));
    int total_queries = 0;
    for(int t = 0; t < NTask; t++) {
        q_disps[t] = total_queries;
        total_queries += all_q_counts[t];
    }
    struct gx_query_t *all_queries = (struct gx_query_t *)
        malloc((size_t)(total_queries > 0 ? total_queries : 1) * sizeof(struct gx_query_t));

    int *q_byte_counts = (int *) malloc(NTask * sizeof(int));
    int *q_byte_disps  = (int *) malloc(NTask * sizeof(int));
    for(int t = 0; t < NTask; t++) {
        q_byte_counts[t] = all_q_counts[t] * (int)sizeof(struct gx_query_t);
        q_byte_disps[t]  = q_disps[t]      * (int)sizeof(struct gx_query_t);
    }
    MPI_Allgatherv(local_queries, n_local_queries * (int)sizeof(struct gx_query_t), MPI_BYTE,
                   all_queries,   q_byte_counts, q_byte_disps, MPI_BYTE,
                   MPI_COMM_WORLD);
    free(q_byte_counts); free(q_byte_disps);

    /* === Step 3: per-rank, walk local pool against each remote rank's queries === */
    /* Pool index list (supply-mask filtered). */
    int num_pool = 0;
    for(int i = 0; i < NumPart; i++) {
        if(P[i].Mass <= 0) continue;
        if(!ghost_type_passes((int)P[i].Type, supply_mask)) continue;
        num_pool++;
    }
    int *pool = (int *) malloc((size_t)(num_pool > 0 ? num_pool : 1) * sizeof(int));
    {
        int p = 0;
        for(int i = 0; i < NumPart; i++) {
            if(P[i].Mass <= 0) continue;
            if(!ghost_type_passes((int)P[i].Type, supply_mask)) continue;
            pool[p++] = i;
        }
    }

    /* Per-peer match bitmask over pool indices (dedup multiple queries → one ghost). */
    char *matched = (char *) calloc((size_t)NTask * (size_t)(num_pool > 0 ? num_pool : 1), sizeof(char));
    for(int t = 0; t < NTask; t++) {
        if(t == ThisTask) continue;
        int q_start = q_disps[t];
        int q_count = all_q_counts[t];
        char *match_for_t = matched + (size_t)t * (size_t)num_pool;
        for(int qi = 0; qi < q_count; qi++) {
            const struct gx_query_t *q = &all_queries[q_start + qi];
            double q_pos_x = q->pos[0], q_pos_y = q->pos[1], q_pos_z = q->pos[2];
            double q_h = q->h;
            double q_h2 = q_h * q_h;
            for(int p = 0; p < num_pool; p++) {
                if(match_for_t[p]) continue;
                int j = pool[p];
                double dx = q_pos_x - P[j].Pos[0];
                double dy = q_pos_y - P[j].Pos[1];
                double dz = q_pos_z - P[j].Pos[2];
                double adx = gx_abs_dx(dx, bsx, bhx, per_x);
                double ady = gx_abs_dx(dy, bsy, bhy, per_y);
                double adz = gx_abs_dx(dz, bsz, bhz, per_z);
                double r2 = adx*adx + ady*ady + adz*adz;
                double thresh2;
                if(search_mode == NGB_SEARCH_ONEWAY) {
                    thresh2 = q_h2;
                } else {
                    double h_j = (double)P[j].KernelRadius * safety_factor;
                    double h_max = (q_h > h_j) ? q_h : h_j;
                    thresh2 = h_max * h_max;
                }
                if(r2 < thresh2) match_for_t[p] = 1;
            }
        }
    }

    /* === Step 4: per-peer counts + index list === */
    int *send_count = (int *) mymalloc("gx_rd_sc", NTask * sizeof(int));
    int *recv_count = (int *) mymalloc("gx_rd_rc", NTask * sizeof(int));
    int *send_disp  = (int *) mymalloc("gx_rd_sd", NTask * sizeof(int));
    int *recv_disp  = (int *) mymalloc("gx_rd_rd", NTask * sizeof(int));
    for(int t = 0; t < NTask; t++) { send_count[t] = 0; recv_count[t] = 0; }
    for(int t = 0; t < NTask; t++) {
        if(t == ThisTask) continue;
        char *match_for_t = matched + (size_t)t * (size_t)num_pool;
        int s = 0;
        for(int p = 0; p < num_pool; p++) if(match_for_t[p]) s++;
        send_count[t] = s;
    }
    MPI_Alltoall(send_count, 1, MPI_INT, recv_count, 1, MPI_INT, MPI_COMM_WORLD);
    int total_send = 0, total_recv = 0;
    for(int t = 0; t < NTask; t++) { total_send += send_count[t]; total_recv += recv_count[t]; }
    send_disp[0] = 0; recv_disp[0] = 0;
    for(int t = 1; t < NTask; t++) {
        send_disp[t] = send_disp[t-1] + send_count[t-1];
        recv_disp[t] = recv_disp[t-1] + recv_count[t-1];
    }

    /* Check space (mirrors legacy guard). */
    if(NumPart + total_recv > All.MaxPart) {
        printf("ERROR: request-driven ghost exchange needs %d ghosts on task %d, only %d free.\n",
               total_recv, ThisTask, All.MaxPart - NumPart);
        endrun(7702);
    }

    /* === Step 5: pack particle data + cell data + home_idx === */
    struct particle_data *send_P = (struct particle_data *) mymalloc("gx_rd_sP",
        (total_send > 0 ? total_send : 1) * sizeof(struct particle_data));
    struct gas_cell_data *send_CellP = (struct gas_cell_data *) mymalloc("gx_rd_sC",
        (total_send > 0 ? total_send : 1) * sizeof(struct gas_cell_data));
    int *send_home_idx = (int *) malloc((total_send > 0 ? total_send : 1) * sizeof(int));
    {
        int *task_offset = (int *) mymalloc("gx_rd_toff", NTask * sizeof(int));
        memcpy(task_offset, send_disp, NTask * sizeof(int));
        for(int t = 0; t < NTask; t++) {
            if(t == ThisTask) continue;
            char *match_for_t = matched + (size_t)t * (size_t)num_pool;
            for(int p = 0; p < num_pool; p++) {
                if(!match_for_t[p]) continue;
                int j = pool[p];
                int off = task_offset[t]++;
                send_P[off] = P[j];
                send_home_idx[off] = j;
                if(P[j].Type == 0 && j < N_gas) send_CellP[off] = CellP[j];
                else memset(&send_CellP[off], 0, sizeof(struct gas_cell_data));
            }
        }
        myfree(task_offset);
    }
    free(matched);

    /* === Step 6: Alltoallv particles + cells + home_idx === */
    gizmo_mpi_alltoallv_typed(send_P, send_count, send_disp,
                              &P[NumPart], recv_count, recv_disp,
                              sizeof(struct particle_data), MPI_COMM_WORLD);
    if(All.TotN_gas > 0) {
        gizmo_mpi_alltoallv_typed(send_CellP, send_count, send_disp,
                                  &CellP[NumPart], recv_count, recv_disp,
                                  sizeof(struct gas_cell_data), MPI_COMM_WORLD);
    }

    /* Update counts now so home_idx receive can land at &P[NumPart_before_ghost+...] */
    NumGhostParticles = total_recv;
    NumPart += total_recv;

    /* Mark dirty for compact_xyzh refresh (same as legacy). */
    if(NumGhostParticles > 0) {
        gpu_compact_xyzh_mark_h_dirty_range(NumPart_before_ghost, NumPart);
    }

    /* Home-index exchange + provenance maps. */
    int *recv_home_idx = (int *) malloc((total_recv > 0 ? total_recv : 1) * sizeof(int));
    gizmo_mpi_alltoallv_typed(send_home_idx, send_count, send_disp,
                              recv_home_idx, recv_count, recv_disp,
                              sizeof(int), MPI_COMM_WORLD);
    ghost_home_rank_map = (int *) malloc((total_recv > 0 ? total_recv : 1) * sizeof(int));
    ghost_home_index_map = recv_home_idx;
    for(int t = 0; t < NTask; t++) {
        for(int g = 0; g < recv_count[t]; g++) {
            ghost_home_rank_map[recv_disp[t] + g] = t;
        }
    }
    /* Preserve comm maps for reverse Alltoallv (ghost writeback). */
    ghost_wb_recv_count = (int *) malloc(NTask * sizeof(int));
    ghost_wb_recv_disp  = (int *) malloc(NTask * sizeof(int));
    ghost_wb_send_count = (int *) malloc(NTask * sizeof(int));
    ghost_wb_send_disp  = (int *) malloc(NTask * sizeof(int));
    memcpy(ghost_wb_recv_count, recv_count, NTask * sizeof(int));
    memcpy(ghost_wb_recv_disp,  recv_disp,  NTask * sizeof(int));
    memcpy(ghost_wb_send_count, send_count, NTask * sizeof(int));
    memcpy(ghost_wb_send_disp,  send_disp,  NTask * sizeof(int));

    double t_ghost_total = timediff(t_ghost_start, my_second());

    if(ThisTask == 0) {
        PRINT_STATUS("Ghost exchange (request-driven, %s, %s): %d local + %d ghost  queries=%d total_queries=%d num_pool=%d  [%.4f s]",
                     (spec->caller_name ? spec->caller_name : "?"),
                     (search_mode == NGB_SEARCH_ONEWAY ? "ONEWAY" : "SYMMETRIC"),
                     NumPart_before_ghost, NumGhostParticles,
                     n_local_queries, total_queries, num_pool, t_ghost_total);
    }

    /* Diagnostic: ghost composition + import-waste ratio (should be ~0% for
     * the request-driven path by construction since per-particle accept ran
     * before pack — provides direct A/B vs the legacy tile-overlap waste). */
    if(gizmo_verbose_diag() && total_recv > 0) {
        int by_type[6] = {0,0,0,0,0,0};
        for(int g = 0; g < total_recv; g++) {
            int gi = NumPart_before_ghost + g;
            int tt = (int)P[gi].Type;
            if(tt >= 0 && tt < 6) by_type[tt]++;
        }
        printf("[GX_GHOSTTYPE rank=%d call=%d caller=%s total_recv=%d  T0(gas)=%d T1=%d T2=%d T3=%d T4(star)=%d T5(sink)=%d]\n",
               ThisTask, this_call, (spec->caller_name ? spec->caller_name : "?"),
               total_recv, by_type[0], by_type[1], by_type[2], by_type[3], by_type[4], by_type[5]);
        fflush(stdout);
    }
    gx_print_waste(spec, this_call, total_recv);

    /* Cleanup local. */
    myfree(send_CellP); myfree(send_P);
    myfree(recv_disp); myfree(send_disp); myfree(recv_count); myfree(send_count);
    free(send_home_idx);
    free(pool); free(all_queries); free(q_disps); free(all_q_counts); free(local_queries);
}

/* Public wrappers — each fills a spec, calls the single _impl. New callers
 * add a wrapper line; do not duplicate logic. */
void ghost_exchange(double safety_factor)
{
    struct ghost_exchange_spec_t sp = {GHOST_TYPE_ALL, GHOST_TYPE_ALL, NGB_SEARCH_SYMMETRIC, safety_factor, "all_types"};
    ghost_exchange_impl(&sp);
}
void ghost_exchange_hydro(double safety_factor)
{
    struct ghost_exchange_spec_t sp = {GHOST_TYPE_GAS, GHOST_TYPE_GAS, NGB_SEARCH_SYMMETRIC, safety_factor, "hydro_symmetric"};
    ghost_exchange_impl(&sp);
}
void ghost_exchange_hydro_oneway(double safety_factor)
{
    struct ghost_exchange_spec_t sp = {GHOST_TYPE_GAS, GHOST_TYPE_GAS, NGB_SEARCH_ONEWAY, safety_factor, "hydro_oneway"};
    ghost_exchange_impl(&sp);
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

    /* Must use the SAME effective radius as ghost_exchange() saved during its
       tile build, otherwise growth in e.g. KernelRadiusDM or AGS_KernelRadius
       would not trigger a re-exchange. Keep this in sync with ghost_exchange(). */
    int p = 0;
    for(i = 0; i < nlocal; i++) {
        if(P[i].Mass <= 0) continue;
        int t = p / tile_target;
        if(t >= ntiles) t = ntiles - 1;
        double hi = P[i].KernelRadius;
#if defined(GALSF_SUBGRID_WINDS) && (GALSF_SUBGRID_WIND_SCALING==2)
        if(P[i].Type == 0 && i < N_gas) {
            if((double)CellP[i].KernelRadiusDM > hi) hi = CellP[i].KernelRadiusDM;
        }
#endif
#ifdef AGS_KERNELRADIUS_CALCULATION_IS_ACTIVE
        if((double)P[i].AGS_KernelRadius > hi) hi = P[i].AGS_KernelRadius;
#endif
        if(hi > current_hmax[t]) current_hmax[t] = hi;
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
    /* Ghost slots are about to leave scope (NumPart shrinks back to local).
     * Any dirty-list entries for indices in [num_local, num_local+num_ghost)
     * would index out of bounds in compact_h_refresh on the next cached call
     * if a smaller-num_total build happens before SIDX invalidation. Force
     * full-pool refresh mode as a fail-safe — the next build's full
     * gpu_spatial_index_build will rebuild compact_xyzh from scratch and
     * clear the state anyway, so the cost here is at most one extra full
     * refresh on a build path that's already paying for SIDX construction. */
    if(NumGhostParticles > 0) { gpu_compact_xyzh_mark_h_dirty_all(); }
    PreviousGhostCount = NumGhostParticles; /* save for domain decomposition headroom */
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
int ghost_get_previous_count(void) { return PreviousGhostCount; }
int ghost_get_num_local(void)  { return (NumPart_before_ghost >= 0) ? NumPart_before_ghost : NumPart; }
int *ghost_get_home_rank(void)  { return ghost_home_rank_map; }
int *ghost_get_home_index(void) { return ghost_home_index_map; }
int *ghost_get_wb_recv_count(void) { return ghost_wb_recv_count; }
int *ghost_get_wb_recv_disp(void)  { return ghost_wb_recv_disp; }
int *ghost_get_wb_send_count(void) { return ghost_wb_send_count; }
int *ghost_get_wb_send_disp(void)  { return ghost_wb_send_disp; }
