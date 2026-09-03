/* gpu_drift.cc
 *
 * Batched device drift. The bulk drift sites hand an index list to
 * drift_particles_batch, which compacts it to the particles that actually need
 * advancing and, when there are enough of them, runs the drift body on the device
 * over compact staged copies instead of walking P[] and CellP[] on the host.
 *
 * The body is the same one the host runs: drift_particle_impl in
 * core/drift_particle_functions.h. There is no second copy of the physics, and no
 * device-only approximation of any part of it.
 *
 * Routing is on the number of particles needing a drift, against the same
 * GPU_MIN_PARTICLES_FOR_OFFLOAD the other batched device loops use. Below it the
 * caller's work is done by the ordinary host loop, so a step with few active
 * particles pays nothing: no staging buffers, no device memory, no copies.
 *
 * A drift that does not happen is a wrong position, not a missing improvement, so
 * every path that cannot reach the device -- no staging memory, no table mirror --
 * falls back to the host loop and still drifts every particle it was given.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

#include <Kokkos_Core.hpp>

#include "../declarations/gpu_all_mirror.h"
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../core/timestep_functions.h"
#include "../declarations/gpu_dispatch_templates.h"
#include "../system/gpu_particles_arena.h"
#include "drift_particle_functions.h"

/* Device-visible copy of the drift and gravkick tables for this translation unit.
   Per-TU because a device symbol cannot be shared across TUs without relocatable
   device code; the allocate-and-fill policy is shared, in the arena. */
static double *drift_kick_table_dev_ = NULL;

/* Batch size, the same cap the cooling loop uses. What it buys is a bounded staging
   footprint: the helper holds this many compact structs on the host and again on the
   device while a call runs. Device stack depth is not the binding reason here -- the
   drift's equation-of-state call reads a cached composition rather than running
   cooling's iterative solve, so this chain is the shallower of the two, and a cap that
   is safe for cooling is safe for it. */
static const int GPU_DRIFT_BATCH_SIZE = 32768;

/* Drift idx[0..n_idx) to time1 on the host, the way the callers did before this
   entry existed. Threaded because all four bulk sites already threaded their loops
   and routing through here must not quietly serialise them; the body acts on one
   particle with no pair coupling and no random draws, so the indices being distinct
   is the whole of the argument. */
static void drift_particles_host_(const int *idx, int n_idx, integertime time1)
{
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 256) if(n_idx >= 16)
#endif
    for(int k = 0; k < n_idx; k++) {drift_particle(idx[k], time1);}
}

/* A null idx means the contiguous range [0, n_idx), which is what the full-drift
   site hands over: materialising an identity array there would be an extra
   allocation and an extra pass to say nothing. */
void drift_particles_batch(const int *idx, int n_idx, integertime time1)
{
    if(n_idx <= 0) {return;}

    /* Compact to the particles that are not already at time1. The drift body returns
       immediately for the rest, so staging them would be copying a whole struct each
       way to do nothing. At the full-drift site this pass is O(NumPart) at a site that
       is already O(NumPart) by definition; the other sites filter a list they hold. */
    std::vector<int> needs_drift;
    needs_drift.reserve((size_t) n_idx);
#ifdef _OPENMP
    /* Threaded above a handful of indices: the full-drift site hands over every
       particle, and reading Ti_current out of the particle array is a strided pass
       over the whole of it, which is a large part of what running the drift in bulk
       is meant to remove. Each thread collects its own contiguous static range and
       the ranges are concatenated in thread order, so the compacted list is the same
       list, in the same order, whatever the thread count. Below that a parallel
       region and its per-thread vectors cost more than the scan they divide. */
    if(n_idx >= 16)
    {
        const int n_threads = omp_get_max_threads();
        std::vector<std::vector<int> > per_thread(n_threads);
#pragma omp parallel
        {
            std::vector<int> &mine = per_thread[omp_get_thread_num()];
            mine.reserve((size_t) n_idx / n_threads + 16);
#pragma omp for schedule(static)
            for(int k = 0; k < n_idx; k++)
            {
                const int i = idx ? idx[k] : k;
                if(P[i].Ti_current != time1) {mine.push_back(i);}
            }
        }
        for(int t = 0; t < n_threads; t++)
        {
            needs_drift.insert(needs_drift.end(), per_thread[t].begin(), per_thread[t].end());
        }
    }
    else
#endif
    {
        for(int k = 0; k < n_idx; k++)
        {
            const int i = idx ? idx[k] : k;
            if(P[i].Ti_current != time1) {needs_drift.push_back(i);}
        }
    }
    const int n_need = (int) needs_drift.size();
    if(n_need <= 0) {return;}

    /* Tiny-N and everything below the offload threshold stays exactly as it was. */
    if(n_need < GPU_MIN_PARTICLES_FOR_DRIFT_OFFLOAD) {
        drift_particles_host_(needs_drift.data(), n_need, time1);
        return;
    }

    GIZMO_GPU_ENSURE_ALL_FRESH();

    /* Built on the host, captured by value: the owners are host globals and a
       kernel that reached for one would read host memory silently. */
    struct EosTableView eos_tables = eos_tables_view();

    struct DriftKickTableView tables;
    if(drift_kick_table_mirror_refresh(&drift_kick_table_dev_, &tables) != 0) {
        drift_particles_host_(needs_drift.data(), n_need, time1);
        return;
    }

    const int batch_cap = (n_need < GPU_DRIFT_BATCH_SIZE) ? n_need : GPU_DRIFT_BATCH_SIZE;
    struct ParticleStagingBatch batch = {};
    if(!particle_staging_acquire(&batch, batch_cap)) {
        drift_particles_host_(needs_drift.data(), n_need, time1);
        return;
    }

    for(int batch_start = 0; batch_start < n_need; batch_start += GPU_DRIFT_BATCH_SIZE)
    {
        int batch_n = n_need - batch_start;
        if(batch_n > GPU_DRIFT_BATCH_SIZE) {batch_n = GPU_DRIFT_BATCH_SIZE;}

        if(!particle_staging_gather(&batch, needs_drift.data() + batch_start, batch_n, P, CellP)) {
            /* The gather staged nothing. These particles still have to be drifted:
               leaving them behind is a wrong position, and the sites that early-return
               on Ti_current would then never revisit them. */
            drift_particles_host_(needs_drift.data() + batch_start, n_need - batch_start, time1);
            break;
        }

        /* Slot j holds a particle, and holds a cell exactly while j < gas_count --
           the same condition the drift body's own Type==0 guard already expresses,
           so the kernel needs no extra gating for the non-gas slots. */
        struct particle_data *kp = batch.dev_P;
        struct gas_cell_data *kc = batch.dev_Cell;
        gizmo_gpu_kernel_launch("drift_particles", batch.count, KOKKOS_LAMBDA(int j) {
            drift_particle_impl(j, time1, kp, kc, &tables, &eos_tables);
        }, batch_start);

        /* Synchronous by construction: the results are home before this returns, so
           the lazy-drift sites that early-return on Ti_current can never observe a
           particle whose Ti_current has advanced while its fields have not. */
        particle_staging_scatter(&batch, P, CellP);

        /* Assert the work product, not a return code: the launch wrapper reports a
           device failure by requesting a controlled stop, which drains at the next
           phase boundary, and returns nothing to its caller; a failed kernel would
           otherwise have its pre-drift staging scattered straight back. Ti_current is what says whether the body ran on a particle, and any
           that is still short of time1 is drifted here on the host. Without this the
           full-drift site would go on to certify the neighbour pool as current over
           positions that were never advanced, and the sweep that would have caught it
           is the one that certification suppresses. Costs one integer compare per
           staged slot on the path where nothing went wrong. */
        int n_undrifted = 0;
        for(int j = 0; j < batch.count; j++) {
            if(P[batch.index[j]].Ti_current != time1) {n_undrifted++;}
        }
        if(n_undrifted > 0) {
            std::vector<int> undrifted;
            undrifted.reserve((size_t) n_undrifted);
            for(int j = 0; j < batch.count; j++) {
                if(P[batch.index[j]].Ti_current != time1) {undrifted.push_back(batch.index[j]);}
            }
            drift_particles_host_(undrifted.data(), (int) undrifted.size(), time1);
        }
    }

    gpu_particles_arena_invalidate();   /* host P/CellP scattered; arena stale */
    particle_staging_release(&batch);
}
