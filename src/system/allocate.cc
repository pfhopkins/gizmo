#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../gravity/forcetree.h"
#include "gpu_particles_arena.h"
#ifdef SUBFIND
#include "../structure/subfind/subfind.h"
#endif

/* UVM-canonical particles: allocate P[] and CellP[] directly in
 * Kokkos::SharedSpace (CudaUVMSpace on GH200) via gpu_particles_uvm_alloc()
 * rather than through mymalloc.  The GPU particles arena then aliases these
 * pointers — no host->device memcpy on arena_acquire, no coherence dance,
 * no invalidate.
 *
 * The kokkos_malloc call lives in gpu_particles_arena.cc (a GPU TU) so that
 * this host TU does not have to include <Kokkos_Core.hpp> directly, which
 * would trip the KOKKOS_ENABLE_CUDA-without-__CUDACC__ guard.
 *
 * P/CellP are never myfree'd in the codebase (verified by grep), so
 * bypassing mymalloc for them is safe — the mymalloc LIFO stack stays
 * intact for everything else.  Storage persists to process exit. */


/* This routine allocates memory for particle storage, both the collisionless and the fluid particles.
 * The memory for the ordered binary tree of the timeline is also allocated.
 */
/*!
 * This file was originally part of the GADGET3 code developed by
 * Volker Springel. The code has been modified
 * in part (cleaned up a bit, dealt with some newer memory
 * structures and allocation strategies) by Phil Hopkins
 * (phopkins@caltech.edu) for GIZMO.
 */

int allocate_memory(int do_collective_preflight)
{
  size_t bytes;

  double bytes_tot = 0;

  int NTaskTimesThreads;

  NTaskTimesThreads = maxThreads * NTask;

  /* GRACEFUL-OOM: allocate_memory() never fatal hard-exits on a
   * capacity failure -- it requests a SOFT controlled-stop and RETURNS a nonzero code;
   * the caller drains it at its own all-rank poll. do_collective_preflight only selects
   * the arena preflight's fit-check:
   *  =1 -- ALL-RANK setup phase (read_ic, after find_files() broadcasts the header to
   *        every rank): collective check (one Allreduce); a UVM/STL failure's soft
   *        bad-stop is reconciled at the caller's poll BEFORE any P/CellP deref.
   *  =0 -- SUBSET/turn caller (restart.cc groupTask turn): LOCAL check only (no MPI ->
   *        no deadlock against peers parked at a barrier); the soft bad-stop drains at
   *        restart's existing per-turn poll, with the failing rank skipping its payload
   *        reads (goto finish_turn). */

  /* Node-scoped persistent-memory preflight (user-info aid): on the all-rank path,
   * warn or cleanly stop BEFORE allocating if the projected per-node persistent reserve
   * cannot fit the node. Collective on the node communicator, so only when do_collective_preflight. */
  if(do_collective_preflight)
    {
      int preflight_stop = gizmo_memory_preflight();
      if(preflight_stop) {return preflight_stop;}
    }

  /* Arena (mymalloc) preflight -- BEFORE any mymalloc / metadata mutation. MUST mirror
   * the arena allocations below: Exportflag/Exportindex/Exportnodecount (NTaskTimesThreads
   * ints each, x3) and Send/Recv count/offset (NTask ints each, x4). ProcessedFlag and
   * (CHIMES) ChimesGasVars are plain host malloc rather than arena blocks so that
   * resize_particle_storage can replace them when the particle capacity changes -- a
   * mymalloc block is LIFO-bound and cannot be grown in place. Like the UVM particle arrays
   * (P/CellP) and the STL timebin vectors they are therefore capacity-checked by
   * attempt-then-soft-bad-stop (see the handler below), not by this preflight; the node-level
   * projection in gizmo_memory_preflight() carries their bytes instead. */
  {
    size_t arena_need = (size_t) 3 * gizmo_mymalloc_rounded_size(NTaskTimesThreads * sizeof(int))
                      + (size_t) 4 * gizmo_mymalloc_rounded_size(NTask * sizeof(int));
    int arena_blocks = 7;
    int arena_fits = do_collective_preflight ? gizmo_alloc_fits_all_ranks(arena_need, arena_blocks)
                                             : gizmo_alloc_fits_this_rank(arena_need, arena_blocks);
    if(!arena_fits)
      {
        gizmo_request_controlled_stop(812, "allocate_memory: particle-storage arena preflight failed (insufficient FreeBytes/MAXBLOCKS)", __FILE__, __LINE__, __FUNCTION__);
        return 812;
      }
  }

  Exportflag = (int *) mymalloc("Exportflag", NTaskTimesThreads * sizeof(int));
  Exportindex = (int *) mymalloc("Exportindex", NTaskTimesThreads * sizeof(int));
  Exportnodecount = (int *) mymalloc("Exportnodecount", NTaskTimesThreads * sizeof(int));

  Send_count = (int *) mymalloc("Send_count", sizeof(int) * NTask);
  Send_offset = (int *) mymalloc("Send_offset", sizeof(int) * NTask);
  Recv_count = (int *) mymalloc("Recv_count", sizeof(int) * NTask);
  Recv_offset = (int *) mymalloc("Recv_offset", sizeof(int) * NTask);

  /* ---- Non-arena allocations (UVM particle arrays + STL timebin vectors) use
   * attempt-then-check (they return NULL / throw, unlike the arena path). Accumulate
   * ONE local failure flag across all of them; the failure handler at the end requests a
   * soft bad-stop + returns (the caller drains). Once the flag is set, later optional
   * allocations are skipped. Nothing between here and that handler dereferences P/CellP,
   * so the caller's poll runs first. An uncaught std::bad_alloc is also barred. */
  int alloc_fail_local = 0;

  /* Per-walk scratch (gravtree memsets it before every use, so it is never read unwritten).
   * Host malloc, not an arena block, so the capacity resize can replace it. */
  if(All.MaxPart > 0)
    {
      bytes = All.MaxPart * sizeof(unsigned char);
      ProcessedFlag = (unsigned char *) malloc(bytes);
      if(ProcessedFlag == NULL) { alloc_fail_local = 1; printf("failed to allocate memory for ProcessedFlag (%g MB).\n", bytes / (1024.0 * 1024.0)); fflush(stdout); }
      else { bytes_tot += bytes; gizmo_mem_account_add(GIZMO_MEM_PARTICLE_SOA, (long long) bytes); }
    }

  try {
    ActiveParticleList.reserve(All.MaxPart);
    NextInTimeBin.resize(All.MaxPart);
    PrevInTimeBin.resize(All.MaxPart);
    /* Account reserved capacity (not size): these vectors hold up to MaxPart ints. */
    gizmo_mem_account_set(GIZMO_MEM_STL_TIMEBIN,
                          (long long)(ActiveParticleList.capacity() + NextInTimeBin.capacity()
                                      + PrevInTimeBin.capacity()) * (long long) sizeof(int));
  } catch(const std::bad_alloc&) { alloc_fail_local = 1; printf("allocate_memory: STL timebin vector alloc (MaxPart=%d) threw bad_alloc.\n", All.MaxPart); fflush(stdout); }


  if(All.MaxPart > 0 && !alloc_fail_local)
    {
      bytes = All.MaxPart * sizeof(struct particle_data);
      P = (struct particle_data *) gpu_particles_uvm_alloc(bytes, "particle_soa_P");
      if(P == NULL) { alloc_fail_local = 1; printf("failed to allocate memory for particle data storage structure `P' (%g MB).\n", bytes / (1024.0 * 1024.0)); fflush(stdout); }
      else { bytes_tot += bytes; gizmo_mem_account_add(GIZMO_MEM_PARTICLE_SOA, (long long) bytes); if(ThisTask == 0) {printf("Allocated %g MByte for particle data storage (UVM canonical, SharedSpace).\n", bytes_tot / (1024.0 * 1024.0));} }

      /* Wakeup dirty sidecar: 1 byte/particle acceleration index for
       * process_wake_ups (SharedSpace so device wakeup kernels can mark it).
       * gpu_particles_uvm_alloc zeros the buffer; WakeupDirtyValid stays 0 so
       * the first process_wake_ups full-scans P[] and rebuilds. */
      if(!alloc_fail_local)
        {
          bytes = All.MaxPart * sizeof(unsigned char);
          WakeupDirty = (unsigned char *) gpu_particles_uvm_alloc(bytes, "particle_soa_wakeupdirty");
          if(WakeupDirty == NULL) { alloc_fail_local = 1; printf("failed to allocate memory for WakeupDirty sidecar (%g MB).\n", bytes / (1024.0 * 1024.0)); fflush(stdout); }
          else { gizmo_mem_account_add(GIZMO_MEM_PARTICLE_SOA, (long long) bytes); }
          WakeupDirtyValid = 0;
        }
    }

  if(All.MaxPartGas > 0 && !alloc_fail_local)
    {
      bytes_tot = 0;

      bytes = All.MaxPartGas * sizeof(struct gas_cell_data);
      CellP = (struct gas_cell_data *) gpu_particles_uvm_alloc(bytes, "particle_soa_CellP");
      if(CellP == NULL) { alloc_fail_local = 1; printf("failed to allocate memory for gas cell data storage structure (%g MB).\n", bytes / (1024.0 * 1024.0)); fflush(stdout); }
      else { bytes_tot += bytes; gizmo_mem_account_add(GIZMO_MEM_PARTICLE_SOA, (long long) bytes); if(ThisTask == 0) {printf("Allocated %g MByte for storage of hydro data (UVM canonical, SharedSpace).\n", bytes_tot / (1024.0 * 1024.0));} }

#ifdef CHIMES
      /* Host malloc, not an arena block, so the capacity resize can replace it. Each element owns
       * heap sub-arrays that are created and released with the gas cell occupying that slot, so
       * unused slots must start with NULL fields: zero the whole buffer. Only allocate when the
       * UVM allocs above succeeded on THIS rank; the failure handler below requests the soft stop. */
      if(!alloc_fail_local)
	{
	  bytes = All.MaxPartGas * sizeof(struct gasVariables);
	  ChimesGasVars = (struct gasVariables *) malloc(bytes);
	  if(ChimesGasVars == NULL) { alloc_fail_local = 1; printf("failed to allocate memory for ChimesGasVars (%g MB).\n", bytes / (1024.0 * 1024.0)); fflush(stdout); }
	  else
	    {
	      memset(ChimesGasVars, 0, bytes);
	      bytes_tot += bytes;
	      gizmo_mem_account_add(GIZMO_MEM_PARTICLE_SOA, (long long) bytes);
	      if(ThisTask == 0) printf("Allocated %g MByte for storage of ChimesGasVars data.\n", bytes_tot / (1024.0 * 1024.0));
	    }
	}
#endif
    }

  /* UVM/STL allocation failure handler -- SOFT bad-stop + return for BOTH callers (no
   * fatal hard-exit, no MPI here). The caller owns the drain: read_ic's all-rank poll, or
   * restart's per-turn poll after it skips this rank's payload reads (goto finish_turn).
   * In every case the poll runs BEFORE any P/CellP is dereferenced, so the NULL arrays
   * here are never read. This is the graceful OOM exit. */
  if(alloc_fail_local)
    {
      gizmo_request_controlled_stop(1, "allocate_memory: UVM/STL particle allocation failed (drains at caller poll)", __FILE__, __LINE__, __FUNCTION__);
      return 1;
    }

  return 0;
}


/* ---------------------------------------------------------------------------------------
 * Particle-storage capacity changes.
 *
 * All.MaxPart is the physical capacity of P[] and of everything indexed alongside it. It is
 * per-rank and may move while the run is going, so the arrays sized by it move together here and
 * the capacity the rest of the code reads stays backed by real memory at every instant.
 *
 * One particle-capacity-sized array is deliberately not migrated: the saved Peano keys the
 * lightweight repartition reuses, which reallocate themselves whenever they are shorter than the
 * capacity. A grow is covered by that; a shrink leaves them oversized, costing a few bytes per
 * slot against the thousands the records here occupy.
 *
 * The arrays are migrated ONE AT A TIME, largest new allocation first, rather than allocating a
 * whole second set and swapping: only one array is ever duplicated, so the transient peak is the
 * current footprint plus the largest single new array instead of twice the footprint. Doubling
 * would be at its worst exactly when memory is tightest, which is when this is called at all.
 *
 * The advertised capacity is lowered before a shrink and raised after a grow -- in both cases
 * min(old,new) covers the migration -- so no array is ever smaller than the capacity the code
 * believes it has, including when an allocation fails part way. A failure therefore does not
 * restore the previous state: it leaves some arrays oversized (harmless and invisible), the
 * accounting truthful, and a controlled stop requested for the caller to drain.
 * -------------------------------------------------------------------------------------- */

/*! Replace a SharedSpace buffer with one of new_bytes, carrying over the first copy_bytes.
 *  Zero new_bytes is a legitimate request (the array does not exist at the new capacity) and
 *  leaves the slot NULL. Returns nonzero only on a real allocation failure, in which case the
 *  slot still holds the old buffer. */
static int resize_migrate_uvm(void **slot, size_t new_bytes, size_t copy_bytes, const char *label)
{
  void *fresh = (new_bytes > 0) ? gpu_particles_uvm_alloc(new_bytes, label) : NULL;
  if(new_bytes > 0 && fresh == NULL) {return 1;}
  if(fresh && *slot && copy_bytes > 0) {memcpy(fresh, *slot, copy_bytes);}
  if(*slot) {gpu_particles_uvm_free(*slot);}
  *slot = fresh;
  return 0;
}

/*! As above for a host-malloc buffer. The whole new buffer is zeroed before the prefix copy so
 *  that tail slots look exactly like a fresh allocation (the CHIMES sub-array pointers in
 *  particular must start NULL); the copied prefix carries ownership of anything the live slots
 *  own, which is why those entries are never released here. */
static int resize_migrate_host(void **slot, size_t new_bytes, size_t copy_bytes)
{
  void *fresh = (new_bytes > 0) ? malloc(new_bytes) : NULL;
  if(new_bytes > 0 && fresh == NULL) {return 1;}
  if(fresh) {memset(fresh, 0, new_bytes);}
  if(fresh && *slot && copy_bytes > 0) {memcpy(fresh, *slot, copy_bytes);}
  if(*slot) {free(*slot);}
  *slot = fresh;
  return 0;
}

int resize_particle_storage(int new_maxpart)
{
  if(new_maxpart == All.MaxPart) {return 0;}   /* must stay the first statement: the common case is
                                                  a domain boundary asking for the capacity it already
                                                  has, and it must cost one comparison */

  const int old_maxpart    = All.MaxPart;
  const int old_maxpartgas = All.MaxPartGas;
  const int growing        = (new_maxpart > old_maxpart);
  /* Whether gas storage exists is a property of the run, fixed when the ICs were read. Read it
   * from the storage that exists, never from All.TotN_gas: that count falls as gas turns into
   * stars, so a run that converted its last gas cell would silently lose CellP[] here. */
  const int new_maxpartgas = (old_maxpartgas > 0) ? new_maxpart : 0;

  if(new_maxpart > All.TreeNodeIndexBase)
    {
      gizmo_request_controlled_stop(7724, "resize_particle_storage: requested particle capacity would overlap the tree node index base fixed at startup", __FILE__, __LINE__, __FUNCTION__);
      return 7724;
    }
#ifdef SUBFIND
  if(subfind_loctree_is_allocated())
    {
      gizmo_request_controlled_stop(7723, "resize_particle_storage: called while the halo-finder local tree is standing (it offsets the shared Nodes pointer by the particle capacity)", __FILE__, __LINE__, __FUNCTION__);
      return 7723;
    }
#endif
  if(!growing)
    {
      /* Releasing capacity is only safe where nothing else is standing on it: imported ghosts
       * live in P[] above the local particles, and a built tree indexes particle slots. Growing
       * is safe in both of those states, which is what makes it usable mid-step. */
      if(new_maxpart < NumPart)
        {
          gizmo_request_controlled_stop(7720, "resize_particle_storage: requested particle capacity is below the live particle count", __FILE__, __LINE__, __FUNCTION__);
          return 7720;
        }
      if(ghost_pool_is_live() || force_tree_is_allocated())
        {
          gizmo_request_controlled_stop(7723, "resize_particle_storage: capacity may only be released with no imported ghosts and no tree standing", __FILE__, __LINE__, __FUNCTION__);
          return 7723;
        }
    }

  /* Drop the derived views of the storage BEFORE anything moves: the arena aliases P/CellP, and
   * the wakeup index is a position-dependent superset of P[]. The next arena acquire re-binds and
   * the next wakeup pass rebuilds from the authoritative P[].wakeup. */
  gpu_particles_arena_release();
  WakeupDirtyValid = 0;

  /* Advertise the smaller of the two for the duration of the migration, so the capacity the rest
   * of the code reads is backed by every array at every instant. The sync that follows does two
   * jobs: it carries that capacity to the device copies of All, and its leading fence drains any
   * kernel still reading the old buffers before the first one is released -- growing is legal
   * with a tree standing and ghosts imported, so in-flight work is not hypothetical there. */
  if(!growing) {All.MaxPart = new_maxpart; All.MaxPartGas = new_maxpartgas;}
  gizmo_gpu_sync_all();

  const size_t copy_particles = (size_t) NumPart;

  enum {SLOT_P, SLOT_CELLP, SLOT_WAKEUPDIRTY, SLOT_PROCESSEDFLAG, SLOT_TIMEBIN,
#ifdef CHIMES
        SLOT_CHIMES,
#endif
        SLOT_COUNT};
  long long new_bytes[SLOT_COUNT], old_bytes[SLOT_COUNT];
  new_bytes[SLOT_P]             = (long long) new_maxpart    * (long long) sizeof(struct particle_data);
  new_bytes[SLOT_CELLP]         = (long long) new_maxpartgas * (long long) sizeof(struct gas_cell_data);
  new_bytes[SLOT_WAKEUPDIRTY]   = (long long) new_maxpart    * (long long) sizeof(unsigned char);
  new_bytes[SLOT_PROCESSEDFLAG] = (long long) new_maxpart    * (long long) sizeof(unsigned char);
  new_bytes[SLOT_TIMEBIN]       = 3LL * (long long) new_maxpart * (long long) sizeof(int);
  old_bytes[SLOT_P]             = (long long) old_maxpart    * (long long) sizeof(struct particle_data);
  old_bytes[SLOT_CELLP]         = (long long) old_maxpartgas * (long long) sizeof(struct gas_cell_data);
  old_bytes[SLOT_WAKEUPDIRTY]   = (long long) old_maxpart    * (long long) sizeof(unsigned char);
  old_bytes[SLOT_PROCESSEDFLAG] = (long long) old_maxpart    * (long long) sizeof(unsigned char);
  old_bytes[SLOT_TIMEBIN]       = 3LL * (long long) old_maxpart * (long long) sizeof(int);
#ifdef CHIMES
  new_bytes[SLOT_CHIMES]        = (long long) new_maxpartgas * (long long) sizeof(struct gasVariables);
  old_bytes[SLOT_CHIMES]        = (long long) old_maxpartgas * (long long) sizeof(struct gasVariables);
#endif

  /* Descending by new size. Which array is largest depends on the physics compiled in -- the gas
   * cell record is the bigger of the two in some configurations and the smaller in others -- so
   * the order is derived from the sizes rather than written down. */
  int order[SLOT_COUNT], nsteps = 0;
  for(int s = 0; s < SLOT_COUNT; s++)
    {
      if(new_bytes[s] == 0 && old_bytes[s] == 0) {continue;}   /* array absent in this run */
      int k = nsteps++;
      while(k > 0 && new_bytes[order[k - 1]] < new_bytes[s]) {order[k] = order[k - 1]; k--;}
      order[k] = s;
    }

  int failed_slot = -1;
  const char *failed_what = NULL;
  for(int step = 0; step < nsteps && failed_slot < 0; step++)
    {
      const int s = order[step];
      int bad = 0;
      switch(s)
        {
        case SLOT_P:
          {
            void *slot = P;
            bad = resize_migrate_uvm(&slot, (size_t) new_bytes[s], copy_particles * sizeof(struct particle_data), "particle_soa_P");
            P = (struct particle_data *) slot;
            failed_what = "P";
          }
          break;
        case SLOT_CELLP:
          /* The copied range is the particle count, NOT the gas count: with imported ghosts
           * present a gas record can sit at any index P[] can reach, above the local gas cells. */
          {
            void *slot = CellP;
            bad = resize_migrate_uvm(&slot, (size_t) new_bytes[s], copy_particles * sizeof(struct gas_cell_data), "particle_soa_CellP");
            CellP = (struct gas_cell_data *) slot;
            failed_what = "CellP";
          }
          break;
        case SLOT_WAKEUPDIRTY:
          {
            void *slot = WakeupDirty;
            bad = resize_migrate_uvm(&slot, (size_t) new_bytes[s], copy_particles * sizeof(unsigned char), "particle_soa_wakeupdirty");
            WakeupDirty = (unsigned char *) slot;
            failed_what = "WakeupDirty";
          }
          break;
        case SLOT_PROCESSEDFLAG:
          /* Nothing to carry over: it is rewritten in full at the start of each gravity walk. */
          {
            void *slot = ProcessedFlag;
            bad = resize_migrate_host(&slot, (size_t) new_bytes[s], 0);
            ProcessedFlag = (unsigned char *) slot;
            failed_what = "ProcessedFlag";
          }
          break;
        case SLOT_TIMEBIN:
          {
            /* Built fresh and swapped in rather than resized in place: shrinking a vector keeps
             * its old capacity, so this is the only form that actually releases the memory, and
             * a throw part way through leaves the live vectors untouched. The threading itself is
             * rebuilt by reconstruct_timebins() after the decomposition; carrying the live prefix
             * only keeps the contents from being garbage in between. */
            std::vector<int> fresh_next, fresh_prev, fresh_active;
            try {
              fresh_next.resize(new_maxpart);
              fresh_prev.resize(new_maxpart);
              fresh_active.reserve(new_maxpart);
              fresh_active.assign(ActiveParticleList.begin(), ActiveParticleList.end());
            } catch(const std::bad_alloc&) { bad = 1; }
            if(!bad)
              {
                const size_t ncopy = (copy_particles < (size_t) new_maxpart) ? copy_particles : (size_t) new_maxpart;
                if(ncopy > 0)
                  {
                    memcpy(fresh_next.data(), NextInTimeBin.data(), ncopy * sizeof(int));
                    memcpy(fresh_prev.data(), PrevInTimeBin.data(), ncopy * sizeof(int));
                  }
                NextInTimeBin.swap(fresh_next);
                PrevInTimeBin.swap(fresh_prev);
                ActiveParticleList.swap(fresh_active);
              }
            failed_what = "timebin lists";
          }
          break;
#ifdef CHIMES
        case SLOT_CHIMES:
          /* Local gas only, and unlike CellP it has no meaning for imported ghosts. Each live
           * entry owns heap sub-arrays; copying the record moves that ownership to the new array,
           * so the old container is released without touching what its entries pointed at. */
          {
            const size_t copy_gas = (size_t) ((N_gas < new_maxpartgas) ? N_gas : new_maxpartgas);
            void *slot = ChimesGasVars;
            bad = resize_migrate_host(&slot, (size_t) new_bytes[s], copy_gas * sizeof(struct gasVariables));
            ChimesGasVars = (struct gasVariables *) slot;
            failed_what = "ChimesGasVars";
          }
          break;
#endif
        }
      if(bad) {failed_slot = s;}
      else if(s == SLOT_TIMEBIN)
        {
          gizmo_mem_account_set(GIZMO_MEM_STL_TIMEBIN,
                                (long long)(ActiveParticleList.capacity() + NextInTimeBin.capacity()
                                            + PrevInTimeBin.capacity()) * (long long) sizeof(int));
        }
      else
        {
          gizmo_mem_account_add(GIZMO_MEM_PARTICLE_SOA, new_bytes[s] - old_bytes[s]);
        }
    }

  if(failed_slot >= 0)
    {
      /* Arrays that already moved keep the new capacity. Oversized is invisible; undersized
       * cannot happen, because the advertised capacity is min(old,new) throughout. The device
       * copy of All is deliberately left alone -- it already reports a capacity every array can
       * satisfy, and the caller drains this stop before anything dispatches. */
      char msg[256];
      snprintf(msg, sizeof(msg), "resize_particle_storage: could not allocate %s at %d slots (%.1f MB); storage left at a capacity every array satisfies",
               failed_what, new_maxpart, (double) new_bytes[failed_slot] / (1024.0 * 1024.0));
      gizmo_request_controlled_stop(7721, msg, __FILE__, __LINE__, __FUNCTION__);
      return 7721;
    }

  All.MaxPart = new_maxpart;
  All.MaxPartGas = new_maxpartgas;
  gizmo_gpu_sync_all();   /* All is mirrored per GPU translation unit; the capacity must reach
                             those copies before the next kernel reads it */
  return 0;
}
