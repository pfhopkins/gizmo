#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "gpu_particles_arena.h"

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
   * ints each, x3), Send/Recv count/offset (NTask ints each, x4), ProcessedFlag (MaxPart
   * uchar), and (CHIMES) ChimesGasVars. The UVM particle arrays (P/CellP) and the STL
   * timebin vectors are NOT in the mymalloc arena and cannot be capacity-checked ahead of
   * kokkos_malloc/std::vector -- they use attempt-then-soft-bad-stop (see the handler below). */
  {
    size_t arena_need = (size_t) 3 * gizmo_mymalloc_rounded_size(NTaskTimesThreads * sizeof(int))
                      + (size_t) 4 * gizmo_mymalloc_rounded_size(NTask * sizeof(int))
                      + gizmo_mymalloc_rounded_size(All.MaxPart * sizeof(unsigned char));
    int arena_blocks = 8;
#ifdef CHIMES
    arena_need += gizmo_mymalloc_rounded_size(All.MaxPartGas * sizeof(struct gasVariables));
    arena_blocks += 1;
#endif
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

  ProcessedFlag = (unsigned char *) mymalloc("ProcessedFlag", bytes = All.MaxPart * sizeof(unsigned char));
  bytes_tot += bytes;

  /* ---- Non-arena allocations (UVM particle arrays + STL timebin vectors) use
   * attempt-then-check (they return NULL / throw, unlike the arena path). Accumulate
   * ONE local failure flag across all of them; the failure handler at the end requests a
   * soft bad-stop + returns (the caller drains). Once the flag is set, later optional
   * allocations are skipped. Nothing between here and that handler dereferences P/CellP,
   * so the caller's poll runs first. An uncaught std::bad_alloc is also barred. */
  int alloc_fail_local = 0;

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
      /* ChimesGasVars is an arena block (counted in the arena preflight above). Only allocate
       * when the UVM allocs above succeeded on THIS rank, so we never mymalloc on an
       * already-failing path; the failure handler below requests the soft bad-stop. */
      if(!alloc_fail_local)
	{
	  ChimesGasVars = (struct gasVariables *) mymalloc("gasVars", bytes = All.MaxPartGas * sizeof(struct gasVariables));
	  bytes_tot += bytes;
	  if(ThisTask == 0) printf("Allocated %g MByte for storage of ChimesGasVars data.\n", bytes_tot / (1024.0 * 1024.0));
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
 * Adaptive growth of the particle arrays.
 *
 * Ghosts are appended into P[]/CellP[] at [NumPart, NumPart+NumGhost), so ghost-import
 * demand is bounded by All.MaxPart = PartAllocFactor * (TotNumPart / NTask). That sizing
 * scales headroom with the LOCAL PARTICLE COUNT, while ghost demand scales with the
 * DOMAIN SURFACE. Those diverge badly under heavy decomposition: at 27001 particles over
 * 48 ranks (562/rank) an evrard collapse wants ~5156 ghosts against 562 local -- a 9:1
 * ghost:local ratio -- so even PartAllocFactor=10 leaves nothing, and ghost_exchange
 * bad-stops with code 7702 having missed by under 2%.
 *
 * Rather than making the user pre-tune PartAllocFactor for the decomposition, grow on
 * demand. This mirrors what the LET foreign arena already does one frame up
 * (forcetree.cc: "growing adaptive floor ... and rebuilding the tree (retry 1/3)").
 *
 * Safe because the particle arrays are plain pointers everywhere: no Kokkos View is
 * constructed over P/CellP (they are UVM-canonical and passed as raw pointers), the arena
 * only records the pointer (gpu_particles_arena_acquire) so re-acquiring is a single call,
 * and the ~10 `args.P = P` sites are all rebuilt per call rather than cached across steps.
 *
 * Growth is LOCAL to a rank -- each rank sizes to its own demand. All.MaxPart is per-rank
 * state and no collective depends on its value, only on the ghost counts already
 * exchanged. Callers must still reach their collective poll in lockstep.
 *
 * Returns 1 on success (arrays grown, All.MaxPart raised), 0 on failure (arrays untouched,
 * caller should fall back to its honest bad-stop).
 * -------------------------------------------------------------------------------------*/
int gizmo_grow_particle_storage(long long needed_slots, const char *why)
{
  if(needed_slots <= (long long) All.MaxPart) {return 1;}          /* already fits */

  /* Headroom beyond the immediate need, so a slowly-growing demand does not realloc every
   * step. Cap at INT_MAX/2 to keep the int-typed MaxPart and all its downstream index
   * arithmetic well clear of overflow. */
  long long want = needed_slots + needed_slots / 8 + 64;
  if(want > (long long) (INT_MAX / 2)) {want = (long long) (INT_MAX / 2);}
  if(want <= (long long) All.MaxPart) {return 0;}
  int new_MaxPart = (int) want;
  int old_MaxPart_local = All.MaxPart;
  int new_MaxPartGas = All.MaxPartGas;
  /* CellP is sized by MaxPartGas. Gas ghosts land in the same index range, so grow it in
   * step whenever it was tracking MaxPart (the common case, incl. restart.cc's
   * MaxPartGas = MaxPart). If it was deliberately smaller, scale it by the same factor. */
  if(All.MaxPartGas >= old_MaxPart_local) {new_MaxPartGas = new_MaxPart;}
  else {
      long long g = (long long) All.MaxPartGas * (long long) new_MaxPart / (long long) (old_MaxPart_local > 0 ? old_MaxPart_local : 1);
      new_MaxPartGas = (int) ((g > (long long)(INT_MAX/2)) ? (INT_MAX/2) : g);
  }

  /* Allocate the replacements FIRST; only commit once every one succeeded, so a partial
   * OOM leaves the run exactly as it was and the caller's bad-stop is still truthful. */
  struct particle_data *newP = (struct particle_data *) gpu_particles_uvm_alloc((size_t) new_MaxPart * sizeof(struct particle_data), "particle_soa_P_grown");
  struct gas_cell_data *newCellP = NULL;
  unsigned char *newWakeupDirty = NULL;
  unsigned char *newProcessedFlag = NULL;
  if(newP && new_MaxPartGas > 0) {newCellP = (struct gas_cell_data *) gpu_particles_uvm_alloc((size_t) new_MaxPartGas * sizeof(struct gas_cell_data), "particle_soa_CellP_grown");}
  if(newP && (newCellP || new_MaxPartGas <= 0)) {newWakeupDirty = (unsigned char *) gpu_particles_uvm_alloc((size_t) new_MaxPart * sizeof(unsigned char), "particle_soa_wakeupdirty_grown");}
  if(newWakeupDirty) {newProcessedFlag = (unsigned char *) malloc((size_t) new_MaxPart * sizeof(unsigned char));}

  if(!newP || (new_MaxPartGas > 0 && !newCellP) || !newWakeupDirty || !newProcessedFlag)
    {
      if(newProcessedFlag) {free(newProcessedFlag);}
      if(newWakeupDirty)   {gpu_particles_uvm_free(newWakeupDirty);}
      if(newCellP)         {gpu_particles_uvm_free(newCellP);}
      if(newP)             {gpu_particles_uvm_free(newP);}
      printf("Task %d: gizmo_grow_particle_storage(%lld -> MaxPart %d) FAILED to allocate; leaving arrays at MaxPart=%d (%s)\n",
             ThisTask, needed_slots, new_MaxPart, All.MaxPart, why ? why : "");
      fflush(stdout);
      return 0;
    }

  /* Carry over live contents. Only [0, NumPart) of P and [0, N_gas) of CellP are live;
   * gpu_particles_uvm_alloc already zeroed the rest. */
  int n_copy = (NumPart < old_MaxPart_local) ? NumPart : old_MaxPart_local;
  if(n_copy > 0 && P)         {memcpy(newP, P, (size_t) n_copy * sizeof(struct particle_data));}
  if(newCellP && CellP)
    {
      int ng_copy = (N_gas < All.MaxPartGas) ? N_gas : All.MaxPartGas;
      if(ng_copy > 0) {memcpy(newCellP, CellP, (size_t) ng_copy * sizeof(struct gas_cell_data));}
    }
  if(n_copy > 0 && WakeupDirty) {memcpy(newWakeupDirty, WakeupDirty, (size_t) n_copy * sizeof(unsigned char));}
  memset(newProcessedFlag, 0, (size_t) new_MaxPart * sizeof(unsigned char)); /* per-walk scratch: zeroed each gravity_tree() anyway */

  struct particle_data *oldP = P; struct gas_cell_data *oldCellP = CellP; unsigned char *oldWakeupDirty = WakeupDirty;
  P = newP; if(newCellP) {CellP = newCellP;} WakeupDirty = newWakeupDirty;
  /* ProcessedFlag came from the mymalloc arena, which is LIFO -- blocks allocated after it
   * sit above it, so it cannot be resized in place. It is never freed (it lives to process
   * exit), so repoint it at a plain allocation and leave the old arena block dead. That
   * block is All.MaxPart bytes (kilobytes here), and the arena is torn down wholesale at
   * exit, so nothing leaks past the run. */
  ProcessedFlag = newProcessedFlag;

  All.MaxPart = new_MaxPart;
  if(newCellP) {All.MaxPartGas = new_MaxPartGas;}

  /* STL timebin vectors are indexed by particle slot, so they track MaxPart too. */
  try {
    ActiveParticleList.reserve(All.MaxPart);
    NextInTimeBin.resize(All.MaxPart);
    PrevInTimeBin.resize(All.MaxPart);
  } catch(const std::bad_alloc&) {
    printf("Task %d: gizmo_grow_particle_storage: timebin vector resize to %d threw bad_alloc\n", ThisTask, All.MaxPart); fflush(stdout);
    return 0;   /* arrays already swapped and self-consistent; caller bad-stops */
  }

  /* The arena stores the raw pointer, so it must be re-pointed after the swap or every
   * device-side consumer keeps dereferencing the freed buffer. */
  gpu_particles_arena_set_site("gizmo_grow_particle_storage");
  gpu_particles_arena_acquire(All.MaxPart, P, CellP);

  gpu_particles_uvm_free(oldWakeupDirty);
  if(newCellP) {gpu_particles_uvm_free(oldCellP);}
  gpu_particles_uvm_free(oldP);

  printf("Task %d: grew particle storage MaxPart %d -> %d (MaxPartGas -> %d) for %s\n",
         ThisTask, old_MaxPart_local, All.MaxPart, All.MaxPartGas, why ? why : "demand");
  fflush(stdout);
  return 1;
}
