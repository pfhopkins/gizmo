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
 * no invalidate.  See HANDOFF_fire_m11i_arena_sweep.md.
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

  /* COLLECTIVE-SYMMETRY (codex 2026-06-04): allocate_memory() is reached from TWO
   * kinds of caller, distinguished by do_collective_preflight:
   *  (a) =1 -- an ALL-RANK setup phase (read_ic, right after find_files() broadcasts
   *      file-0's header to EVERY rank, so the global counts are known symmetrically).
   *      Here a collective arena preflight is safe, and a UVM/STL failure sets a SOFT
   *      bad-stop that the caller's immediately-following all-rank poll reconciles
   *      BEFORE any P/CellP is dereferenced. This is the graceful OOM path.
   *  (b) =0 -- a SUBSET/turn caller (restart.cc groupTask turn). It MUST NOT enter any
   *      MPI_COMM_WORLD collective here (peers are parked at a barrier -> deadlock), so
   *      arena OOM keeps the mymalloc emergency-hold floor and a UVM/STL failure keeps
   *      the local emergency-hold floor. (restart's own all-rank preflight is a tracked
   *      follow-up; until then this path is the Class-2 floor.) */

  if(do_collective_preflight)
    {
      /* All-rank arena (mymalloc) preflight. MUST mirror the arena allocations below:
       * Exportflag/Exportindex/Exportnodecount (NTaskTimesThreads ints each, x3),
       * Send/Recv count/offset (NTask ints each, x4), ProcessedFlag (MaxPart uchar),
       * and (CHIMES) ChimesGasVars. The UVM particle arrays (P/CellP) and the STL
       * timebin vectors are NOT in the mymalloc arena and cannot be capacity-checked
       * ahead of kokkos_malloc/std::vector -- they use attempt-then-soft-bad-stop +
       * the caller's poll instead (see the failure handler at the end). */
      size_t arena_need = (size_t) 3 * gizmo_mymalloc_rounded_size(NTaskTimesThreads * sizeof(int))
                        + (size_t) 4 * gizmo_mymalloc_rounded_size(NTask * sizeof(int))
                        + gizmo_mymalloc_rounded_size(All.MaxPart * sizeof(unsigned char));
      int arena_blocks = 8;
#ifdef CHIMES
      arena_need += gizmo_mymalloc_rounded_size(All.MaxPartGas * sizeof(struct gasVariables));
      arena_blocks += 1;
#endif
      if(!gizmo_alloc_fits_all_ranks(arena_need, arena_blocks))
        {
          gizmo_request_controlled_stop(812, "allocate_memory: particle-storage arena preflight won't fit on >=1 rank", __FILE__, __LINE__, __FUNCTION__);
          return 1;
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
   * ONE local failure flag across all of them; the failure handler at the end routes
   * it by caller context (soft bad-stop + caller poll under do_collective_preflight,
   * else the emergency-hold floor -- see the note at the top). Once the flag is set,
   * later optional allocations are skipped. Nothing between here and that handler
   * dereferences P/CellP, so the all-rank caller's poll runs first on the soft path.
   * An uncaught std::bad_alloc is also barred. */
  int alloc_fail_local = 0;

  try {
    ActiveParticleList.reserve(All.MaxPart);
    NextInTimeBin.resize(All.MaxPart);
    PrevInTimeBin.resize(All.MaxPart);
  } catch(const std::bad_alloc&) { alloc_fail_local = 1; printf("allocate_memory: STL timebin vector alloc (MaxPart=%d) threw bad_alloc.\n", All.MaxPart); fflush(stdout); }


  if(All.MaxPart > 0 && !alloc_fail_local)
    {
      bytes = All.MaxPart * sizeof(struct particle_data);
      P = (struct particle_data *) gpu_particles_uvm_alloc(bytes);
      if(P == NULL) { alloc_fail_local = 1; printf("failed to allocate memory for particle data storage structure `P' (%g MB).\n", bytes / (1024.0 * 1024.0)); fflush(stdout); }
      else { bytes_tot += bytes; if(ThisTask == 0) {printf("Allocated %g MByte for particle data storage (UVM canonical, SharedSpace).\n", bytes_tot / (1024.0 * 1024.0));} }
    }

  if(All.MaxPartGas > 0 && !alloc_fail_local)
    {
      bytes_tot = 0;

      bytes = All.MaxPartGas * sizeof(struct gas_cell_data);
      CellP = (struct gas_cell_data *) gpu_particles_uvm_alloc(bytes);
      if(CellP == NULL) { alloc_fail_local = 1; printf("failed to allocate memory for gas cell data storage structure (%g MB).\n", bytes / (1024.0 * 1024.0)); fflush(stdout); }
      else { bytes_tot += bytes; if(ThisTask == 0) {printf("Allocated %g MByte for storage of hydro data (UVM canonical, SharedSpace).\n", bytes_tot / (1024.0 * 1024.0));} }

#ifdef CHIMES
      /* ChimesGasVars is an arena block (counted in the do_collective_preflight arena
       * estimate above). Only allocate when the UVM allocs above succeeded on THIS rank,
       * so we never mymalloc on an already-failing path; the failure handler below then
       * routes the bad-stop (soft+caller-poll under preflight, else emergency-hold). */
      if(!alloc_fail_local)
	{
	  ChimesGasVars = (struct gasVariables *) mymalloc("gasVars", bytes = All.MaxPartGas * sizeof(struct gasVariables));
	  bytes_tot += bytes;
	  if(ThisTask == 0) printf("Allocated %g MByte for storage of ChimesGasVars data.\n", bytes_tot / (1024.0 * 1024.0));
	}
#endif
    }

  /* UVM/STL allocation failure handler. */
  if(alloc_fail_local)
    {
      if(do_collective_preflight)
        {
          /* All-rank setup phase: SOFT bad-stop only. The caller's immediately-following
           * all-rank poll (Allreduce) reconciles this rank's (asymmetric) failure with its
           * peers and finalizes cleanly -- and it runs BEFORE any P/CellP is dereferenced,
           * so the NULL arrays here are never read. This is the graceful OOM exit. */
          gizmo_request_controlled_stop(1, "allocate_memory: UVM/STL particle allocation failed (all-rank setup phase; drains at caller poll)", __FILE__, __LINE__, __FUNCTION__);
          return 1;
        }
      /* Subset/turn caller (restart): no symmetric poll reachable here -> emergency-hold
       * floor (Class-2; scancel-killable, no MPI_Abort). Tracked: restart all-rank preflight. */
      gizmo_emergency_hold_reviewed(1, "TEMP_HARD_CANDIDATE_OOM: UVM/STL particle allocation failed (allocate_memory subset/turn-called; no symmetric poll)", __FILE__, __LINE__, __FUNCTION__);
    }

  return 0;
}
