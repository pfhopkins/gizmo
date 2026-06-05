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

void allocate_memory(void)
{
  size_t bytes;

  double bytes_tot = 0;

  int NTaskTimesThreads;

  NTaskTimesThreads = maxThreads * NTask;

  /* COLLECTIVE-SYMMETRY NOTE (codex 2026-06-02): allocate_memory() is called
   * from SUBSET/turn contexts -- file_io/read_ic.cc read_file() (one rank, or
   * one group, per driver-loop round, with the other ranks waiting at the
   * MPI_Barrier) and the per-rank file_io/restart.cc groupTask turn. It MUST
   * therefore contain NO MPI_COMM_WORLD collective: an Allreduce here would
   * deadlock against the peers parked at that barrier. An earlier draft added a
   * collective arena preflight at this point; it has been REMOVED. Arena
   * (mymalloc) OOM now routes through mymalloc's own emergency-hold path, and
   * the UVM/STL failure below routes through a local emergency-hold path.
   * NEITHER is the target -- gizmo_emergency_hold_reviewed is the SSOT last-resort
   * (print/flush/fence, then a scancel-killable host hold; NO MPI_Abort in the
   * default path as of 2026-06-04 `15153b17`): a safe floor, but Class-2, not
   * graceful. TEMP follow-up (named allocator Pass-B workstream): a real graceful
   * allocator needs a separate
   * all-rank "compute sizes/header -> allocate arrays" phase run BEFORE the
   * per-file/per-rank IO, where a collective preflight is symmetric. */

  Exportflag = (int *) mymalloc("Exportflag", NTaskTimesThreads * sizeof(int));
  Exportindex = (int *) mymalloc("Exportindex", NTaskTimesThreads * sizeof(int));
  Exportnodecount = (int *) mymalloc("Exportnodecount", NTaskTimesThreads * sizeof(int));

  Send_count = (int *) mymalloc("Send_count", sizeof(int) * NTask);
  Send_offset = (int *) mymalloc("Send_offset", sizeof(int) * NTask);
  Recv_count = (int *) mymalloc("Recv_count", sizeof(int) * NTask);
  Recv_offset = (int *) mymalloc("Recv_offset", sizeof(int) * NTask);

  ProcessedFlag = (unsigned char *) mymalloc("ProcessedFlag", bytes = All.MaxPart * sizeof(unsigned char));
  bytes_tot += bytes;

  /* ---- Non-arena allocations (UVM + STL) use attempt-then-check (they return
   * NULL / throw, unlike the arena path). Accumulate ONE local failure flag
   * across all of them. Because allocate_memory() is subset/turn-called (see
   * note above), this failure is NOT checked by an MPI collective -- it routes
   * to the single emergency-hold path (TEMP_HARD_CANDIDATE_OOM, Class-2) at the
   * end (an earlier draft used a combined Allreduce here; removed -- it would
   * deadlock). Once the flag is set, later optional allocations are skipped
   * (surgical; we are going to emergency-hold anyway). Nothing between here and the
   * final check dereferences P/CellP. An uncaught std::bad_alloc is also barred. */
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
      /* ChimesGasVars is arena (covered by the combined preflight above). Only
       * allocate when the UVM allocs above succeeded on THIS rank, so we never
       * mymalloc on an already-failing path; the final collective check below
       * still makes all ranks bad-stop together. */
      if(!alloc_fail_local)
	{
	  ChimesGasVars = (struct gasVariables *) mymalloc("gasVars", bytes = All.MaxPartGas * sizeof(struct gasVariables));
	  bytes_tot += bytes;
	  if(ThisTask == 0) printf("Allocated %g MByte for storage of ChimesGasVars data.\n", bytes_tot / (1024.0 * 1024.0));
	}
#endif
    }

  /* UVM/STL allocation failure: NO collective here (allocate_memory() is
   * subset/turn-called -- see note above). Route this rank's failure through the
   * single reviewed TEMPORARY hard path (no MPI). NOT permanent; requires a
   * Vista receipt + the all-rank-allocation-phase follow-up. */
  if(alloc_fail_local)
    {
      gizmo_emergency_hold_reviewed(1, "TEMP_HARD_CANDIDATE_OOM: UVM/STL particle allocation failed (allocate_memory subset/turn-called; no symmetric poll)", __FILE__, __LINE__, __FUNCTION__);
    }




}
