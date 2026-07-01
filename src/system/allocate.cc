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

  /* GRACEFUL-OOM (codex 2026-06-04): allocate_memory() never fatal hard-exits on a
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
