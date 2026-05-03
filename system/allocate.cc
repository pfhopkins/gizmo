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

  Exportflag = (int *) mymalloc("Exportflag", NTaskTimesThreads * sizeof(int));
  Exportindex = (int *) mymalloc("Exportindex", NTaskTimesThreads * sizeof(int));
  Exportnodecount = (int *) mymalloc("Exportnodecount", NTaskTimesThreads * sizeof(int));

  Send_count = (int *) mymalloc("Send_count", sizeof(int) * NTask);
  Send_offset = (int *) mymalloc("Send_offset", sizeof(int) * NTask);
  Recv_count = (int *) mymalloc("Recv_count", sizeof(int) * NTask);
  Recv_offset = (int *) mymalloc("Recv_offset", sizeof(int) * NTask);

  ProcessedFlag = (unsigned char *) mymalloc("ProcessedFlag", bytes = All.MaxPart * sizeof(unsigned char));
  bytes_tot += bytes;

  ActiveParticleList.reserve(All.MaxPart);

  NextInTimeBin.resize(All.MaxPart);
  PrevInTimeBin.resize(All.MaxPart);


  if(All.MaxPart > 0)
    {
      bytes = All.MaxPart * sizeof(struct particle_data);
      P = (struct particle_data *) gpu_particles_uvm_alloc(bytes);
      if(!P)
	{
	  printf("failed to allocate memory for particle data storage structure `P' (%g MB).\n", bytes / (1024.0 * 1024.0));
	  endrun(1);
	}
      bytes_tot += bytes;

      if(ThisTask == 0) {printf("Allocated %g MByte for particle data storage (UVM canonical, SharedSpace).\n", bytes_tot / (1024.0 * 1024.0));}
    }

  if(All.MaxPartGas > 0)
    {
      bytes_tot = 0;

      bytes = All.MaxPartGas * sizeof(struct gas_cell_data);
      CellP = (struct gas_cell_data *) gpu_particles_uvm_alloc(bytes);
      if(!CellP)
	{
	  printf("failed to allocate memory for gas cell data storage structure (%g MB).\n", bytes / (1024.0 * 1024.0));
	  endrun(1);
	}
      bytes_tot += bytes;

      if(ThisTask == 0) {printf("Allocated %g MByte for storage of hydro data (UVM canonical, SharedSpace).\n", bytes_tot / (1024.0 * 1024.0));}

#ifdef CHIMES 
      if (!(ChimesGasVars = (struct gasVariables *) mymalloc("gasVars", bytes = All.MaxPartGas * sizeof(struct gasVariables)))) 
	{
	  printf("failed to allocate memory for 'ChimesGasVars' (%g MB).\n", bytes / (1024.0 * 1024.0));
	  endrun(1); 
	}
      bytes_tot += bytes; 

      if(ThisTask == 0)
	printf("Allocated %g MByte for storage of ChimesGasVars data.\n", bytes_tot / (1024.0 * 1024.0));
#endif
    }




}
