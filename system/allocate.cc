#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../gravity/forcetree.h"


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

  /* The blocks sized by the particle capacity are allocated MOVABLE, in this order, because
   * resize_particle_storage() grows them in place: a LIFO block can only be resized when
   * everything above it in the arena can be shifted out of the way, which is what the movable
   * flag records.  Nothing else about them changes -- they are never freed, and the blocks
   * allocated above them are still freed in the usual LIFO order. */
  ProcessedFlag = (unsigned char *) mymalloc_movable(&ProcessedFlag, "ProcessedFlag", bytes = All.MaxPart * sizeof(unsigned char));
  bytes_tot += bytes;

  ActiveParticleList.reserve(All.MaxPart);

  NextInTimeBin.resize(All.MaxPart);
  PrevInTimeBin.resize(All.MaxPart);


  if(All.MaxPart > 0)
    {
      if(!(P = (struct particle_data *) mymalloc_movable(&P, "ParticleData_P", bytes = All.MaxPart * sizeof(struct particle_data))))
	{
	  printf("failed to allocate memory for particle data storage structure `P' (%g MB).\n", bytes / (1024.0 * 1024.0));
	  endrun(1);
	}
      bytes_tot += bytes;

      if(ThisTask == 0) {printf("Allocated %g MByte for particle data storage.\n", bytes_tot / (1024.0 * 1024.0));}
    }

  if(All.MaxPartGas > 0)
    {
      bytes_tot = 0;

      if(!(CellP = (struct gas_cell_data *) mymalloc_movable(&CellP, "GasCellData_P", bytes = All.MaxPartGas * sizeof(struct gas_cell_data))))
    {
	  printf("failed to allocate memory for gas cell data storage structure (%g MB).\n", bytes / (1024.0 * 1024.0));
	  endrun(1);
	}
      bytes_tot += bytes;

      if(ThisTask == 0) {printf("Allocated %g MByte for storage of hydro data.\n", bytes_tot / (1024.0 * 1024.0));}

#ifdef CHIMES 
      if (!(ChimesGasVars = (struct gasVariables *) mymalloc_movable(&ChimesGasVars, "gasVars", bytes = All.MaxPartGas * sizeof(struct gasVariables)))) 
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


/* ---------------------------------------------------------------------------------------
 * Particle-storage capacity changes.
 *
 * All.MaxPart is the physical size of P[] and of everything indexed alongside it, and it is also
 * the gravity tree's node-index base (Nodes = Nodes_base - All.MaxPart; the root node is
 * All.MaxPart and pseudo-particles start at All.MaxPart + MaxNodes).  Those node indices travel
 * between ranks inside exported node lists, and the domain exchange compares each rank's load
 * against the capacity while iterating over globally-reduced counts.  The capacity is therefore
 * rank-identical, and raising it is a collective decision: the caller is responsible for passing
 * the same arguments on every rank (see domain_raise_particle_capacity in domain.cc).  Because
 * the base moves with the capacity, this may only run with no tree standing -- the domain
 * boundary, where the tree has just been freed and will be rebuilt from the new value.
 *
 * Capacity only ever rises.  A release would have to be agreed collectively too, and the
 * refinement that asks for growth is not a transient: the memory would be wanted straight back.
 *
 * The blocks are grown in place, bottom-up through the arena, so each grow shifts only the blocks
 * above it and each of those is still at its old size when it moves.  Nothing is duplicated, so
 * the transient footprint is the final one -- which matters, because this runs precisely when
 * memory is tight.  What it costs instead is that a grow cannot be undone, so the whole set is
 * checked against the arena before the first one is touched.
 * -------------------------------------------------------------------------------------- */

/*! Grow one movable arena block to new_bytes, carrying its contents, and publish the new pointer
 *  through *slot.  Growth only: shrinking an arena block gains nothing here and would have to
 *  shift every block above it twice. */
static void resize_grow_block(void **slot, size_t new_bytes)
{
  if(new_bytes > mymalloc_block_size(*slot)) {*slot = myrealloc_movable(*slot, new_bytes);}
}

/*! Everything that can refuse a capacity change, with no side effect, so the collective decision can be
 *  gated on every rank's answer before any rank moves a block.  Returns 0 when the change would go through
 *  and the matching code from resize_particle_storage otherwise; complain selects whether the reason is
 *  printed, so the predicate used for the gate stays silent and the attempt that follows explains itself. */
static int resize_particle_storage_check(int new_maxpart, int new_maxpartgas, int complain)
{
  if(new_maxpart < All.MaxPart || new_maxpartgas < All.MaxPartGas)
    {
      if(complain) {printf("Task=%d: resize_particle_storage asked to release capacity (%d/%d gas slots, from %d/%d); capacity only rises.\n",
                           ThisTask, new_maxpart, new_maxpartgas, All.MaxPart, All.MaxPartGas); fflush(stdout);}
      return 7720;
    }
  if(new_maxpart < NumPart || new_maxpartgas < N_gas)
    {
      if(complain) {printf("Task=%d: resize_particle_storage asked for %d/%d gas slots, below the %d/%d already here.\n",
                           ThisTask, new_maxpart, new_maxpartgas, NumPart, N_gas); fflush(stdout);}
      return 7720;
    }
  if(new_maxpart > All.MaxPartExpandable)
    {
      if(complain && ThisTask == 0)
        {
          printf("Particle capacity cannot be raised to %d slots: this run's ceiling is %d, fixed from Max_Memory_Per_MPI_Task_in_MB (%d MB) when it started. "
                 "Spread the run over more ranks, or give each rank a larger memory budget and start a fresh run.\n",
                 new_maxpart, All.MaxPartExpandable, All.MaxMemSize); fflush(stdout);
        }
      return 7721;
    }
  if(force_tree_is_allocated())
    {
      if(complain) {printf("Task=%d: resize_particle_storage called with a tree standing; the capacity is that tree's node-index base.\n", ThisTask); fflush(stdout);}
      return 7722;
    }

#ifndef DISABLE_MEMORY_MANAGER
  /* One grow cannot be undone, so the whole set is priced against the arena together: every block must have
   * only movable blocks above it, and the extra bytes they all want must already be free.  Checking them one
   * at a time would let the second request fail with the first already committed. */
  {
    void *const blocks[] = {(void *) ProcessedFlag, (void *) P, (void *) CellP
#ifdef CHIMES
                            , (void *) ChimesGasVars
#endif
                           };
    const size_t wanted[] = {(size_t) new_maxpart * sizeof(unsigned char),
                             (size_t) new_maxpart * sizeof(struct particle_data),
                             (size_t) new_maxpartgas * sizeof(struct gas_cell_data)
#ifdef CHIMES
                             , (size_t) new_maxpartgas * sizeof(struct gasVariables)
#endif
                            };
    size_t extra = 0;
    for(int b = 0; b < (int) (sizeof(blocks) / sizeof(blocks[0])); b++)
      {
        if(blocks[b] == NULL) {continue;}   /* array absent in this run (no gas cells) */
        const char *why = "the memory manager refused it";
        if(!mymalloc_movable_can_resize(blocks[b], wanted[b], &why))
          {
            if(complain) {printf("Task=%d: cannot raise the particle capacity to %d slots: %s.\n", ThisTask, new_maxpart, why); fflush(stdout);}
            return 7723;
          }
        const size_t now = mymalloc_block_size(blocks[b]);
        const size_t then = mymalloc_rounded_size(wanted[b]);
        if(then > now) {extra += then - now;}
      }
    if(extra > FreeBytes)
      {
        if(complain)
          {
            dump_memory_table();
            printf("Task=%d: cannot raise the particle capacity from %d to %d slots: it needs %g MB more and only %g MB of the %d MB budget is free. "
                   "Raise Max_Memory_Per_MPI_Task_in_MB, or spread the run over more ranks.\n",
                   ThisTask, All.MaxPart, new_maxpart, extra / (1024.0 * 1024.0), FreeBytes / (1024.0 * 1024.0), All.MaxMemSize);
            fflush(stdout);
          }
        return 7724;
      }
  }
#endif

  return 0;
}

int resize_particle_storage_is_possible(int new_maxpart, int new_maxpartgas)
{
  return (resize_particle_storage_check(new_maxpart, new_maxpartgas, 0) == 0);
}

int resize_particle_storage(int new_maxpart, int new_maxpartgas)
{
  if(new_maxpart == All.MaxPart && new_maxpartgas == All.MaxPartGas) {return 0;}   /* must stay the first
                                                                                      statement: the common
                                                                                      case is a domain
                                                                                      boundary asking for the
                                                                                      capacity it already has,
                                                                                      and it must cost two
                                                                                      comparisons */

  const int bad = resize_particle_storage_check(new_maxpart, new_maxpartgas, 1);
  if(bad) {return bad;}

  /* Bottom-up through the arena, in allocation order, so each block that gets shifted is shifted
   * at its old size and only once. */
  {
    void *slot = (void *) ProcessedFlag;
    resize_grow_block(&slot, (size_t) new_maxpart * sizeof(unsigned char));
    ProcessedFlag = (unsigned char *) slot;
  }
  if(P)
    {
      void *slot = (void *) P;
      resize_grow_block(&slot, (size_t) new_maxpart * sizeof(struct particle_data));
      P = (struct particle_data *) slot;
    }
  if(CellP)
    {
      void *slot = (void *) CellP;
      resize_grow_block(&slot, (size_t) new_maxpartgas * sizeof(struct gas_cell_data));
      CellP = (struct gas_cell_data *) slot;
    }
#ifdef CHIMES
  if(ChimesGasVars)
    {
      void *slot = (void *) ChimesGasVars;
      resize_grow_block(&slot, (size_t) new_maxpartgas * sizeof(struct gasVariables));
      ChimesGasVars = (struct gasVariables *) slot;
      /* Each live entry owns heap sub-arrays; the grow carried those pointers with the records it
       * moved.  The slots added above them hold whatever the arena last had there, so clear them:
       * a cell arriving in one of them must find the same state a cell arriving at startup does. */
      if(new_maxpartgas > All.MaxPartGas)   /* still the old capacity: it is published at the end */
        {memset(ChimesGasVars + All.MaxPartGas, 0, (size_t)(new_maxpartgas - All.MaxPartGas) * sizeof(struct gasVariables));}
    }
#endif

  /* The timebin threading is rebuilt by reconstruct_timebins() after the decomposition; growing
   * these only keeps their contents from being read past the end in between. */
  ActiveParticleList.reserve(new_maxpart);
  NextInTimeBin.resize(new_maxpart);
  PrevInTimeBin.resize(new_maxpart);

  All.MaxPart = new_maxpart;
  All.MaxPartGas = new_maxpartgas;
  return 0;
}


/*! Fix the largest particle capacity this run may ever reach.  Called once, where the initial
 *  capacity is set, and carried through restart files from there: the ceiling a run starts with is
 *  the ceiling it keeps, so its index layout never moves under it.
 *
 *  It is derived from the memory-manager budget, because that budget is a fixed-size arena and is
 *  what growth actually spends.  The per-slot cost counts everything a slot drags along -- the
 *  particle and gas-cell records, the tree's per-particle side arrays and its share of nodes, the
 *  Peano keys and the timebin lists -- so the share below is a share of the whole marginal cost of
 *  a slot, and what it leaves over is for the work buffers, whose size follows the communication
 *  buffer rather than the particle count.  Erring high costs nothing here, since whether the
 *  memory is really there stays the allocator's answer at the moment it is asked; erring low would
 *  refuse a capacity the budget could have supplied.
 *
 *  A wide multiple of the starting capacity holds it down as well, so a small problem given a
 *  large budget does not advertise slots it will never hold.  Every term is identical on every
 *  rank, which is what keeps the capacity itself rank-identical. */
void set_particle_capacity_ceiling(void)
{
  long long bytes_per_slot = (long long) sizeof(struct particle_data)
                           + (long long) sizeof(unsigned char)        /* ProcessedFlag */
                           + 3LL * (long long) sizeof(int)            /* timebin lists */
                           + 2LL * (long long) sizeof(peanokey)       /* domain keys, live and persistent */
                           + 2LL * (long long) sizeof(int);           /* tree Father + Nextnode */
  if(All.MaxPartGas > 0)
    {
      bytes_per_slot += (long long) sizeof(struct gas_cell_data);
#ifdef CHIMES
      bytes_per_slot += (long long) sizeof(struct gasVariables);
#endif
    }
  bytes_per_slot += (long long)(All.TreeAllocFactor * ((double) sizeof(struct NODE) + (double) sizeof(struct extNODE)));

  long long cap = PARTICLE_CAPACITY_MAX_GROWTH_FACTOR * (long long) All.MaxPart;
  const long long budget = (long long) All.MaxMemSize * 1024 * 1024;
  const long long cap_budget = (long long)(PARTICLE_CAPACITY_SHARE_OF_BUDGET * (double) budget) / bytes_per_slot;
  if(cap_budget < cap) {cap = cap_budget;}
  if(cap < (long long) All.MaxPart) {cap = (long long) All.MaxPart;}   /* never below what is already allocated */
  if(cap > (long long)(INT_MAX / 4)) {cap = (long long)(INT_MAX / 4);} /* node and pseudo-particle indices sit above it */
  All.MaxPartExpandable = (int) cap;

  if(ThisTask == 0)
    {
      printf("Allocating %d particle slots per rank (%d gas); the capacity may rise to %d within the %d MB per-rank budget.\n",
             All.MaxPart, All.MaxPartGas, All.MaxPartExpandable, All.MaxMemSize);
      fflush(stdout);
    }
}
