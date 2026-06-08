#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../declarations/allvars.h"
#include "../core/proto.h"

/* custom memory allocation routines */

/*!
 * This file was originally part of the GADGET3 code developed by
 * Volker Springel. The code has been modified
 * somewhat by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 * ALIGN issues fixed with patch from Peter Bell Feb. 2020. Since then more heavy
 * modifications to allow for new compilers and machines have been made.
 */

#define MAXBLOCKS 500
#define MAXCHARS  16
#define MIN_ALIGNMENT 32

static size_t TotBytes;
static void *Base;

static unsigned long Nblocks;

static void **Table;
static size_t *BlockSize;
static char *MovableFlag;
static void ***BasePointers;

static char *VarName;
static char *FunctionName;
static char *FileName;
static int *LineNumber;


void mymalloc_init(void)
{
  size_t n;

  BlockSize = (size_t *) malloc(MAXBLOCKS * sizeof(size_t));
  Table = (void **) malloc(MAXBLOCKS * sizeof(void *));
  MovableFlag = (char *) malloc(MAXBLOCKS * sizeof(char));
  BasePointers = (void ***) malloc(MAXBLOCKS * sizeof(void **));
  VarName = (char *) malloc(MAXBLOCKS * MAXCHARS * sizeof(char));
  FunctionName = (char *) malloc(MAXBLOCKS * MAXCHARS * sizeof(char));
  FileName = (char *) malloc(MAXBLOCKS * MAXCHARS * sizeof(char));
  LineNumber = (int *) malloc(MAXBLOCKS * sizeof(int));

  memset(VarName, 0, MAXBLOCKS * MAXCHARS);
  memset(FunctionName, 0, MAXBLOCKS * MAXCHARS);
  memset(FileName, 0, MAXBLOCKS * MAXCHARS);

  n = All.MaxMemSize * ((size_t) 1024 * 1024);

#ifdef DISABLE_ALIGNED_ALLOC
  Base = malloc(n);
#else
  Base = aligned_alloc(MIN_ALIGNMENT, n);
#endif
  {
    /* Large planned SYMMETRIC allocation -> collective guard on the result.
     * Every rank calls mymalloc_init() once, so this Allreduce is symmetric.
     * On any-rank failure: bad-stop + return WITHOUT touching Base; begrun's
     * post-mymalloc_init poll drains out gracefully (no MPI_Abort, no Vista
     * CG-stuck wedge). This is the production-OOM path we want off MPI_Abort. */
    int base_fail_local = (Base == NULL) ? 1 : 0, base_fail_any = 0;
    MPI_Allreduce(&base_fail_local, &base_fail_any, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if(base_fail_any)
      {
        if(base_fail_local) {printf("Failed to allocate memory for `Base' (%d Mbytes).\n", All.MaxMemSize); fflush(stdout);}
        gizmo_request_controlled_stop(122, "mymalloc_init: Base arena allocation failed on >=1 rank", __FILE__, __LINE__, __FUNCTION__);
        return;
      }
  }

  TotBytes = FreeBytes = n;

  AllocatedBytes = 0;
  Nblocks = 0;
  HighMarkBytes = 0;
}

/* LOCAL arena capacity check (NO MPI). Returns 1 iff `bytes` fits in THIS rank's
 * arena (FreeBytes) AND `nblocks` more blocks fit in the fixed block table
 * (MAXBLOCKS) -- a request can fit by bytes yet still exhaust the block table,
 * which would route the real mymalloc onto the emergency-hold path. Safe in ANY
 * context, including subset/turn (no collective): it is the building block for
 * caller-side OOM preflight. On a 0 return the caller bad-stops + skips the
 * allocation, draining gracefully instead of holding deep in the allocator. */
int gizmo_alloc_fits_this_rank(size_t bytes, int nblocks)
{
  int local_short = ((nblocks < 0) || (bytes > FreeBytes) || (Nblocks + (unsigned int)nblocks > MAXBLOCKS)) ? 1 : 0;
  return local_short ? 0 : 1;
}

/* COLLECTIVE preflight for large SYMMETRIC arena allocations. Returns 1 iff the
 * request fits on THIS rank AND every other rank; else 0 on ALL ranks (one
 * Allreduce over the local check). MUST be called only where all ranks
 * participate -- NEVER inside the generic mymalloc path, nor a subset/turn
 * context (use gizmo_alloc_fits_this_rank there). */
int gizmo_alloc_fits_all_ranks(size_t bytes, int nblocks)
{
  int local_short = gizmo_alloc_fits_this_rank(bytes, nblocks) ? 0 : 1, any_short = 0;
  MPI_Allreduce(&local_short, &any_short, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  return any_short ? 0 : 1;
}

/* The arena size a request of `n` bytes actually consumes -- mirrors the
 * rounding in mymalloc_fullinfo() (MIN_ALIGNMENT round-up + minimum block).
 * Use to build accurate preflight totals so alignment overhead can't sneak a
 * real allocation onto the residual TEMP-hard path. */
size_t gizmo_mymalloc_rounded_size(size_t n)
{
  if((n % MIN_ALIGNMENT) > 0) {n = (n / MIN_ALIGNMENT + 1) * MIN_ALIGNMENT;}
  if(n < MIN_ALIGNMENT) {n = MIN_ALIGNMENT;}
  return n;
}

void report_detailed_memory_usage_of_largest_task(size_t * OldHighMarkBytes, const char *label,
						  const char *func, const char *file, int line)
{
  size_t *sizelist, maxsize, minsize;
  double avgsize;
  int i, task;

  sizelist = (size_t *) mymalloc("sizelist", NTask * sizeof(size_t));
  MPI_Allgather(&AllocatedBytes, sizeof(size_t), MPI_BYTE, sizelist, sizeof(size_t), MPI_BYTE,
		MPI_COMM_WORLD);

  for(i = 1, task = 0, maxsize = minsize = sizelist[0], avgsize = sizelist[0]; i < NTask; i++)
    {
      if(sizelist[i] > maxsize)
	{
	  maxsize = sizelist[i];
	  task = i;
	}
      if(sizelist[i] < minsize)
	{
	  minsize = sizelist[i];
	}
      avgsize += sizelist[i];
    }

  myfree(sizelist);


  if(maxsize > 1.1 * (*OldHighMarkBytes))
    {
      *OldHighMarkBytes = maxsize;

      avgsize /= NTask;

      if(ThisTask == task)
	{
	  printf
	    ("\nIn '%s', %s()/%s/%d: Largest Allocation = %g Mbyte (on task=%d), Smallest = %g Mbyte, Average = %g Mbyte\n",
	     label, func, file, line, maxsize / (1024.0 * 1024.0), task, minsize / (1024.0 * 1024.0),
	     avgsize / (1024.0 * 1024.0));
	  dump_memory_table();
	}
      fflush(stdout);
      MPI_Barrier(MPI_COMM_WORLD);
    }
}




void dump_memory_table(void)
{
#ifndef OUTPUT_ADDITIONAL_RUNINFO
    if(ThisTask==0 && All.HighestActiveTimeBin == All.HighestOccupiedTimeBin)
#endif
    {
        unsigned int i;
        size_t totBlocksize = 0;
        
        printf("------------------------ Allocated Memory Blocks----------------------------------------\n");
        printf("Task   Nr F          Variable      MBytes   Cumulative         Function/File/Linenumber\n");
        printf("----------------------------------------------------------------------------------------\n");
        for(i = 0; i < Nblocks; i++)
        {
            totBlocksize += BlockSize[i];
            
            printf("%4d %4d %d  %16s  %10.4f   %10.4f  %s()/%s/%d\n",
                   ThisTask, i, MovableFlag[i], VarName + i * MAXCHARS, BlockSize[i] / (1024.0 * 1024.0),
                   totBlocksize / (1024.0 * 1024.0), FunctionName + i * MAXCHARS,
                   FileName + i * MAXCHARS, LineNumber[i]);
        }
        printf("----------------------------------------------------------------------------------------\n");
    }
}

void *mymalloc_fullinfo(const char *varname, size_t n, const char *func, const char *file, int line)
{
  if((n % MIN_ALIGNMENT) > 0) {n = (n / MIN_ALIGNMENT + 1) * MIN_ALIGNMENT;}
  if(n < MIN_ALIGNMENT) {n = MIN_ALIGNMENT;}

  if(Nblocks >= MAXBLOCKS)
    {
      printf("Task=%d: No blocks left in mymalloc_fullinfo() at %s()/%s/line %d. MAXBLOCKS=%d\n", ThisTask,
	     func, file, line, MAXBLOCKS);
      gizmo_emergency_hold_reviewed(813, "mymalloc: no free blocks (MAXBLOCKS exhausted) -- REVIEWED_HARD_OOM_FLOOR", __FILE__, __LINE__, __FUNCTION__);
    }

  if(n > FreeBytes)
    {
      dump_memory_table();
      printf
	("\nTask=%d: Not enough memory in mymalloc_fullinfo() to allocate %g MB for variable '%s' at %s()/%s/line %d (FreeBytes=%g MB).\n",
	 ThisTask, n / (1024.0 * 1024.0), varname, func, file, line, FreeBytes / (1024.0 * 1024.0));
      gizmo_emergency_hold_reviewed(812, "mymalloc: out of arena memory (FreeBytes) -- REVIEWED_HARD_OOM_FLOOR", __FILE__, __LINE__, __FUNCTION__);
    }
  Table[Nblocks] = (char*)Base + (TotBytes - FreeBytes);
  FreeBytes -= n;

  strncpy(VarName + Nblocks * MAXCHARS, varname, MAXCHARS - 1);
  strncpy(FunctionName + Nblocks * MAXCHARS, func, MAXCHARS - 1);
  strncpy(FileName + Nblocks * MAXCHARS, file, MAXCHARS - 1);
  LineNumber[Nblocks] = line;

  AllocatedBytes += n;
  BlockSize[Nblocks] = n;
  MovableFlag[Nblocks] = 0;

  Nblocks += 1;

  if(AllocatedBytes > HighMarkBytes)
    HighMarkBytes = AllocatedBytes;

  return Table[Nblocks - 1];
}


void *mymalloc_movable_fullinfo(void *ptr, const char *varname, size_t n, const char *func, const char *file,
				int line)
{
  if((n % MIN_ALIGNMENT) > 0) {n = (n / MIN_ALIGNMENT + 1) * MIN_ALIGNMENT;}
  if(n < MIN_ALIGNMENT) {n = MIN_ALIGNMENT;}

  if(Nblocks >= MAXBLOCKS)
    {
      printf("Task=%d: No blocks left in mymalloc_fullinfo() at %s()/%s/line %d. MAXBLOCKS=%d\n", ThisTask,
	     func, file, line, MAXBLOCKS);
      gizmo_emergency_hold_reviewed(816, "mymalloc_movable: no free blocks (MAXBLOCKS exhausted) -- REVIEWED_HARD_OOM_FLOOR", __FILE__, __LINE__, __FUNCTION__);
    }

  if(n > FreeBytes)
    {
      dump_memory_table();
      printf
	("\nTask=%d: Not enough memory in mymalloc_fullinfo() to allocate %g MB for variable '%s' at %s()/%s/line %d (FreeBytes=%g MB).\n",
	 ThisTask, n / (1024.0 * 1024.0), varname, func, file, line, FreeBytes / (1024.0 * 1024.0));
      gizmo_emergency_hold_reviewed(817, "mymalloc_movable: out of arena memory (FreeBytes) -- REVIEWED_HARD_OOM_FLOOR", __FILE__, __LINE__, __FUNCTION__);
    }
  Table[Nblocks] = (char*)Base + (TotBytes - FreeBytes);
  FreeBytes -= n;

  strncpy(VarName + Nblocks * MAXCHARS, varname, MAXCHARS - 1);
  strncpy(FunctionName + Nblocks * MAXCHARS, func, MAXCHARS - 1);
  strncpy(FileName + Nblocks * MAXCHARS, file, MAXCHARS - 1);
  LineNumber[Nblocks] = line;

  AllocatedBytes += n;
  BlockSize[Nblocks] = n;
  MovableFlag[Nblocks] = 1;
  BasePointers[Nblocks] = (void **) ptr;

  Nblocks += 1;

  if(AllocatedBytes > HighMarkBytes)
    HighMarkBytes = AllocatedBytes;

  return Table[Nblocks - 1];
}



void myfree_fullinfo(void *p, const char *func, const char *file, int line)
{
  if(Nblocks == 0)
    {gizmo_request_controlled_stop(76878, "myfree: Nblocks==0 (nothing to free)", __FILE__, __LINE__, __FUNCTION__); return;}

  if(p != Table[Nblocks - 1])
    {
      dump_memory_table();
      printf("Task=%d: Wrong call of myfree() at %s()/%s/line %d: not the last allocated block!\n", ThisTask,
	     func, file, line);
      fflush(stdout);
      /* invariant violation: bad-stop + immediate return BEFORE the Nblocks-- / BlockSize[] mutation below (would index out of bounds / free the wrong block) */
      gizmo_request_controlled_stop(814, "myfree: not the last allocated block", __FILE__, __LINE__, __FUNCTION__); return;
    }

  Nblocks -= 1;
  AllocatedBytes -= BlockSize[Nblocks];
  FreeBytes += BlockSize[Nblocks];
}



void myfree_movable_fullinfo(void *p, const char *func, const char *file, int line)
{
  unsigned int i;

  if(Nblocks == 0)
    {gizmo_request_controlled_stop(768728, "myfree_movable: Nblocks==0 (nothing to free)", __FILE__, __LINE__, __FUNCTION__); return;}

  /* first, let's find the block */
    //unsigned int nr; //
    int nr; // actually shouldn't be unsigned for the bcheck below //

  for(nr = Nblocks - 1; nr >= 0; nr--)
    if(p == Table[nr])
      break;

  if(nr < 0)
    {
      dump_memory_table();
      printf
	("Task=%d: Wrong call of myfree_movable() from %s()/%s/line %d - this block has not been allocated!\n",
	 ThisTask, func, file, line);
      fflush(stdout);
      gizmo_request_controlled_stop(8152, "myfree_movable: block not allocated", __FILE__, __LINE__, __FUNCTION__); return;
    }

  if(nr < Nblocks - 1)		/* the block is not the last allocated block */
    {
      /* check that all subsequent blocks are actually movable */
      for(i = nr + 1; i < Nblocks; i++)
	if(MovableFlag[i] == 0)
	  {
	    dump_memory_table();
	    printf
	      ("Task=%d: Wrong call of myfree_movable() from %s()/%s/line %d - behind block=%d there are subsequent non-movable allocated blocks\n",
	       ThisTask, func, file, line, nr);
	    fflush(stdout);
	    gizmo_request_controlled_stop(81252, "myfree_movable: non-movable block behind target", __FILE__, __LINE__, __FUNCTION__); return;
	  }
    }


  AllocatedBytes -= BlockSize[nr];
  FreeBytes += BlockSize[nr];

  size_t offset = -BlockSize[nr];
  size_t length = 0;

  for(i = nr + 1; i < Nblocks; i++)
    length += BlockSize[i];

  if(nr < Nblocks - 1)
    memmove((char*)Table[nr + 1] + offset, Table[nr + 1], length);

  for(i = nr + 1; i < Nblocks; i++)
    {
      Table[i] = (char*)Table[i] + offset;
      *BasePointers[i] = (char*)*BasePointers[i] + offset;
    }

  for(i = nr + 1; i < Nblocks; i++)
    {
      Table[i - 1] = Table[i];
      BasePointers[i - 1] = BasePointers[i];
      BlockSize[i - 1] = BlockSize[i];
      MovableFlag[i - 1] = MovableFlag[i];

      strncpy(VarName + (i - 1) * MAXCHARS, VarName + i * MAXCHARS, MAXCHARS - 1);
      strncpy(FunctionName + (i - 1) * MAXCHARS, FunctionName + i * MAXCHARS, MAXCHARS - 1);
      strncpy(FileName + (i - 1) * MAXCHARS, FileName + i * MAXCHARS, MAXCHARS - 1);
      LineNumber[i - 1] = LineNumber[i];
    }

  Nblocks -= 1;
}






void *myrealloc_fullinfo(void *p, size_t n, const char *func, const char *file, int line)
{
    if((n % MIN_ALIGNMENT) > 0) {n = (n / MIN_ALIGNMENT + 1) * MIN_ALIGNMENT;}
    if(n < MIN_ALIGNMENT) {n = MIN_ALIGNMENT;}

  if(n < 8)
    n = 8;

  /* NOTE: unlike myfree* (void), a myrealloc* invariant violation can't safely
   * bad-stop+return: the caller would treat the returned pointer as a
   * successfully-resized buffer and write past the old size. Until the two real
   * callers (domain.cc, mpi_util.cc) are guarded, route to the emergency-hold
   * path (Class-2; the caller-guard is the named allocator Pass-B workstream). */
  if(Nblocks == 0)
    gizmo_emergency_hold_reviewed(76879, "myrealloc: Nblocks==0 -- REVIEWED_HARD_ALLOCATOR_INVARIANT", __FILE__, __LINE__, __FUNCTION__);

  if(p != Table[Nblocks - 1])
    {
      dump_memory_table();
      printf("Task=%d: Wrong call of myrealloc() at %s()/%s/line %d - not the last allocated block!\n",
	     ThisTask, func, file, line);
      fflush(stdout);
      gizmo_emergency_hold_reviewed(815, "myrealloc: not the last allocated block -- REVIEWED_HARD_ALLOCATOR_INVARIANT", __FILE__, __LINE__, __FUNCTION__);
    }

  AllocatedBytes -= BlockSize[Nblocks - 1];
  FreeBytes += BlockSize[Nblocks - 1];

  if(n > FreeBytes)
    {
      dump_memory_table();
      printf
	("Task=%d: Not enough memory in myremalloc(n=%g MB) at %s()/%s/line %d. previous=%g FreeBytes=%g MB\n",
	 ThisTask, n / (1024.0 * 1024.0), func, file, line, BlockSize[Nblocks - 1] / (1024.0 * 1024.0),
	 FreeBytes / (1024.0 * 1024.0));
      gizmo_emergency_hold_reviewed(90002001, "myrealloc: out of arena memory (FreeBytes) -- REVIEWED_HARD_OOM_FLOOR", __FILE__, __LINE__, __FUNCTION__);
    }
  Table[Nblocks - 1] = (char*)Base + (TotBytes - FreeBytes);
  FreeBytes -= n;

  AllocatedBytes += n;
  BlockSize[Nblocks - 1] = n;

  if(AllocatedBytes > HighMarkBytes)
    HighMarkBytes = AllocatedBytes;

  return Table[Nblocks - 1];
}

void *myrealloc_movable_fullinfo(void *p, size_t n, const char *func, const char *file, int line)
{
  unsigned int i;

  if((n % MIN_ALIGNMENT) > 0) {n = (n / MIN_ALIGNMENT + 1) * MIN_ALIGNMENT;}
  if(n < MIN_ALIGNMENT) {n = MIN_ALIGNMENT;}

  if(Nblocks == 0)
    gizmo_emergency_hold_reviewed(768799, "myrealloc_movable: Nblocks==0 -- REVIEWED_HARD_ALLOCATOR_INVARIANT", __FILE__, __LINE__, __FUNCTION__);

  /* first, let's find the block */
   //unsigned int nr; //
   int nr; // actually shouldn't be unsigned for the bcheck below //

  for(nr = Nblocks - 1; nr >= 0; nr--)
    if(p == Table[nr])
      break;

  if(nr < 0)
    {
      dump_memory_table();
      printf
	("Task=%d: Wrong call of myrealloc_movable() from %s()/%s/line %d - this block has not been allocated!\n",
	 ThisTask, func, file, line);
      fflush(stdout);
      gizmo_emergency_hold_reviewed(8151, "myrealloc_movable: block not allocated -- REVIEWED_HARD_ALLOCATOR_INVARIANT", __FILE__, __LINE__, __FUNCTION__);
    }

  if(nr < Nblocks - 1)		/* the block is not the last allocated block */
    {
      /* check that all subsequent blocks are actually movable */
      for(i = nr + 1; i < Nblocks; i++)
	if(MovableFlag[i] == 0)
	  {
	    dump_memory_table();
	    printf
	      ("Task=%d: Wrong call of myrealloc_movable() from %s()/%s/line %d - behind block=%d there are subsequent non-movable allocated blocks\n",
	       ThisTask, func, file, line, nr);
	    fflush(stdout);
	    gizmo_emergency_hold_reviewed(8152, "myrealloc_movable: non-movable block behind target -- REVIEWED_HARD_ALLOCATOR_INVARIANT", __FILE__, __LINE__, __FUNCTION__);
	  }
    }


  AllocatedBytes -= BlockSize[nr];
  FreeBytes += BlockSize[nr];

  if(n > FreeBytes)
    {
      dump_memory_table();
      printf
	("Task=%d: at %s()/%s/line %d: Not enough memory in myremalloc_movable(n=%g MB). previous=%g FreeBytes=%g MB\n",
	 ThisTask, func, file, line, n / (1024.0 * 1024.0), BlockSize[nr] / (1024.0 * 1024.0),
	 FreeBytes / (1024.0 * 1024.0));
      gizmo_emergency_hold_reviewed(90002002, "myrealloc_movable: out of arena memory (FreeBytes) -- REVIEWED_HARD_OOM_FLOOR", __FILE__, __LINE__, __FUNCTION__);
    }

  size_t offset = n - BlockSize[nr];
  size_t length = 0;

  for(i = nr + 1; i < Nblocks; i++)
    length += BlockSize[i];

  if(nr < Nblocks - 1)
    memmove((char*)Table[nr + 1] + offset, Table[nr + 1], length);

  for(i = nr + 1; i < Nblocks; i++)
    {
      Table[i] = (char*)Table[i] + offset;

      *BasePointers[i] = (char*)*BasePointers[i] + offset;
    }

  FreeBytes -= n;
  AllocatedBytes += n;
  BlockSize[nr] = n;

  if(AllocatedBytes > HighMarkBytes)
    HighMarkBytes = AllocatedBytes;

  return Table[nr];
}
