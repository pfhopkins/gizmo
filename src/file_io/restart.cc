#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/file.h>
#include <unistd.h>

#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../domain/domain.h"
#include "../gravity/let_data.h"

static FILE *fd;

#ifdef CHIMES 
static ChimesFloat *gasAbundancesBuf;
#endif 

static void in(int *x, int modus);
static void byten(void *x, size_t n, int modus);

int old_MaxPart = 0, new_MaxPart;


/*! Total number of particles recorded in the restart set, or -1 if it cannot be read. Used at
 *  startup to size the memory arena, which has to be sized before anything is allocated from it
 *  and therefore before the restart files are read properly.
 *
 *  Deliberately modest about what it is for. The whole run-parameter block is the first thing in
 *  a restart file, so this reads it into a throwaway copy and takes one number out; it never
 *  touches the live parameters, and restart() below remains the only thing that actually loads
 *  them. The number is also not required to be exact: a run's particle count grows as gas spawns
 *  winds and stars, so the primary file and the backup can legitimately disagree, and the size
 *  this feeds carries enough margin to absorb that. Anything unreadable simply returns -1 and the
 *  caller falls back to a conservative size, because a restart set that is genuinely broken is
 *  restart()'s business to report, in its own place, with its own message.
 */
long long peek_total_particles_in_restart(void)
{
    long long total = -1;

    if(ThisTask == 0)
    {
        char path[DEFAULT_PATH_BUFFERSIZE_TOUSE];
        FILE *f;
        int attempt;

        for(attempt = 0; attempt < 2 && total < 0; attempt++)
        {
            if(attempt == 0)
                {snprintf(path, DEFAULT_PATH_BUFFERSIZE_TOUSE, "%s/restartfiles/%s.0", All.OutputDir, All.RestartFile);}
            else
                {snprintf(path, DEFAULT_PATH_BUFFERSIZE_TOUSE, "%s/restartfiles/%s.0.bak", All.OutputDir, All.RestartFile);}

            if((f = fopen(path, "r")))
            {
                struct global_data_all_processes stored;
                if(fread(&stored, sizeof(stored), 1, f) == 1) {total = (long long) stored.TotNumPart;}
                fclose(f);
            }
        }
    }

    MPI_Bcast(&total, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    return total;
}


/*! This function reads or writes the restart files.
 * Each processor writes its own restart file, with the
 * I/O being done in parallel. To avoid congestion of the disks
 * you can tell the program to restrict the number of files
 * that are simultaneously written to NumFilesWrittenInParallel.
 *
 * If modus>0  the restart()-routine reads, 
 * if modus==0 it writes a restart file. 
 */

/*!
 * This file was originally part of the GADGET3 code developed by
 * Volker Springel. The code has been modified heavily
 * (adding/removing read/write items, allowing for different variables
 * to be changed or re-initialized on restarts, and changing variable units,
 * allowing run-time option modifications, rewriting for new libraries,
 * reconfiguring how some memory is structured, etc.)
 * by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

void restart(int modus)
{
    char buf[DEFAULT_PATH_BUFFERSIZE_TOUSE], buf_bak[DEFAULT_PATH_BUFFERSIZE_TOUSE], buf_mv[DEFAULT_PATH_BUFFERSIZE_TOUSE];
    double save_PartAllocFactor;
    int nprocgroup, primaryTask, groupTask;
    struct global_data_all_processes all_task0 = {};   /* zero-init so the required groupTask==0 Bcast is safe even if rank 0 failed its header read */
    int rekeyed_MaxPartAssignable = 0;   /* worked out during the read, put in force at the end of it */
    int nmulti = MULTIPLEDOMAINS, regular_restarts_are_valid = 1, backup_restarts_are_valid = 1;
    

#ifdef CHIMES 
    int partIndex, abunIndex; 
#endif 
    
    if(ThisTask == 0 && modus == 0) // writing re-start files: move old files to .bak
    {
        snprintf(buf, DEFAULT_PATH_BUFFERSIZE_TOUSE, "%s/restartfiles", All.OutputDir);
        mkdir(buf, 02755);
        int i_Task_iter;
        for(i_Task_iter=0; i_Task_iter<NTask; i_Task_iter++)
        {
            snprintf(buf, DEFAULT_PATH_BUFFERSIZE_TOUSE, "%s/restartfiles/%s.%d", All.OutputDir, All.RestartFile, i_Task_iter);
            snprintf(buf_bak, DEFAULT_PATH_BUFFERSIZE_TOUSE, "%s/restartfiles/%s.%d.bak", All.OutputDir, All.RestartFile, i_Task_iter);
#ifdef IO_REDUNDANT_BACKUP_RESTARTFILE_FREQUENCY
            if( (((int)(CPUThisRun/All.CpuTimeBetRestartFile)) % ((int)IO_REDUNDANT_BACKUP_RESTARTFILE_FREQUENCY)) == 0)
            {
                char buf_bak2[DEFAULT_PATH_BUFFERSIZE_TOUSE]; snprintf(buf_bak2, DEFAULT_PATH_BUFFERSIZE_TOUSE, "%s/restartfiles/%s.%d.bak2", All.OutputDir, All.RestartFile, i_Task_iter);
                rename(buf_bak,buf_bak2); // move old backup restart files to .bak2 files //
            }
#endif
            rename(buf,buf_bak); // move old restart files to .bak files //
        }
    }
    if(modus == 1) // reading re-start files. make sure to check all the files to read exist!
    {
        int i_Task_iter;
        for(i_Task_iter=0; i_Task_iter<NTask; i_Task_iter++)
        {
            snprintf(buf, DEFAULT_PATH_BUFFERSIZE_TOUSE, "%s/restartfiles/%s.%d", All.OutputDir, All.RestartFile, i_Task_iter);
            snprintf(buf_bak, DEFAULT_PATH_BUFFERSIZE_TOUSE, "%s/restartfiles/%s.%d.bak", All.OutputDir, All.RestartFile, i_Task_iter);
            if(!(fd = fopen(buf, "r"))) {regular_restarts_are_valid=0;} else {fclose(fd);} // check if regular restart exists
            if(!(fd = fopen(buf_bak, "r"))) {backup_restarts_are_valid=0;} else {fclose(fd);} // check if backup restart exists
        }
        if(ThisTask == 0)
        {
            if((regular_restarts_are_valid == 0) && (backup_restarts_are_valid == 0))
            {
                printf("Fatal error. Full set of restart files ('%s' or '%s') not found - check the restarts are uncorrupted and your MPI process number has not changed.\n", buf, buf_bak);
            }
            if((regular_restarts_are_valid == 0) && (backup_restarts_are_valid == 1))
            {
                printf("Default restartfiles ('%s') not found - they are incomplete or corrupted [number of files matching MPI process number not found. But apparently valid set of backup restartfiles ('%s') found. Attempting to use those.\n", buf, buf_bak);
            }
        }
        /* All-rank bad-stop: every rank ran the file-existence scan above, so
         * (regular==0 && backup==0) is identical on all ranks. endrun lives
         * OUTSIDE the rank-0 printf so the flip turns it into a symmetric
         * bad-stop; the poll before the per-rank file-IO loop drains it. */
        if((regular_restarts_are_valid == 0) && (backup_restarts_are_valid == 0)) { endrun(7871); }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    snprintf(buf, DEFAULT_PATH_BUFFERSIZE_TOUSE, "%s/restartfiles/%s.%d", All.OutputDir, All.RestartFile, ThisTask);
    if((modus == 1) && (regular_restarts_are_valid == 0) && (backup_restarts_are_valid == 1)) {snprintf(buf, DEFAULT_PATH_BUFFERSIZE_TOUSE, "%s/restartfiles/%s.%d.bak", All.OutputDir, All.RestartFile, ThisTask);}
    snprintf(buf_bak, DEFAULT_PATH_BUFFERSIZE_TOUSE, "%s/restartfiles/%s.%d.bak", All.OutputDir, All.RestartFile, ThisTask);
    snprintf(buf_mv, DEFAULT_PATH_BUFFERSIZE_TOUSE, "mv %s %s", buf, buf_bak);
    
    if((NTask < All.NumFilesWrittenInParallel))
    {
        if(ThisTask == 0) {printf("Fatal error.\nNumber of processors must be greater than or equal to `NumFilesWrittenInParallel'.\n");}
        endrun(2131);   /* symmetric (NTask + All identical on all ranks) -> all-rank bad-stop */
    }

    nprocgroup = NTask / All.NumFilesWrittenInParallel;
    
    if((NTask % All.NumFilesWrittenInParallel))
    {
        nprocgroup++;
    }

  primaryTask = (ThisTask / nprocgroup) * nprocgroup;

  /* Bad-stop poll BEFORE the per-rank restart file-IO loop. Drains the two
   * symmetric setup bad-stops above (7871 missing restart set, 2131 NTask <
   * NumFilesWrittenInParallel) after the Stage-1d flip. All ranks reach here
   * (the MPI_Barrier above is unconditional); the intervening code is pure
   * arithmetic, so this is the first collective-symmetric drain point before
   * any restart data is touched. */
  if(gizmo_poll_controlled_stop()) return;

  for(groupTask = 0; groupTask < nprocgroup; groupTask++)
    {
      if(ThisTask == (primaryTask + groupTask))	/* ok, it's this processor's turn */
	{
	  /* Graceful per-rank restart-IO faults. read_status != 0 means this rank's restart
	   * payload cannot be safely read/written; we set a soft bad-stop, skip the rest of THIS
	   * rank's local file IO (goto finish_turn), still execute the required groupTask==0
	   * Bcast, and drain at the per-turn poll below. No peer P2P in a turn, so no rank is
	   * stranded. (NOTE: a truncated/corrupt mid-read still routes through my_fread's own
	   * hold at io.cc:5392 (778) until the io.cc pass converts it.) */
	  int restart_status = 0;
	  if(modus)
	    {
	      if(!(fd = fopen(buf, "r")))
		{
		  if(!(fd = fopen(buf_bak, "r")))
		    {
		      printf("Restart file '%s' nor '%s' found.\n", buf, buf_bak);
		      fflush(stdout);
		      endrun(7870); restart_status = 7870;   /* soft: this rank's restart file (nor backup) found */
		    }
		}
	    }
	  else
	    {
	      if(!(fd = fopen(buf, "w")))
		{
		  printf("Restart file '%s' cannot be opened.\n", buf);
		  fflush(stdout);
		  endrun(7878); restart_status = 7878;   /* soft: cannot open this rank's restart file for write */
		}
	    }


	  save_PartAllocFactor = All.PartAllocFactor;

	  /* common data (skip if the file failed to open -- fd is NULL) */
	  if(!restart_status)
	    {
	      byten(&All, sizeof(struct global_data_all_processes), modus);
	      if(ThisTask == 0 && modus > 0) {all_task0 = All;}
	    }

	  if(modus > 0 && groupTask == 0)	/* read -- required Bcast: must fire on every participant even on failure */
	    {
	      MPI_Bcast(&all_task0, sizeof(struct global_data_all_processes), MPI_BYTE, 0, MPI_COMM_WORLD);
	    }

	  if(restart_status) {goto finish_turn;}   /* file open failed: skip the local payload, drain at the per-turn poll */


	  if(modus)		/* read */
	    {
	      /* The most local particles a rank may be assigned is PartAllocFactor times the balanced
	       * count, and both of those can differ from what the run that wrote this file had: the
	       * factor because the parameter file may have been edited, the count because particles are
	       * created and destroyed while a run proceeds.  Work it out again from what is true now, so
	       * that resuming a run keeps the MEANING of the factor rather than the value it happened to
	       * imply at the start.  The same expression on every rank, over a particle count every rank
	       * already agrees on, so the result is identical everywhere without communicating it -- the
	       * property the cap depends on.  Written exactly as when the initial conditions were read,
	       * so a resumed run and a fresh one at the same size reach the same number.
	       *
	       * An edited factor and a moved particle count go through the same arithmetic here on
	       * purpose.  They used to be separate: only an edit worked anything out again, so resuming
	       * with the factor untouched held the cap at whatever the initial conditions implied,
	       * however far the run had grown past them, while retyping the same factor with one digit
	       * changed worked it out in full.  Two restarts that ask for the same thing have to do the
	       * same thing, and a factor that is defined relative to the particle count has to follow it. */
	      const int factor_was_raised = (save_PartAllocFactor > All.PartAllocFactor);
	      All.PartAllocFactor = save_PartAllocFactor;

	      long long cap_wanted = (long long) (All.PartAllocFactor * (All.TotNumPart / NTask));
	      if(cap_wanted < 1) {cap_wanted = 1;}

	      /* The run's capacity ceiling was fixed when the ICs were read and travels in the restart
	       * file; it bounds the tree's node index base and its per-particle side arrays, so nothing
	       * may be assigned past it.  It is a bound on INDICES, and nothing is allocated from it: it
	       * is a whole node's memory share priced entirely in particle slots, many times what a run
	       * ever holds.  Storage, on the other hand, IS sized from the cap -- so adopting the
	       * ceiling as the cap would ask for that whole share in particle arrays.  A cap that wants
	       * to go past it therefore does not move at all: the run keeps the cap it already had,
	       * which is what it was running at anyway, and is told why.  A factor the user RAISED is refused instead: that
	       * is a request that cannot be met, and saying so is more use than quietly doing something
	       * else.  A factor the user LOWERED is not refused, since it asks for less than changing
	       * nothing would have.  A ceiling of zero, which can only mean the file does not carry one,
	       * needs no special case: nothing sits below it, so the cap simply holds still and the
	       * reads below report the file's real trouble. */
	      if(cap_wanted > (long long) All.MaxPartExpandable)
		{
		  if(factor_was_raised)
		    {
		      /* Rank-uniform inputs, so every rank reaches this together and one of them says it. */
		      if(ThisTask == 0)
			{
			  printf("At %lld particles, the PartAllocFactor %g asked for here needs %lld particle slots per rank, past the %d this run can index, fixed when it started. Restart at the previous value, which resumes, or start a fresh run with more memory available per rank.\n",
				 (long long) All.TotNumPart, All.PartAllocFactor, cap_wanted, All.MaxPartExpandable);
			  fflush(stdout);
			}
		      /* Skip this rank's payload: allocate_memory + the reads below would run at the very
		       * capacity just rejected.  Drains at the per-turn poll, like the other soft stops here. */
		      endrun(90001023); restart_status = 90001023; goto finish_turn;
		    }
		  if(ThisTask == 0)
		    {
		      printf("At %lld particles, PartAllocFactor %g asks for %lld particle slots per rank, past the %d this run can index, fixed when it started. Keeping the %d it already had. If the decomposition then cannot stay within that, lower PartAllocFactor, or start a fresh run with more memory available per rank.\n",
			     (long long) All.TotNumPart, All.PartAllocFactor, cap_wanted, All.MaxPartExpandable,
			     All.MaxPartAssignable);
		      fflush(stdout);
		    }
		  cap_wanted = (long long) All.MaxPartAssignable;   /* hold still: see above */
		}

	      if((int) cap_wanted != All.MaxPartAssignable)
		{
		  if(ThisTask == 0)
		    {
		      printf("Resuming with %lld particles and PartAllocFactor %g: the most local particles a rank may be assigned moves from %d to %d.\n",
			     (long long) All.TotNumPart, All.PartAllocFactor, All.MaxPartAssignable, (int) cap_wanted);
		      fflush(stdout);
		    }
		  /* Not in force yet: the top tree serialized in this file was refined against the
		   * WRITER's cap, and the read below deserializes it into an allocation this same cap
		   * sizes (domain_allocate -> MaxTopNodes).  Lower the cap first and that allocation is
		   * too small for the top tree about to be read into it, which is a heap overrun rather
		   * than a stop, since NTopnodes is read from the file unchecked.  So the new cap takes
		   * effect where the new capacity does, once every payload has been read. */
		  rekeyed_MaxPartAssignable = (int) cap_wanted;

		  /* How much storage this rank holds is a separate question from what it may be assigned,
		   * and only one direction of it belongs here.  Storage is raised now when the cap has
		   * outgrown it, because the decomposition that consumes the cap has to find the room
		   * already there.  It is never LOWERED here: this file is about to be read into it, and a
		   * rank can legitimately hold more particles than the cap alone suggests, since particles
		   * created between decompositions are bounded by storage rather than by the cap.  Releasing
		   * what a smaller cap no longer needs is the domain boundary's business, where it is done
		   * only once the saving is worth the migration.
		   *
		   * When it is raised, All.MaxPart goes back to the writer's value a few lines below so the
		   * serialized tree payload stays readable, and reaches the new one at the END OF THIS READ, with the cap.
		   * The two are therefore deliberately unequal for the duration of this read: do not "tidy"
		   * them into agreement, and do not assert storage >= cap globally -- that assertion is
		   * resize_particle_storage's business, and its first call comes after both are in force. */
		  if(cap_wanted > (long long) All.MaxPart)
		    {
		      old_MaxPart = All.MaxPart;	/* the capacity the payload below was written at */
		      All.MaxPart = (int) cap_wanted;
		      gizmo_set_gas_capacity_from_maxpart();
		      new_MaxPart = All.MaxPart;
		    }
		}

	      if(all_task0.Time != All.Time)
		{
		  printf("The restart file on task=%d is not consistent with the one on task=0\n", ThisTask);
		  fflush(stdout);
		  /* soft: inconsistent restart. This is an EXIT case -- skip the rest of the local
		   * payload (allocate_memory + reads on inconsistent sizes could OOM / overflow)
		   * and drain at the per-turn poll. */
		  endrun(16); restart_status = 16; goto finish_turn;
		}

	      /* allocate_memory(0): subset/turn caller (per-rank groupTask turn) -> LOCAL-only
	       * preflight inside, NO collective (peers parked at a barrier; an Allreduce here would
	       * deadlock). On an arena (local-preflight) or UVM/STL OOM it requests a soft bad-stop
	       * and returns nonzero WITHOUT holding; we skip this rank's payload reads (goto
	       * finish_turn) and drain at the per-turn poll below. No duplicate endrun -- the
	       * controlled-stop is already requested with a descriptive reason inside allocate_memory. */
	      {
		int alloc_status = allocate_memory(0);
		if(alloc_status) {restart_status = alloc_status; goto finish_turn;}
	      }
	    }

	  in(&NumPart, modus);
	  if(NumPart > All.MaxPart)
	    {
	      printf("it seems you have reduced(!) 'PartAllocFactor' below the value of %g needed to load the restart file.\n", NumPart / (((double) All.TotNumPart) / NTask));
	      printf("fatal error\n");
	      fflush(stdout);
	      /* soft: would overflow P[] in the byten(&P[0],...) below. Skip the payload, drain at the poll. */
	      endrun(22); restart_status = 22; goto finish_turn;
	    }

	  if(modus)		/* read */
	    {
	      /* Restore the WRITER's MaxPart FOR THE REST OF THIS READ, AND NO LONGER -- the swap back
	       * is at the end of restart(), and the two belong together.  What this used to be FOR was
	       * the tree: force_treeallocate published the particle-slot count from it, so the writer's
	       * value had to be in place for the serialized Nextnode pseudo segment to land at the
	       * offset the writer used.  That reason is gone -- the slot count now comes from the file
	       * explicitly (below) -- and it is kept because removing it is a decision in its own right,
	       * not a side effect of changing what the tree reads.  Storage here was allocated at the
	       * RAISED capacity, and the NumPart guard above ran against that same value.
	       * What this window may NOT do is outlive the read.  It once did, on the reasoning that
	       * nothing looks at All.MaxPart before the domain boundary; that was true of the tree and
	       * false of the live step, where a ghost import grows storage against it and a wind spawn
	       * refuses against it -- both reading a capacity smaller than the decomposition is already
	       * allowed to assign. */
	      if(old_MaxPart) {All.MaxPart = old_MaxPart;}
	    }


	  /* Particle data  */
	  byten(&P[0], NumPart * sizeof(struct particle_data), modus);

	  in(&N_gas, modus);
	  if(N_gas > 0)
	    {
	      if(N_gas > All.MaxPartGas)
		{
		  printf("GAS: it seems you have reduced(!) 'PartAllocFactor' below the value of %g needed to load the restart file.\n", N_gas / (((double) All.TotN_gas) / NTask));
		  printf("fatal error\n");
		  fflush(stdout);
		  /* soft: would overflow CellP[] in the byten(&CellP[0],...) below. Skip the payload, drain at the poll. */
		  endrun(222); restart_status = 222; goto finish_turn;
		}
	      /* fluid-cell data  */
	      byten(&CellP[0], N_gas * sizeof(struct gas_cell_data), modus);

#ifdef CHIMES 
	      gasAbundancesBuf = (ChimesFloat *) malloc(N_gas * ChimesGlobalVars.totalNumberOfSpecies * sizeof(ChimesFloat));

	      if (!modus) /* write */
		{
		  /* Read abundance arrays into buffer */
		  for (partIndex = 0; partIndex < N_gas; partIndex++)
		    {
		      for (abunIndex = 0; abunIndex < ChimesGlobalVars.totalNumberOfSpecies; abunIndex++)
                {gasAbundancesBuf[(partIndex * ChimesGlobalVars.totalNumberOfSpecies) + abunIndex] = ChimesGasVars[partIndex].abundances[abunIndex];}
		    }
		}

	      /* Abundance buffer */
	      byten(&gasAbundancesBuf[0], N_gas * ChimesGlobalVars.totalNumberOfSpecies * sizeof(ChimesFloat), modus);
	      /* GasVars */
	      byten(&ChimesGasVars[0], N_gas * sizeof(struct gasVariables), modus);
			  
	      if (modus) /* read */
		{
		  for (partIndex = 0; partIndex < N_gas; partIndex++)
		    {
		      /* Allocate memory for abundance arrays */
		      allocate_gas_abundances_memory(&(ChimesGasVars[partIndex]), &ChimesGlobalVars);
		      
		      /* Read abundances from buffer */
		      for (abunIndex = 0; abunIndex < ChimesGlobalVars.totalNumberOfSpecies; abunIndex++)
                {ChimesGasVars[partIndex].abundances[abunIndex] = gasAbundancesBuf[(partIndex * ChimesGlobalVars.totalNumberOfSpecies) + abunIndex];}

#ifdef CHIMES_TURB_DIFF_IONS 
		      chimes_update_turbulent_abundances(partIndex, 1, P, CellP); 
#endif 
		    }
		}
	      free(gasAbundancesBuf); 
#endif
	    }

	  /* write state of random number generator */
	  byten(random_generator.s, sizeof(random_generator.s), modus);
	  byten(&SelRnd, sizeof(SelRnd), modus);

#ifdef TURB_DRIVING
      byten(StRng.s, sizeof(StRng.s), modus);
	  byten(&StNModes, sizeof(StNModes), modus);
	  byten(StOUPhases, StNModes*6*sizeof(double),modus);
	  byten(StAmpl, StNModes*3*sizeof(double),modus);
	  byten(StAka, StNModes*3*sizeof(double),modus);
	  byten(StAkb, StNModes*3*sizeof(double),modus);
	  byten(StMode, StNModes*3*sizeof(double),modus);
	  byten(&StTPrev, sizeof(StTPrev),modus);
#endif

	  /* write flags for active timebins */
	  byten(TimeBinActive, TIMEBINS * sizeof(int), modus);

	  /* now store relevant data for tree */
        in(&Gas_split, modus);
#ifdef GALSF
        in(&Stars_converted, modus);
#endif


	  /* now store relevant data for tree */

	  in(&nmulti, modus);
        in(&NTopleaves, modus);
        in(&NTopnodes, modus);
        /* The tree node arrays are sized from the domain's local-particle cap, which changes
           with the decomposition, so the node count this tree was built with travels in the
           restart file: the reader reproduces the writer's node and foreign capacities for an
           unedited restart, keeping the serialized node, pseudo-particle and Nextnode indices
           below valid.
           The LET foreign-node capacity travels too, and is handed to force_treeallocate verbatim
           below rather than re-derived: the serialized node pointers encode it (pseudo-particles
           start at TreeNodeIndexBase + MaxNodes + MaxForeignNodes), and one of its inputs,
           PartAllocFactor, is a value the user may edit on a restart.
           RuntimeMinLETForeignNodes, the run's ratcheted LET foreign-arena floor, travels so later
           trees, which derive their own capacity, keep the run's ratchet. */
        in(&MaxNodes, modus);
        in(&MaxForeignNodes, modus);
        byten(&RuntimeMinLETForeignNodes, sizeof(RuntimeMinLETForeignNodes), modus);
        if(modus && MaxForeignNodes < 0)
          {
            printf("Restart file on task=%d carries a negative LET foreign-node capacity (%d), so the tree layout it describes cannot be reproduced.\n",
                   ThisTask, MaxForeignNodes);
            fflush(stdout);
            /* soft: allocating and deserializing against an unusable layout would corrupt the tree.
             * Skip this rank's payload, drain at the per-turn poll. */
            endrun(90001026); restart_status = 90001026; goto finish_turn;
          }
	  if(modus != 0 && nmulti != MULTIPLEDOMAINS)
	    {
	      if(ThisTask == 0)
		printf
		  ("Looks like you changed MULTIPLEDOMAINS from %d to %d.\nWe will need to discard tree stored in restart files and construct a new one.\n",
		   nmulti, (int) MULTIPLEDOMAINS);

	      /* In this case we must do a new domain decomposition! */
	    }
	  else
	    {

	      if(modus)		/* read */
		{
		  domain_allocate();
		  /* The reader must rebuild the layout the WRITER serialized, so the slot count comes
		   * from the file, not from this rank's current capacity. The two are equal only while
		   * the capacity never moves; once it can grow mid-step the writer's tree side arrays
		   * were allocated with the smaller count, and its pseudo-particle segment sits at that
		   * smaller offset in Nextnode[]. Allocating at the grown capacity would deserialize the
		   * pseudo segment from the wrong offset -- the same physical-layout reason MaxNodes and
		   * MaxForeignNodes are read from the file rather than re-derived. */
		  force_treeallocate(MaxNodes, All.TreeParticleSlots, MaxForeignNodes);
		  /* tree-alloc UVM OOM: a NULL base means force_treeallocate soft-flagged. NO collective
		   * in this per-turn (subset) context -- skip the byten payload (local file reads) and
		   * drain at restart's existing all-rank poll. */
		  if(!Nodes_base || !Extnodes_base || !Nextnode || !Father) {restart_status = 223; goto finish_turn;}
		}

	      in(&Numnodestree, modus);

	      if(Numnodestree > MaxNodes)
		{
		  printf
		    ("Tree storage: it seems you have reduced(!) 'PartAllocFactor' below the value needed to load the restart file (task=%d). "
		     "Numnodestree=%d  MaxNodes=%d\n", ThisTask, Numnodestree, MaxNodes);
		  fflush(stdout);
		  /* soft: would overflow Nodes_base in the byten(...) below. Skip the payload, drain at the poll. */
		  endrun(221); restart_status = 221; goto finish_turn;
		}

	      byten(Nodes_base, Numnodestree * sizeof(struct NODE), modus);
	      byten(Extnodes_base, Numnodestree * sizeof(struct extNODE), modus);

	      byten(Father, NumPart * sizeof(int), modus);

	      byten(Nextnode, NumPart * sizeof(int), modus);
	      byten(Nextnode + All.TreeParticleSlots, NTopnodes * sizeof(int), modus);

	      byten(DomainStartList, NTask * MULTIPLEDOMAINS * sizeof(int), modus);
	      byten(DomainEndList, NTask * MULTIPLEDOMAINS * sizeof(int), modus);
	      byten(TopNodes, NTopnodes * sizeof(struct topnode_data), modus);
	      byten(DomainTask, NTopnodes * sizeof(int), modus);
	      byten(DomainNodeIndex, NTopleaves * sizeof(int), modus);

	      byten(DomainCorner, 3 * sizeof(double), modus);
	      byten(DomainCenter, 3 * sizeof(double), modus);
	      byten(&DomainLen, sizeof(double), modus);
	      byten(&DomainFac, sizeof(double), modus);
	    }

	finish_turn:
	  if(fd) {fclose(fd);}   /* fd is NULL on the 7870/7878 open-failure path */
	}
      else			/* wait inside the group */
	{
	  if(modus > 0 && groupTask == 0)	/* read */
	    {
	      MPI_Bcast(&all_task0, sizeof(struct global_data_all_processes), MPI_BYTE, 0, MPI_COMM_WORLD);
	    }
	}

      /* All-rank drain replaces the per-turn barrier (one collective, same sync): a per-rank
       * restart-IO fault set a soft bad-stop above; collect it here and finalize cleanly BEFORE
       * the next turn or domain_Decomposition() below runs on junk state. No MPI_Abort. */
      gizmo_exit_bad_stop_if_requested("restart:groupTask_turn");
    }

  /* Every payload has been read, so the writer's numbers have done their job and this run's take
   * over here -- both of them, together, because they are two halves of one capacity.
   *
   * The advertised storage may not stay at the writer's value for any longer than the read, or a
   * consumer that grows storage mid-step would ask for less than the decomposition is already
   * allowed to assign and be refused for it.  The assignment cap may not move any EARLIER than
   * this, because the top tree read above was sized by it.  The same swap remains at the domain
   * boundary, where it is now a no-op that costs one comparison and keeps that path correct on
   * its own terms. */
  if(modus != 0 && rekeyed_MaxPartAssignable) {All.MaxPartAssignable = rekeyed_MaxPartAssignable;}
  if(modus != 0 && old_MaxPart) {All.MaxPart = new_MaxPart; old_MaxPart = 0;}

  if(modus != 0 && nmulti != MULTIPLEDOMAINS)	/* in this case we must force a domain decomposition */
    {
        if(ThisTask == 0) {printf("Doing extra domain decomposition because you changed MULTIPLEDOMAINS\n"); fflush(stdout);}

      domain_Decomposition(0, 0, 0);
    }
}



/* reads/writes n bytes 
 */
void byten(void *x, size_t n, int modus)
{
  if(modus)
    my_fread(x, n, 1, fd);
  else
    my_fwrite(x, n, 1, fd);
}


/* reads/writes one int 
 */
void in(int *x, int modus)
{
  if(modus)
    my_fread(x, 1, sizeof(int), fd);
  else
    my_fwrite(x, 1, sizeof(int), fd);
}
