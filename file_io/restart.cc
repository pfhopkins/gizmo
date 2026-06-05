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

static FILE *fd;

#ifdef CHIMES 
static ChimesFloat *gasAbundancesBuf;
#endif 

static void in(int *x, int modus);
static void byten(void *x, size_t n, int modus);

int old_MaxPart = 0, new_MaxPart;


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
	      if(All.PartAllocFactor != save_PartAllocFactor)
		{
		  old_MaxPart = All.MaxPart;	/* old MaxPart */

		  if(ThisTask == 0)
		    printf("PartAllocFactor changed: %f/%f , adapting bounds ...\n",
			   All.PartAllocFactor, save_PartAllocFactor);

		  All.PartAllocFactor = save_PartAllocFactor;
		  All.MaxPart = (int) (All.PartAllocFactor * (All.TotNumPart / NTask));
		  All.MaxPartGas = (int) (All.PartAllocFactor * (All.TotN_gas / NTask));
#ifdef ALLOW_IMBALANCED_GASPARTICLELOAD
          All.MaxPartGas = All.MaxPart; // PFH: increasing All.MaxPartGas according to this line can allow better load-balancing in some cases. however it leads to more memory problems
#endif
          new_MaxPart = All.MaxPart;

		  save_PartAllocFactor = -1;
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
	      if(old_MaxPart) {All.MaxPart = old_MaxPart;}	/* such that tree is still valid */
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
		  force_treeallocate((int) (All.TreeAllocFactor * All.MaxPart) + NTopnodes, All.MaxPart);
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
	      byten(Nextnode + All.MaxPart, NTopnodes * sizeof(int), modus);

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
