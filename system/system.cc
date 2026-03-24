#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <unistd.h>
#include <signal.h>
#include <gsl/gsl_rng.h>


#include "../declarations/allvars.h"
#include "../core/proto.h"

/*! various routines are collected here, needed to communicate with the 
 *  actual system, stdin/out, abort runs, etc
 */

/*!
 * This file (and the relevant routines) were originally part of the GADGET3
 * code developed by Volker Springel. Some routines
 * have been collected and re-arranged a bit, but not substantially modified, by 
 * Phil Hopkins (phopkins@caltech.edu) for GIZMO. Many others have been written from
 * scratch by PFH, particularly any that deal with the new modular memory allocation,
 * parallelization, neighbor communication, and multi-threading methods.
 */


/* get random number -- old format took ID as seed to generate reliably identical numbers from a table, but this is not safe for large simulations */
double get_random_number(MyIDType id)
{
    return gsl_rng_uniform(random_generator);
}

/* returns the number of cpu-ticks in seconds that have elapsed. (or the wall-clock time) */
double my_second(void)
{
  return MPI_Wtime();

  /* note: on AIX and presumably many other 32bit systems,
   * clock() has only a resolution of 10ms=0.01sec
   */
}

double measure_time(void)	/* strategy: call this at end of functions to account for time in this function, and before another (nontrivial) function is called */
{
  double t, dt;

  t = my_second();
  dt = t - WallclockTime;
  WallclockTime = t;

  return dt;
}

double report_time(void)       /* strategy: call this to measure sub-times of functions*/
{
  double t, dt;

  t = my_second();
  dt = t - WallclockTime;

  return dt;
}


/* returns the time difference between two measurements
 * obtained with my_second(). The routine takes care of the
 * possible overflow of the tick counter on 32bit systems.
 */
double timediff(double t0, double t1)
{
  double dt;

  dt = t1 - t0;

  if(dt < 0)			/* overflow has occured (for systems with 32bit tick counter) */
    {
      dt = 0;
    }

  return dt;
}




#ifdef X86FIX

#define _FPU_SETCW(x) asm volatile ("fldcw %0": :"m" (x));
#define _FPU_GETCW(x) asm volatile ("fnstcw %0":"=m" (x));
#define _FPU_EXTENDED 0x0300
#define _FPU_DOUBLE   0x0200

void x86_fix(void)
{
  unsigned short dummy, new_cw;
  unsigned short *old_cw;

  old_cw = &dummy;

  _FPU_GETCW(*old_cw);
  new_cw = (*old_cw & ~_FPU_EXTENDED) | _FPU_DOUBLE;
  _FPU_SETCW(new_cw);
}

#endif


void minimum_large_ints(int n, long long *src, long long *res)
{
  int i, j;
  long long *numlist;

  numlist = (long long *) mymalloc("numlist", NTask * n * sizeof(long long));
  MPI_Allgather(src, n * sizeof(long long), MPI_BYTE, numlist, n * sizeof(long long), MPI_BYTE,
                MPI_COMM_WORLD);

  for(j = 0; j < n; j++)
    res[j] = src[j];

  for(i = 0; i < NTask; i++)
    for(j = 0; j < n; j++)
      if(res[j] > numlist[i * n + j])
        res[j] = numlist[i * n + j];

  myfree(numlist);
}

void sumup_large_ints(int n, int *src, long long *res)
{
  int i, j, *numlist;

  numlist = (int *) mymalloc("numlist", NTask * n * sizeof(int));
  MPI_Allgather(src, n, MPI_INT, numlist, n, MPI_INT, MPI_COMM_WORLD);

  for(j = 0; j < n; j++)
    res[j] = 0;

  for(i = 0; i < NTask; i++)
    for(j = 0; j < n; j++)
      res[j] += numlist[i * n + j];

  myfree(numlist);
}

void sumup_longs(int n, long long *src, long long *res)
{
  int i, j;
  long long *numlist;

  numlist = (long long *) mymalloc("numlist", NTask * n * sizeof(long long));
  MPI_Allgather(src, n * sizeof(long long), MPI_BYTE, numlist, n * sizeof(long long), MPI_BYTE,
		MPI_COMM_WORLD);

  for(j = 0; j < n; j++)
    res[j] = 0;

  for(i = 0; i < NTask; i++)
    for(j = 0; j < n; j++)
      res[j] += numlist[i * n + j];

  myfree(numlist);
}

size_t sizemax(size_t a, size_t b)
{
  if(a < b)
    return b;
  else
    return a;
}


/* routine to invert square matrices of side Ndims x Ndims, used in a number of places in code for e.g. gradients, etc. */
double matrix_invert_ndims(Mat3<double>& T, Mat3<double>& Tinv)
{
    double FrobNorm = T.frobenius_norm_sq(); /* Frobenius norm of T */
    Tinv = Mat3<double>{}; /* initialize Tinv to null */
#if (NUMDIMS==1) /* one-dimensional case */
    double detT = T[0][0]; if((detT != 0) && !isnan(detT)) {Tinv[0][0] = 1./detT;} else {Tinv[0][0]=1;} /* only one non-trivial element in 1D! */
#elif (NUMDIMS==2) /* two-dimensional case */
    double detT = T[0][0]*T[1][1] - T[0][1]*T[1][0];
    if((detT != 0) && !isnan(detT))
    {
        Tinv[0][0] =  T[1][1] / detT; Tinv[0][1] = -T[0][1] / detT;
        Tinv[1][0] = -T[1][0] / detT; Tinv[1][1] =  T[0][0] / detT;
    }
#else /* three-dimensional case */
    double detT = T.invert(Tinv);
#endif
    double FrobNorm_inv = Tinv.frobenius_norm_sq(); /* Frobenius norm of inverse matrix */
    double ConditionNumber = DMAX( sqrt(FrobNorm*FrobNorm_inv) / NUMDIMS , 1 ); /* this = sqrt( ||T^-1||*||T|| ) :: should be ~1 for a well-conditioned matrix */
    return ConditionNumber;
}

/* legacy overload accepting raw arrays — forwards to Mat3 version */
double matrix_invert_ndims(double T[3][3], double Tinv[3][3])
{
    return matrix_invert_ndims(*reinterpret_cast<Mat3<double>*>(T), *reinterpret_cast<Mat3<double>*>(Tinv));
}



long long report_comittable_memory(long long *MemTotal,
				   long long *Committed_AS, long long *SwapTotal, long long *SwapFree)
{
  FILE *fd;
  char buf[1024];

  if((fd = fopen("/proc/meminfo", "r")))
    {
      while(1)
	{
	  if(fgets(buf, 500, fd) != buf)
	    break;

	  if(bcmp(buf, "MemTotal", 8) == 0)
	    {
	      *MemTotal = atoll(buf + 10);
	    }
	  if(strncmp(buf, "Committed_AS", 12) == 0)
	    {
	      *Committed_AS = atoll(buf + 14);
	    }
	  if(strncmp(buf, "SwapTotal", 9) == 0)
	    {
	      *SwapTotal = atoll(buf + 11);
	    }
	  if(strncmp(buf, "SwapFree", 8) == 0)
	    {
	      *SwapFree = atoll(buf + 10);
	    }
	}
      fclose(fd);
    }

  return (*MemTotal - *Committed_AS);
}

/* task to assess available memory and print it. also returns an estimate of the maximum available memory for an MPI task (assuming equal numbers of tasks per node) */
double mpi_report_comittable_memory(long long BaseMem, int verbose)
{
    long long *sizelist, maxsize[6], minsize[6], Mem[6];
    int i, imem, mintask[6], maxtask[6];
    double avgsize[6];
    char label[DEFAULT_PATH_BUFFERSIZE_TOUSE];
    
    Mem[0] = report_comittable_memory(&Mem[1], &Mem[2], &Mem[3], &Mem[4]);
    Mem[5] = Mem[1] - Mem[0];

    for(imem = 0; imem < 6; imem++)
    {
        sizelist = (long long *) malloc(NTask * sizeof(long long));
        MPI_Allgather(&Mem[imem], sizeof(long long), MPI_BYTE, sizelist, sizeof(long long), MPI_BYTE, MPI_COMM_WORLD);
        
        for(i = 1, mintask[imem] = 0, maxtask[imem] = 0, maxsize[imem] = minsize[imem] =
            sizelist[0], avgsize[imem] = sizelist[0]; i < NTask; i++)
        {
            if(sizelist[i] > maxsize[imem])
            {
                maxsize[imem] = sizelist[i];
                maxtask[imem] = i;
            }
            if(sizelist[i] < minsize[imem])
            {
                minsize[imem] = sizelist[i];
                mintask[imem] = i;
            }
            avgsize[imem] += sizelist[i];
        }
        free(sizelist);
    }

    if(verbose) /* print outputs for the user */
    {
        if(ThisTask == 0)
        {
            printf("-------------------------------------------------------------------------------------------\n");
            for(imem = 0; imem < 6; imem++)
            {
                switch (imem)
                {
                    case 0:
                        snprintf(label,DEFAULT_PATH_BUFFERSIZE_TOUSE,  "AvailMem");
                        break;
                    case 1:
                        snprintf(label, DEFAULT_PATH_BUFFERSIZE_TOUSE, "Total Mem");
                        break;
                    case 2:
                        snprintf(label, DEFAULT_PATH_BUFFERSIZE_TOUSE, "Committed_AS");
                        break;
                    case 3:
                        snprintf(label, DEFAULT_PATH_BUFFERSIZE_TOUSE, "SwapTotal");
                        break;
                    case 4:
                        snprintf(label, DEFAULT_PATH_BUFFERSIZE_TOUSE, "SwapFree");
                        break;
                    case 5:
                        snprintf(label, DEFAULT_PATH_BUFFERSIZE_TOUSE, "AllocMem");
                        break;
                }
                printf
                ("%s:\t Largest = %10.2f Mb (on task=%d), Smallest = %10.2f Mb (on task=%d), Average = %10.2f Mb\n",
                 label, maxsize[imem] / (1024.), maxtask[imem], minsize[imem] / (1024.), mintask[imem],
                 avgsize[imem] / (1024. * NTask));
            }
            printf("-------------------------------------------------------------------------------------------\n");
        }
        if(ThisTask == maxtask[2])
        {
            printf("Task with the maximum commited memory");
            system("echo $HOST");
        }
        fflush(stdout);
    }

    /* do some quick checks on estimating available memory to be safe */
    int number_of_shared_memory_nodes = getNodeCount();
    int number_of_mpi_tasks_per_node = NTask / number_of_shared_memory_nodes;
    double min_memory_mb_in_single_node = minsize[0]/(1024.);
    double safe_memory_mb_per_mpitask_nobuffer = min_memory_mb_in_single_node / number_of_mpi_tasks_per_node;
    double safe_memory_mb_per_mpitask_withbuffer = safe_memory_mb_per_mpitask_nobuffer - All.BufferSize;
    return safe_memory_mb_per_mpitask_withbuffer;
}

