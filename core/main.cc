/*! \file main.c
 *  \brief start of the program
 */

/*!
 * These routines evolved from GADGET3 code developed by Volker Springel.
 * They have been extensively modified by Phil Hopkins (phopkins@caltech.edu)
 * for GIZMO.
 */


#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>
#include <time.h>
#include <math.h>

#include "../declarations/allvars.h"
#include "../core/proto.h"
/* Kokkos lifecycle wrappers declared in proto.h, defined in cooling/cooling.cc */



/*!
 *  This function initializes the MPI communication packages, and sets
 *  cpu-time counters to 0. Then begrun() is called, which sets up
 *  the simulation either from IC's or from restart files.  Finally,
 *  run() is started, the main simulation loop, which iterates over
 *  the timesteps.
 */
int main(int argc, char **argv)
{
  int i;

#ifdef IMPOSE_PINNING
  get_core_set();
#endif

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
  MPI_Comm_size(MPI_COMM_WORLD, &NTask);
  gizmo_kokkos_initialize(argc, argv);  /* must come after MPI_Init; sets up CUDA device and thread pool */
  /* Check if CUDA/Kokkos init changed CPU floating-point mode (FTZ/DAZ).
     On ARM (aarch64), check FPCR bits 24 (FZ) and 19 (FZ16).
     On x86, check MXCSR bits 15 (FZ) and 6 (DAZ). */
  {
#if defined(__aarch64__)
      unsigned long fpcr;
      __asm__ __volatile__("mrs %0, fpcr" : "=r"(fpcr));
      int fz = !!(fpcr & (1UL << 24));   /* flush-to-zero */
      int fz16 = !!(fpcr & (1UL << 19)); /* flush-to-zero for half-precision */
      if(ThisTask == 0) {printf("[FP_MODE] FPCR after Kokkos init: 0x%016lx  FZ=%d FZ16=%d\n", fpcr, fz, fz16); fflush(stdout);}
      if(fz) {
          fpcr &= ~(1UL << 24); /* clear FZ bit to restore IEEE denormal handling */
          __asm__ __volatile__("msr fpcr, %0" :: "r"(fpcr));
          __asm__ __volatile__("mrs %0, fpcr" : "=r"(fpcr));
          if(ThisTask == 0) {printf("[FP_MODE] FPCR after FZ fix: 0x%016lx  FZ=%d\n", fpcr, !!(fpcr & (1UL << 24))); fflush(stdout);}
      }
#elif defined(__x86_64__)
      unsigned int mxcsr;
      __asm__ __volatile__("stmxcsr %0" : "=m"(mxcsr));
      int ftz = !!(mxcsr & (1U << 15));
      int daz = !!(mxcsr & (1U << 6));
      if(ThisTask == 0) {printf("[FP_MODE] MXCSR after Kokkos init: 0x%08x  FTZ=%d DAZ=%d\n", mxcsr, ftz, daz); fflush(stdout);}
      if(ftz || daz) {
          mxcsr &= ~((1U << 15) | (1U << 6)); /* clear FTZ and DAZ */
          __asm__ __volatile__("ldmxcsr %0" :: "m"(mxcsr));
          if(ThisTask == 0) {printf("[FP_MODE] MXCSR after FTZ/DAZ fix: restored IEEE denormals\n"); fflush(stdout);}
      }
#endif
  }

#ifdef IMPOSE_PINNING
  pin_to_core_set();
#endif

  double safe_memorypertask = mpi_report_comittable_memory(0,1);
  MPI_Barrier(MPI_COMM_WORLD);

  /* initialize OpenMP thread pool and bind (implicitly though OpenMP runtime) */
  if(ThisTask == 0)
    {
      char *username = getenv("USER");
      char hostname[201]; hostname[200] = '\0';
      int have_hn = gethostname(hostname,200);
      time_t rawtime;
      struct tm * timeinfo;
      time ( &rawtime );
      timeinfo = localtime ( &rawtime );

      printf("\nSystem time: %s", asctime(timeinfo) );
      printf("This is GIZMO, version %d, running on %s as %s.\n",
              GIZMO_VERSION,
              have_hn == 0 ? hostname : "?",
              username ? username : "?"
      );
#ifdef BUILDINFO
      printf(BUILDINFO", " __DATE__ " " __TIME__ "\n");
#endif
      printf("\nCode was compiled with settings:\n\n");
      output_compile_time_options();
   }

#ifdef _OPENMP
#pragma omp parallel
  {
#pragma omp master
    {
      maxThreads = omp_get_num_threads();
    }
  }
#endif

  for(PTask = 0; NTask > (1 << PTask); PTask++);

  if(argc < 2)
    {
      if(ThisTask == 0)
	{
	  printf("Parameters are missing.\n");
	  printf("Call with <ParameterFile> [<RestartFlag>] [<RestartSnapNum>]\n");
	  printf("\n");
	  printf("   RestartFlag    Action\n");
	  printf("       0          Read initial conditions and start simulation\n");
	  printf("       1          Read restart files and resume simulation\n");
	  printf("       2          Restart from specified snapshot dump and continue simulation\n");
	  printf("       3          Run FOF and optionally SUBFIND if enabled\n");
	  printf("       4          Convert snapshot file to different format\n");
	  printf("       5          Calculate power spectrum and two-point function\n");
	  printf("       6          Calculate velocity power spectrum for the gas particles\n");
	  printf("\n");
	}
      endrun(0);
    }

  strcpy(ParameterFile, argv[1]);

  if(argc >= 3)
    RestartFlag = atoi(argv[2]);
  else
    RestartFlag = 0;

  if(argc >= 4)
    RestartSnapNum = atoi(argv[3]);
  else
    RestartSnapNum = -1;

  /* initialize CPU-time/Wallclock-time measurement */
  for(i = 0; i < CPU_PARTS; i++) {All.CPU_Sum[i] = CPU_Step[i] = 0;}

  CPUThisRun = 0;
  WallclockTime = my_second();

  begrun();			/* set-up run  */

  run();			/* main simulation loop */

  gizmo_kokkos_finalize();  /* must come before MPI_Finalize */
  MPI_Finalize();		/* clean up & finalize MPI */

  return 0;
}
