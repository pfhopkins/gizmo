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
#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif

#include "../declarations/allvars.h"
#include "../core/proto.h"
/* Kokkos lifecycle wrappers declared in proto.h, defined in cooling/cooling.cc */

static unsigned long long ExecutableFNV1a64 = 0; /*!< 64-bit FNV-1a hash of the running executable (0 if it could not be read) */
static long long ExecutableSizeInBytes = -1;     /*!< size of the running executable in bytes (-1 if it could not be read) */
unsigned long long executable_fnv1a64(void) {return ExecutableFNV1a64;}
long long executable_size_in_bytes(void) {return ExecutableSizeInBytes;}

/* Fingerprint of the running executable (its size and 64-bit FNV-1a hash), so two runs can be checked for
   having used the same build. This identifies builds; it is not a security hash. Task 0 reads the file once
   and every task receives the result, since any task may write snapshot headers. */
static void record_executable_fingerprint(const char *argv0)
{
    if(ThisTask == 0)
    {
        char path[MAX_PATH_BUFFERSIZE_TOUSE]; path[0] = 0;
#if defined(__APPLE__)
        uint32_t path_size = sizeof(path);
        if(_NSGetExecutablePath(path, &path_size) != 0) {path[0] = 0;}
#elif defined(__linux__)
        snprintf(path, sizeof(path), "/proc/self/exe");
#endif
        FILE *fd = path[0] ? fopen(path, "rb") : NULL;
        if(!fd && argv0) {fd = fopen(argv0, "rb");}
        if(fd)
        {
            static unsigned char buffer[1 << 16];
            unsigned long long hash = 14695981039346656037ULL; long long size = 0; size_t n;
            while((n = fread(buffer, 1, sizeof(buffer), fd)) > 0) {for(size_t k = 0; k < n; k++) {hash ^= buffer[k]; hash *= 1099511628211ULL;} size += (long long) n;}
            fclose(fd);
            ExecutableFNV1a64 = hash; ExecutableSizeInBytes = size;
        }
    }
    MPI_Bcast(&ExecutableFNV1a64, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ExecutableSizeInBytes, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
}



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
  /* Vista never-hang fatal policy: install fail-fast handlers as early as rank is
     known and BEFORE Kokkos/CUDA init, so an involuntary fault (fabric/UCX abort,
     signal) or a reported MPI error becomes an immediate no-cleanup _exit rather
     than an abrupt-kill with a live CUDA context (which wedges GH200 nodes in
     SLURM CG). See core/gizmo_fatal.h. */
  gizmo_install_fatal_signal_handlers();
  gizmo_install_mpi_error_handler();
  gizmo_node_comm_init();  /* persistent node-local communicator, used for node-scoped memory reporting */
  gizmo_kokkos_initialize(argc, argv);  /* must come after MPI_Init; sets up CUDA device and thread pool */
  gizmo_kokkos_mem_register();  /* observe Kokkos allocations from here on (after Kokkos init, before GIZMO allocations) */
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
  record_executable_fingerprint(argv[0]);

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
      printf("Effective compile-time settings (the above, with every default and derived option they switch on; hash %016llx):\n\n",
             compile_time_effective_options_fnv1a64());
      for(const char *line = compile_time_effective_options(); *line; )
      {
          const char *end = strchr(line, '\n');
          printf("        %.*s\n", (int) (end - line), line);
          line = end + 1;
      }
      printf("\nCompiler: %s\n", compile_time_compiler_version());
      if(executable_size_in_bytes() >= 0) {printf("Executable: %lld bytes, FNV-1a 64 hash %016llx\n", executable_size_in_bytes(), executable_fnv1a64());}
      else {printf("Executable: could not be read to fingerprint it\n");}
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

  if(strlen(argv[1]) >= sizeof(ParameterFile)) /* every task sees the same arguments, so all leave here together */
    {
      if(ThisTask == 0)
	{
	  printf("The parameterfile name '%s' is %d characters long, but at most %d are allowed.\n"
	         "Run from the directory that holds the parameterfile and pass just its name.\n",
	         argv[1], (int) strlen(argv[1]), (int) sizeof(ParameterFile) - 1);
	  fflush(stdout);
	}
      MPI_Finalize();
      exit(1);
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
  cpu_chain_sync(my_second());

  begrun();			/* set-up run  */

  report_memory_ledger("startup");  /* all ranks reach this point after begrun(); its node-scoped reduce is collective */

  /* Bad-stop poll after setup (Vista no-MPI_Abort policy). If begrun() flagged
   * an unrecoverable bad stop (bad params / missing table / IO setup failure),
   * every rank collected it here; skip the run loop and fall through to the
   * clean gizmo_kokkos_finalize() + MPI_Finalize() shutdown. This is a
   * top-level global point all ranks reach, so the collect is symmetric. */
  gizmo_collect_controlled_stop();
  if(gizmo_controlled_stop_code() == 0)
    {
      run();			/* main simulation loop */
    }
  else if(ThisTask == 0)
    {
      const char *r = gizmo_controlled_stop_local_reason();
      printf("Bad-stop during setup (code=%d): %s. Skipping run loop; "
             "finalizing cleanly.\n", gizmo_controlled_stop_code(),
             r ? r : "(reason set on other rank only)");
      fflush(stdout);
    }

  /* A bad stop (here or inside run()) exits nonzero AFTER a clean finalize.
   * The full original code/reason are printed above / by the run loop; the
   * process status is a small fixed nonzero (large codes aren't portable as
   * exit statuses). Node still releases cleanly: MPI_Finalize, not MPI_Abort. */
  int badstop_exit = (gizmo_controlled_stop_code() != 0) ? 1 : 0;

  gizmo_kokkos_finalize();  /* must come before MPI_Finalize */
  MPI_Finalize();		/* clean up & finalize MPI */

  return badstop_exit;
}
