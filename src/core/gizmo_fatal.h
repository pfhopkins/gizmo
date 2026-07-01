#ifndef GIZMO_FATAL_H
#define GIZMO_FATAL_H

#include <mpi.h>

/* Vista Never-Hang Fatal Policy — the FAIL-FAST (local, cleanup-FORBIDDEN) half
 * of GIZMO termination. The GRACEFUL (collective, cleanup-allowed) half lives in
 * core/run.cc (controlled-stop) and is the ONLY path permitted to call
 * gizmo_kokkos_finalize()/MPI_Finalize().
 *
 * Doctrine (permanent): after fabric/device/MPI state is poisoned, EVERY cleanup
 * action -- Kokkos::fence/finalize, MPI_Abort/Finalize, cudaDeviceReset, or even
 * normal C++ static destructors -- is a node-lock risk (a rank wedged in the
 * driver goes uninterruptible D-state -> SLURM CG -> node reboot). The only safe
 * action on a fatal/involuntary error is to STOP touching GPU/MPI and let the OS
 * reap a clean process via _exit(). _exit (never exit()) skips atexit + Kokkos/
 * CUDA static destructors that would otherwise run on a dirty context.
 *
 * GRACEFUL is only for self-detected, symmetric, poll-reachable errors (see
 * run.cc gizmo_exit_bad_stop_if_requested + endrun(nonzero)). Everything
 * involuntary or mid-protocol (fatal signals, UCX/MPI transport corruption,
 * allocator floors with no poll, launcher SIGTERM to survivors) is FAIL-FAST. */

/* SSOT fail-fast. STRICTLY async-signal-safe: write() a preformatted line then
 * _exit(code?code:1). No printf/malloc/MPI/Kokkos/CUDA. reason_static must be a
 * string literal or otherwise long-lived pointer (read in signal context). */
[[noreturn]] void gizmo_fatal_fast_exit(int code, const char *reason_static);

/* Install fatal-signal handlers (SIGABRT/SEGV/BUS/FPE/ILL/TERM). Call once, right
 * after MPI_Comm_rank sets ThisTask and BEFORE Kokkos/CUDA init. */
void gizmo_install_fatal_signal_handlers(void);

/* Route MPI-reported errors to fail-fast instead of MPI's default abort (which
 * can wedge). gizmo_install_mpi_error_handler() covers MPI_COMM_WORLD +
 * MPI_COMM_SELF; call gizmo_mpi_set_failfast_errhandler(comm) after every
 * communicator split/create so derived communicators are covered too. */
void gizmo_install_mpi_error_handler(void);
void gizmo_mpi_set_failfast_errhandler(MPI_Comm comm);

/* Reviewed hard-fatal exit for the rare mid-protocol / allocator-floor sites that
 * cannot reach a symmetric poll. RENAME of the former gizmo_emergency_hold_reviewed
 * -- the infinite "hold" is retired. Normal context: bounded fprintf/fflush
 * diagnostic, then gizmo_fatal_fast_exit(code). No fence/sleep/MPI_Abort/finalize
 * in the default path. */
[[noreturn]] void gizmo_fatal_hard_exit_reviewed(int code, const char *reason,
                                                 const char *file, int line,
                                                 const char *func);

#endif /* GIZMO_FATAL_H */
