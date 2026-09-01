/* Vista Never-Hang Fatal Policy — fail-fast (local, cleanup-forbidden) termination.
 * See core/gizmo_fatal.h for the doctrine. Companion to run.cc's graceful
 * (collective, cleanup-allowed) controlled-stop path. */

#include "gizmo_fatal.h"

#include <mpi.h>
#include <signal.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

extern int ThisTask;  /* set in main.cc right after MPI_Comm_rank */

/* Rank captured at handler-install time so the async-signal-safe path never
 * reads a global that might be mid-update. -1 until installed. */
static volatile sig_atomic_t g_fatal_rank = -1;

/* --- async-signal-safe helpers (no libc formatting, no locks) -------------- */

/* Append the decimal form of v to buf at *pos (bounded). Async-signal-safe. */
static void as_append_int(char *buf, int cap, int *pos, long v)
{
    if(*pos >= cap) { return; }
    if(v < 0) { buf[(*pos)++] = '-'; v = -v; }
    char tmp[24];
    int n = 0;
    if(v == 0) { tmp[n++] = '0'; }
    while(v > 0 && n < (int)sizeof(tmp)) { tmp[n++] = (char)('0' + (v % 10)); v /= 10; }
    while(n > 0 && *pos < cap) { buf[(*pos)++] = tmp[--n]; }
}

static void as_append_str(char *buf, int cap, int *pos, const char *s)
{
    if(!s) { return; }
    while(*s && *pos < cap) { buf[(*pos)++] = *s++; }
}

/* Append v as 0x-prefixed hex. Async-signal-safe. Addresses are far more use in
 * hex than decimal: the magnitude alone separates a null-pointer offset from a
 * wild index. */
static void as_append_hex(char *buf, int cap, int *pos, unsigned long v)
{
    as_append_str(buf, cap, pos, "0x");
    char tmp[20];
    int n = 0;
    if(v == 0) { tmp[n++] = '0'; }
    while(v > 0 && n < (int)sizeof(tmp)) { const int d = (int)(v & 0xf); tmp[n++] = (char)(d < 10 ? ('0' + d) : ('a' + d - 10)); v >>= 4; }
    while(n > 0 && *pos < cap) { buf[(*pos)++] = tmp[--n]; }
}

/* The single safe primitive both signal handlers and normal-context callers use.
 * STRICTLY async-signal-safe: one bounded stack buffer, write(2), then _exit(). */
[[noreturn]] void gizmo_fatal_fast_exit(int code, const char *reason_static)
{
    char buf[512];
    int pos = 0;
    as_append_str(buf, (int)sizeof(buf), &pos, "GIZMO FATAL fast-exit rank=");
    as_append_int(buf, (int)sizeof(buf), &pos, (long)g_fatal_rank);
    as_append_str(buf, (int)sizeof(buf), &pos, " code=");
    as_append_int(buf, (int)sizeof(buf), &pos, (long)code);
    as_append_str(buf, (int)sizeof(buf), &pos, " : ");
    as_append_str(buf, (int)sizeof(buf), &pos, reason_static ? reason_static : "(no reason)");
    as_append_str(buf, (int)sizeof(buf), &pos, "\n");
    ssize_t w = write(2, buf, (size_t)pos);
    (void)w;
    _exit(code ? code : 1);
}

/* --- fatal signal handler -------------------------------------------------- */

static void gizmo_fatal_signal_handler(int sig, siginfo_t *info, void *ucontext)
{
    (void)ucontext;
    char buf[192];
    int pos = 0;
    as_append_str(buf, (int)sizeof(buf), &pos, "GIZMO FATAL signal=");
    as_append_int(buf, (int)sizeof(buf), &pos, (long)sig);
    as_append_str(buf, (int)sizeof(buf), &pos, " rank=");
    as_append_int(buf, (int)sizeof(buf), &pos, (long)g_fatal_rank);
    /* The faulting address, for the signals that carry one. Without it a
     * segfault report says only that one happened; with it, the magnitude
     * distinguishes a null dereference from an out-of-range index at a glance,
     * which is the difference between one diagnostic run and several. */
    if(info && (sig == SIGSEGV || sig == SIGBUS || sig == SIGILL || sig == SIGFPE)) {
        as_append_str(buf, (int)sizeof(buf), &pos, " addr=");
        as_append_hex(buf, (int)sizeof(buf), &pos, (unsigned long)info->si_addr);
        as_append_str(buf, (int)sizeof(buf), &pos, " code=");
        as_append_int(buf, (int)sizeof(buf), &pos, (long)info->si_code);
    }
    as_append_str(buf, (int)sizeof(buf), &pos, " -- no cleanup, immediate _exit\n");
    ssize_t w = write(2, buf, (size_t)pos);
    (void)w;
    _exit(128 + sig);
}

void gizmo_install_fatal_signal_handlers(void)
{
    g_fatal_rank = (sig_atomic_t)ThisTask;

    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_sigaction = gizmo_fatal_signal_handler;
    sigfillset(&sa.sa_mask);      /* block all other signals inside the handler */
    sa.sa_flags = SA_RESETHAND    /* a fault inside the handler re-raises the default disposition */
                | SA_SIGINFO;     /* deliver siginfo_t so the handler can report the faulting address */

    /* Involuntary faults + the launcher's SIGTERM to survivors. On GH200 a clean
     * reap beats a final flush; a healthy time-limit stop is handled internally
     * (TimeLimitCPU / controlled-stop), not by trusting external SIGTERM cleanup. */
    const int sigs[] = { SIGABRT, SIGSEGV, SIGBUS, SIGFPE, SIGILL, SIGTERM };
    for(unsigned k = 0; k < sizeof(sigs) / sizeof(sigs[0]); ++k) {
        sigaction(sigs[k], &sa, NULL);
    }
}

/* --- MPI error routing ----------------------------------------------------- */

/* One shared fail-fast errhandler, created lazily; set on MPI_COMM_WORLD +
 * MPI_COMM_SELF at install and on every communicator this code splits/creates
 * (else split comms keep MPI's default fatal disposition, which can wedge). */
static MPI_Errhandler g_failfast_errhandler = MPI_ERRHANDLER_NULL;

static void gizmo_mpi_error_handler(MPI_Comm *comm, int *err, ...)
{
    (void)comm;
    /* Doctrine: once MPI reports an error the fabric may be poisoned -- do NOT
     * touch MPI again (no MPI_Error_string). Route the integer code straight to
     * the async-safe fail-fast, which write()s code+reason and _exit()s. */
    gizmo_fatal_fast_exit(err ? *err : 1, "MPI transport error (no cleanup)");
}

void gizmo_mpi_set_failfast_errhandler(MPI_Comm comm)
{
    if(g_failfast_errhandler == MPI_ERRHANDLER_NULL) {
        MPI_Comm_create_errhandler(gizmo_mpi_error_handler, &g_failfast_errhandler);
    }
    MPI_Comm_set_errhandler(comm, g_failfast_errhandler);
}

void gizmo_install_mpi_error_handler(void)
{
    gizmo_mpi_set_failfast_errhandler(MPI_COMM_WORLD);
    gizmo_mpi_set_failfast_errhandler(MPI_COMM_SELF);
}

/* --- reviewed hard-fatal exit (renamed from gizmo_emergency_hold_reviewed) --- */

[[noreturn]] void gizmo_fatal_hard_exit_reviewed(int code, const char *reason,
                                                 const char *file, int line,
                                                 const char *func)
{
    /* Normal context: a bounded diagnostic BEFORE the async-safe fail-fast. */
    fprintf(stderr,
            "GIZMO FATAL (reviewed hard-exit, NO cleanup) on task=%d, function '%s()', "
            "file '%s', line %d: code %d: %s\n",
            (int)g_fatal_rank, func ? func : "(?)", file ? file : "(?)", line, code,
            reason ? reason : "(no reason given)");
    fflush(stderr);

#ifdef GIZMO_UNSAFE_LOCAL_COREDUMP_HOLD
    /* LOCAL-DEBUG ONLY, never defined on Vista/production. Preserves a core-dump-
     * friendly hold for a local debugger. This path is a KNOWN node-lock risk and
     * is compiled out of every production build (grep gate asserts the default
     * fatal path is clean). */
    fprintf(stderr, "  [GIZMO_UNSAFE_LOCAL_COREDUMP_HOLD] holding for local debugger; abort() for core\n");
    fflush(stderr);
    abort();
#endif

    gizmo_fatal_fast_exit(code, "reviewed hard-exit (mid-protocol / allocator floor)");
}
