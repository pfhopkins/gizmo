# Hierarchical host timers

The timer API registers regions on first use and tracks their actual nesting.
It measures host elapsed time with `std::chrono::steady_clock`. It keeps its own
statistics, separate from `CPU_Step`, `CPU_Sum`, `WallclockTime`, and restart data.

## Enable and run

The normal GIZMO build includes the timing implementation. No configuration flag
or additional dependency is needed beyond the branch's existing C++17, MPI and
Kokkos requirements. Timers are disabled by default. Set these environment
variables on rank zero before launching:

```sh
GIZMO_TIMERS=1 mpiexec -n 4 ./src/GIZMO parameters.txt
```

| Variable | Default | Meaning |
| --- | --- | --- |
| `GIZMO_TIMERS` | off | `1` or `true` enables timers. |
| `GIZMO_TIMERS_INTERVAL` | `1` | Report at CPU log boundaries whose step number is a multiple of this positive integer. |
| `GIZMO_TIMERS_KOKKOS` | off | `1` or `true` also forwards regions to Kokkos profiling tools. Requires timers to be enabled. |

Rank zero broadcasts these settings once during initialization, so differing
rank environments cannot make reporting collectives diverge. A final report is
written regardless of the interval on normal completion or a controlled stop.

Rank zero appends `timers.txt` (readable tree) and `timers.tsv` (tab separated
records) to `OutputDir`. Each process launch adds a session marker and TSV header;
restarts begin a new session with fresh totals. The TSV header can therefore recur
between sessions. No timing state is loaded from restart files.

## Instrument a host region

Include `system/timer.h` using the relative path appropriate for the source file.
The header has no Kokkos, MPI, or GIZMO configuration dependency.

```cpp
#include "../system/timer.h"

void solve()
{
    GIZMO_TIMER_SCOPE("Tree");
    prepare();
    {
        GIZMO_TIMER_SCOPE("TreeSolve");
        {
            GIZMO_TIMER_SCOPE("small_function");
            small_function();
        }
        finish_solve();
    }
    finish_tree();
}
```

These regions appear as `Tree`, `Tree::TreeSolve`, and
`Tree::TreeSolve::small_function`. If the caller is already inside `Step`, the
paths begin with `Step::Tree`. The active stack determines the parent every time;
a shared function called in different contexts gets separate records.

Explicit boundaries are also supported:

```cpp
TimerStart("Tree");
TimerStart("Tree::TreeSolve");
// work
TimerStop("Tree::TreeSolve");
TimerStop("Tree");
```

Use short names when possible. A qualified name must identify an immediate child
of the full active path; it does not create missing ancestors. For example, inside
`Step::Tree`, use `TreeSolve` or `Step::Tree::TreeSolve`. Names must be nonempty and
cannot contain tabs or newlines; colons are reserved for `::` path separators. Avoid
embedding particle IDs, timestep numbers, or other unbounded values in names.

`TimerStop()` without an argument closes the current region. A named stop accepts
the leaf name or full path and validates the top of the stack. Invalid starts and
stops increment an error counter and emit a bounded diagnostic without changing
the stack. They do not abort the simulation. Prefer `GIZMO_TIMER_SCOPE` for normal
control flow: it closes on block exit, early return, and exception unwinding.
Scope objects cannot be copied or moved and must be destroyed on their creating
thread. Do not manually stop a scope owned by a `TimerScope` object.

## Read the measurements

- **Inclusive:** elapsed time in a region, including its nested regions.
- **Exclusive / self:** elapsed time excluding direct children. Time not covered
  by child timers remains in the parent's self time.
- **Calls:** started invocations, including active invocations.
- **Completed / active:** completed and still-open invocation counts.

Every snapshot is cumulative within a process launch. Do not add snapshots to
obtain totals. Live invocations contribute elapsed time up to the snapshot,
without being stopped or double counted at their eventual stop. Their changing
self time can include gaps between instrumentation points.

MPI reports reconcile complete paths across ranks, regardless of registration
order or rank-local branches. They report rank minimum, mean and maximum time,
ranks present, and summed call/completion/active counts. Absent regions contribute
zero to the rank mean and minimum. Maxima from different rows may belong to
different ranks and need not add up. Subtraction of child times happens locally,
before any MPI aggregation. Reporting is collective and can be expensive at high
rank counts; increase `GIZMO_TIMERS_INTERVAL` when needed.

Each host thread has its own stack. Thread zero is the thread that initializes
timing. Worker threads start with an independent root; they do not inherit the
main thread's active region. The report keeps `main` rows separate from `workers`
rows, which sum worker thread seconds on each rank. Worker sums are not wall time
and must not be subtracted from main-thread timers. Snapshots are synchronized
with worker updates; retired workers' statistics remain available until exit.

## Kokkos and device execution

Only instrument host code. Do not place the API inside `KOKKOS_LAMBDA`,
`KOKKOS_INLINE_FUNCTION`, per-interaction loops, or bodies that can execute on
either the device or the host. Use kernel dispatch regions to measure that work.

The initial instrumentation covers setup, each run-loop iteration, timestep
selection, domain decomposition, gravity solve/build/primary and Ewald walks,
hydro phases, the shared kernel launch helpers, and direct gravity walk launches.
It does not yet cover every direct Kokkos launch or every MPI exchange phase.

The timer adds no fences itself. Existing launch helpers already fence, so their
regions include host launch overhead, completion waiting and error checking.
Direct asynchronous operations may outlive their host region. These numbers are
host elapsed times, not pure device kernel times; use a GPU profiler for device
execution, transfers and overlap.

`timer_kokkos.cc` forwards the same paths through Kokkos `pushRegion`/`popRegion`
when requested. Forwarding covers main-thread regions; worker statistics are still
collected internally, without assuming external tools support concurrent region
stacks. The bridge is compiled with the GPU compiler, while the core and MPI
reporter use the host compiler. An attached profiling tool may request fences at
region boundaries; forwarding can therefore alter execution overlap. This API
does not install or replace Kokkos callbacks. GIZMO's existing allocation telemetry
still installs allocation/free callbacks, which needs separate attention when an
external tool also needs those events.

## Lifecycle and implementation limits

GIZMO initializes timing after Kokkos initialization. Reporting occurs on the main
thread at existing collective boundaries. Its local shutdown closes remaining
main-thread regions before Kokkos finalization. Fatal `_exit` paths intentionally
skip reports and cleanup, preserving the branch's fatal error policy. Automatic
scope cleanup cannot run through `exit` or `_exit`; partial data can be lost after
the last report. Quiesce instrumented workers before shutdown.

Disabled timers return before reading the clock, allocating records, or invoking
hooks; scope construction/destruction still has a small call/flag-check cost.
Enabled timers use an uncontended per-thread mutex and a lookup among the current
parent's children. Labels and paths are stored at registration, rather than
concatenated on each call. There is no shared lock or MPI call on each ordinary
entry/exit after the thread has registered. Thread registration and first visits
allocate memory. Prefer phases and batches over individual tiny operations.

For standalone use, call `gizmo::timing::TimerInitialize(true)` once before worker
threads start. `TimerSnapshot()` returns per-thread records;
`TimerShutdown()` disables instrumentation while retaining the records. The
optional `RegionHooks` pair must be nonthrowing and must not reenter the timing
API. Hooks that run on workers must support concurrent calls; set their
`main_thread_only` member to restrict forwarding to the initializing thread.
`TimerReportMPI` in `timer_mpi.h` requires every rank in its communicator to
call it from the main thread in the same collective order. Its stream arguments
are used only on communicator rank zero and may be null.

## Focused checks

Run from the repository root:

```sh
make -f dev/Makefile.timers check
make -f dev/Makefile.timers check-mpi
```

These standalone checks need C++17 and MPI, but no GIZMO configuration or Kokkos.
They exercise nested accounting, active snapshots, repeated calls, full paths,
context changes, recursion, early returns, exceptions, invalid boundaries,
disabled timers, concurrent workers, scope ownership, shutdown, and MPI reports
with differing registration order and missing regions. `TIMER_BUILD`, `CXX`,
`MPICXX`, `MPIEXEC`, and `CXXFLAGS` can be overridden on the make command line.
