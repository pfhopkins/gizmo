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
| `GIZMO_TIMERS_SORT` | `max` | Sort `timers.txt` siblings by descending inclusive rank maximum (`max`) or mean (`mean`). |
| `GIZMO_TIMERS_MODE` | `cumulative` | Write cumulative reports, individual sync-point interval reports (`step`), or both (`both`). |

Rank zero broadcasts these settings once during initialization, so differing
rank environments cannot make reporting collectives diverge. A final report is
written regardless of the interval on normal completion or a controlled stop.

Rank zero appends `timers.txt` (readable tree) and `timers.tsv` (tab separated
records) to `OutputDir` in cumulative mode. Step mode writes `timers-step.txt`
and `timers-step.tsv`; `both` writes both pairs. Each process launch adds a session marker and TSV header;
restarts begin a new session with fresh totals. The TSV header can therefore recur
between sessions. No timing state is loaded from restart files.

## Naming and coverage convention

The runtime reports should provide a readable breakdown of the simulation and
make expensive regions easy to locate in the source. Whole-function scopes use
the actual function name, supplied automatically by `GIZMO_TIMER_FUNCTION()`.
Place it immediately inside the function definition, before early returns.
The macro expands to `GIZMO_TIMER_SCOPE(__func__)`; it does not automatically
instrument other functions.

Use `GIZMO_TIMER_SCOPE("phase_name")` for a portion of a function, such as the
`evolution_step` iteration or a communication/packing phase. Kernel launch scopes normally
use the Kokkos kernel label. A distinct label such as `ewald_kernel` identifies
the kernel subset of an enclosing function with the same name as the Kokkos
label. An unusually cumbersome function name may use a short explicit alias,
with a source comment stating the function and measured boundaries.

Each scope must cover the work its name describes. A whole-function timer ends
on function exit; a phase timer needs its own block if it ends before the
function. Avoid wrapping a call and instrumenting the callee with the same name,
which creates two indistinguishable levels. Missing child coverage remains in
self time, so the report must not be read as a complete function call graph.

The instrumented entry functions cover setup/run, timestep selection and drift,
statistics, full/light domain decomposition, gravity and tree build/walk,
long-range gravity, hydro phases, cooling, additional forces and stellar
feedback. Shared GPU launch helpers automatically time their named launches
with the helpers' existing fences; explicit direct-launch scopes cover selected
gravity kernels. This is a runtime summary whose finer breakdown can be extended
at large self-time regions. The timers add no GPU fences.

Bulk phases now have the following child coverage. Whole-function rows retain
their source names; phase rows describe loops or batches within those functions.

| Region | Child costs exposed |
| --- | --- |
| Evolution loop | First/second kicks, particle kick loops, motion-bound updates, active-list construction, drift, logging and controlled-stop polls. |
| Timestep selection | Fast-wave scan, displacement-constraint scan, particle timestep assignment, wakeups and their MPI reductions. |
| Hydro | Corridor preparation/ghost refresh, active-list/subgroup preparation, neighbour-runner routes, convergence/compaction and finalization. |
| Neighbour runner | Candidate collection, drift/filter, pair evaluation, arena/context preparation, local/remote CPU and GPU dispatch routes, ghost exchange and convergence reductions. |
| Gravity/tree | Tree build, topology preparation/build/finalization, Morton-sort batches, moment precomputation, moment kernel pipeline, copies/writeback, LET packing/encoding/exchange/install and walk completion reductions. |
| Domain decomposition | Extent and cost calculation, top-tree construction, sorting, memory checks, particle packing, receive layout and payload exchange. |
| Data movement | Particle staging gather/scatter, device/shared allocations, explicit host/device copies and shared-memory prefetch. |

Blocking MPI calls are measured under their caller, for example
`gravity_tree::MPI_Allreduce_walk_completion` and
`force_exchange_pseudodata_complete::MPI_Waitall`. Their elapsed time includes
both arrival skew and communication/progress time. A long wait identifies where
a rank waits; compare it with the preceding computation and packing regions to
investigate why other ranks arrive later. It is not a direct measurement of
network transfer time or proof that the collective itself is the bottleneck.
Nonblocking submission rows end in `_post`; `MPI_Waitall` measures the remaining
wait after any intervening work. These scopes add no MPI operations.

Tree kernels that already fence individually have a child named after the
kernel. The moment-refresh kernels share a single existing completion fence, so
they are measured together as `refresh_moments_kernels`; no extra fence is added
to separate them. Use GPU profiling to break down that asynchronous pipeline.
`pack_peer_LETs` measures the caller's wall time across the OpenMP packing loop;
packing/encoding on other threads appears separately in the worker lane as
thread-seconds. Per-particle and per-pair functions are deliberately left inside
their enclosing batch timer to keep overhead manageable.

This naming convention replaces earlier aliases such as `Tree`, `TreeSolve`,
`TreeBuild`, `Density` and `PrimaryWalk` with source function names. Downstream
filters should use the new paths for new runs; previously written sessions keep
their original labels.

## Instrument a host region

Include `system/timer.h` using the relative path appropriate for the source file.
The header has no Kokkos, MPI, or GIZMO configuration dependency.

```cpp
#include "../system/timer.h"

void small_function()
{
    GIZMO_TIMER_FUNCTION();
    // work
}

void gravity_tree()
{
    GIZMO_TIMER_FUNCTION();
    small_function();
}

void compute_grav_accelerations()
{
    GIZMO_TIMER_FUNCTION();
    gravity_tree();
}
```

These regions appear as `compute_grav_accelerations`,
`compute_grav_accelerations::gravity_tree`, and
`compute_grav_accelerations::gravity_tree::small_function`. In the main loop,
the paths begin with `run::evolution_step::compute_grav_accelerations`.
The active stack determines the parent every time;
a shared function called in different contexts gets separate records.

Explicit boundaries are also supported:

```cpp
TimerStart("prepare_walk");
TimerStart("prepare_walk::pack_targets");
// work
TimerStop("prepare_walk::pack_targets");
TimerStop("prepare_walk");
```

Use short names when possible. A qualified name must identify an immediate child
of the full active path; it does not create missing ancestors. For example, inside
`prepare_walk`, use `pack_targets` or `prepare_walk::pack_targets`. Names must be nonempty and
cannot contain tabs or newlines; colons are reserved for `::` path separators. Avoid
embedding particle IDs, timestep numbers, or other unbounded values in names.

`TimerStop()` without an argument closes the current region. A named stop accepts
the leaf name or full path and validates the top of the stack. Invalid starts and
stops increment an error counter and emit a bounded diagnostic without changing
the stack. They do not abort the simulation. Prefer `GIZMO_TIMER_SCOPE` for normal
control flow: it closes on block exit, early return, and exception unwinding.
Scope objects cannot be copied or moved and must be destroyed on their creating
thread. Do not manually stop a scope owned by a `TimerScope` object.

## Find finer-grained costs

Start with `timers.txt` and find an expensive region with a large **self** time.
Add smaller scopes around distinct work inside that region, rebuild GIZMO, and
rerun the same workload. For example, inside the already-instrumented
`gravity_tree()` function in `src/gravity/gravtree.cc`, add blocks like:

```cpp
{
    GIZMO_TIMER_SCOPE("prepare_walk");
    // existing preparation work
}
{
    GIZMO_TIMER_SCOPE("evaluate_walk");
    // existing walk work
}
{
    GIZMO_TIMER_SCOPE("exchange_results");
    // existing communication work
}
```

Each timer lasts until the end of its block. These names are automatically
reported under the active path, for example
`run::evolution_step::compute_grav_accelerations::gravity_tree::evaluate_walk`. No enum, registration table or report
format needs to be edited. Add another nested block to investigate a child.
Wrapping a function call includes all synchronous work in that call.

For explicit boundaries, `TimerStart("prepare_walk")` and
`TimerStop("prepare_walk")` measure only the statements between them. Scope timers
are preferable when the region has early exits. Start with batches or phases to
keep instrumentation overhead small. Device function bodies need GPU profiling;
host scope timers belong outside the kernel, around its dispatch.

For an investigation, report at every step boundary:

```sh
GIZMO_TIMERS=1 GIZMO_TIMERS_INTERVAL=1 \
  mpiexec -n 4 ./src/GIZMO parameters.txt
```

Reports are cumulative by default. Use timestep mode below to compare individual
steps. Do not subtract successive rank maxima: the slowest rank may change, so
correct interval maxima require differences on each rank before MPI aggregation.
Increase the reporting interval again for large production runs.
The TSV `path` column allows filtering a subtree;
`inclusive_mean_s`, `inclusive_max_s`, and `exclusive_mean_s` distinguish a
region's overall cost, rank imbalance, and time not attributed to its children.

## Compare individual timesteps

To retain aggregate totals and also write one-step measurements:

```sh
export GIZMO_TIMERS=1
export GIZMO_TIMERS_MODE=both
export GIZMO_TIMERS_INTERVAL=1
```

`timers.txt` / `timers.tsv` remain cumulative. `timers-step.txt` /
`timers-step.tsv` contain differences for each individual run-loop iteration,
including the timers nested under `evolution_step`. Differences are computed separately
for each thread/path on each MPI rank before reduction. Timer counters are not
reset and live scopes keep their nesting. Calls and completions count events in
the interval; `open_scopes` (TSV `active_sum`) remains the number of scopes open at its end.

The interval starts just after CPU-log reporting at the beginning of a run-loop
iteration, and ends at the next CPU-log reporting boundary. It includes the
iteration's work and the following statistics calculation. Startup work is
excluded from ordinary evolution intervals. A separate initial interval covers
`run()`'s initial force evaluation (or restart tree restoration), before the
evolution loop. It is marked `initial` in the TSV and shown as "initial
sync-point work" in the tree; the earlier `begrun()` setup remains in cumulative
reports. Initial work is reported even when the counter is not an interval
multiple. A forced final/controlled-stop report is marked `partial`, or `startup`
if no interval began. For `INTERVAL > 1`,
the selected output is still one step, not an accumulation over skipped steps.
Use interval 1 when investigating timestep-to-timestep changes.

### Sync-point numbering

Reports use GIZMO's existing `All.NumCurrentTiStep` counter and the same
`Sync-Point` terminology as the simulation log. A fresh run starts at sync-point
0: initialization and the initial force evaluation happen there, before the
evolution loop. Sync-point 1 is the first evolution advance and can have only a
few active particles. It is not a label for the large startup calculation.
Restart files retain their restored counter; timers do not renumber it.

Evolution interval headers show both endpoints, for example:

```text
Timer report: evolution interval | Sync-Point 1 | ranks=8 | errors(total)=0
Window: Sync-Point 0 -> 1
```

The TSV counter column is named `sync_point` (formerly `step`), and the appended
`from_sync_point` column identifies the interval's beginning. Initial work can
have equal endpoints, such as 0 -> 0, because no evolution advance occurred.
`from_sync_point=-1` denotes a cumulative or unbounded startup report. Parse each
session's TSV header when combining files written by different timer versions.

Each report also records the simulation time and system timestep, total particle
count, global active-particle and active-gas counts, per-rank active count means
and maxima, and highest active time bin. Workload is captured after domain
repartition and active-list construction, before the main force/physics work.
Active particles use the live active-list size; active gas uses the scheduled
active time-bin gas counts. These are timestep workload descriptors, not exact
per-kernel input sizes (imports and neighbour counts can differ). In cumulative
reports they describe the latest step, not a cumulative particle-update count.
Unavailable counts use -1, including startup and finalization before a workload
was captured. Metadata is repeated on each TSV timer row to allow filtering by
path and plotting `inclusive_max_s` against `active_particles_sum` or
`active_gas_sum` directly.

The `Interval-entry workload` line and appended TSV
`previous_active_particles_{sum,mean,max}` / `previous_active_gas_{sum,mean,max}`
fields describe the active set at the start of the interval, before timestep
selection and the first half-kick. The ordinary `Workload` line describes the
new force set after the advance. These sets can differ substantially: a sparse
new force set does not imply a cheap preceding timestep selection or kick.
Some algorithms also scan all particles regardless of active count; for example,
MFV kick loops scan `NumPart` and update active particles or gas. In cumulative
reports both workload descriptions refer to the latest interval.

Per-step snapshots and MPI reporting add overhead. Mode `step` omits cumulative
reporting; mode `both` performs both MPI aggregations at each selected boundary.

## Read the measurements

The readable tree's time columns are seconds summed across calls in the selected
window (session totals for cumulative mode, one iteration for step mode).
`mean` and `max` refer to MPI ranks, not individual calls:

| Column | Meaning |
| --- | --- |
| `incl_mean` | Mean across ranks of a region's elapsed time, including nested timers. |
| `incl_max` | Largest inclusive time on any rank. |
| `self_mean` | Mean across ranks of elapsed time excluding nested timers. |
| `self_max` | Largest self time on any rank. |
| `incl_imb` | Inclusive max / mean across ranks; 1 means balanced. |
| `self_imb` | Self max / mean across ranks; 1 means balanced. |

An imbalance of 2 means the slowest rank took twice the mean rank time. Zero-time
rows use 1. Missing regions contribute zero to the mean, so a region executed on
one of N ranks can show an imbalance of N. These ratios describe elapsed-time
imbalance; MPI waits can mask differences in computation. TSV columns are named
`inclusive_imbalance` and `exclusive_imbalance`.

For example, a region taking 10 seconds on a rank, with 7 seconds inside child
timers, contributes 10 seconds inclusive and 3 seconds self on that rank. Self
time includes uninstrumented work, MPI waiting, and GPU completion waiting within
the region; it is not necessarily CPU compute time.

Both readable trees sort top-level regions and each parent's immediate children by
descending `incl_max` by default. Parents stay before their complete subtrees;
the main/worker thread sections remain separate. ASCII branches show which rows
share a parent. The entire time column is not globally descending: a child stays
with its parent even if another subtree contains a more expensive child.
Ties use path names. Max ordering
highlights slowest-rank costs for MPI performance work. Compare max against mean
to identify imbalance; a large self time suggests where to add finer scopes.
Maxima in separate rows can come from different ranks and are not a reconstructed
critical path through the program.

To order by average rank cost instead, set this runtime variable before launch:

```sh
export GIZMO_TIMERS_SORT=mean
```

Mean and sum ordering are equivalent at fixed rank count, so mean ordering also
highlights aggregate rank time. TSV rows retain canonical path order for analysis.
Sorting changes output order only; timer collection and MPI reductions use the
same canonical paths in both modes.

- **Inclusive:** elapsed time in a region, including its nested regions.
- **Exclusive / self:** elapsed time excluding direct children. Time not covered
  by child timers remains in the parent's self time.
- **Calls:** started invocations, including active invocations.
- **Completed / open scopes:** completed and still-open invocation counts.
  These count timer invocations, not particles. An interval can show
  `calls_sum=0` and `open_scopes=8` for `run`: its scope began before the interval
  and remains open on all eight ranks.

The timer core's snapshots are cumulative within a process launch. Cumulative
reports must not be added to obtain totals; step reports contain separate
interval differences. Live invocations contribute elapsed time up to the snapshot,
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
main thread's active region. Readable reports use one "Main threads" section and,
when present, one "Worker threads" section instead of repeating a lane on every
row. The TSV retains `lane=main` / `lane=workers` for machine processing. Main
means the initializing host thread on each MPI rank, not MPI rank zero or a GPU
execution stream. Worker rows sum worker thread seconds on each rank. Worker sums are not wall time
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
