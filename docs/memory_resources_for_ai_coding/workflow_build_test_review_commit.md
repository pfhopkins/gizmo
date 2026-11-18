# Build, test, review, commit: the workflow for agent-assisted GIZMO changes

## 1. Build

- Select the machine in `~/.gizmo` or `src/Makefile.systype` (`SYSTYPE="..."`; never commit your change to the latter). An exported
  `SYSTYPE` environment variable overrides both. Never pass `SYSTYPE=` on the `make` command line: it stays unquoted, matches no
  machine block, and the build silently falls back to a generic compiler.
- **Clean build every time** (`make clean` in `src/`, which deletes every `.o`, the binary, `GIZMO_config.h` and
  `compile_time_info.cc`). The Makefile has no header-dependency tracking, so a partial rebuild can link old and new code together;
  a result from an incremental build after a header or `Config.sh` change is evidence of nothing.
- A full build takes minutes: run it in the background, then check the binary's timestamp and count the `error` lines in the log.
  Never trust an exit code alone.
- `GIZMO_config.h` and the run's "Code was compiled with settings" lines only echo `Config.sh`; many flags are switched on implicitly
  by `declarations/precompiler_logic.h`. What is actually defined is the run's "Effective compile-time settings" block; read the `#if`
  nesting at the line you care about.

## 2. Compile coverage (CPU and GPU)

- Any module compiles on a CPU (Kokkos-OpenMP) build: to cover your change, build a Config with the flags that gate it. Code inside
  an `#ifdef` you did not enable was not compiled.
- `test/compile_suite/run_compile_suite.sh` builds many flag combinations. It overwrites `src/Config.sh` and `src/Makefile.systype`
  (restoring them only on a clean finish) and cleans the build at the end. Each mode writes a specific machine block named in the script
  (the default CPU mode, `--gpu` for CUDA, `--frontier` for HIP), so on another machine edit those lines first.
  Use `STOP_ON_FAIL=1` to halt at the first failure and `--resume` after the fix: failures usually share one root cause.
- **A GPU compile (CUDA and HIP) is required** for changes to anything device code compiles: device-callable functions and their
  annotations, neighbor-loop Specs, structures or layouts the device reads, device mirrors, allocations, launch policies, the
  neighbor-loop runner, headers included by many files, or `#if` logic that decides what reaches device code. A CPU build cannot
  detect a host function called from device code at all. If the change can alter what device code computes, compilation is not
  enough: compare a short device run against a CPU run of the same problem. On CUDA,
  treat the `#20011-D` / `#20014-D` warnings (host function called from device code) and the `warning #2001*` / `#2009*` class as
  errors; on HIP, read every warning that is not from Kokkos itself.
- In a `*_functions.h` header, follow the annotation convention of the module's existing headers (some define a fallback for
  `KOKKOS_INLINE_FUNCTION`, some require it to be defined before inclusion). Never include `Kokkos_Core.hpp` unconditionally there
  (host-only files that include it break CUDA builds), and never define the macro yourself just to hide a wrong include order.
- Every name in an added `#if` / `#ifdef` must exist (`src/Template_Config.sh`, the User Guide, a `#define` in `src/`, or the
  Makefile). A misspelled or retired name turns the condition into a constant without any warning.
- After deleting a symbol, grep the whole tree for it: code under flags you did not build may still use it.

## 3. Running tests

- Run a test the way it is run: from its directory, with its own `Config.sh`, parameterfile and initial conditions, the parameter
  file named by its bare name. Read the test's `test_<name>.py` first; if it generates its ICs, use the constants it passes, not
  the generator's command-line defaults. Never bolt a flag onto an unrelated test's Config.
- **At least 2 MPI tasks** (a single task skips the exchange between tasks, where many bugs live). Set `OMP_NUM_THREADS`
  explicitly (Kokkos otherwise takes every core) and leave room for other work on the machine.
- **Bound every run before launching it:** `TimeLimitCPU` (alias `MaxSimulationWallTime_in_Seconds`) just above what you need (the
  run stops cleanly once past ~85% of it, checked between steps, so one long step can overshoot), a short `TimeMax`, and an output
  volume you have computed (snapshots times snapshot size, plus the restart files a clean stop writes). Output goes in a run
  directory the developer designated, with room for it (checked first), never in system temporary directories and never as large
  output inside the source checkout. Stopping at `MinSizeTimestep` is the default: set a sane value and never add
  `CONTINUE_BELOW_MINTIMESTEP` to get past it. To stop a healthy run early, create a file named `stop` in its output directory.
- **Before reading any number:** read the run's standard error, and `grep -i` stdout and stderr for
  `endrun|error|failed|being ignored`. Unknown tags, duplicate tags (only the first is used) and tags whose module is compiled out
  are not errors: GIZMO prints "... is being ignored" and runs the default. Check `parameters-usedvalues` shows the values you set.
- **Prove the changed code ran:** the flags are on, the particles it acts on exist, and a temporary print or counter fires (then
  remove it). A pass that never reached your code proves nothing; every check needs a known positive that would fail.
- **Compare against a reference,** not the exit code: the test's own assertions (`pytest test/<name>`, see the User Guide), a
  known solution, or the parent commit's output. Snapshot comparisons match particles by sorted ID (check the IDs are unique first)
  and report magnitudes. Bitwise identity is a valid bar only for a provably inert change on a deterministic CPU run (2 tasks,
  1 thread); threaded and GPU runs are compared to the test's tolerances. Compare runs at the same physical time, never the same
  wall time; compare the earliest snapshot first when results look wrong.
- Gate in proportion: a mechanical change gets a compile and one reference problem. Most checks should take minutes; one that
  takes hours is usually a badly chosen check.
- `test/` holds physics test problems for users, not per-change regression checks. One-off checks live outside the repository.
  Adding a test problem needs the maintainer's agreement that it tests real physics no other problem covers or has teaching value.

## 4. Performance work

- Measure the target on the machine and configuration the claim is about before optimising it.
- Timing builds include `OUTPUT_ADDITIONAL_RUNINFO` (`cpu.txt` every step instead of only on full steps, plus `timebin.txt`,
  `info.txt`, `energy.txt`).
- Report absolute seconds first: wall time to the last common physical time, each timing category for both runs and their
  difference, and steps binned by number of active elements. Percentages and conclusions come after the numbers.
- A GPU result is judged against a CPU-only run using the same cores (MPI tasks times threads) as the GPU run: the GPUs are added
  to those cores, so the GPU run must beat the CPU run on the same node count.
- Under load imbalance, a blocking collective absorbs the wait of everything before it: time every phase between collectives,
  with mean and maximum across tasks.

## 5. Review

- Read your own complete staged diff (`git diff --cached`), not the tool's summary of what it changed. Batched or scripted edits
  that failed partway may still have written some files.
- Get an independent review of every non-trivial change: another agent or model, given the diff and the relevant code, not your
  explanation. Verify each finding in the code before acting on it. Imported pull requests get the same review.
- Check the staged diff for: `#error` added, `getenv`, debug prints and counters, comments referring to plans or history, new
  files, changed defaults, flags or parameters missing from `Template_Config.sh` and the User Guide.
- If review finds defects in your fix, stop patching: write down the invariant the fix maintains (each piece of cached or derived state,
  what changes it, what signals the change, and who reads it),
  agree that with the developer, and implement the smallest thing that satisfies it.

## 6. Commit and pull request

- Stage files by explicit path; check `git diff --cached --stat` before committing. Nothing generated, no outputs, binaries, ICs,
  logs, plots or agent notes.
- If the working tree holds other unrelated changes, check that the commit alone builds (e.g. build it from `git archive` of the
  commit or a separate worktree).
- One logical change per commit and per pull request. The message says what changed and why, which modules and configurations it
  affects, and what was run to check it.
- Before pushing, `git log origin/<branch>..HEAD`: a plain push sends every local commit. Pushing, and anything that changes a
  shared branch, waits for the developer's explicit go.
- Pull requests follow the User Guide's contribution steps.
