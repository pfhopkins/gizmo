# Rules for AI agents working on GIZMO

The User Guide's "Code Development & Extensions" section is the authoritative statement of the principles; read it first. The rules
below apply them. "The developer" means the human you work for: every decision marked as theirs is theirs alone.

## 1. Code

1. **Single source of truth.** One implementation of each equation, limiter, unit conversion or eligibility rule. Extend the
   existing routine or loop; never add a parallel copy, a `_device` duplicate or a second path that computes less. When you move
   code, delete the old copy in the same change.
2. **Never silently remove or weaken physics or numerics.** Not by dropping an `#ifdef` branch, adding an `#error` for a flag
   combination you did not handle, returning a constant, falling back to an older path, loosening a tolerance, or removing a
   limiter or safety check (even one a rewrite makes look unnecessary). A change covers every configuration of the code it touches.
   Bug fixes are welcome (see behaviour rule 4 for how).
3. **Physics and numerics stay separate.** Physics goes through the existing templates (local per-element work, the neighbor-loop
   Spec, the gravity tree walk, the global solvers; see the User Guide). Dispatch and optimisation decisions key on structural
   properties (number of active elements, search type, memory), never on which module is calling.
4. **Work scales with the active set.** On a step where few elements are active, nothing may loop over all particles or nodes, or
   copy whole particle arrays to or from the device. Check every new loop and every new device transfer against this.
5. **Portable device code.** Kokkos only: no CUDA- or HIP-specific code. `GIZMO_GPU_COMPILER` means "built by a GPU compiler"
   (host and device passes alike); code that must differ between the device and host passes tests
   `defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)`, never `__CUDA_ARCH__` alone (it does not exist on HIP). Functions
   called from device code in another file are `KOKKOS_INLINE_FUNCTION` in a `*_functions.h` header, included where used (no
   relocatable device code). Never guard a physics call out of the device build to make it compile.
6. **Use the device well.** A device loop that is parallel at one level only (one lane per source) needs a stated reason. When
   there is work but few sources, give each source a team of threads (a Kokkos `TeamPolicy`, one team per source) sharing its
   neighbor list or tree walk.
7. **Bounded memory.** State the size of every new buffer as a function of the active count, task count and received traffic;
   no permanent growth, no retained buffers without a bound.
8. **No hidden switches.** No environment variables controlling behaviour. A user setting is a `Config.sh` flag or a parameterfile
   parameter, documented in `src/Template_Config.sh` and the User Guide in the same change, or it is fixed in the code.
9. **Small footprint.** No new files for small helpers (they go in the module's existing header or `core/proto.h`); no new
   validation frameworks, telemetry or defensive machinery nobody asked for; no argument lists assembled from `#ifdef` blocks.
10. **Comments are for a stranger with only the code.** Say what the code does and why, cite papers for formulas. Never mention
    plans, phases, stages, tickets, private notes, the agent, or the history of the change. Never write "deferred" or "TODO" for
    work you left out: either do it or tell the developer.
11. **Temporary code stays separable.** Debug prints, counters and test switches live in their own block or patch, never on a line
    that also carries real code, and are removed before commit. Decide when each one is removed at the moment you write it.

## 2. Behaviour

1. **"Stop" means stop.** Halt, report the state, wait.
2. **One item at a time.** Finish it, report, ask before starting the next. A to-do list is not permission to continue.
3. **Scope reductions are the developer's decision, asked as a full stop.** Any change that leaves out a configuration, a branch
   or a capability: stop work, ask the question on its own, with no "I'll proceed with X unless you object", and wait.
4. **A failure you find gets reported and resolved as the developer decides,** whether or not your change caused it; "pre-existing"
   and "out of scope" are not answers, and neither is quietly widening your change to fix it. Report it, asking "intended, or a
   bug?" when the behaviour might be deliberate.
5. **Claims need artifacts.** "Fixed", "validated", "passing" or "the cause is X" requires a clean build of the code you describe,
   the run's own output, and a comparison against a reference. Otherwise say what the run shows and label the explanation a
   hypothesis. Lead with the numbers; every summary word ("faster", "everywhere") cites them. A surprising result means first
   that you have a bug.
6. **Read before you theorise.** Read the diagnostics the run already wrote, the input data, and the existing records before
   building a new instrument or proposing a new hypothesis. When a symptom is unchanged across several changes, stop changing
   things and instrument.
7. **Know what others have landed.** Before designing or committing: `git fetch`, read the recent history of the branches you
   build on, and merge them first.
8. **Destructive or outward actions need the developer's yes, each time:** deleting files, any backward git motion (`reset`,
   `restore`, `checkout -- <path>`, `stash`, `clean`, `revert`, force-push, branch deletion), pushing, posting, sending, submitting
   jobs. Approval of one does not carry to the next.
9. **Never kill a process,** your own included: list it (PID, age, command, why) and let the developer decide. Design every run
   to end by itself; you may stop your own healthy run cleanly with its `stop` file, never anyone else's.
10. **Approval has a source.** Before acting on "approved", be able to quote who said what. A question the developer asked is not
    a ruling; your own earlier message is never approval.
11. When the task is done, say so and stop. Do not invent follow-up work.

## 3. Never

- Commit your own notes, plans, logs, run outputs, snapshots, binaries, ICs, plots or large files; never `git add -A` or `git add .`.
- Write simulation output or large files to system temporary directories, or anywhere outside the directories the developer named.
- Log in to, copy files to or from, or submit jobs on remote computing facilities unless the developer has set that up
  deliberately; never retry a failed login (repeated failures under multi-factor authentication can lock out an account or a
  whole group).
- Call `MPI_Abort`, `MPI_Finalize` or `exit()` to handle an error, or let an error branch on some tasks skip a collective the others
  enter: use `endrun()` and keep the exit graceful (see "Errors, exits and shutdown" in `code_facts_and_pitfalls.md`).
- Blame the compiler or floating point for a CPU-versus-GPU difference larger than roundoff: it has almost always been a bug in
  the code. Claim a compiler defect only with a minimal reproducer.
- Add a test problem to `test/` for a one-off check (see `workflow_build_test_review_commit.md`).
