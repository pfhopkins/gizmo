# GIZMO facts that agents get wrong

Each of these has been mis-diagnosed or re-investigated many times. Check here before treating one of them as a bug.

## Configuration and parameters

- `FIRE_CRS=0` does not turn cosmic rays off; for a run without them, leave the flag out.
- `NOGRAVITY` is a retired name for `SELFGRAVITY_OFF`; an `#if !defined(NOGRAVITY)` is always true. Treat any unfamiliar flag name
  in a condition as suspect until you find where it is defined.
- An `#ifdef` placed before `allvars.h` is included sees neither the `Config.sh` flags (`GIZMO_config.h` comes in there) nor the
  flags `precompiler_logic.h` switches on implicitly: put flag-dependent includes after `allvars.h`.
- Parameters: unknown tags, duplicate tags (only the first is used) and tags whose module is compiled out do not stop the run; it
  prints "Tag ... was specified, but it is being ignored" on stdout and uses the default. Some tags are only read with
  `DEVELOPER_MODE`.
- Flag combinations that genuinely cannot work are checked in one place, the `#error` checks in `declarations/precompiler_logic.h`;
  a new one belongs there only for a combination a real user would plausibly try and that cannot work.
- The parameterfile name is limited in length: keep the parameterfile in the run directory and pass its bare name.
- `RestartFlag 1` needs the same memory layout and number of tasks as the run that wrote the restart files; it rebuilds the domain
  and tree on resume.
- `EOS_DAMAGE_POROSITY` and some other solid-body flags switch `EOS_TILLOTSON` on; in mixed-material builds, physics choices key
  on each cell's material (`eos_branch_of`), never on which EOS is compiled.

## Numerics that look like bugs and are not

- `NumNgb` is stored as its `NUMDIMS`-th root (in 3D a cube root: 3.17 for 32 neighbors).
- The kernel-length iteration stops anywhere within a tolerance band, so any change in summation order lands on a different, equally
  valid kernel length; in chaotic problems the difference then grows to order unity. Test: tighten `MaxNumNgbDeviation` by 100x
  (needs `DEVELOPER_MODE`); if the difference collapses to roundoff, this is the cause.
- With adaptive (individual) timesteps only total mass is conserved to machine precision; momentum, energy and other conserved
  quantities are conserved to integration accuracy. To tell an integration effect from a real leak, rerun with a single global
  timestep (`MinSizeTimestep = MaxSizeTimestep`, below the smallest needed step).
- Positions in an ordinary snapshot are drifted to the output time while velocities are at each particle's last kick, so energies
  computed from a snapshot are not a synchronized state.
- Hydro evaluates every pair from both sides on purpose (each side accumulates its own fluxes; only timestep wake-up flags and MFV mass fluxes are written to the neighbor, atomically): on GPUs that is cheaper than
  evaluating once and scattering to the neighbor with atomics. Do not propose "evaluate each pair once".
- Tree node moments are deliberately not refreshed when star formation turns a gas cell into a star of the same mass; mass changes
  that matter to gravity (e.g. sink formation) request a refresh (`TreeMomentsStaleFlag`).
- Particles spawned by some modules (e.g. `SINK_WIND_SPAWN`) share IDs by design, so per-particle snapshot comparisons are
  impossible there; compare totals and statistics.

## Code facts

- Particle slots, array indices and pointers are not identities and cannot be kept across domain decomposition, reordering,
  merging or splitting, spawning, or migration between tasks. If persistent identity is needed, use the module's established key
  and verify that it is unique; `P[i].ID` is not unique in every configuration.
- Directly changing a particle's position, type, mass or ordering can invalidate the tree, imported neighbor copies, cached bounds
  and device copies. Use the existing mutation or lifecycle helper for that change (for example, `mesh/gpu_neighbor_list.h` has
  ones for position, type and kernel-radius changes), check every piece of derived state that reads what you changed, and never
  invent a second change-tracking system.
- A change to a structure whose layout depends on `Config.sh` changes the restart-file layout, and the structure must be defined
  under identical `#if` guards in every source file that includes it.
- `struct particle_data` is over-aligned: never `malloc`/`calloc`/`realloc` it; use `mymalloc` or `new[]`.
- The tree's node indices start at `All.TreeNodeIndexBase`, not `All.MaxPart` (which changes during a run).
- For neighbor and hydro separations, periodic, shearing and other box boundaries are handled only by `NEAREST_XYZ` and
  `NGB_PERIODIC_BOX_LONG_{X,Y,Z}`, which take all three coordinates. Never write per-axis wrapping by hand. (Gravity can
  deliberately use different periodicity, e.g. `GRAVITY_NOT_PERIODIC`.)
- Random numbers: each loop has its own stream (`gizmo_loop_rng_salt("<loop name>")`; a new stochastic draw gets its own name,
  never another loop's), and pair draws are keyed on both particle IDs. Changing the
  order of draws in a stochastic routine changes results without being wrong; bitwise identity is not the bar there.
- An atomic on a neighbor-loop Spec's own accumulator is never needed (it is private to the source); `atomic_add` and `+=` round
  differently, so do not expect bitwise identity when swapping them.
- `core/init.cc` can overwrite fields read from the initial conditions: when an IC field is missing from the first snapshot, look there
  first.

## Errors, exits and shutdown

A run that dies the wrong way on a GPU machine can leave processes stuck in the device driver, which on a cluster can hold nodes
until an administrator reboots them. `MPI_Abort` used to be called throughout the code and is not safe there; it has been removed.
- **Never add `MPI_Abort`, a direct `MPI_Finalize`, or `exit()`.** Report an error with `endrun(code)`, `code` nonzero.
- `endrun(0)` is not an error path: it finalizes MPI and exits at once on the calling task. It is only for places where the run ends
  on purpose on every task (e.g. post-processing restart modes).
- On the host, `endrun()` with a nonzero code records a request for a controlled stop and RETURNS. The run is already invalid:
  stop using the invalid state at once (return, or skip the remaining work), but keep the sequence of collective calls intact so
  every task reaches the existing all-task check and the run drains to a clean shutdown. Never enter a collective from an error
  branch only some tasks take, and never return early on one task while the others will enter a collective (deadlock). What is
  computed after `endrun` does not matter; whether the shutdown is clean does.
- In device code `endrun` records the failure and stops the kernel; the host turns that into the same controlled stop.
- `gizmo_fatal_hard_exit_reviewed` and the fail-fast handlers (`core/gizmo_fatal.h`) are only for failures where no all-task check
  can be reached (corrupted communication, fatal signals); never use them for an ordinary error.
- At the normal end of a run, `core/main.cc` calls `gizmo_kokkos_finalize()` before `MPI_Finalize()`; keep that order in any new
  shutdown path.

## Device (GPU) facts

- `gizmo_gpu_kernel_launch` waits for the kernel to finish; `gizmo_gpu_kernel_launch_async` does not. Never free, resize, return to
  a pool or overwrite a buffer until the work using it has completed, including in shared (unified) memory.
- Neighbor-loop Specs read global settings only through their `CallScalars`. Other GPU source files that launch their own kernels
  read `All.*` through a device mirror and must call `GIZMO_GPU_ENSURE_ALL_FRESH()` at the dispatch boundary; on HIP a missing call
  silently reads every global as zero.
- A `__managed__` variable owned by one source file must only be accessed through functions compiled in that file; strong copies of
  such functions emitted from another file read the wrong copy on CUDA while the CPU build is correct.
- Some NVIDIA compilers constant-fold a local `int` used as a branch gate inside a device lambda; such gates computed from particle
  data are declared `volatile int`.
- `fflush(stdout)` and other host-only calls are illegal in device functions (plain `printf` is fine).
- Host edits to tree node links between rebuilds must also update the device copy of the links, or a later walk can loop forever.
- A crash that disappears when a small source change moves the compiler's register layout is not fixed; it is still a bug.
