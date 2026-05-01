# GPU Port — Current State (as of 2026-04-30)
Branch: `gpu_kokkos_unified`
Targets validated: Kokkos OpenMP (Apple Silicon, `MacBookCellar_Kokkos`); Kokkos CUDA (NVIDIA GH200, `Vista`).
Frontier (AMD MI250X, HIP) is the next planned target.

This document supersedes the earlier mid-Phase-4 handoff. The full thirteen-step modernization roadmap is complete; what follows is a snapshot of the resulting state and the small number of explicitly deferred items.

---

## 1. Status by roadmap step

| Step | Scope | Status |
|------|-------|--------|
| 1 | Per-kernel physics audit | DONE |
| 2 | Multi-rank validation | DONE |
| 3 | Compile-flag matrix (106/106 Mac, 144/144 Vista) | DONE |
| 4 | First-pass benchmarks | DONE — see §5 |
| 5 | Legacy CPU tree-walk retirement | DONE |
| 6 | Ghost-exchange infrastructure | DONE |
| 7 | Dead-code cleanup (header consolidation, `*_gpu.h` → `*_gpu_decls.h`, `mesh/ngb.cc` removal, `rt_CGmethod.cc`/`potential.cc` retirement) | DONE |
| 8 | Halo-transfer OOM handling | substantively DONE; dynamic realloc deferred behind mymalloc refactor |
| 9 | Remaining physics loops on device | DONE (one open known issue, §6) |
| 10 | Global-solver evaluation (HeFFTe / Hypre GPU) | DONE; HeFFTe explicitly deferred until a Vista module is available |
| 11 | Ancient/dead-code pruning | DONE |
| 12 | Documentation first pass | DONE |
| 13 | Domain decomposition + gravity tree on device | DONE (Phases 1–10 closed) |
| 14 | Documentation second pass (this document; user-guide GPU section) | IN PROGRESS |
| 15 | Scaling-test suite (Vista weak/strong, Frontier port) | not started |
| 16 | New-physics roadmap (two-temp plasma, batteries, MCRT, planet formation, multi-fluid, GR-MHD) | not started |

---

## 2. Device-resident inventory

Everything in the list below executes on the Kokkos device (GPU when CUDA/HIP backend is active; OpenMP threads when the OpenMP backend is active). Host responsibilities are limited to MPI traffic, file I/O, the time-loop, and a handful of conservation-diagnostic reductions.

**Hydro chain.** Ghost import (host MPI dispatch into device-resident ghost arena) → density h-convergence → symmetric neighbor-list build (SFC tiles + BVH, two-pass CSR) → gradient (MLS) → hydro force (Riemann + flux, atomic j-writes) → ghost writeback. Source files: `hydro/density.cc`, `hydro/hydro_evaluate_gpu.cc`, `hydro/gpu_neighbor_list.cc`, `hydro/gpu_gradient.cc`, `hydro/gpu_hydro_force.cc`, `system/ghost_exchange.cc`.

**Self-gravity tree pipeline (Step 13).**
- `gravity/gpu_topology_build.{cc,h}` + `gpu_topology_finalize.{cc,h}`: domain Peano-Hilbert key build and topology assembly on device.
- `gravity/gpu_morton.{cc,h}` + `gpu_morton_functions.h`: Morton sorting infrastructure for the BVH build.
- `gravity/gpu_peano_walk.{cc,h}` + `gpu_peano_walk_functions.h`: GPU SFC walk for tree construction.
- `gravity/gpu_gravity_tree.{cc,h}`: persistent SoA mirror of `Nodes_base`, `Nextnode`, `Father` (and the dirty-tracking ledger).
- `gravity/gpu_moment_refresh.cc`: incremental moment recomputation.
- `gravity/gpu_pseudo_update.{cc,h}` + `gpu_nextnode_thread.cc`: pseudo-particle updates and Nextnode threading.
- `gravity/let_pack.cc` + `let_data.h`: LET packing on device, exchanged via `MPI_Iallgatherv`.
- `gravity/gpu_gravtree.{cc,h}`: speculative Barnes-Hut walk, force / potential / tidal-tensor / jerk / adaptive-softening / PMGRID short-range table.
- `gravity/gpu_force_drift.cc`, `gpu_force_update.cc`: per-substep drift/cost/kick on device.
- `system/gpu_particles_arena.cc`: persistent UVM/SharedSpace arena for `P[]`, `CellP[]`, tree arrays.

**Source/subgrid physics.** Cooling, chemistry, RT subcycles, mechanical/thermal/radiative feedback, sink formation/accretion/swallow, FIRE radiative feedback, AGS density/force, SIDM, dm_dispersion, MHD div-B (CG/SSOR; HYPRE solve still host-side unless built `--with-cuda`), elastic solids, nuclear burning (`aprox13` device-callable; SkyNet/Torch external).

**Per-particle drift, predictor, timestep.** Folded into the same device parallel-region pattern.

---

## 3. Configuration model

There are no `Config.sh` flags that toggle GPU vs CPU dispatch — both code paths are always compiled in, and the Kokkos backend selects which runs. The retired flags (now hard-wired on with any Kokkos build) are: `OPENMP_GPU_OFFLOAD`, `GIZMO_USE_NEIGHBOR_LIST_FOR_DENSITY`, `GIZMO_GPU_GRAVTREE`. Builds without Kokkos are no longer supported.

The persistent particle arena (`P[]`, `CellP[]`, `Nodes[]`, `Nextnode[]`, `Father[]`) lives in Kokkos `SharedSpace` (CUDA Unified Memory on NVIDIA, page-migrated on HIP, plain host memory on OpenMP). Device kernels access this arena zero-copy; there are no host↔device deep-copies on the per-step hot path.

`__managed__` `All_dev` mirrors the runtime `All` parameter struct for read-only device access.

Optional infrastructure flag added during Step 13: `GIZMO_MIXED_PRECISION_GRAVITY` (default OFF). When set, exposes the `MyGravFloat = float` typedef and switches force-carrying gravity fields to single precision while leaving positions in double. Follows pkdgrav3 / Bonsai conventions. Diagnostic flag also added: `GIZMO_DEBUG_RT_COOLING` (high-volume stdout, intended for targeted RT/cooling debug; slated for retirement once §6 is closed).

---

## 4. Validation status

All currently-supported configurations are bitwise-identical between the GPU and (now-retired) legacy CPU tree-walk reference, on the same hardware, at 1- and 2-rank scale. Headline validations:

- `soundwave`, `dustywave`, `square`, `field_loop`, `mhd_wave`, `gmc_cooling`, `evrard`, `hernquist_sidm`, `isodisk_mechfb_sinks`, `isodisk_mechfb_cr`, `gravtree_vanilla`, `gravtree_vanilla_eval`, `gmc_cooling_pmgrid`: pass on Mac (Kokkos OpenMP) and Vista (Kokkos CUDA).
- Compile-flag matrix: 106/106 Mac, 144/144 Vista.

See `docs/gpu_port_verification_rules.md` for the canonical verification protocol and `MEMORY.md` index for per-test details.

---

## 5. Performance (NVIDIA GH200, single GPU vs. single Vista CPU node)

| Test | N | CPU (s/step) | GPU (s/step) | Speedup |
|------|---|--------------|--------------|---------|
| `poisson_box` (hydro + gravity) | 125k | 1.92 | 0.27 | 7.2× |
| `poisson_box` (hydro + gravity) | 1M | 17.0 | 2.14 | 7.9× |
| `gmc_cooling` (MHD + cooling + SF) | 512k | 25.8 | 4.8 | 5.4× |

Per-phase breakdown for `gmc_cooling` 512k: hydro force ~32×, gradient ~6.7×, cooling ~3.5×, density ~3.2×.

At ≲100k particles per rank the GPU build is overhead-bound. MHD and STARFORGE sweeps, plus a clean log-log scaling study, are the Step 15 deliverable.

---

## 6. Known open items

1. **`gmc_cooling_rt` ~25% divergence** (T, xe, urad_FUV) when used with the GPU RT-chemistry path. Bug is isolated; debug branch `gpu_kokkos_ngbtest_gmccoolingrtdebug` retains diagnostics. Not blocking any other work; `gmc_cooling_rt` is excluded from the validation gate in favor of `soundwave` + `gmc_cooling`.

2. **Dynamic realloc for the persistent arena** (Step 8 tail). Blocked behind a planned `mymalloc` refactor; the static `MaxPart`/`MaxNodes` allocations cover all current production runs.

3. **HeFFTe GPU PM** (Step 10 tail). Closed pending appearance of a HeFFTe Vista module; CPU FFTW3 + CPU Hypre paths are intact and PMGRID is not on the hot path for current science targets.

4. **Gravity tree optimization backlog** (seven items, benchmark-gated): see `MEMORY.md` → `project_gravity_tree_optimization_backlog.md`. Each item has a concrete trigger threshold to be evaluated against Step 15 scaling data; do not open speculatively.

---

## 7. Hard rules carried forward

These are invariants for any further work on the GPU/Kokkos path. Full statements live in `docs/gpu_port_verification_rules.md`; the short form:

- Test = comparison against reference, not exit code.
- Always multi-rank (`mpirun -np 2` minimum).
- Never loosen tolerances to make a test pass.
- Confirm the kernel actually fires (printf check) before declaring a port validated.
- Never use `gmc_cooling_rt` as the sole validator.
- Every port validates against both a standard-reference Config and an activating Config.
- No `_device.h` duplicate files; one canonical copy per device function.
- Never silently reduce scope or drop `#ifdef` branches in a port.
- No test artifacts in commits.
- Watch for nvcc errors `2001x` / `2009x` in build logs.
- Verify `Config.sh` against `GIZMO_config.h` after every build.
- GPU dispatchers use the active list, never `NumPart`; never early-return before MPI collectives.
- No `Config.sh` flags toggle GPU vs CPU dispatch — that is a Kokkos-backend concern.
- Ask before advancing task lists.
