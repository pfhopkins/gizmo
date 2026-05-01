# GIZMO GPU Port Verification Rules
# Machine-readable reference for Codex (AI agent) validation of GPU port changes
# Date: 2026-04-22
# Source: memory feedback files + gpu_port_handoff_20260423.md

---

## 1. Absolute Rules (BLOCKING — never violate)

1. **Test = comparison, not exit code.** A clean `exit 0` / shutdown is NOT validation. Always compare snapshot output against a reference solution using pytest assertions. Many GPU/MPI bugs produce wrong physics silently.

2. **Always run multi-rank.** Minimum `mpirun -np 2` for all test runs. Single-rank bypasses ghost exchange and masks MPI coordination bugs. Use 1 rank only when specifically isolating a single-rank failure.

3. **Never loosen tolerances.** If `pytest` fails, STOP. Report the failure output and plots to the user. Never increase `rtol`/`atol`, widen bounds, or skip assertions. The only acceptable re-run reasons: wrong flags/Config.sh/systype, or a missing IC file (fix and re-run).

4. **The kernel must actually fire.** Compiling a flag into Config.sh does NOT prove the kernel runs. Confirm execution with a temporary `printf("[KERNEL_NAME-FIRED]")` or counter check. Remove the printf before committing. If PRINT_STATUS shows "0 active sources / 0 pairs", the kernel never ran — that is NOT a valid test.

5. **Never use gmc_cooling_rt as the sole validation test.** It has known radiation-solver bugs that can mask regressions. Minimum sets:
   - 1 test: `soundwave` OR `gmc_cooling` (never `gmc_cooling_rt` alone)
   - 2 tests: `soundwave` + `gmc_cooling`
   - 3 tests: `soundwave` + `gmc_cooling` + `gmc_cooling_rt`

6. **Every GPU port needs two Configs.** (a) A standard-reference Config where the ported kernel is NOT active — confirms no regression to unrelated paths. (b) An activating Config with the flags + IC/params that actually trigger the kernel.

7. **No `_device.h` duplicate files.** Never create `*_device.h` files duplicating functions from other files. One canonical copy per function. Use `*_functions.h` with `KOKKOS_INLINE_FUNCTION` to replace (not duplicate) the original.

8. **Never reduce scope or remove functionality.** When porting to GPU, never remove `#ifdef` branches, simplify logic, or drop edge cases. Guard GPU-incompatible branches with `#ifndef GIZMO_GPU_COMPILER`; do not delete them.

9. **Do not commit test-run artifacts.** Never `git add` output directories, plot files, ICs, TREECOOL, spcool_tables, or log files unless the user explicitly directs adding a new regression reference.

10. **Watch for nvcc error classes 2001x and 2009x.** After GPU compile, grep the build log for `error: 2001` and `error: 2009`. Surface any hits before proceeding.

11. **Verify Config.sh and binary match EVERY test.** After every build, run:
    ```bash
    cat Config.sh
    grep "" GIZMO_config.h | head -60
    ```
    Stale `.o` files from a different config silently produce wrong binaries. `make clean` before switching Config.sh.

12. **GPU dispatcher must use the active-list, NOT NumPart.** Active-list pattern is required for all GPU neighbor dispatchers. Never early-return before MPI collectives.

13. **Kokkos builds automatically activate all GPU infrastructure.** GPU neighbor-list, GPU kernels, and GPU gravity tree are always on with SYSTYPE=Vista or MacBookCellar_Kokkos. No extra Config.sh flags are needed to enable them.

14. **`TreeRebuild_ActiveFraction` uses the standard default (0.005).** The TakeLevel gate that previously required forcing this to 2.0 has been fixed; normal rebuild-fraction values work correctly. Validation params use 0.005.

15. **Ask before advancing task lists.** After completing each discrete task, report and ask for approval before moving to the next item.

---

## 2. Local Mac Verification Workflow

### 2.1 Build GPU variant (Kokkos OpenMP, MacBookCellar_Kokkos)

```bash
# Set Makefile.systype:
echo 'SYSTYPE="MacBookCellar_Kokkos"' > Makefile.systype

# Confirm Config.sh has the activating flags for the kernel under test
# (GPU neighbor-list and GPU kernels are always active with MacBookCellar_Kokkos)

make clean
make -j8 2>&1 | tee build_gpu.log

# Check for nvcc error classes:
grep -E "error: 200[19]" build_gpu.log

# Verify binary config matches intent:
grep "" GIZMO_config.h | head -60
```

### 2.2 Build CPU reference (no Kokkos)

```bash
# Use SYSTYPE="MacBookCellar" (non-Kokkos variant) for a CPU-only reference build.
# (MacBookCellar_Kokkos always enables GPU infrastructure — not suitable for reference.)

make clean
make -j8 2>&1 | tee build_cpu.log
```

Reference snapshot storage convention: `test/<problem>/reference_cpu_treewalk/`

### 2.3 Add a kernel-fires printf (before running)

Inside the ported kernel or its dispatcher, add:
```c
printf("[KERNEL_NAME-FIRED] n_active=%d\n", n_active);
```
This must appear in run output with `n_active > 0`. Remove before committing.

### 2.4 Run both variants multi-rank

```bash
# GPU run:
mpirun -np 2 ./GIZMO test/<name>/<name>.params 1>gizmo_gpu.out 2>gizmo_gpu.err

# CPU reference run (if generating fresh reference):
mpirun -np 2 ./GIZMO_CPU test/<name>/<name>.params 1>gizmo_cpu.out 2>gizmo_cpu.err
```

Check stdout for `[KERNEL_NAME-FIRED]` with nonzero count.

### 2.5 Run pytest assertions

```bash
# Standard test with existing reference:
pytest test/<name>/test_<name>.py -v

# When comparing GPU output against freshly-generated CPU reference:
GIZMO_TEST_SKIP_BUILD_RUN=1 pytest test/<name>/test_<name>.py -v

# Generate plots (always — user reviews plots, not just pass/fail):
# pytest framework auto-generates plots during the run above
```

### 2.6 What counts as PASS (local)

- `pytest` exits 0 with all assertions passing (see tolerance rules in Section 7).
- `[KERNEL_NAME-FIRED]` printf appeared with `n_active > 0`.
- Both standard-reference Config (kernel inactive) and activating Config (kernel active) pass.
- Run was multi-rank (`-np 2` minimum).
- Config.sh and binary were verified to match before run.

---

## 3. Vista Verification Workflow

### 3.1 Environment setup

```bash
# GPU build (nvidia + CUDA, for GH200):
source ~/.bashrc
# Makefile.systype must be: SYSTYPE="Vista"

# CPU build (gcc):
source ~/.bashrc_gcc
# Makefile.systype must be: SYSTYPE="Vista_CPU"

# Non-interactive SSH quirk — always prefix:
ssh vista 'source ~/.bashrc && cd /scratch/01799/phopkins/gizmo && ...'
```

NEVER load `module load gcc/...` after sourcing `.bashrc` — replaces nvidia/25.9 and breaks CUDA.

Key module set loaded by `.bashrc`:
```
nvidia/25.9  cuda  kokkos/4.5.01-cuda  openmpi  hdf5/2.0.0  fftw3  gsl
```

### 3.2 Sync source to Vista

```bash
# ALWAYS exclude Makefile.systype and Config.sh (per-machine files):
rsync -av --exclude='Makefile.systype' --exclude='Config.sh' \
    /path/to/local/gizmo/ vista:/scratch/01799/phopkins/gizmo/
```

Populate run directory with IC files from Mac before submitting:
```bash
rsync -av test/<name>/ics.hdf5 test/<name>/TREECOOL test/<name>/spcool_tables/ \
    vista:/scratch/01799/phopkins/gizmo/test/<name>/
# Also sync reference/exact files for pytest comparison.
```

### 3.3 Build on Vista

```bash
ssh vista 'source ~/.bashrc && cd /scratch/01799/phopkins/gizmo && make clean && make -j72'
```

Use `-j72` on Vista (not `-j8`). Login node can handle parallel make.

### 3.4 SLURM submission

Always copy from the canonical fiducial template: `test/gmc_cooling/run_gmc.bsub`

Template structure (header is one combined line):
```bash
#!/bin/bash
#SBATCH -J jobname -p gh-dev -N 1 --ntasks-per-node=2 -t 00:15:00 -A TG-NAIRR260139
source $HOME/.bashrc
ibrun ./GIZMO ./test/<name>/<name>.params 1>gizmo.out 2>gizmo.err
```

Key rules:
- Use `ibrun` NOT `mpirun` (TACC-specific)
- Account: `TG-NAIRR260139` (not truncated form `TG-NAIRR26013`)
- Extension: `.bsub` (not `.slurm`)
- `gh-dev` partition: short runs ≤15 min wall; max 2 in queue, 1 running concurrently
- `gh` partition: longer wait but higher per-user limit; use when gh-dev is contended
- Submit: `sbatch run.bsub`; check queue: `squeue -u $USER`
- **ONE job at a time** unless user explicitly approves concurrent submissions

**NEVER run GIZMO tests via pytest directly on Vista. Always use SLURM sbatch with ibrun.**

### 3.5 Sync output back and run assertions

After the job completes:
```bash
# Sync output back to Mac:
rsync -av vista:/scratch/01799/phopkins/gizmo/test/<name>/output/ \
    test/<name>/output_vista/

# Run pytest assertions locally against Vista-produced snapshots:
GIZMO_TEST_SKIP_BUILD_RUN=1 pytest test/<name>/test_<name>.py -v
```

Alternatively, run pytest on Vista directly:
```bash
PYTHON311=/opt/apps/gcc14/cuda12/python3/3.11.8/bin/python3.11
export LD_LIBRARY_PATH=/opt/apps/gcc/14.2.0/lib64:/opt/apps/gcc14/cuda12/python3/3.11.8/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/home1/01799/phopkins/.local/lib/python3.11/site-packages:/scratch/01799/phopkins/gizmo/python_src
GIZMO_TEST_SKIP_BUILD_RUN=1 $PYTHON311 -m pytest test/<name>/test_<name>.py -v
```

**SLURM clean exit is NOT validation. pytest assertions are mandatory.**

### 3.6 Test sequencing protocol on Vista

1. Submit one test via `sbatch`.
2. Wait for completion.
3. Sync output back; run pytest assertions; report pass/fail.
4. If PASS: inform user, run next scheduled test.
5. If FAIL: diagnose. If fix is a pure bug-fix with no functionality change → fix and resubmit. If fix would change functionality, roll back code, or defer a planned port → present summary to user and wait for approval.

---

## 4. Test-to-Kernel Map

Which test actually fires which GPU module. Compiling a flag in is not enough — the IC+params must trigger the physics.

| GPU Module / Kernel Family | Canonical Test | Reference Solution? |
|---|---|---|
| Hydro chain: density, gradient, hydro forces | `soundwave` | YES (`assert_snapshots_are_close(rtol=1e-5)`) |
| Hydro chain + cooling | `gmc_cooling` | YES (`gmc_cooling_exact.hdf5`, `rel=0.10`) |
| RT chemistry | `gmc_cooling_rt` | YES (but NEVER solo — see Rule 5) |
| Grain-gas drag (solids) | `dustywave` | YES |
| MHD / div-B | `square` (or `field_loop`, `mhd_wave`) | YES |
| Sinks (sink_environment, sink_feed, sink_swallow) | `isodisk_mechfb_sinks` (FIRE_BHS + SINK_GRAVACCRETION=1) | Asserts sink mass + baryon conservation |
| Mechanical feedback (SNe) | `isodisk_mechfb` / `isodisk_mechfb_cr` / `isodisk_mechfb_sinks` | No universal reference; compare pre/post migration |
| Thermal feedback | `isodisk_thermalfb` | No reference; compare pre/post |
| RT source injection | `fire_rtsources` or isodisk RT-active variant | No reference; compare pre/post |
| AGS density / force | `hernquist_sidm` (AGS_KERNELRADIUS_CALCULATION_IS_ACTIVE) | No reference; compare pre/post |
| SIDM / CBE integrator | `hernquist_sidm` | No reference; compare pre/post |
| Cosmic rays (CR pipeline) | `isodisk_mechfb_cr` | No reference; compare pre/post |
| Grain-grain collisions | custom config with GRAIN_COLLISIONS; no standard test | No reference; compare pre/post |
| GPU gravity tree walk | `gravtree_vanilla`, `gravtree_vanilla_eval`, `evrard` | Compare pre/post; bitwise identical at ≤2 ranks |
| GPU gravity + PMGRID | `gmc_cooling_pmgrid` | Compare pre/post |

### Notes on "No reference" tests — compare old vs new workflow

When no reference snapshot exists:
1. Build and run at the parent commit (before migration). Save `output/` as `reference_cpu_treewalk/`.
2. Build and run at the migration commit. Save `output/`.
3. h5diff or pytest-style comparison between the two (expect bitwise equality up to FP rounding).
4. The migration must be in its own commit with no other changes.

### Tests that do NOT exercise common modules

`soundwave` and `gmc_cooling` do NOT fire: sinks, AGS, mechfb, RT source injection, dispersion-grad, grain physics, SIDM, cosmic rays, gravity tree walk. A passing result on these tests says nothing about those modules.

---

## 5. Common Failure Modes (things that look like passes but aren't)

| Failure Mode | Symptom | How to Catch |
|---|---|---|
| Wrong binary / stale `.o` files | Compiles fine, wrong physics silently | `grep "" GIZMO_config.h` after every build; `make clean` before switching Config.sh |
| Single-rank only | No ghost exchange bugs caught | Always `-np 2` minimum |
| Kernel never fired | "0 active sources/pairs" in log; test passes but proves nothing | `[KERNEL_NAME-FIRED]` printf with count check |
| Wrong IC/params for module | Flag compiled in but physics never activates (no sinks formed, no SNe, no RT sources) | Inspect params: SinkFormationDensity, SNe rate, particle types present |
| Vista job exits cleanly but output is wrong | Silent memory/arena bug; SLURM says COMPLETED | Must rsync + run pytest assertions |
| gmc_cooling_rt used alone | Known radiation-solver bug masks regressions | Always pair with soundwave or gmc_cooling |
| TakeLevel ≥ 0 gate for gravity walk | *Fixed* — GPU walk now fires correctly with default rebuild fraction | If re-observed, check `gpu_gravtree_walk_primary()` TakeLevel logic |
| Stale gpu_gravity_tree SoA | Wrong forces after tree rebuild | Ensure `gpu_gravity_tree_invalidate()` called after each GPU walk |
| Stale arena after GPU write | Wrong P[i] reads on next host access | Ensure `gpu_particles_arena_invalidate()` called after GPU writes GravAccel |
| All.G double-applied | Forces off by G factor | GPU kernel must write RAW forces; post-walk loop applies G |
| FORALL blocked by SYMMETRIZE | Compile error or `#error` gate | ADAPTIVE_GRAVSOFT_FORALL requires Tier 1c ported first |
| Wrong SLURM account ID | Submission fails or charged to wrong allocation | Use `TG-NAIRR260139` (not truncated `TG-NAIRR26013`) |
| Heredoc variable expansion in ssh | SLURM script has wrong paths or expanded env vars | Use single-quoted heredoc: `ssh vista 'cat > file << '"'"'EOF'"'"' ...'` |

---

## 6. Verifying the GPU gravity walk fires

The historical `TreeRebuild_ActiveFraction=2.0` workaround is obsolete — the `TakeLevel` gate that previously skipped the GPU walk under cost-measurement mode has been fixed, and the standard default (`0.005`) works correctly. Validation params should use the standard default unless a specific test calls for otherwise.

To confirm the walk actually fired (mandatory for any GPU gravity port — see Rule 4):
```bash
grep "GPU gravity" gizmo.out   # or whatever prefix the dispatcher printf uses
# Must show nonzero particle counts.
```

---

## 7. What Counts as PASS

### Tolerance rules by test

| Test | Passing condition | Tolerance |
|---|---|---|
| `soundwave` | `assert_snapshots_are_close(rtol=1e-5)` | 1e-5 relative |
| `gmc_cooling` | pytest assertions vs `gmc_cooling_exact.hdf5` | `rel=0.10` |
| `dustywave` | pytest assertions vs reference | see test file |
| `square`, `field_loop`, `mhd_wave` | pytest assertions vs reference | see test file |
| GPU gravity tree walk (bitwise) | h5diff or snapshot comparison vs CPU tree-walk | Bitwise identical (FP rounding only) |
| "compare old vs new" (no reference) | h5diff between pre-migration and post-migration output | Bitwise identical or within genuine FP rounding |

**NEVER loosen a tolerance.** If a tolerance check fails, that is a real regression until proven otherwise.

### Bit-comparison expectations

For pure kernel migrations (no algorithm change, no new physics):
- 1-rank and 2-rank runs should be bitwise identical to the CPU tree-walk reference.
- Any difference beyond floating-point associativity reordering is a bug.
- "FP rounding" means at the level of `ulp` differences due to order-of-operations changes; it does NOT mean percent-level differences.

### Additional required checks for every port

- [ ] `[KERNEL_NAME-FIRED]` printf appeared with `n_active > 0` (activating Config).
- [ ] Standard-reference Config (kernel inactive) passes its existing pytest suite without tolerance change.
- [ ] Activating Config run with at least `mpirun -np 2`.
- [ ] `grep "" GIZMO_config.h` verified to match the intended Config.sh before each run.
- [ ] No nvcc errors 2001x or 2009x in build log.
- [ ] No test artifacts committed to the repo.

---

## 8. Test Suite Reference

### SHORT set (local Mac, after any refactor)
```
gmc_cooling
gmc_cooling_rt    # formally fails — surface plots+results for user review; never use as sole validator
soundwave
dustywave
square
```

### FULL set (only after Item 6B + explicit user approval)
```
square
dustywave
field_loop
gmc_cooling
gmc_cooling_rt
mhd_wave
soundwave
nuclear_unit
nuclear_xrb
nuclear_detonation
compiler_suite      # VERY IMPORTANT — exercises Config flag combinations
```
Note: TURB_DIFF / TURB_DIFF_DYNAMIC / TURB_DRIVING have no regression test yet (standing TODO).

### Vista GPU subset
Same as SHORT set. `gmc_cooling_rt` formally fails; surface plots+results for user approval.

### Test hierarchy (required gates before merge)
1. After any refactor: SHORT set locally (multi-rank Kokkos-OpenMP) → user review
2. After that passes + Item 6B done: FULL set locally → user review
3. After full passes + user approval: short set on Vista GPU → user review
4. Promote to commit/PR only after all three gates pass
