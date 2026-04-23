# GPU Port Handoff — 2026-04-23
Branch: `gpu_kokkos_domaintree`
Target: Kokkos OpenMP (Mac dev) / Kokkos CUDA (Vista GH200)

---

## 1. Completed and Committed

| Step | Module | Status |
|------|--------|--------|
| 1-12 | density, neighbor list, hydro gradients/forces, ghost exchange, MHD div-B, simple_chemistry cooling, rt_chem, RT transport subcycle, mechanical/thermal/radiation feedback, sink formation/swallowing, AGS density/force, SIDM, elastic solids, dm_dispersion | GPU-ported, Vista-validated |
| 13 Phase 1 | Persistent P[]/CellP[] arena (SharedSpace UVM), zero-copy GPU access | Committed |
| 13 Phase 3 | GPU gravity tree SoA (`gpu_gravity_tree.cc/h`), mirrors Nodes_base + Nextnode | Committed |
| 13 Phase 4 Tier 1a | GPU speculative gravity tree walk, core walk only (`7641063b`) | Committed; bitwise identical 1-rank + 2-rank |
| 13 Phase 4 Tier 1b.1+1b.2 | EVALPOTENTIAL inline potential + PMGRID shortrange_table (`5c07b4ac`) | Committed; bitwise identical 2-rank |

---

## 2. In Progress (working tree, not committed)

**Step 13 Phase 4 Tier 1b.3 — ADAPTIVE_GRAVSOFT_FORGAS**
- Code complete in `gravity/gpu_gravtree.cc`
- Compile-tested against all configs including FORGAS
- Runtime tests in progress: evrard, vanilla_eval, gmc_cooling+PMGRID
- NOT yet committed

---

## 3. Key Files Changed in Phase 4

| File | Role |
|------|------|
| `gravity/gpu_gravtree.cc` | Walk kernel + dispatcher (main file) |
| `gravity/gpu_gravtree.h` | Public interface (always-callable stub) |
| `gravity/gpu_gravity_tree.cc` + `.h` | SoA mirror of tree (Phase 3) |
| `gravity/gravtree.cc` | Calls `gpu_gravtree_walk_primary()` before primary loop (~line 173) |
| `gravity/forcetree.cc` | Changed `static float shortrange_table[]` to non-static for extern access |
| `test/gravtree_vanilla/Config.sh` + `.params` | Validation test (no AGS) |
| `test/gravtree_vanilla_eval/Config.sh` + `.params` | EVALPOTENTIAL variant |
| `test/evrard/evrard_gpu_val.params` | AGS validation; requires `TreeRebuild_ActiveFraction=2.0` |
| `test/gmc_cooling_pmgrid/Config.sh` + `.params` | PMGRID validation |

---

## 4. Supported Configs (compile + runtime pass)

| Config | Status |
|--------|--------|
| Vanilla (HYDRO_MESHLESS_FINITE_MASS + EOS_GAMMA + OUTPUT_IN_DOUBLEPRECISION + DEVELOPER_MODE + GIZMO_USE_NEIGHBOR_LIST_FOR_DENSITY + GIZMO_GPU_GRAVTREE) | bitwise identical 1-rank + 2-rank |
| Vanilla + EVALPOTENTIAL | runtime in progress |
| Vanilla + ADAPTIVE_GRAVSOFT_FORGAS + GIZMO_GPU_GRAVTREE | runtime in progress |
| BOX_PERIODIC + GRAVITY_NOT_PERIODIC + PMGRID + GIZMO_GPU_GRAVTREE | runtime in progress |

---

## 5. Unsupported / #error Gates in `gravity/gpu_gravtree.cc`

### Tier 1c (next planned)
- `ADAPTIVE_GRAVSOFT_SYMMETRIZE_FORCE_BY_AVERAGING` — not ported; blocks FORALL
- `ADAPTIVE_GRAVSOFT_FORALL` — auto-enables SYMMETRIZE via `precompiler_logic.h:152-156`; blocked

### Tier 2 (FIRE/STARFORGE payloads)
- `RT_USE_GRAVTREE`, `RT_USE_TREECOL_FOR_NH`
- `SINK_CALC_DISTANCES`, `SINK_PHOTONMOMENTUM`, `SINK_DYNFRICTION_FROMTREE`, `SINK_COMPTON_HEATING`
- `SINGLE_STAR_STARFORGE_DEFAULTS`, `SINGLE_STAR_SINK_DYNAMICS`, `SINGLE_STAR_TIMESTEPPING`, `SINGLE_STAR_FIND_BINARIES`, `SINGLE_STAR_FB_TIMESTEPLIMIT`
- `COSMIC_RAY_SUBGRID_LEBRON`, `GALSF_FB_FIRE_RT_LONGRANGE`, `CHIMES_STELLAR_FLUXES`

### Tier 3 (deferred)
- `ADAPTIVE_GRAVSOFT_FROM_TIDAL_CRITERION` — needs tidal tensor
- `COMPUTE_TIDAL_TENSOR_IN_GRAVTREE`, `COMPUTE_JERK_IN_GRAVTREE`
- `DM_SCALARFIELD_SCREENING`, `GRAVITY_SPHERICAL_SYMMETRY`
- `COUNT_MASS_IN_GRAVTREE`, `GRAVTREE_CALCULATE_GAS_MASS_IN_NODE`
- `GALSF_MERGER_STARCLUSTER_PARTICLES`, `ADAPTIVE_GRAVSOFT_MAX_SOFT_HARD_LIMIT`, `SINGLE_STAR_AND_SSP_NUCLEAR_ZOOM` (type-specific softening overrides)

### Phase 5 / Tier 3
- `HERMITE_INTEGRATION`, `ADAPTIVE_TREEFORCE_UPDATE`, `NEIGHBORS_MUST_BE_COMPUTED_EXPLICITLY_IN_FORCETREE`

### Later / not planned near-term
- `BOX_PERIODIC` without `GRAVITY_NOT_PERIODIC` (periodic Ewald walk)
- `SELFGRAVITY_OFF`

---

## 6. Known Gotchas

### 6.1 TakeLevel gate (CRITICAL for validation)
- `gpu_gravtree_walk_primary()` returns 0 immediately when `TakeLevel >= 0` (cost-measurement mode)
- Default `TreeRebuild_ActiveFraction=0.005` causes `TakeLevel >= 0` for typical test sizes → GPU walk never fires
- ALL validation params MUST have `TreeRebuild_ActiveFraction=2.0`
- Check: add a `printf` in the dispatcher to confirm GPU path is taken

### 6.2 All.G double-apply
- GPU walk writes RAW forces (no G factor)
- Post-walk loop in `gravity_tree()` applies `P[i].GravAccel *= All.G` unconditionally
- Same for `P[i].Potential` when EVALPOTENTIAL
- Do not add G inside the GPU kernel

### 6.3 force_drift_node stale SoA
- CPU updates `Nodes[no].u.d.s` in-place during walk
- `gpu_gravity_tree_invalidate()` must be called after each GPU walk to force reseed on next call
- Already implemented; do not remove

### 6.4 Arena invalidate
- `gpu_particles_arena_invalidate()` must be called after GPU writes `P[i].GravAccel`
- Already implemented; do not remove

### 6.5 FORALL blocked
- `ADAPTIVE_GRAVSOFT_FORALL` auto-enables `SYMMETRIZE_BY_AVERAGING` via `precompiler_logic.h:152-156`
- FORALL cannot be enabled until Tier 1c is ported

### 6.6 Dead code — force_treeevaluate_potential
- `force_treeevaluate_potential` is DEAD CODE when EVALPOTENTIAL is set
- `compute_potential()` is gated `#if !defined(EVALPOTENTIAL)` in `core/run.cc`
- Do not try to GPU-port or call it when EVALPOTENTIAL is active

### 6.7 Stale build artifacts
- Always `make clean` before switching Config.sh
- Stale .o files from a different config silently produce wrong binaries
- Verify which config built which binary: `grep GIZMO_config.h` after build

---

## 7. Next Atomic Commits

| Commit | Description | Blocker |
|--------|-------------|---------|
| Tier 1b.3 | ADAPTIVE_GRAVSOFT_FORGAS — commit after runtime validation | runtime tests pending |
| Tier 1c | ADAPTIVE_GRAVSOFT_SYMMETRIZE_FORCE_BY_AVERAGING + FORALL | depends on 1b.3 |
| Tier 2 | RT_USE_GRAVTREE, SINK_*, SINGLE_STAR_* payloads | depends on 1c |
| Tier 3 | COMPUTE_TIDAL_TENSOR, COSMIC_RAY_SUBGRID_LEBRON, DM_SCALARFIELD_SCREENING, GRAVITY_SPHERICAL_SYMMETRY, HERMITE_INTEGRATION | depends on Tier 2 |
| Phase 5 | ADAPTIVE_TREEFORCE_UPDATE GPU extrapolation | depends on Tier 3 |
| Later | BOX_PERIODIC Ewald walk, TreePM GPU, HeFFTe FFT | long-term |

---

## 8. Hard Rules for This Work

- NEVER loosen comparison tolerances — failing physics tests mean a real bug
- NEVER assume TakeLevel=-1 without verifying `TreeRebuild_ActiveFraction=2.0` in params
- NEVER skip multi-rank test — 1-rank masks domain boundary bugs
- NEVER commit without both a standard-reference Config AND an activating Config passing
- NEVER create dual code paths or fall back to CPU tree-walk; fix GPU infrastructure
- Always verify Config.sh + `grep GIZMO_config.h` after build before running tests
- Always call `gpu_gravity_tree_invalidate()` + `gpu_particles_arena_invalidate()` after GPU walk
- Add a `printf` to confirm the GPU dispatch path fires before declaring a config validated
