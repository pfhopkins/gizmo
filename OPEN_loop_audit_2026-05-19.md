# Open Loop Audit, 2026-05-19

Merged union of Claude + Codex loop-audit passes. Do not remove an item until Phil explicitly marks it done, skipped, or otherwise closed.

Ratings:
- P0: straightforward cleanup / consistency issue.
- P1: strong missed-port or performance candidate.
- P2: real loop/custom path needing design review.
- P3: likely special-case / diagnostic / infrastructure, but still open.
- P4: reviewed/done at top level for this audit; known other-branch or intentionally out-of-scope workstream, still tracked here for merge awareness only.
- CLOSED: reviewed and accepted as no-action for this audit; retained only as a historical cross-reference.
- SPECIAL-INFRA: reviewed as infrastructure/diagnostic, not a loop-port target for this audit.

P4 top-level status: reviewed/done for this loop-audit pass. Do not start local work on P4 items from this branch unless Phil explicitly reopens one after the parallel branch merges.

P0 status: **COMPLETE 2026-05-19** — both P0 items done (A1 `2950b7e1`, A4 `11f69d40`), on `wave5_next` (rebased onto `gpu_kokkos_unified`, pushed).

## A. Dead, Orphaned, Merge-Gap, Naming/Comment Cleanup

| ID | Item | Rating | Notes |
|---|---|---:|---|
| A1 | `galaxy_sf/mechanical_fb_gpu.cc` still compiled, apparently dead after `mechfb_loop` runner port | P0 — **DONE** | **DONE 2026-05-19, commit `2950b7e1`** (rebased onto `gpu_kokkos_unified`; was `0a74ee08` pre-rebase). Confirmed dead (no caller; not a live oracle — no toggle). Retired: file deleted, removed from `GPU_OBJS` + build rule, dead decl + doc block removed from `galsf_gpu_decls.h`, dangling line-refs / live-dual-path comments cleaned up. Mac builds clean `GALSF_FB_MECHANICAL` off and on (`test/isodisk_mechfb` Config). |
| A2 | `sidm/cbe_integrator_gpu.cc` live GPU-only EP loop / possible merge gap | CLOSED | CBE-owned parallel-branch item. Inventory says CBE EP-only work landed elsewhere, but this branch still has old `cbe_drift_kick_evaluate_gpu()` called from `core/kicks.cc`. If still live after merge, CBE branch should decide whether tiny-N CPU dual-dispatch/sync hygiene is needed; no local Wave-5 action. |
| A3 | `solids/grain_drag_gpu.cc` naming | P3 | Behavior is live/canonical dual-dispatch, not legacy GPU-only code. `_gpu.cc` suffix is misleading compared with `cooling.cc`, `turb_driving.cc`, `nuclear.cc`. Rename/fold only if desired. |
| A4 | stale comments in `ags_density_loop` / `ags_force_loop` | P0 — **DONE** | **DONE 2026-05-19, commit `11f69d40`.** Scoped to `ags_density_loop.{h,cc}` only — `ags_force_loop` needed nothing (its oracle text is the live `AgsForceSpec` runner oracle; ghost-writeback comments already cleaned by `1081fcc2`). Retired stale "still in-tree / still present" legacy-file claims, dead gate tokens (`LEGACY_PARITY` / `VERIFY_REVERSE_COMM` / `FORCE_CROSS_RANK_WAKEUP`), present-tense "two-binary parity is the validation route" wording, the misleading "oracle catches drift" merge_accum note, and the `set_oracle_brute_pass` terminate() string advising an impossible legacy build. Comment/diagnostic-string only; oracle hard-stub unchanged; provenance comments preserved. Mac compile clean. |
| A5 | `hydro/density_gpu.cc` still holding `gradient_evaluate_gpu` / `hydro_evaluate_gpu` | CLOSED | Density/gradients/hydro corridor, owned by active parallel branch. Listed for merge awareness only; no local action from this audit. |
| A6 | `*_gpu_decls.h` consolidation headers | P3 | Claude classified as active/fine. Keep until explicitly accepted; some decls may still expose dead paths such as A1. |

## B. Strong Missed Pure-Local / Dual-Dispatch Candidates

| ID | Item | Rating | Notes |
|---|---|---:|---|
| B1 | `galaxy_sf/sfr_eff.cc::star_formation_parent_routine()` | P1 | CPU-only active gas loop: SF probability, RNG, gas-to-star conversion/spawn bookkeeping, delay timers. Real hotspot candidate, but topology/type mutation may require splitting pure-local physics from host-special mutation. |
| B2 | `solids/ism_dust_chemistry.cc::update_dust_processes()` | P1 | Major missed CPU-only dust chemistry loop. Called via `finish_cooling_host_deferred_dust_updates()` after cooling, including GPU cooling path, so dust chemistry remains host-side per gas particle. Behind `GALSF_ISMDUSTCHEM_*`. Strong dual-dispatch candidate. |
| B3 | `radiation/rt_utilities.cc::rt_update_driftkick()` | P1 | Heavy per-cell local RT drift/kick kernel, nested over RT bins/intensity bins with stiff update logic. Called from predict/kicks and transport subcycle. Strong candidate. |
| B4 | `core/transport_subcycle.cc::transport_subcycle_kick()` and surrounding active RT loops | P1 | Wrapper around B3 plus opacity/Eddington updates; runs once per RT subcycle. Port/review together with B3. |
| B5 | `eos/eos.cc` coefficient loops: nonideal MHD, conduction/viscosity, turbulent diffusion coefficients | P2 | Host per-active coefficient assignment loops behind `MHD_NON_IDEAL`, `CONDUCTION`, `VISCOSITY`, `TURB_DIFFUSION`. Pure-local candidates, medium priority unless profiling says hot. |
| B6 | `galaxy_sf/stellar_evolution.cc::update_stellarnumber_and_timedistribofstarformation()` | P2 | Active star loop for distributed IMF/star-number timing. Likely local; assess sparsity and RNG/state side effects. |
| B7 | `nuclear/nuclear.cc` dual-dispatch hygiene | P2 | Already CPU/GPU with `GPU_MIN_PARTICLES_FOR_OFFLOAD`, but CPU tiny-N path compact-gathers and `GIZMO_GPU_ENSURE_ALL_FRESH()` is before branch. Review carefully because nuclear fixup/network staging may intentionally use compact arrays. |
| B8 | `radiation/rt_chem.cc` dual-dispatch hygiene | P2 | Already CPU/GPU, but CPU tiny-N path compact-gathers and GPU sync is before branch. Good candidate for grain_drag/turb_driving-style direct-host CPU path if SSOT kernel supports it. |
| B9 | `cooling/cooling.cc` canonical dual-dispatch hygiene | P3 | Current canonical template still compacts on CPU path and syncs before branch. Lower priority; only touch if deliberately updating the canonical template. |

## C. Core Integrator / Every-Step Pure-Local Loops

| ID | Item | Rating | Notes |
|---|---|---:|---|
| C1 | `core/predict.cc::gizmo_full_drift_to()` | CLOSED | **Reviewed 2026-05-19 — no work.** Full-N OMP drift, but idempotent-cache-gated (`g_last_full_drift_Ti`) and invoked only at domain-decomp boundaries / output / restart — NOT per timestep. The per-step drift is the lazy active-only `move_particles()` (C2). No tiny-N globals violation; the whole-pool h-dirty mark is a self-documented optional narrowing, not a defect. Dual-dispatch payoff weak (rare invocation). |
| C2 | `core/predict.cc::move_particles()` | P2 | Active-list drift is currently serial over `ActiveParticleList`; lazy-drift/h-dirty semantics make this delicate. Candidate for local parallelization or explicit waiver. |
| C3 | `core/kicks.cc` first/second half kicks | P2 | **Reviewed 2026-05-19 — characterized, kept as a profiling-gated dual-dispatch candidate (no longer P3 cleanup).** Host-only core kick loop (`do_first/second_halfstep_kick` → `do_the_kick`); no device path. The MFV `NumPart` scan is intentional — inactive gas can receive fluxes from active neighbors, so the gas-side conserved-mass update (`do_the_kick` kicks.cc:250-280, run independently of `ActiveParticleList`) is needed for manifest mass conservation; trimming to active-only is a numerics/conservation change, NOT cleanup (`kicks.cc:46` itself flags it would need revised MFV methods). A marginal host trim (split MFV into all-local-gas + active-non-gas instead of full `NumPart`) helps only mixed gas+collisionless runs — not worth doing absent profiling. A future Kokkos dual-dispatch port of `do_the_kick()` is conceptually possible (pure per-particle) but has a very broad option surface (gravity/PM/cooling-split/MFV mass-conservation/MHD/RT/CR/elastic/EOS/fuzzy-DM/Hermite) plus arena/mirror sync costs — needs its own design + validation, not casual cleanup. Profiling-gated; do not start the port casually. |
| C4 | `core/kicks.cc::apply_long_range_kick()` | CLOSED | **Reviewed 2026-05-19 — no work.** Full-N OMP PM kick, but called only on PM (long-range) steps (`All.PM_Ti_begstep/endstep == All.Ti_Current`) — top-bin steps where every particle is active, so full-N is the correct work set; never called on tiny-N steps. No violation. Trivial per-particle work; standalone dual-dispatch payoff minimal — rides along if the PM path (I1/I2) GPU-ports. |
| C5 | `core/timestep.cc` timestep assignment loops | P2 | Active-list local loops every step. Cheaper per particle than cooling, but broad and systematic. |
| C6 | `core/timestep.cc` DEDNER fastest-wave reduction | CLOSED | **Reviewed 2026-05-19 — no work.** Host OpenMP `NumPart` gas reduction + MPI max, behind `#ifdef DIVBCLEANING_DEDNER` and gated on `HighestActiveTimeBin == HighestOccupiedTimeBin` (or one-time `FastestWaveSpeed==0` init) — runs only on top-bin steps (all-active); skipped on tiny-N steps. Correctly gated, no violation. Do not port standalone unless broader timestep/hydro work reopens it. |

## D. Neighbor/Search-Like Custom Paths Outside Runner

| ID | Item | Rating | Notes |
|---|---|---:|---|
| D1 | `mesh/merge_split.cc` direct `gpu_ngb_list_build` + host decision/execution | P2 | Real bespoke NGL path. Merge/split mutates topology and has local-only semantics, so may stay custom; still needs explicit classification. |
| D2 | `galaxy_sf/radfb_local.cc::HII_heating_singledomain()` | CLOSED | Special custom loop: serial-greedy singledomain semantics, no ghost writeback. Already has tiny-source local-walker vs large-source GPU-NGL dispatch. No runner work. |
| D3 | `turb/turb_powerspectra.cc` nearest-gas slab fill | CLOSED | Scheduled turbulence-spectrum diagnostic, not timestep tiny-N physics. Direct arbitrary-source GPU NGL is for FFT-grid slab cells; runner/dual-dispatch not applicable. |
| D4 | `structure/twopoint.cc` direct NGL plus old `twopoint_ngb_treefind_variable` remnants | CLOSED | Diagnostic/interim restart-only path. Dead legacy `twopoint_ngb_treefind_variable`/`twopoint_count_local` remnants removed; future production-scale improvement would be a gravity-tree pair-count walk, not runner/dual-dispatch work. |
| D5 | `structure/group_search.cc::group_search_build_cross_type_nl()` | CLOSED | Shared FOF/Subfind catalogue helper wrapping cross-type neighbor lists. Not a timestep loop or standalone port target. |
| D6 | `structure/fof.cc` FOF neighbor-list + CPU grouping/property aggregation | CLOSED | Catalogue/on-the-fly halo finder. Scheduled by snapshot/on-the-fly FoF cadence; CPU grouping/properties are special infrastructure. |
| D7 | `structure/subfind/*` modern and local treefind paths | CLOSED | Snapshot/catalogue infrastructure. Key density/link/nearest-two paths already route through modern group-search helpers; not runner/dual-dispatch work. |
| D8 | `structure/lineofsight.cc` line-of-sight loops | CLOSED | Retired ancient unreachable `OUTPUT_LINEOFSIGHT*` diagnostic. The GADGET-era path had no scheduler call in GIZMO; future work should use/port LIGHTCONE-style machinery instead of carrying this dead code. |

## E. Modified Gradient / Hydro/MHD Known Or Other-Branch Items Still On List

| ID | Item | Rating | Notes |
|---|---|---:|---|
| E1 | **MG / Modified Gradient solver audit**: `hydro/mg_gradient_correction.cc` | P2 | Single holistic audit under `MHD_MODIFIED_GRADIENT`; do not split into separate Wave items unless Phil chooses staged implementation. Includes: modern CSR matrix build vs dead legacy `mg_build_matrix()` cleanup/confirmation; CPU/OpenMP `mg_cg_solve()` sparse matvec/SSOR over `N_gas` up to `MG_CG_MAX_ITER`; external CPU `mg_solve_hypre/pardiso()` alternatives; strategy decision for GPU matvec/solver vs explicit CPU waiver; validation/numerical-equivalence plan. Earlier E1/E2/E3/E7 are merged here. |
| E4 | `hydro/hydro_toplevel.cc` DEDNER cell-centered cleanup | CLOSED | Density/gradients/hydro corridor, owned by active parallel branch. Small CPU active gas post-loop after GPU hydro kernel; merge-awareness only here. |
| E5 | conduction/viscosity/CR diffusion/turbulent metal diffusion pair callbacks inside hydro kernel | CLOSED | Density/gradients/hydro corridor, owned by active parallel branch. Claimed already GPU via `hydro_accumulate_neighbor`; keep for merge-review confirmation only. |
| E6 | ANEOS / Helmholtz EOS Newton solvers | CLOSED | Reviewed as intentional/nonlocal to this audit. Device-callable and covered inside GPU cooling/EOS paths; not a separate loop-port target unless future profiling, stack, or numerics work reopens it. |

## F. Sink Local / Topology Loops

| ID | Item | Rating | Notes |
|---|---|---:|---|
| F1 | `sinks/sink_util.cc::sink_start()` | CLOSED | Host setup/bookkeeping: counts active sinks, assigns compact temp indices, allocates/fills `SinkTempInfo`. Not a runner/dual-dispatch target. |
| F2 | `sinks/sink_util.cc::sink_properties_loop()` | P2 | Local per-sink physics: mdot, drag, long-range radiation-prep. Sparse but real pure-local-ish candidate. |
| F3 | `sinks/sink.cc::sink_final_operations()` | CLOSED | Host finalization after runner sink loops: mass/momentum updates, timebin summaries, file output, optional promotion/removal. Not a clean GPU/local-kernel target. |
| F4 | `sinks/sink_swallow_and_kick.cc::spawn_sink_wind_feedback()` | CLOSED | Particle-creation/topology path followed by `rearrange_particle_sequence()`. Keep host-special; no runner/dual-dispatch work. |
| F5 | `sinks/sink_swallow_and_kick.cc` spawned-particle initialization loop | CLOSED | Serial/order-sensitive initialization of appended particles, active/timebin lists, IDs, conservation updates, and dirty-radius markers. Keep host-special. |

## G. Cooling / Local Heating / Small Physics Loops

| ID | Item | Rating | Notes |
|---|---|---:|---|
| G1 | `cooling/selfshield_local_incident_uv_flux()` | CLOSED | Small active-gas local UV attenuation/self-shielding pass. Now index-based OpenMP; no GPU/runner work warranted. |
| G2 | `cooling/disk_betacool.h::disk_betacool_parent_routine()` | CLOSED | Already OMP active-list local loop; optional test-problem physics, mutually exclusive with `COOLING`. No GPU/runner work warranted. |
| G3 | `cooling/planet_heating.h::planet_heating_parent_routine()` | CLOSED | Already OMP active-list local loop; small local heating update. No GPU/runner work warranted. |
| G4 | `cooling/chimes/*` network/table loops | CLOSED | Intentional CPU CHIMES dependency. Broad library-style chemistry/table machinery; current cooling path routes CHIMES away from GPU. Not a runner/dual-dispatch item unless Phil opens a dedicated CHIMES port campaign. |

## H. Cosmic Ray / EOS Local Functions

| ID | Item | Rating | Notes |
|---|---|---:|---|
| H1 | `eos/cosmic_ray_fluid/cosmic_ray_utilities.cc::CalculateAndAssign_CosmicRay_DiffusionAndStreamingCoefficients()` | CLOSED | Gradients/hydro corridor item: called from `hydro/gradients.cc`, not a standalone loop-port target in this audit. Defer to the parallel gradients/hydro branch. |
| H2 | `eos/cosmic_ray_fluid/cosmic_ray_functions.h::CR_cooling_and_losses*()` | CLOSED | Runs inside `cooling.cc::do_the_cooling_for_particle()`, so it already rides the cooling CPU/GPU dual-dispatch path. Any remaining performance/stack concern belongs under B9 cooling hygiene, not a CR-specific port. |
| H3 | `CR_initialize_multibin_quantities()` | CLOSED | Startup-only CR spectral-bin/LUT initialization from `begrun.cc`; host-only is intentional. |

## I. Gravity / Tree / PM Infrastructure

| ID | Item | Rating | Notes |
|---|---|---:|---|
| I1 | `gravity/pm_periodic.cc` long-range PM gravity | P1/P3 | Huge CPU-only FFT/mesh block: particle-grid assignment, Green's function, finite difference, scatter. Real cosmological hotspot, but hard MPI-FFT port. |
| I2 | `gravity/pm_nonperiodic.cc` long-range PM gravity | P1/P3 | Same class as I1 for nonperiodic PM. |
| I3 | `gravity/gpu_gravtree.cc` custom Kokkos gravity walk | P3 | Already GPU, custom infrastructure, not runner. Classify special. |
| I4 | `gravity/gpu_moment_refresh.cc` custom Kokkos tree moment loops | P3 | Already GPU infrastructure. |
| I5 | `gravity/forcetree.cc` host tree-node loops | P3 | Tree build/maintenance CPU loops. |
| I6 | `gravity/forcetree_update.cc` host active-list updates | P3 | Gravity maintenance/update loops. |
| I7 | `gravity/gpu_force_update.cc` custom Kokkos force kick | P3 | Already GPU-ish force update path, custom. |
| I8 | `gravity/gpu_force_drift.cc` drift/table loops | P3 | Custom GPU/OMP gravity support. |
| I9 | `gravity/analytic_gravity.h` many active-list host loops | P3 | Analytic/test gravity loops; probably audit-out but keep. |
| I10 | `gravity/binary.cc` Hermite/Kepler binary integration | P3 | Optional CPU ODE solver path behind Hermite/binary configs. |
| I11 | short-range tree gravity | P3/done-candidate | Claimed GPU-ported primary path via `gpu_gravtree.cc`; CPU path fallback remains. Keep until accepted. |

## J. Domain / Ordering / System / Diagnostics

| ID | Item | Rating | Notes |
|---|---|---:|---|
| J1 | `domain/domain.cc` OMP domain decomposition loops | SPECIAL-INFRA | Domain decomposition / particle migration / load-balance infrastructure, not a physics kernel or loop-port target. Closed for this audit. |
| J2 | `system/peano.cc` non-gas ordering/key loops | SPECIAL-INFRA | Ordering/key-generation and sort/reorder infrastructure. Already OMP-task sorted; no dual-dispatch/runner action here. Closed for this audit. |
| J3 | `mesh/ghost_exchange.cc` request-driven/tile-overlap loops | SPECIAL-INFRA | Core communication/search plumbing for runner and legacy paths; no dual-dispatch concept applies. Closed for this audit unless future ghost-exchange performance work is opened. |
| J4 | `mesh/ghost_writeback.cc` ghost writeback loops | P3 — **CLOSED** | **DONE 2026-05-19, commit `7fd60abe`.** Retired the 3 dead physics-name wrappers — `ghost_writeback_{zero_,}swallowtime`, `ghost_writeback_mechfb`, `ghost_writeback_{zero_,}grainbackrx` (all no callers; live Specs use the generic bundle / `mechfb_writeback_detail` callback) — plus their bespoke structs and the now-unused `ghost_writeback_ops.h` + `mechanical_fb_types.h` includes; each replaced with a short retired-note. Stale-comment cleanup in mechfb_loop.cc / mechanical_fb_types.h / grain_drag_gpu.cc / grain_evolution_functions.h. Kept: hydro wrappers (live), generic bundle scaffold, ghost-write detector. `ags_force_loop` needed nothing (already-correct past-tense provenance). −456/+55 LOC; pure dead-code + comments, no physics change. Mac compile clean under GALSF_FB_MECHANICAL + GRAIN_FLUID/GRAIN_BACKREACTION. **Follow-on (commit `a88c7e36`):** J4 compile verification under `test/gravtree_sinks` exposed a pre-existing `gpu_kokkos_unified` sink-port bug — `SinkSwkSpec`/`SinkFeedSpec` `populate_call_scalars` read `#ifdef`-gated `All.*` members (`Sink_accreted_fraction`, `Sink_outflow_velocity`, `Sink_Rad_MomentumFactor`) without matching guards, breaking any `SINK_PARTICLES` build lacking `SINK_WIND_*`/`SINK_PHOTONMOMENTUM`. Fixed by gating the `CallScalars` fields + reads to match the member-defining `#if`s; no physics change, both gate branches compile-verified. |
| J5 | `mesh/state_hash.cc` active-list hashing | SPECIAL-INFRA | Env-gated diagnostic/checksum loop. Keep as diagnostic; possible future cleanup/removal only if diagnostics are retired. Closed for this audit. |
| J6 | `core/run.cc` assorted active-list bookkeeping loops | SPECIAL-INFRA | Driver/bookkeeping orchestration. Track specific physics loops elsewhere; no broad port target here. Closed for this audit. |
| J7 | `system/code_block_*` legacy export/import templates | CLOSED | Deleted after grep+compile: no live external include sites, only stale comments/internal self-includes. Stale references cleaned in hydro/transport/difffilter comments. |
| J8 | `system/parallel_sort*`, MPI utility loops | SPECIAL-INFRA | MPI/distributed sort utility infrastructure. Not physics loop/dual-dispatch work. Closed for this audit. |

## K. Closed / Accepted Done Items

| ID | Item | Rating | Notes |
|---|---|---:|---|
| K1 | `turb/turb_driving.cc` | CLOSED | Already audited/cleaned dual-dispatch; pure per-particle spectral kernel, not neighbor loop. |
| K2 | `turb/dynamic_diffusion*.cc` | CLOSED | Host drivers for already-ported `DiffFilterSpec` / `DynDiffSpec`; no loop-audit action. |
| K3 | `solids/grain_drag_gpu.cc` behavior | CLOSED | Behavior done: true active-grain tiny-N path. Naming residue remains separately tracked as A3. |
| K4 | all current `*_loop.{h,cc}` runner ports | CLOSED | Umbrella accepted as done. Track only specific future bugs in specific ports. |
| K5 | pure-local dual-dispatch sites: cooling, nuclear, rt_chem, turb_driving, grain_drag | CLOSED | Umbrella closed. Remaining specific hygiene is tracked by B7/B8/B9. |
| K6 | `solids/grain_physics_loop.*` | CLOSED | Runner-ported and validated. |
| K7 | `sidm/dm_fuzzy_loop.*` | CLOSED | Runner-ported and validated; separate DM_FUZZY blowup physics issue is outside this audit. |
| K8 | `radiation/rt_source_injection_loop.*` | CLOSED | Runner-ported source-injection loop. |
| K9 | `sinks/*_loop.*` runner ports | CLOSED | Runner ports done. Related host sink loops remain tracked in F. |
| K10 | `galaxy_sf/thermal_fb_loop.*`, `radfb_rp_loop.*`, `dm_dispersion_loop.*`, `mechfb_loop.*` | CLOSED | Runner-ported feedback/dispersion loops. Residual cleanup tracked separately by A1/A4. |
| K11 | `gravity/ags_density_loop.*`, `ags_force_loop.*` | CLOSED | Runner-ported AGS loops. Residual stale comments tracked separately by A4. |

## Suggested Review Order

1. P0 cleanup: **COMPLETE 2026-05-19** — A1 (`2950b7e1`) + A4 (`11f69d40`). No P0 items remain.
2. Big missed hot paths: B2 dust chemistry, B3/B4 RT drift-kick/subcycle, E1 MG/modified-gradient audit, I1/I2 PM gravity.
3. Pure-local dual-dispatch hygiene: B8 `rt_chem`, B7 `nuclear`, B9 cooling template.
4. Custom search/NGL: D1 merge/split, D2 HII, D3 powerspectra, D4 twopoint, D5-D7 structure/group/subfind/FOF.
5. Core-local loops: C1-C6 predict/kick/timestep.
6. Sparse/special physics: F sinks, G cooling-adjacent, H cosmic rays, I/J infrastructure.
