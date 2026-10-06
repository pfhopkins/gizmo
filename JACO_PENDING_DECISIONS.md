# jaco integration: pending decisions

State as of 2026-10-06 (section 0 summarizes what changed since 2026-10-04). GIZMO branch `gizmo_jaco_dev` (this tree) and jaco branch `gizmo_integration`
(`~/code/jaco_gizmo`), pushed 2026-10-06. Evidence lines cite the agent runs or code reads that produced them;
"recommendation" is mine, not a decision; "Decision (MYG, date)" lines are the user's.

## 0. Where things stand (2026-10-06)
- Tips (pushed 2026-10-06): GIZMO `gizmo_jaco_dev` 1db4547e for code (later commits on it are docs only), jaco
  `gizmo_integration` 49f95d4. The pip-installed jaco in `~/python_work` points at `~/code/jaco_gizmo`.
- Models: `starforge`, `starforge_legacy`, `starforge_legacy_RT` (12 unknowns: u, T, H+, He+, He++, H2, photon_EUV in
  photons/H, photon_FUV/NUV/ONIR/IR in eV/H, T_dust on its steady-state dust balance), `starforge_legacy_RT_EUV`.
  Every matter-radiation term is a process in `src/jaco/models/starforge/radiation.py`; GIZMO only transports bands.
  Acceptance from the merged tips: 23 passed, 5 documented xfails (RT subsuite, Iliev gate, non-RT variants).
- There is NO non-legacy RT model yet. `radiation.py` is used only by the legacy RT models. A physical `starforge_RT`
  needs, in place of legacy's "donation": recombination emitting photons into the bands (instead of photoionization
  donating the photon energy to ONIR), the photoelectric effect as a yield split of FUV dust absorption, gas IR
  absorption at c into the heat row, line/continuum cooling as photon products of the cooling processes, and LW
  dissociation consuming LW photons if that band is ever added. Not started.
- Agenda to full-physics STARFORGE RT parity (approved 2026-10-06): `~/.claude/plans/have-a-look-at-delightful-umbrella.md`,
  phases P0-P7. Decision (MYG, 2026-10-06), design of `starforge_RT`:
  - an abstract band set in jaco: processes declare sigma(E)/j(E), codegen projects them onto a band spec, and GIZMO
    takes its band count and indices from the generated header; `starforge_RT` is one spec, and refining a band
    means editing only the spec;
  - `starforge_legacy_RT` is ported onto the spec, with byte-identical generated code as the pass bar;
  - in-band shape: piecewise power law with a fixed slope (nu E_nu = const) in every band except the T_rad-tracked
    IR band; the EUV slope is fitted to GIZMO's 4e4 K blackbody band averages (`rt_chem.cc` `rt_get_sigma`);
  - recombination: case B, with a switch for case A (legacy mixes the two; GIZMO_RT_MICROPHYSICS_ISSUES.md 1.3);
  - every matter-radiation exchange, energy and momentum alike (flux absorption, radiation force, work terms),
    lives inside the jaco solve, with local photon+matter momentum checks; GIZMO keeps transport only;
  - stellar sources enter the solve as a per-band source term, fed by a scatter-side accumulator (a cell-side gather
    was rejected on cost).
- Legacy GIZMO is being fixed on a separate branch, `rt_microphysics_fixes` (see GIZMO_RT_MICROPHYSICS_ISSUES.md):
  the IR double count, the discarded IR gas share, Rad_Je, the Iliev photon count, the restart H2 rebuild, the sound
  speed's stale gamma, and an uninitialized `dt_hydrostep_i` in the hydro flux limiters. Consequences for jaco:
  (a) once that branch lands, `starforge_legacy_RT` should drop its reproduced IR double count
  (`Model.without(...)`) and match the fixed legacy kick, and its gmc_cooling_rt/shu_M120 comparisons move to the
  regenerated benchmarks; (b) the sound-speed and `dt_hydrostep_i` fixes change every STARFORGE legacy baseline the
  jaco variants were compared against, so the jaco acceptance must be rerun after merging that branch;
  (c) `gizmo_jaco_dev` needs a merge of origin/starforge_dev (now abd5e309) and later of the fix branch.
  Decision (MYG, 2026-10-06), the GIZMO baseline for the jaco work:
  - Only the option-2 fixes (F1-F6 + `dt_hydrostep_i`) go into it. Deferred, each to its own later batch: legacy
    case-B recombination rates, the RSOL band/gas convention, and the first-call root-find.
  - Before landing: the suite on the fix-branch tip c8312cc5, plus a gmc_cooling_rt benchmark regenerated from a COLD
    NEUTRAL start (InitGasTemp ~100 K).
  - Order: land on starforge_dev -> merge into `gizmo_jaco_dev` -> `starforge_legacy_RT` tracks F1/F2 -> regenerate
    the golden codegen hashes -> rerun the jaco acceptance. That frozen state is the reference for the band
    abstraction's byte-identical gate.
  - The P1 full-physics mini test is a small turbulent cloud forming protostars plus one massive star placed near
    the end of its life, so jets, winds, RT, MHD and a SN all fire within the test.

## 1. Model physics (jaco side)

### 1.1 Clumping factor in coherent flows (`starforge` only)
Both models use legacy's estimator, C_2 = 1 + (0.5 |grad v| dx / c_s)^2, uncapped, on the full velocity-gradient norm.
Legacy applies it only to its H2 chemistry and zeroes f_H2 above 3e5 K, so it never feels the blow-up. `starforge`
applies C_2 to every two-body rate (by design: "all clumping factors in starforge") and gets C_2 ~ 325 in SN ejecta
(27% deficit in SN_singlestar radial momentum; C_2 = 1 reproduces legacy exactly) and 1.8-4 in gmc_cooling.
Options: (a) cap C_2; (b) use the trace-free part of grad v only, so bulk expansion/shock jumps do not count as
sub-grid turbulence; (c) clump only cold/neutral gas; (d) chemistry-only in `starforge` too.
Recommendation: (b). `starforge_legacy` is unaffected by construction.
Decision (MYG, 2026-10-06): (b), the trace-free part of grad v. Not yet implemented.

### 1.2 Reproduced legacy energy creation in `starforge_legacy_RT`
To match the RT benchmarks the model reproduces four legacy behaviours that do not conserve energy, each
documented in the model docstring: the IR double count (a separate process, removable with `Model.without`),
dust heated by the full gas IR absorption, gas IR heating deposited at c~/c rather than c, and photoelectric heating
and LW photodissociation not depleting their bands. Whether `starforge_legacy_RT` ships with these on (benchmark
fidelity) or off (correct physics, benchmarks re-baselined) should follow the legacy A/B in section 3.
The kick's exponent cap was lowered from 50 to 10 inside jaco for solver cost; beyond it the band keeps e^-10.
Decision (MYG, 2026-10-06): `starforge_legacy_RT` exists solely to behave like baseline GIZMO in its most developed
state on starforge_dev. It tracks that branch term for term: it drops each legacy term that a landed fix removes
(starting with the IR double count and the gas IR share once `rt_microphysics_fixes` lands) and keeps reproducing
whatever legacy still does, energy-creating or not. Physical corrections belong in `starforge_RT`, not here.

### 1.3 H2 source for the star-formation criterion under JACO
Commit 779305d0 makes `Get_Gas_Molecular_Mass_Fraction` return the network's H2 under JACO (as GIZMO does for
CHIMES/GRACKLE). In isodisk_thermalfb at 0.1 Zsun no H2 forms within 30 Myr, so the molecular SF criterion sees
f_H2 ~ 1e-4 instead of the KG2010 fit's 0.68: ~100 stars / 4 Msun/yr vs legacy's ~400 / 12. The fit reproduces legacy.
Recommendation: keep the network value; the commit is isolated and revertable.
Decision (MYG, 2026-10-06): galaxy-scale ISM is not a jaco target; jaco is for the high-resolution ISM setups only.
The `isodisk_thermalfb[jaco]` variant is removed from the test. 779305d0 stays: the network H2 is the consistent
answer for every other `Get_Gas_Molecular_Mass_Fraction` consumer.

### 1.4 `starforge` dense-gas electrons
With legacy's strong CR attenuation and WD01 grain recombination of Mg+, `starforge` x_e is 4-13x below legacy at
1e3-1e4 cm^-3. Matters if Ne feeds non-ideal MHD. Mg is undepleted. No action taken.
Decision (MYG, 2026-10-06): find out why. If the gap comes only from solving all the chemistry in one
self-consistent system (rather than legacy's prescribed budget), accept it; if a process is missing or wrong, fix it.
Investigation open.

### 1.5 Subcycled HII region and the lowest-density FUV bin
Photon-conserving ownership of the ionizing band puts the subcycled HII region 1.8% below legacy-subcycled at
t = 1 (legacy's own ionized mass moves 30% between non-subcycled and subcycled). Cold-start gmc_cooling_rt FUV is
5.1% off in the lowest-density bin only (halves with dt/4; legacy moves more). Both are accepted as documented
deviations; say if either should be chased.
Decision (MYG, 2026-10-06): do not chase anything involving subcycling for now. The FUV lowest-density bin is not a
subcycling item; it stays an accepted, documented deviation.

## 2. Solver

### 2.1 Tier-1 budget stall
When a neutral budget reaches round-off, the fraction-to-boundary bound shrinks the step 100x per iteration until the
8-step limit; tier 2 then finishes at ~25 extra evaluations. ~1-1.5% of driver cases, ~0.2% of total evaluations,
never a wrong answer. The old C code escaped 245/406 sweep cases only through a -ffast-math rounding accident that
the budget table removed. Named driver cases exist for the stall and for the x_e -> 0 spurious-root guard.
Options: (1) ignore the bound below 4 eps (recovers +410/+672 tier-1 answers, fails the guard case: 2640 K vs 8785 K);
(2) detect the stall and hand to tier 2 (identical answers, saves the wasted iterations); (3) pin the eliminated
neutral in the active set at its floor and move along the constraint (principled, untested).
Recommendation: (2) now; (3) when floors for eliminated species are generated by R6.
Decision (MYG, 2026-10-06): (2). Re-measure the stall share once the rest settles (the full-physics mini test, and
again after `starforge_RT`). Not yet implemented.

### 2.2 Driver loose ends
Four `starforge_legacy_RT` driver failures at dt = 1e20 with bands held (n = 1e8, protostellar field): T_dust cycles
across the 160 K composition switch in the fixed-T chemistry; unphysical held-band states GIZMO never produces.
The driver's equilibrium check assumes a unique fixed-T root; the dust balance has two stable roots across a switch,
so the reference scan must continue T_dust from the answer's branch. Eight pre-existing `starforge` equilibrium
mismatches at n = 1e3, T0 = 1e7, dt = 1e13 are the check's linearized gate at the H-recombination cliff, not the solver.

## 3. Legacy GIZMO defects (moved)
Findings about GIZMO's own RT and microphysics (the IR kick double count and its A/B, the RSOL band/gas drains, the
cooling-radiation return limiter, the first-cooling-call branch jump, the idealized-test and build-system defects,
the two non-RT defects) now live in GIZMO_RT_MICROPHYSICS_ISSUES.md. The parts that are decisions for the jaco
branches: whether `starforge_legacy_RT` keeps reproducing legacy's energy-creating terms (1.2 above), and the
gmc_cooling_rt benchmark, which encodes the first-cooling-call defect and must be regenerated before its jaco
variants can leave xfail.

## 4. Tests and tolerances
- New r_IF assertions (HII_region_simple, HII_region, HII_region_subcycle) and the Iliev gate use 10%; measured
  agreement is 1-3%. Tightening to 5% is defensible.
  Decision (MYG, 2026-10-06): leave the tolerances at 10%.
- shu_M120's jaco variant needs 8 ranks x 2 threads to fit the test's 600 s timeout.
- gmc_cooling: `jaco_legacy` asserts against the benchmark; `jaco` is run-to-completion only.

## 5. Housekeeping
- Pushed 2026-10-06 as a backup (user's decision): `gizmo_jaco_dev` and jaco `gizmo_integration`. A merge of
  origin/starforge_dev is due once the fix branch lands.
- Most subagent commits carry `Co-Authored-By: Claude Opus 5.5 (1M context)`.
  Decision (MYG, 2026-10-06): leave the co-author lines as they are (accurate); do not amend.
- Side worktrees fully merged and removable: `gizmo_jaco_{eos,solver,tests}`, `jaco_{sync,design,design2}`.
  Keep branch `jaco_solver_cpp` (evaluated and rejected prototype, unmerged).
- The GIZMO solver `#error`s without a jaco that carries the R6 metadata: keep the pip-installed jaco in sync.
- Queued: the equal-precision jaco-vs-legacy performance comparison (recipe in memory: `starforge_legacy`, tolerances
  matched both ways, gmc_cooling/SN/wind, cooling time from cpu.txt on an idle node).
