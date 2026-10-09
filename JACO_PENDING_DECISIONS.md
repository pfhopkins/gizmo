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
- Band layer core built (jaco branch `band_spec` 4ebfed5, `~/code/jaco_bands`, design note docs/band_spec.md).
  Decision (MYG, 2026-10-06), band-layer design:
  - Split the ionizing band at 24.59 eV in `starforge_RT`: one fixed slope cannot match both G_HI and nu_eff, while
    two bands each matched to GIZMO's 4e4 K blackbody reproduce sigma_HI, G_HI and nu_eff within 2.3%.
    `starforge_legacy_RT` keeps its single band through overrides.
  - In-band cross-sections: exact quadrature of the declared sigma(E) against the band's fixed-slope spectrum, at
    codegen (not the paper's edge power law).
  - Thermal emission into fixed-shape bands: per-band Kirchhoff (chi_B := chi_E), so that detailed balance holds per
    band and optically thick gas thermalises exactly.
  - Photon vs energy bookkeeping in ionizing bands (measured after the split: 6.6% / 5.6% per sub-band, mostly
    physical): photons are conserved exactly, and the deferred hardening energy (<h nu>_b - E_abs per absorption) is
    booked as an explicit energy-ledger term. No second moment. Check: an ~8-sub-band HII_region run as the converged
    reference for the 2-band front temperature and radius.
- Legacy GIZMO is being fixed on a separate branch, `rt_microphysics_fixes` (see GIZMO_RT_MICROPHYSICS_ISSUES.md):
  the IR double count, the discarded IR gas share, Rad_Je, the Iliev photon count, the restart H2 rebuild, the sound
  speed's stale gamma, and an uninitialized `dt_hydrostep_i` in the hydro flux limiters. Consequences for jaco:
  (a) once that branch lands, `starforge_legacy_RT` should drop its reproduced IR double count and match the fixed
  legacy kick, and its gmc_cooling_rt/shu_M120 comparisons move to the regenerated benchmarks: DONE 2026-10-08 (jaco
  `legacy_rt_fixed` 9cdb5ee; 1.2): gmc_cooling_rt[jaco_rt] passes against the cold-start benchmark (urad_FIR 0.38 ->
  0.009) and is no longer xfail; shu_M120[jaco_rt] passes against its benchmark, which was not regenerated; (b) the
  sound-speed and `dt_hydrostep_i` fixes change every STARFORGE legacy baseline the jaco variants were compared
  against, so the jaco acceptance must be rerun after merging that branch;
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
Decision (MYG, 2026-10-06): (b), the trace-free part of grad v.
Implemented (branches `jaco_p0`, 2026-10-06): a declared parameter `∇v_tf` (host: `jaco_tracefree_gradient_norm`);
`starforge_legacy` keeps the full norm, its generated code unchanged. Effect is small: in SN_singlestar the trace-free
norm is 0.9-0.98 of the full one (the shell is radial compression plus tangential stretching, which is trace-free, not
homologous), so radial momentum at t = 0.05 goes from 0.787 to 0.803 of legacy (C_2 = 1: 1.016). gmc_cooling: median C_2
2.2 -> 1.6 at t = 1. A planar shock, grad v = diag(-a, 0, 0), keeps sqrt(2/3) = 0.82 of the norm, 2/3 of C_2 - 1.
Decision (MYG, 2026-10-06): also clump cold gas only: C_2 = 1 + (C_2,tf - 1) f(T), f a logistic in log T with
T_mid = 5000 K, w = 0.1 dex (jaco `symbols.py` constants; f = 0.98 at 2000 K, 0.047 at 1e4 K). Implemented (jaco
`jaco_p0` 60831f2): SN_singlestar radial momentum 0.975 (t = 0.015) and 0.991 (t = 0.05) of legacy; shell (1e4-1e5 K)
C_2 p50/p90/p99 1.01/1.04/1.23. gmc_cooling unchanged (cold gas). Solver tiers unchanged or better.

### 1.2 Reproduced legacy energy creation in `starforge_legacy_RT`
To match the RT benchmarks the model reproduces three legacy behaviours that do not conserve energy, each
documented in the model docstring: dust heated by the full gas IR absorption, gas IR heating deposited at c~/c rather
than c, and photoelectric heating and LW photodissociation not depleting their bands. (The IR double count, a fourth,
was dropped when F1 landed; see below.) Whether `starforge_legacy_RT` ships with these on (benchmark
fidelity) or off (correct physics, benchmarks re-baselined) should follow the legacy A/B in section 3.
The kick's exponent cap was lowered from 50 to 10 inside jaco for solver cost; beyond it the band keeps e^-10.
Decision (MYG, 2026-10-06): `starforge_legacy_RT` exists solely to behave like baseline GIZMO in its most developed
state on starforge_dev. It tracks that branch term for term: it drops each legacy term that a landed fix removes
(starting with the IR double count and the gas IR share once `rt_microphysics_fixes` lands) and keeps reproducing
whatever legacy still does, energy-creating or not. Physical corrections belong in `starforge_RT`, not here.
Done (2026-10-08, P0.4; jaco `legacy_rt_fixed` 9cdb5ee, GIZMO `legacy_rt_fixed`): `starforge_legacy_RT` tracks the
fixed kick (F1 a5d1924c, F2 bf6e4745, merged here at 6d472eda).
- F1: the process that reproduced the direct donation (`radiation.legacy_ir_donation_copy`) is deleted, not kept
  behind `Model.without`. T_rad_new re-derived from the fixed kick (`rt_utilities.cc` 846-857, 924-927 at 6d472eda):
  E_abs_tot_toIR dt is never in e0 (counted at T_rad); it joins the band's own absorbed energy in e_absorbed,
  weighted at T_dust for the dust's share. The model's dust emission already carries the donors' dust absorption at
  T_dust, so T_rad_new now starts from the band's initial photons alone (`ir_radiation_temperature` loses
  `prior_sources`). Not reproduced, before or after: the kick also counts the gas share f_gas e_absorbed, which goes
  to the gas, as photons at max(T_rad, T_gas) (f_gas < 1e-2 outside dense ionized gas).
- F2 verified, no change: `kick_gas_share` = 2 (1 - e^-x/2) / x of the gas absorption rate is the full share of both
  half-kicks, 1:1 with the band's loss, which is what bf6e4745 now delivers (before it, unsplit cells got the same
  total with the opening share lost and the closing one doubled). Its standing approximations are unchanged: x and
  the share use the dust absorption opacity alone, where GIZMO divides by the total (0,0) opacity.
- jaco tests: `test_ir_band_gets_the_donations_once` (was `_twice`) and a fixed-kick transcription in
  `test_ir_radiation_temperature_output` (exact to 1e-12; the old weighting gave 40.1 K vs 45.1 K in a thin case).
  Fast suite 824 passed (test_CIE: missing data file, as before), slow test_legacy_rt 2 passed. Golden hashes: only
  starforge_legacy_RT's `microphysics_func_jac.cc` and `microphysics_outputs.cc` change.
- GIZMO (4 ranks x 1 thread, ccalin030): gmc_cooling_rt[jaco_rt] against the cold-start benchmark, max rel per
  statistic before -> after: urad_FIR 0.381 -> 0.009, Trad 0.013 -> 0.001, T 0.021 -> 0.015, Tdust 0.012 -> 0.012,
  urad_FUV 0.055 -> 0.048, xe 0.009 -> 0.011. It passes, so its xfail is removed: this branch's jaco_rt needs jaco at
  or after 9cdb5ee (with the pip-installed `gizmo_integration` jaco it fails on urad_FIR).
  shu_M120[jaco_rt] passes (run outside the test's 600 s limit, which 4 ranks exceed, then asserted on its snapshots
  with GIZMO_TEST_SKIP_BUILD_RUN=1) against its old benchmark, still made with the double count: T 0.055, Tdust
  0.057, Trad 0.071 (tolerance 0.1), sink mass 1.465 vs 1.461. Against the fixed legacy run at 6d472eda: T and Tdust
  within 0.038 (jaco 2-4% colder inside 2.5e-3 pc), Trad within 0.084 (at 4-6e-3 pc), urad_FIR (not asserted) 3-14%
  lower inside 2.5e-3 pc and within 1% outside 0.01 pc.
- Merging `band_spec` later: expect conflicts where it edits `legacy_ir_donation_copy` (deleted here),
  `ir_radiation_temperature`'s `prior_sources`/P_IR, and the model's `RADIATION_PROCESSES`/`_outputs`; take
  band_spec's lines without the copy and P_IR.

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
Result (2026-10-06; scripts in the session scratchpad xe_gap/): mostly legitimate, plus one clumping defect.
- The quoted 4-13x comes from the DENSE states in jaco's test_ionization_balance.py (G0=0, C2=1, N_H 1.8e23-6e23,
  50-100x gmc_cooling's columns). It is not from CR attenuation: both models attenuate zeta identically, once.
- Pure chemistry, legitimate. `starforge` moves the CR charge onto Mg+ by charge transfer and neutralises it on
  small grains (WD01), so x_e ~ zeta/n. Legacy calls its charge "Mg+" but recombines it at k_ei = 9.77e-8
  (cooling.cc:2359), a dissociative-recombination rate ~3700x Mg+'s radiative one, and adds a second
  sqrt(zeta/beta n) molecular-ion term; x_e ~ sqrt(zeta/n) by construction. Gap at G0=0, C2=1: 2.1x at 1e3, 3.4x at
  1e4, 13.5x at 1e5.
- Assumption-dependent (not a bug; see the decision below): C2 on the steady-state ion balance (ionization_balance.py:37-42, :71; grain_assisted_recombination.py:260)
  lowers x_e by a further 2.6-6.3x. The ions equilibrate in ~150 yr at 1e4, against a ~1.5e5 yr sub-grid crossing
  time, so each density holds its local equilibrium x ~ zeta/(alpha n), and the mass-weighted mean is
  zeta/(alpha n_mean): no clumping term. A lognormal sub-grid integration reproduces C2=1 within 3-25%.
- At gmc_cooling's own states (1e2-1e3), the sign flips: Mg photoionization by the attenuated G0, which legacy lacks,
  puts `starforge` 6-16x ABOVE legacy, and 5-20x above McKee's 1e-5 n^-1/2. Undepleted Mg and the band-averaged dust
  attenuation inflate it.
- Against McKee (1989), legacy tracks within 2.8x everywhere (it is built that way). `starforge` matches at 1e4 only
  with C2=1, and sits 37x below at 1e5 even with C2=1: WD01's PAH-rich grains make the dense-core x_e a lower bound.
  That is a modelling choice.
Decision (MYG, 2026-10-06): keep C2 on every two-body rate, the ion balance included (a modelling convention:
sub-grid abundances are taken as uniform within a cell). Gas-phase Mg: a simple constant for now, F_GAS_MG = 1
(undepleted, GIZMO's assumption), in jaco `ionization_balance.py`, with a TODO pointing to Jenkins 2009
(2009ApJ...700.1299J: Eq. 10, Table 4's Mg row, whose F*-<n(H)> fit is in sight-line mean densities). The generated
code is byte-identical (golden hashes pass).

### 1.5 Subcycled HII region and the lowest-density FUV bin
Photon-conserving ownership of the ionizing band puts the subcycled HII region 1.8% below legacy-subcycled at
t = 1 (legacy's own ionized mass moves 30% between non-subcycled and subcycled). Cold-start gmc_cooling_rt FUV is
5.1% off in the lowest-density bin only (halves with dt/4; legacy moves more). Both are accepted as documented
deviations; say if either should be chased.
Decision (MYG, 2026-10-06): do not chase anything involving subcycling for now. The FUV lowest-density bin is not a
subcycling item; it stays an accepted, documented deviation.

### 1.6 No optically-thick cooling cap in the non-RT models (found 2026-10-08)
Legacy non-RT GIZMO caps net radiative cooling at a slab blackbody rate:
|Q| <= sigma T^4 A_eff / ((1+tau) n_H), with tau = kappa(T) Sigma and Sigma from TreeCol (`cooling.cc:1438-1483`).
Under JACO, `call_jaco` returns before that code (`cooling.cc:280-281`), and no jaco model, nor the solver, has a cap.
So `starforge` and `starforge_legacy` cool optically thin at any column. They never form a first core: thin LTE H2
cooling at ~1000 K and 1e-8 g/cm^3 exceeds compressional heating by ~15x. The existing gates are below the densities
where the cap binds, so none of them sees this. Dense non-RT jaco runs (`shu_jets[jaco]`, sink neighbourhoods) are
affected.

RT models are not affected: they trap IR in the M1 band.

The second-collapse work adds a cap to `starforge`: an option, on by default, ported from GIZMO's formula, with
kappa packed by the host. `starforge_legacy` still lacks it; that is a parity defect against the model's purpose.

Related, accuracy only: GA08's LTE H2 + H dissociation rate with the Forrey three-body formation rate puts the
equilibrium n_H^2/n_H2 at 7.6x Saha at 2000 K (30x at 1500 K, ~1 at 4000 K), in legacy and jaco alike. H2 dissociates
~150 K cooler than in Masunaga & Inutsuka (2000). Remedy: derive the three-body rate from dissociation by detailed
balance.

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
again after `starforge_RT`).
Implemented (branch `jaco_p0`, 2026-10-06): two consecutive steps held below tol by a budget within 4 eps of zero return
tier-1 status -7 (budget stall). Driver: answers bitwise identical, no tier changes; 2.8k of 57.9k solves stall
per model, total evaluations -0.9% (starforge) / -0.7% (legacy), ~5-7 per stalled case. gmc_cooling: no stalls in
either model (its tier-1 failures are the iteration budget on trace He+/H2 near 1e4 K).

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
gmc_cooling_rt benchmark, which encoded the first-cooling-call defect (regenerated from a cold start; jaco_rt left
xfail on 2026-10-08, 1.2).

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
