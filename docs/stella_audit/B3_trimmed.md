# Mechanical feedback: cold ejecta, and ejecta mass of every mode in the weighting pass

Series: fix on the base, general, no Stella needed. Branch `up/B3-trimmed` (4 commits: test, weighting pass, cold ejecta and dilution, docs). The audit branch holds the same source change; its two test-text lines still mention the old deferred-heat branch and are replaced by the branch's form when the line is re-cut.
Source +22 -22 (`mechfb_loop.cc` 20/20, `mechanical_fb.cc` 1/1, `mechanical_fb_functions.h` 1/1).
Order: independent; any time before the single-star feedback PRs (slice2/10 to 12), which rely on the same ejecta-mass logic.
Base: <BASE>. Tip: <TIP>. Numbers below were measured on 070c5b78f (the base of the audit line) unless marked older, and are to be re-measured on <TIP>.
Tests: `test/mechfb_energy` (new, broad: energy against 1e51 erg per SN and against the wind luminosity, star:cell mass 1:1 to 1000:1, moving star) and `test/wind_singlestar` (existing).
Full suite: <SUITE RESULT>.

## Please look first: two changes to code already in the base
1. **`test/wind_singlestar` energy bound.** The adiabatic energy check of the local injection mode (wind mode 2) had a 35 % bound and an expected-failure marker. This PR sets the bound to 10 %, the same as the spawn mode, and keeps the expected-failure marker only for mode 2 with cooling (the shell radius there, about 28 % inside Weaver, is a separate question). On the base, adiabatic mode 2 passes unexpectedly: energy / (L_w t) = 1.268, worst deviation 26.8 %. With this PR: 1.013, worst 1.9 %.
2. **The `KNOWN OPEN DEFECT` comments.** Two long comments in `test_wind_singlestar.py` say that mode 2 delivers about 25 to 30 % too much energy and that the cause is unknown. This PR removes them and replaces the marker comment with three lines: adiabatic mode 2 now conserves energy (`test/mechfb_energy`), only the cooling case stays marked. The README line changes from "to 10 % for spawning" to "to 10 % in both wind modes".

The adiabatic spawn variant of `test/wind_singlestar` fails on the base (worst 12.5 % against 10 %) and still fails with this PR. Mode 1 does not use this coupling; this PR does not touch it.

## What and why
Mechanical feedback deposits each source's ejecta with neighbour weights from a weighting pass. Two energy errors remain on the base after #106 and #107:
- the ejecta join a cell at the cell's own specific energy (`TE_injected = Mf uf - M0 u0`), so mass alone adds heat;
- a cell that gets mass but no heat is not diluted (the apply step runs only when `dTE > 0`).
At high star:cell mass ratios and for mass return this gives up to 3 % too much energy per SN and up to 48 % too much for winds. A third, smaller point: the weighting pass takes the SN channel for every model but FIRE, and falls back to the whole star mass for a source that returns mass without a SN (STARFORGE single-star winds, mode 2).

## Evidence before
`test/mechfb_energy` on the base, 3 of 7 cases fail (20^3 lattice, n_H = 0.1, adiabatic, no gravity; per SN in 1e51 erg, winds as gained / offered):

| case | base | bar |
|---|---|---|
| SN 1:1 | 1.0011 | 3 % |
| SN 100:1 | 1.0056 | 3 % |
| SN 1000:1 | **1.0325** | 3 % |
| SN 1:1, star at 300 km/s | 1.0010 | 3 % |
| SN 1:1, star at 1000 km/s | 1.0010 | 3 % |
| winds FIRE-3, 100:1 | **1.2844** | 2 % |
| winds FIRE-3, 100:1, star at 300 km/s | **1.4809** | 2 % |

Bars were written before any run: energy per SN within 3 % of 1e51 erg (a blast drifts about 1.5 % in this lattice), wind energy within 2 %, the same with the star moving through gas at rest (energy in the star's frame).

## Change
No new flag; the change applies to every build with `GALSF_FB_MECHANICAL`.
1. Weighting pass (`mechfb_loop.cc`): the ejecta mass summed over every mode the source is active in, for every model (`mech_fb_ejecta_mass_this_step`). This replaces the channel block of #106. For FIRE it differs from the base only in a step with both a SN and mass return (sum of both; base: SN only).
2. Cold ejecta (`mechanical_fb_functions.h`): `TE_injected = Mf (uf - u0)`; the added mass carries no ambient u. #108 described the base form as intended; it is what remains of the energy error.
3. Dilution (`mechanical_fb.cc`): the apply step runs `if(dTE > 0 || dm > 0)`, so a cell given mass and no heat has its u diluted. Companion of 2.

Not in it: a per-cell energy closure in the star frame (an earlier version), which created energy when gas moves relative to the star. `U_thermal_residual_tocouple` is unchanged.

## Relation to #106, #107 and the single-star model
- #106: the channel block is replaced by the sum over modes; for steps with only SNe or only mass return the result is the same.
- #107: heat is always injected on the base; this PR has no deferred-heat branch.
- Single stars (Stella series): they keep their own ejecta-mass override (`single_star_ejecta_mass_this_step`) after this weighting pass, because this pass calls `particle2in` with channel 0. On the audit line single stars dilute u before the apply step and use dU = dTE / mf there, so the dilution is not done twice.

## CR energy
The base takes `U_thermal_residual_tocouple` before the CR block removes the CR share. With `CR_DYNAMICAL_INJECTION_IN_SNE` and v_eff > 1000 km/s the gas gets the whole E and the CRs another f E on top. This PR leaves that order (read from the code, not measured).

## Cost
None: no new data, no collectives, nothing on quiet steps. Step 1 calls the per-mode ejecta-mass function once per active source in the weighting pass.

## Evidence after
`test/mechfb_energy`, 7 of 7 pass:

| case | base | base + cold ejecta only | this PR | bar |
|---|---|---|---|---|
| SN 1:1 | 1.0011 | | 1.0008 | 3 % |
| SN 100:1 | 1.0056 | | 1.0007 | 3 % |
| SN 1000:1 | **1.0325** | 1.0008 | 1.0008 | 3 % |
| SN 1:1, star at 300 km/s | 1.0010 | | 1.0008 | 3 % |
| SN 1:1, star at 1000 km/s | 1.0010 | | 1.0008 | 3 % |
| winds FIRE-3, 100:1 | **1.2844** | 1.0066 | 1.0066 | 2 % |
| winds FIRE-3, 100:1, star at 300 km/s | **1.4809** | 1.0053 | 1.0053 | 2 % |

The moving-star cases catch the per-cell closure version (1.1677 and 2.939 per SN, winds 1.0834, all fail); the base passes them since it has no closure.

First SN at 1000:1, coupled momentum: 7.82e4 on the base and with this PR (analytic limit 7.73e4). This PR does not change the coupled momentum.

Existing tests, base vs this PR (2 ranks x 2 threads):

| test | base | this PR |
|---|---|---|
| `test/mechfb_energy` | 3 failed, 4 passed | 7 passed |
| `test/wind_singlestar` adiabatic, local injection | unexpected pass: 1.268 of L_w t (worst 26.8 %, bound 35 %) | pass: 1.013 (worst 1.9 %, bound 10 %) |
| `test/wind_singlestar` adiabatic, spawn | fail, worst 12.5 % against 10 % | fail, same |
| `test/wind_singlestar` with cooling | local expected failure, spawn pass | same |
| `test/fire` (2 variants) | pass | pass; gas thermal 895.46 -> 895.37 code units (-0.01 %), kinetic and stellar mass unchanged |
| `test/isodisk_mechfb` | pass | pass; at t = 0.03 gas thermal x 0.459, kinetic x 0.475 (cold ejecta alone: x 0.458, x 0.474) |
| `test/isodisk_mechfb_sinks` | pass | pass; thermal +6.5 % at t = 0.003 |

Also measured on the base, for context: #107 removed the deferred-heat loss. 1:1 lattice at n_H = 10, `MaxSizeTimestep` 10 kyr: 1.0013 x 1e51 erg per SN (0.020 on an older base before #107); with 1 kyr steps 1.0011 (was 0.729).

## Who is exposed
Exposure follows the ejecta mass per cell mass and the ambient u, not the config name.

| setup | what changes |
|---|---|
| FIRE-2/3 SN | cold ejecta: `test/fire`, 704 SNe, thermal -0.01 % |
| FIRE-3 mass return | cold ejecta: winds 1.28 -> 1.007 at 100:1 in the lattice |
| example model (`GALSF_FB_MECHANICAL` alone), heavy stars | cold ejecta: 1.0325 -> 1.0008 at 1000:1 |
| STARFORGE single-star winds, mode 2 | ejecta mass of the mass-return mode in the weights, cold ejecta: adiabatic 1.268 -> 1.013 of L_w t |
| winds in spawn mode, jets | no mechanical coupling, not exposed |

## Flag off
No flag: every build with `GALSF_FB_MECHANICAL` changes on purpose; builds without it do not compile these files. In the object comparison of the audit line against the base (FIRE-3 config), the changed functions are `verify_and_assign_local_mechfb_integrals`, `mechfb_loop` and `neighbor_loop_runner`; everything else byte- or instruction-identical (to be re-run on <TIP>).

## Limits
- The adiabatic spawn failure of `test/wind_singlestar` (12.5 %) is not addressed.
- The mode-2 shell radius with cooling (about 28 % inside Weaver) stays an expected failure.
- The CR order above is unchanged.
- `test/mechfb_energy` is adiabatic at n_H = 0.1; it does not test cooling gas.

## Depends on / followed by
Nothing. Followed by the single-star feedback PRs (slice2/10 to 12).
