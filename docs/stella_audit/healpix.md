# Mechanical feedback weights series (hp/01, hp/02)

Series name: hp. Base: <BASE>. Tip: <TIP>. Branch names are series names.
Covers: the angular split of mechanical feedback (SN energy, wind and AGB momentum, star-side radiation kick) in the mechanical loop. hp/01 adds the reference implementation's (stella_dev) HEALPix option, opt-in. hp/02 adds the reference's production weighting, mass x kernel, which is the Stella default.
Order: slice2 -> chem (chem/12 before tc) -> tc -> gap -> cr (cr/04 before pi/06) -> met -> pi -> ext -> tr -> kj. hp/01 sits on par/08-stella-defaults-configs; hp/02 sits on hp/01.

| branch | src lines | tests |
|---|---|---|
| hp/01-mechfb-healpix-weights | 76 (76 added, none removed; flagged) | test/mechfb_healpix_weights: 4 passed, 1 and 2 ranks (narrow) |
| hp/02-mechfb-mass-weights | 114 (107 added, 7 removed) | test/mechfb_mass_kernel_weights: 20 passed, 1 and 2 ranks (broad: closure of energy, momentum, mass, metals) |

Source lines without Template_Config, docs and tests. Both under 300 source lines. All numbers below were measured on an older base (cc7c11d5d for the 144-run measurement; hp/01 and hp/02 on par/08) (measured on an older base; to be re-measured on the audit branch). Full suite: <SUITE RESULT>.

---
## Measurement design (bars written before any run)

Question: does the angular split of feedback with the base's face-area weights and vector balancing match the reference with `GALSF_RESOLVEDISM_FB_HEALPIX`?

Codes: the reference with the flag ("on"), the reference without ("off"), the port. Production cloud config in all three; the reference adds only its test star flag (`GALSF_RESOLVEDISM_ISOLATED_FB_TEST`).
Setups, 1 Msun cells, one star, 8 seeds each (jitter of the cells; same cells in the three codes): (a) uniform n_H 100; (b) gradient n_H = 100 exp(z/0.5 pc), between 1 and 1000; (c) star on the surface of a clump n_H 1000, radius 3 pc, in n_H 1. Events: one 20 Msun SN (energy), one wind deposit from a 35 Msun star (momentum).
Measured from snapshots just before and a few 1 yr steps after the deposit (hydro has moved nothing): per receiving cell dE, dM, d(m v) as seen from the star; fractions in the 12 NSIDE-1 HEALPix pixels and on the dense side (`f_dense`: cells with direction pointing to the denser side: +z in b, -z in c; in a, +z as a control).

Equivalent only if all hold in all three setups, for energy and for momentum:
1. TV(<port>, <reference>) <= 95th percentile of TV(ref_s, ref_s') over seed pairs (TV = half the sum over pixels of |difference|, <> the mean over seeds). In uniform pixel filling that percentile is about 0; then 0.02 is the floor.
2. Every pixel: |<port>_b - <ref>_b| <= 2 sqrt(var_ref_b + var_port_b), var over seeds.
3. |<f_dense port> - <f_dense ref>| <= 2 sqrt(var_ref + var_port).

For mass x kernel the same bar applies with "off" in place of "on". In addition: closure per event (energy, momentum, mass, metals each <= 1e-12 relative; the reference's own wind ledger in the 48 runs: energy <= 1.3e-12, |p| against booked <= 1.3e-15, net vector / sum |p| <= 7e-16), measured with hydro frozen in a unit test; the test fails on the base and passes with the change, never marked expected-failure; flag off identical by objcmp; `#error` for the new flag together with the other weight flags; a smoke run with the Stella cloud config (no pass/fail threshold; a crash is a fail).

---
## Result on the base without the change (144 runs)

Verdict: the base is not equivalent to the reference with the flag. TV(port, on) per row (limit in brackets):

| setup | energy (SN) | momentum (wind) |
|---|---|---|
| a uniform | 0.181 (0.020) FAIL | 0.179 (0.024) FAIL |
| b gradient | 0.066 (0.312) ok | 0.064 (0.363) ok |
| c clump | 0.202 (0.331) ok, 2 of 12 pixels FAIL | 0.162 (0.357) ok |

f_dense differences stay inside 2 sigma in all rows (a: 0.049, 0.045; b: 0.212, 0.167; c: 0.126, 0.096). In the gradient, f_dense is 0.525 +- 0.119 for the base, 0.737 +- 0.149 for the reference with the flag, 0.921 +- 0.045 without, 0.924 for mass x kernel. The base is not mass-weighted either: TV to mass x kernel in the gradient is 0.45 (reference without the flag 0.015), to uniform 0.11 (with the flag 0.14). The base's weights are solid-angle like, but not the same split. Limits of the bar: in b and c the scatter is large, so a pass there is weak.
Run faults found and redone before the verdict: four seeds with IC files damaged by two runs writing them at once; all ICs were checked afterwards.

---
## hp/01: Mechanical feedback, HEALPix weights, equal share per occupied pixel

**What and why.** The reference's `GALSF_RESOLVEDISM_FB_HEALPIX` as an option of the mechanical loop. The sky of a source is cut in the 12 NSIDE-1 pixels; each occupied pixel gets the same share, and the cells of a pixel inside the source's own kernel share it equally. The face-area weights do not give this split (table above).

**Evidence before.** The test on the base: 4 failed, a cell's share off by a factor up to 21; the 144-run measurement above.

**Change.** New flag `GALSF_FB_MECHANICAL_HEALPIX_WEIGHTS`, default off. No new parameters.
- declarations/particle_data.h: 12 pixel cell counts behind the weights.
- galaxy_sf/mechfb_loop.{h,cc}: one extra loop mode -3 before -2, counts the cells per pixel.
- galaxy_sf/mechanical_fb_functions.h: NSIDE-1 pixel, weight 1 / (occupied pixels x cells in the pixel), receivers inside the source's kernel only, vector q = w (r - mean of w r) in place of the face-wise balancing, as the reference; pure-thermal SNe use the plain weight (heat needs no balance); all gas along one ray: radial, as the reference.

**Cost.** 96 B per particle with the flag (12 doubles of `Area_weighted_sum`, in every particle as the existing 12; 96 B per source in `MechFBLocalIn`). One more neighbour pass per step with an active source. No collective added. No work on quiet steps.

**Evidence after** (measured on an older base; to be re-measured on the audit branch). test/mechfb_healpix_weights: 4 passed on 1 and 2 ranks. The same 48 runs with the flag (production config): bar met in all six rows; TV to the reference with the flag at most 0.001, f_dense differences 0.000. The test changed once while it was brought up (thresholds unchanged): the momentum sum runs over all gas, not the receivers, because hot receivers push on their neighbours; the run is 2 yr, not 4 yr, so hydro does not turn the kicks.

**Flag off.** Base par/08 against the tip: gmc_cooling 145/147 byte-identical, FIRE3 144/147, the rest the same instructions (asserts' line numbers and strings only) (measured on an older base; to be re-measured on the audit branch). Flag off: to be re-run on <TIP>.

**Depends on / followed by.** par/08; hp/02. Not in any default config. Open question for the reviewer: should Stella turn it on? The reference's cloud config also leaves it off.

---
## hp/02: Mechanical feedback, mass x kernel weights, Stella default

**What and why.** The reference's production weighting as an option of the mechanical loop, and the Stella default: each cell in the source's kernel gets mass x kernel; winds and AGB go along q = w (r - mean of w r), normalised by sum |q|, so net momentum is zero and energy closes per cell. The base's face-area weights stay the default outside Stella; HEALPix stays opt-in.

**Evidence before.** The test on the base: 20 failed, a cell's share off by up to 0.15, energy 6e-5, no recoil, 0.7 % of the momentum lost where two winds overlap. The 144-run measurement: gradient, SN energy to the dense side 0.92 in the reference, 0.53 in the base.

**Change.** New flag `GALSF_FB_MECHANICAL_MASS_KERNEL_WEIGHTS`, default off, set by STELLA_DEFAULTS; `GALSF_FB_MECHANICAL_FACE_AREA_WEIGHTS` keeps the base weights in a Stella build; `#error` with the HEALPix flag or the base's flag, and without single stars from the tables. No new parameters.
- galaxy_sf/mechanical_fb_functions.h: weight, booked momentum for winds and AGB.
- galaxy_sf/mechanical_fb.cc: heat lands in the deposit; overlapping sources add their momenta in full and the kinetic energy the sum does not take stays as heat.
- galaxy_sf/mechfb_loop.{h,cc}: the star recoils with what the gas took unbalanced; this also applies to the HEALPix flag (a change to hp/01 in that corner).
- galaxy_sf/radfb_rp_loop.{h,cc}: star-side radiation kick with the same weight and direction, over the star's kernel.

**Cost.** 0 B per particle. 24 B per source in the loop's accumulator; 48 B per source in the radiation loop. No extra pass, no new collective, nothing on quiet steps. Radiation search over one kernel in place of 2 + tau_IR: 129 receivers against 1385 in the gradient.

**Evidence after** (measured on an older base; to be re-measured on the audit branch).

Bar 1, the split: met in all six rows (reference without the flag as reference).

| setup | energy TV (limit) | momentum TV (limit) |
|---|---|---|
| a | 0.0077 (0.351) | 0.0010 (0.334) |
| b | 0.0054 (0.289) | 0.0007 (0.279) |
| c | 0.0078 (0.275) | 0.0014 (0.255) |

No pixel off by more than 2 sigma; f_dense differences at most 0.0047. A pass on the limits alone is weak: the seed scatter of the reference gives limits of 0.26 to 0.35 and 2 sigma on f_dense up to 0.57 (setup c). The real number is the TV itself. The SN energy TV is not 0.000 as for HEALPix; this was traced to the ICs: the reference cannot read the last wind age, so it deposits 1.32 Msun of wind at t = 1 yr and then 17.13 Msun of SN at t = 2 yr, while the port's star has its winds marked as delivered. Both effects together predict the reference's mass shares to TV <= 0.001, per-seed correlation 0.99 to 1.00. Kernel radii agree to 0.1 % and alone give TV 0.0002. The leftover is a harness difference, not a weighting difference.

Bar 2, closure. Unit test (hydro frozen, 1 and 2 ranks): SN, wind, AGB energy, momentum, mass, metals all <= 1e-12; also moving gas (lab frame, star included), one ray with recoil, two overlapping winds, radiation kick. Grid, star's frame, per mechanical loop (ledger print in a throw-away copy):

| channel | deposits | energy | net momentum |
|---|---|---|---|
| SN | 24 | 4.4e-16 | 8.7e-26 (of sqrt(2EM)) |
| AGB | 24 | 4.4e-16 (|p| 5.6e-16) | 3.3e-15 |
| wind, first deposit, 0.11 Msun | 24 | 7.3e-15 | 6.0e-16 |
| wind, later deposits, 4e-7 Msun | 168 | **2.6e-9, NOT MET** | **1.3e-11, NOT MET** |

Later wind deposits miss the 1e-12 limit. Momentum is within one ulp of the cells' stored momentum (0.21 ulp), so that one is at the resolution of the state. Energy is not: 46 ulp at worst, so the cause is not shown; this is an unexplained difference. It has no drift: the sign alternates, mean -2e-11 against rms 9e-10, same on 1 rank, and summed over a run it is at most 2.7e-14 of the first deposit. The reference's ledger is not comparable here: its wind in these runs is a fixed 0.31 and 0.01 Msun, far from the state's resolution. An earlier version of this ledger was in the lab frame and showed 2.8e-8 net momentum, which was the ejecta carrying the star's drift; fixed in the meter, not in the code.

Bar 3, the test: fails on the base 20 of 20, passes 20 of 20 with the change, on 1 and 2 ranks. Edits to the test during bring-up: a cell's share is compared absolutely (1e-9), since it comes from differences of masses near 1; the radiation test reads the first interval with a kick. The 1e-12 closure limit did not change.

Bar 4, flag off: gmc 146/147 and FIRE3 146/147 byte-identical, core/main.o same instructions, strings only. `#error` with both flags and with the base's flag checked by preprocessor; STELLA_DEFAULTS sets the flag unless HEALPix or the base's flag is set; non-Stella builds do not get it. Compile rows added; the compile suite itself was not run. Flag off: to be re-run on <TIP>.

AGB (same conditions): inside the mechanical loop in mode 1, so the flag covers it. Bar met in all rows, but weak: the reference's own AGB kernel is not converged, 0.2 to 165 neighbours in setups b and c by the kernel count from the ICs (the SN star in the same cells: about 50 to 70, port within 1 %). Per-seed TV reaches 0.48 for that reason. Why the reference's kernel is off for this star is not shown.

**IR kick.** Reading the base: the star-side kick is in `radfb_rp_loop` with a different weight (h_j^2 / (eps^2 + r^2) x taper, radius 2 + tau_IR kernels), radial, no mean direction, no recoil. Reading the reference: mass x kernel over the star's kernel along r - m. But the run shows that is not what happens there: 0 kick lines in 48 runs with a lit star; budget lines book dp > 0. At pass -2 the budget is not loaded (`particle2in` loads it only for loop_iteration == 1), so with no wind event `evaluate` returns at once and the weight sum stays 0; at pass 1 the star has momentum 8e-16 and weight sum 0, and returns. So the reference's production run computes and books the star-side IR kick and delivers none of it to the gas. Size in setup a: star-side 1.7e33 against 7e35 g cm/s booked per step, 0.2 %; larger in dense gas. The direct term per cell is a separate pass and is not affected. The reference was not changed (read only).

The bar's reference for the IR kick therefore does not exist as written. Used instead: a throw-away copy of the reference with that one line repaired (pass -2 loads the budget), which is what its source intends. This reference was chosen after seeing the result. Against the repaired copy the base's split is not equivalent: pixels fail in b and c, net momentum up to 48 %, 1385 receivers against 129, sum |p| up to 1.56 times. With the flag (2 ranks) the bar is met: TV 0.0000, 0.0047, 0.0002 in a, b, c; sum |p| ratio port/reference 0.9995 to 1.0015 (setup b 0.9538 in one seed). The 3.84e-2 net-momentum line and the 0.9538 in setup b are print lines lost between two ranks, not physics: that seed on 1 rank has 138 of 138 cells, net 4e-16; sum |p| differs by under 1 % on 1 rank (0.9905 and 1.0039 in the two seeds rerun; cause not shown).

Open question for the reviewer: the IR kick is in the PR (about 45 lines, in the flag). With it the port pushes where the reference's production run does not. Dropping `radfb_rp_loop.{h,cc}` from the PR gives the production behaviour as it runs, with the base's kick; neither gives "no kick". Keep it in this PR or not?

**Smoke: small cloud, gmc_Z1_m1.0, 12 Myr.**

| build | ranks | end | stars | M* [Msun] | SN |
|---|---|---|---|---|---|
| mass x kernel | 32 | 12 Myr | 1323 | 775 | 2 |
| mass x kernel | 16 | 12 Myr | 804 | 479 | 2 |
| mass x kernel | 24 | **crash 6.48 Myr** | 698 | 412 | 0 |
| base weights | 16 | 12 Myr | 1847 | 1057 | 1 |
| base weights | 24 | **crash 5.86 Myr** | 1303 | 790 | 0 |
| base weights | 32 | **crash 4.73 Myr**, twice | 283 | 212 | 0 |
| hp/01 base | 32 | **crash 4.73 Myr** | 283 | 212 | 0 |

The crash is not caused by this change: the base crashes at the same step as the base weights, and both weightings crash at some rank count. It is a NaN node mass in the tree: gpu_moment_refresh fails in force_treebuild (code 90000068), then LET finalize. Source of the NaN not found. A crash is a fail by the smoke rule, so the smoke is not clean. Stellar mass moves by a factor 1.6 with rank count alone, so this says nothing on mass x kernel against the base weights.

**Flag off.** gmc 146/147, FIRE3 146/147 byte-identical, main.o strings only (measured on an older base; to be re-measured on the audit branch). Flag off: to be re-run on <TIP>.

**Limits.** On a base with eb4a2bd57 (pair separation gas minus star, now in the base) the direction sum of the radiation part takes the opposite sign; the re-cut carries that change (81560c6fd, star-side radiation pressure pointing away from the star, with its test 207328da2). The numbers above were measured before that base change.

**Depends on / followed by.** hp/01. Open before this lands: the wind energy residual (later deposits), whether to keep the radiation part (question above), the tree NaN.

---
## Note: IC metals check and reruns

The IC generator wrote header `Flag_Metals=1` in the port ICs, so the port read 1 of the metal columns and the port's gas was metal-free (the reference reads its own IC and was fine). Fixed: `Flag_Metals` = number of columns written; same cells, same RNG, all other datasets bit-identical to the old ICs (checked for all 24 setup/seed ICs). The par/08-based branches (hp/01 and hp/02 tips) have NUM_METAL_SPECIES 11, so those ICs carry the first 11 columns (Z, He, C, N, O, Ne, Mg, Si, S, Ca, Fe). Metals in the first code-written snapshot match the reference to 1 % (3 % in the clump setup c); before the fix the port had 1 column of 27 and Z 0.0104 against 0.0140.
Rerun, setups a, b, c, SN and wind, all 8 seeds, same harness and bars: every row, TV, f_dense and bar verdict equal to the earlier numbers within 0.001 (base still not equivalent: a TV 0.181/0.179, c energy 2 pixels; hp/01 flag-on TV <= 0.001, bar met; hp/02 TV <= 0.0077, bar met). The split depends on geometry and masses, not on the metal content, over the 1 yr steps measured. Not rerun: the closure unit test, ledger grid, IR kick and smoke tests (they use other ICs).
