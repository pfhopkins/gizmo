# Tall-box series (tb/01 to tb/03)

Series name: tb. Base: <BASE>. Tip: <TIP>. Branch names are series names.
Covers: self-gravity periodic in x and y and open in z (tb/01), a test that TreeCol columns and fluxes wrap the same way (tb/02), and the interstellar-radiation-field background scaled with recent star formation in builds without SGChem (tb/03).
Order: slice2 -> chem (chem/12 before tc) -> tc -> gap -> cr (cr/04 before pi/06) -> met -> pi -> ext -> tr -> kj. tb/01 stands alone on the base. tb/02 follows tb/01 and gap/03; tb/03 follows tb/01 and cr/06. tb/02 and tb/03 sit on older stacks and need a re-cut.

| PR | branch | src lines | tests |
|---|---|---|---|
| 1 | tb/01-periodic-xy-images-long-box | 79 | test/tallbox_gravity (broad: against a direct sum over images and an analytic field) |
| 2 | tb/02-treecol-columns-under-xy-images | 0 (test only) | test/treecol_tallbox (broad) |
| 3 | tb/03-ghosdt-isrf-background-and-tallbox-test | 53 | test/isrf_recent_sfr_tallbox (narrow) |

All PRs are under 300 source lines. All numbers below were measured on older bases (7ebc666df for tb/01, gap/03 9a932336d for tb/02, cr/06 fd82665d7 for tb/03) (measured on an older base; to be re-measured on the audit branch). Full suite: <SUITE RESULT>.

---
## tb/01: Periodic x/y images, long box

**What and why.** Self-gravity that is periodic in x and y and open in z, for tall boxes such as a kpc-wide patch of a disc with 20 kpc of height. Today a long periodic box aborts at start-up (`BOX_LONG_*` with periodic gravity), and `GRAVITY_NOT_PERIODIC` drops the periodicity altogether, so a patch of a disc feels no pull from its own copies. The new flag `GRAVITY_PERIODIC_XY_IMAGES` makes the tree wrap separations in x and y only and add the (2N+1)^2-1 images of every accepted source around its nearest image. It sits in the shared pair evaluation, so the host and device walks both get it, and the radiation walks that already use the gravity box-wrap (columns, fluxes) follow it.

**Evidence before.** `test/tallbox_gravity` on the base: the run stops in `read_parameter_file()` (begrun.cc:3631, "The gravitational solver does not allow stretched-periodic boxes"). With `GRAVITY_NOT_PERIODIC`, the best the base can do, the slab acceleration differs from the direct sum over images by a median of 105 % and the probes 0.2/0.5/1.0 above the sheet feel a_z = 4.16/2.10/0.81 instead of 6.15/5.95/5.62.

**Change.**
- `gravtree_force_kernel.h`: image sum in `grav_pair_evaluate_core`, via a helper using `grav_force_pair`, the tidal and jerk accumulators.
- `gravity_box_distance.h`: one gate `GRAVITY_BOX_WRAPS` for the four wrap helpers.
- `precompiler_logic.h`: requires `BOX_PERIODIC` with periodic x/y and `BOX_OUTFLOW_Z` or `BOX_REFLECT_Z`; refuses `BOX_SHEARING`, `PMGRID`, `GRAVITY_SPHERICAL_SYMMETRY`, `SINGLE_STAR_DIRECT_GRAVITY`; defines `GRAVITY_NOT_PERIODIC`, so no Ewald and no PM.
- `begrun.cc`, `global_data_all_struct.h`, `Template_Config.sh`.
- New flag `GRAVITY_PERIODIC_XY_IMAGES` (default off); new parameter `GravityPeriodicXYImages` (int N >= 0, required with the flag).

**Cost.** 4 bytes in `All` under the flag; no per-particle or per-cell state; no collectives. Work: (2N+1)^2-1 extra pair evaluations per accepted interaction (288 for N = 8), only under the flag.

**Evidence after** (measured on an older base; to be re-measured on the audit branch). `test_tallbox_gravity`, 1 and 2 ranks identical: tight walk vs numpy direct sum over the same images median 5e-14, p99 4.8e-3, max 5.9e-3; probes within 1.7e-7 of the analytic field of the image patch; sheet self-force 2.3e-3 of 2 pi G Sigma; net force 3e-5, momentum 5e-14, mass exact.

**Open question for the reviewer: the production-walk bar is not met.** Production walk vs tight walk: median 1.1e-2, p99 3.3e-2, against the limits set before the first run (4e-3, 1.5e-2); the test fails on this and is left failing. The same walk in an open box gives 1.9e-3 / 3.8e-3. Measurements (absolute difference between production and tight walk, slab IC; `ErrTolForceAcc` is the criterion that matters, `ErrTolTheta=0.5` with `ErrTolForceAcc=1e-6` equals the tight walk exactly): N image rings 0 / 1 / 2 / 4 / 8 / 16: median |da| 2.7e-2 / 1.9e-2 / 3.1e-2 / 3.5e-2 / 2.8e-2 / 1.6e-2, relative to the median |a| 6.8e-2 / 3.9e-2 / 3.6e-2 / 2.1e-2 / 1.1e-2 / 6.5e-3. So it is not the ring edge (N = 0, nearest-image wrap only, has the same absolute error, and it does not grow with N). Split by target position at N = 8, median |da|: targets in the middle of the box (no half-box plane inside it) 1.3e-2; the other bins 2.3-3.1e-2. Nodes that straddle the half-box account for about half; the other half is open (the wrap alone, not the images, costs a factor ~9 over the open box: 2.8e-2 vs 3.0e-3). A bar that covers the measured value would be 1.7e-2 / 5e-2. Should the bar be adopted, or the cause found first?

**Comparison with the reference implementation (stella_dev 22017b407)** (built with the same tall-box flags, same ICs and parameters, 1 rank): sheet probes in the tight walk identical to 4 digits (6.1501 / 5.9509 / 5.6215 above and below, analytic 6.1501 / 5.9509 / 5.6215), sheet self-force 1.65e-3 of 2 pi G Sigma in both. Slab, tight walk vs direct sum over images: here median 5e-14, p99 4.8e-3; reference median 2.2e-3, p99 5.1e-3. Slab, reference vs here: tight median 9.5e-4, p99 3.5e-3; production median 1.6e-3, p99 5.0e-3. The production h = 0.2 probe: reference 6.1217, here 6.1501 (analytic 6.1501). Ewald check: cubic box (z periodic, Ewald) against the tall box, in-plane acceleration, slab of three layers: both codes differ from the Ewald result by 1.2e-2 rms (relative to the rms |a|), identical for the two codes. This is not a test of the images: the cubic box has z copies and the Ewald force a neutralising background, which an open-z image sum does not (a 17^3 explicit image sum does not converge to the Ewald force either, 0.16 off). The 5e-5 in the reference's notes is not reproduced by this set-up.

**Flag off.** objcmp against the base (base built at a path of the same length, since the source directory is compiled into strings), gmc_cooling and FIRE3 configs: 140 objects each, 136 byte-identical; the other four (cooling.o, begrun.o, main.o, io.o) have the same instructions and differ only in line-number immediates or strings (measured on an older base; to be re-measured on the audit branch). No FIRE feedback or star-formation code is touched. Flag off: to be re-run on <TIP>.

**Depends on / followed by.** None / tb/02, tb/03.

Open: potential and the target's own images are not computed (no net force); star direct gravity is refused with the flag.

---
## tb/02: TreeCol columns under x/y images

Base: gap/03-treecol-16bit-storage-default plus tb/01.

**What and why.** With tb/01 the TreeCol pixel columns, stellar band fluxes and G0, accumulated inside the gravity walk with its box wrap, wrap in x and y and not in z in a tall box. No source change; the test pins it.

**Evidence before.** With the unwrapped reference (what a `GRAVITY_NOT_PERIODIC` run corresponds to) the median column error of cells within 8 pc of an x/y face is 0.21 against 0.023 in the interior (analysis of the first run's cell dump; not a run of the old code).

**Change.** Test only, no source.

**Cost.** None.

**Evidence after** (measured on an older base; to be re-measured on the audit branch). `test/treecol_tallbox::test_columns_and_fluxes_wrap_in_xy_only`, 1 and 2 ranks: fluxes pass the treecol_pixels limits (6-13.6 eV 1.2e-4 median, pixels 1.8e-4, band totals 3e-8), and the wrap gate (edge cells 1.9e-2 vs 2.3e-2 for the others, limit 1.5x) passes. The column limit of treecol_pixels (1e-2 / 5e-2) is not met: 1.9e-2 / 5.9e-2. Interior cells show 2.3e-2, so this looks like the tree error of this geometry and not the wrap. The test is left failing for review. Open question for the reviewer: a run in a cubic non-periodic box of the same lattice would give the baseline and separate the two; is it wanted before this goes in?

**Flag off.** No source change.

**Depends on / followed by.** tb/01, gap/03.

---
## tb/03: Recent-SFR interstellar radiation field and CR background without SGChem

Base: cr/06-sgchem-cr-fluid-energy-exchange (fd82665d7) plus tb/01.

**What and why.** The recent-SFR scaling of cr/03 covers the SGChem inputs. The reference's version also scales the `RT_ISRF_BACKGROUND` field and the background cosmic-ray energy density in builds without SGChem (its own test `test/ghosdt_g0_cr`). This PR takes the macro `ISRF_BACKGROUND_STRENGTH` (max(F x strength, 0.324e-2/1.7)) into the places that read `InterstellarRadiationFieldStrength` (`cooling.cc` x2, `rt_functions.h` x6, `cosmic_ray_functions.h`) and F x sqrt(strength) x 1.6e-12 for the CR background. The flag now needs SGCHEM or RT_ISRF_BACKGROUND.

**Evidence before.** `test/isrf_recent_sfr_tallbox` on cr/06 + tb/01: the build stops at the `#error` "... set SGCHEM".

**Change.** 53 changed source lines in the files above. No new state or collectives.

**Cost.** None beyond the above.

**Evidence after** (measured on an older base; to be re-measured on the audit branch). The test passes (2 ranks, 65 calls, 1112 stars formed): F equals the full scan and the final-snapshot rebuild to 1e-12 at every call; background field as the macro. Against the reference (same IC, same flags, own run): F(0) = 115.2 and maximum 116.99 identical; minimum 99.5 / 97.4 and end 107.2 / 106.0 differ because star formation is stochastic.

**Open.**
- The running sum does not see young stars removed by the open z boundary; an earlier version of the test with the gas on the z = 0 face lost 303 particles and the sum drifted 0.4 % above the full scan. The test keeps the cloud away from the faces. It needs a hook in the device drift (`outflow_removed_mass`).
- The non-SGChem CR background path and the field in `cooling.cc` / `rt_functions.h` are not observed by the test beyond the printed G0 and the CR energy density of cell 0 (the CR density is not compared with the reference).
- Self-gravity is off in the test.

**Flag off.** objcmp against cr/06, gmc_cooling and FIRE3 configs, base at a same-length path: 147 objects, 143 byte-identical, the other four (cooling.o, begrun.o, main.o, io.o) same instructions with only immediate/string differences (cooling.o shows one immediate shift of -654336, same instruction stream) (measured on an older base; to be re-measured on the audit branch). This includes tb/01's changes. Flag off: to be re-run on <TIP>.

**Depends on / followed by.** tb/01, cr/06.
