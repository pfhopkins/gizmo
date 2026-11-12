# Parity series (par/01 to par/08)

Series name: par. Base: <BASE>. Tip: <TIP>. Branch names are series names.
Covers: places where the base differs from the reference implementation (stella_dev) for single-star runs, each fixed with a test and a comparison number: split threshold, chemistry state on merges, wake-up target bin, gas timestep without gravity from band fluxes, direct radiation pressure per cell, snapshot radiation fields, snapshot chemistry rates, and the Stella default configs.
Order: slice2 -> chem (chem/12 before tc) -> tc -> gap -> cr (cr/04 before pi/06) -> met -> pi -> ext -> tr -> kj. The par branches are stacked par/01 ... par/08 on gap/03; each says below which series PR it belongs to.

**Duplicates.** par/05 and par/06 are the same changes as tc/06b and tc/06c and are sent once, as tc/06b and tc/06c (see the TreeCol series texts). Checked commit by commit: the two test commits are byte-identical patches; the two source commits make the same change, and differ only in context (on the par stack the new per-cell field uses the 16-bit pixel storage type of gap/03, and neighbouring Template_Config lines differ). The sections below keep only the comparison numbers measured on the par stack; the tc/06c CR line is `sgchem_cr_ion_rate`.

Other placements: par/01 goes into slice2/05 (the test into single_star_record), par/02 into chem/05, par/03 into slice2/15, par/04 into tc/04, par/07 into chem/07. par/08 stays a PR of its own on gap/03. `test/single_stars_merge_split` is tc/01b.

Source lines (added/removed against the previous branch; tests, docs, configs and Template_Config not counted): par/01 4, par/02 6, par/03 21, par/04 3/-1, par/05 124/-2, par/06 171, par/07 116, par/08 47. All under 300. Commit order in each PR: test, change, docs. Numbers below were measured on an older base (gap/03 ff2993cf8, then 43025c83; reference implementation (stella_dev) at 22017b407) and are to be re-measured on <TIP>. Full suite: <SUITE RESULT>.

## Bars (written before any result; one in-repo assert was replaced after a result, see par/04)

Same ICs and same analysis script for the reference and the port. Test breadth: the radiation-pressure and wake-up tests are broad (against a reference or a conservation law); the threshold, merge-state, timestep and header tests are narrow.

- Radiation pressure (par/05). One star in a uniform box, chemistry on, gravity on, bands on, radiation pressure the only stellar channel (the base has no photoionisation; the reference twin is built without it). Momentum = sum over gas of m v.r_hat around the star from snapshots, run minus twin without radiation pressure, same code. Pass: port within 5 % of the reference at the end time, at n_H = 100 (thin deposit) and n_H = 1e4 (photon cap active), for production bands (port `COUNT_6_8EV_TWICE` vs reference default) and disjoint bands (port default vs reference `EXPERIMENTAL_RADIATION_DISJOINT_BANDS`). "Before" = the same measurement on gap/03; the difference must exceed 5 % in at least one case or the defect was not visible on this setup, which is then reported. A case at n_H = 1 was added (bar not moved): the box is optically thin there; n_H = 100 turned out thick in the UV (tau ~ 6), so the cap is active there too.
- Wake-up target bin (par/03). Falling Sedov (the reference's test/wakeup_falling_sedov IC, adiabatic exception). Pass: steps and cumulative active updates within 5 % of the reference with WakeupTargetBin=2; v_cm error < 1e-3; default of the tag = 2 under single stars.
- Gas timestep (par/04). Cold gas around a point mass with the acceleration criterion much tighter than tidal: minimum gas timestep at the first step within 1 % of the reference.
- Split threshold (par/01). Header MaxMassForParticleSplit = 2.01 x max IC cell mass to 1e-6, as the reference.
- Chemistry state on merges (par/02). Total m x_i of TracAbund species, m T, m T_dust conserved across a merge pass to 1e-6 (one step, no network solve between); survivor equals the reference's mass-weighted formula.
- Snapshot fields (par/06). Every snapshot field the reference's cloud runs read exists under the same name and units; values within 5 % of the reference on the one-star box where the quantity is defined by the same inputs.
- Configs (par/08). Config and params for default, cloud, tall box; STELLA_DEFAULTS expands to the production flag set; compile rows pass.
- All: flag-off objcmp with gmc_cooling and FIRE3 identical to gap/03.

Data for the radiation-pressure bar: reference ICs built from the reference state just after the star's birth (40 Msun star, gas it collected removed from its 81 donors, same time).

---
## par/01-split-threshold-single-stars

**What and why.** Under single stars a cell splits above 2.01 x the largest IC cell, as the reference (init.cc:1230); the port kept 3.01.

**Evidence before.** `test/single_stars_merge_split` `test_split_threshold`: 3.01.

**Change.** src/core/init.cc, under `GALSF_SINGLE_STARS_FROM_IMF`. No flags or parameters.

**Cost.** None.

**Evidence after** (older base; to be re-measured on <TIP>). 2.01. Bar 2.01 to 1e-6: met.

**Flag off.** See the end of this file.

**Limits.** Applies only under single stars; the base value 3.01 stays elsewhere.

**Depends on / followed by.** The slice2 star flag; par/02.

---
## par/02-sgchem-state-on-merges

**What and why.** A merged cell kept the survivor's abundances: H2, H+, CO of the cell it swallowed were lost. The reference mass-weights TracAbund, DustTemp, Temp.

**Evidence before.** 47 merges of 0.3 Msun H2-rich cells: total m x(H2) 5.64 -> 8.8e-6, all of it network formation (to be re-measured on <TIP>).

**Change.** src/mesh/merge_split.cc under SGCHEM. Species per H mix by mass at the one H fraction of the port chemistry. The reference's X_H-weighted form differs only under variable X_H; with `GALSF_SGCHEM_VARIABLE_XH_AND_ABHE` on, ours is not conserved (off in production).

**Cost.** None.

**Evidence after** (older base; to be re-measured on <TIP>). Total m x: H2 5.64 -> 5.6400001, H+ 0.4501 -> 0.45009983, CO conserved; < 1e-6 relative at 1e-10 code step. Bar met for species. Dust temperature and temperature are not measurable in this test (the IC cannot carry DustTemp, the solve overwrites both on active steps): formula identical to the reference, unmeasured.

**Flag off.** See the end of this file.

**Limits.** Not conserved with `GALSF_SGCHEM_VARIABLE_XH_AND_ABHE` (off in production). Dust temperature and temperature mixing unmeasured.

**Depends on / followed by.** chem/05; par/03.

---
## par/03-wakeup-target-bin

**What and why.** The reference sends a woken cell to the highest active bin (WakeupTargetBin=2); the port sends it to the waker bin minus 3.

**Evidence before.** The tag is absent; the run log has no default.

**Change.** `WAKEUP_TARGET_BIN_PARAMETER` (set by `GALSF_SINGLE_STARS_FROM_IMF`), parameter `WakeupTargetBin`: 0 = waker bin minus log2(WAKEUP), 1 = waker bin, 2 = highest active bin (default). Files: core/timestep.cc, core/begrun.cc, global_data_all_struct.h, precompiler_logic.h, Template_Config.sh. Stella feedback marks on the port are positive (mechfb_loop), so mode 2 is what gives the reference's -1 path.

**Cost.** One int in `All`. No collectives.

**Evidence after** (older base; to be re-measured on <TIP>). test/wakeup_falling_sedov: default 2; v_cm error 4.4e-4 (mode 2), 2.1e-4 (mode 0), bar 1e-3. The bar on steps is not met: port 1006 steps, 2.526e6 updates in both modes; reference mode 2 587 / 2.046e6, mode 0 714 / 2.051e6 (update counts to be re-measured on <TIP>). The modes barely change the port. The first hot-cell bin is already one lower on the port (bin 47 vs 48) before any wake-up, and 2547 vs 1220 cells are within 4 x the shortest step at t = 0.05, so the extra steps are not the target bin; cause not measured.

**Flag off.** See the end of this file.

**Limits.** The step-count bar against the reference is not met (1006 against 587 steps); cause not measured.

**Depends on / followed by.** slice2/15; par/04.

---
## par/04-gas-timestep-no-gravity-accel

**What and why.** Band fluxes define `RT_USE_GRAVTREE`, which kept gravity in the gas acceleration criterion; the fluxes add no force, and the reference drops it.

**Evidence before.** Cold gas 20+ pc from a 2e5 Msun point mass, same IC, kpc units: shortest gas step wanted 6.57e-5, median 7.72e-5; the reference 2.004e-4, 2.830e-4 (to be re-measured on <TIP>).

**Change.** src/core/timestep.cc, one condition.

**Cost.** None.

**Evidence after** (older base; to be re-measured on <TIP>). 2.005e-4, 2.829e-4; bar 1 %: met (0.02 %). In-repo test (pc units, narrow): shortest step 0.00625 before, 0.025 after; acceleration criterion 0.013; assert > 1.5 x that (bins are powers of two). That assert replaced the first one (shortest step > 0.5 x the tidal step) after a run with the change still failed it (0.010 against 0.099); the 1 % bar against the reference was not changed.

**Flag off.** See the end of this file.

**Limits.** One condition; checked on one setup.

**Depends on / followed by.** tc/04; par/05.

---
## par/05-radpressure-direct-per-cell (same change as tc/06b)

**What and why.** The reference's production run deposits the direct term per gas cell from the attenuation-free pixel fluxes, in the absorbing cell, capped at the emitted luminosity; only the IR boost stays at the star. The port had it all at the star. Also a sign bug of tc/06: the star-side kick followed dp_ij, which points at the star, so all momentum pushed the gas inward.

**Star-side path.** It stays as the path of builds without band fluxes (the reference's own branch, used by the tall box), plus the opt-out `GALSF_FB_RADPRESSURE_DIRECT_TERM_AT_STAR` for comparisons. Removing it would leave the tall box without radiation pressure.

**Evidence before.** See the table: the net momentum was negative (pushing inward) at all densities.

**Change.** 126 source lines.
- gravtree_treecol_pixels.h: four accumulators in the walk (sigma-weighted flux vector along the pixel centres, scalar), same pixel as the flux.
- single_star_luminosity.cc: `single_star_radpressure_per_cell`.
- radfb_rp_loop.{cc,h}: call, IR only at the star, sign.
- cell_data.h.
- `COUNT_6_8EV_TWICE` also counts LW twice in deposit and cap, as the reference's production run does (resolvedism_fb_momentum.cc at 22017b407: cap line 1015, deposit lines 1041 and 1107); the star-side path does not.

**Cost.** 20 B per gas cell, 32 B per target during the walk, no pixel arrays. Active cells only. The reference's cap loops over all cells and stars each step; here a running sum: each active cell swaps its share, full recount on all-active steps, L_emit from the root node. One MPI_Allreduce of one double per step: flag on only, does not scale with N, but it is on quiet steps. Inactive cells enter the cap with the share of their last update (the reference: current mass and Z).

**Evidence after** (older base; to be re-measured on <TIP>). Momentum in Msun km/s at t = 20 kyr, radial about the star, run minus twin without radiation pressure:

Reference column recomputed from the kept snapshots with the same script; the port snapshots (before, after) are no longer on disk, so those two columns are to be re-measured on <TIP>.

| n_H, bands | reference | before | after | after / reference |
|---|---|---|---|---|
| 1, production | 34.7 | -2.48 | 34.5 | 0.995 |
| 1, disjoint | 24.4 | -2.19 | 24.3 | 0.995 |
| 100, production | 110.5 | -42.3 | 110.8 | 1.003 |
| 100, disjoint | 76.6 | -37.1 | 76.8 | 1.002 |
| 1e4, production | 114.8 | -108.7 | 120.4 | 1.049 |
| 1e4, disjoint | 79.3 | -92.1 | 84.1 | 1.061 |

Bar 5 %: met in 5 of 6; 1e4 disjoint misses by 1.1 %. Summed log momentum there: port 95.0 vs reference 96.4 (-1.4 %), production 136.1 vs 137.8. So the total deposited agrees; the radial projection is 5-6 % higher in the collapsing dense box; cause not measured.
test/single_stars_radpressure_cells: thin 591.6 (direct sum 591.5), before 204.4; thick 3903 / 5538 (L_emit/c + IR 3903 / 5538); outward +0.94, before -0.51.

**Differences to the reference.** The cosmological a^-2 is added to the flux; the reference has none (identical at a = 1).

**Flag off.** See the end of this file.

**Limits.** 1e4 disjoint misses the 5 % bar by 1.1 %. One Allreduce of one double per step with the flag, also on quiet steps.

**Depends on / followed by.** tc/05; par/06.

---
## par/06-snapshot-radiation-fields (same change as tc/06c; CR line goes to cr/01)

**What and why.** Snapshots hold no radiation field on the port; analysis needs G0_FUV, G0_LW, G0_FUV_atten, ShieldFac_Dust, G0_NUV, G0_OPT, CR_IonizationRate, UV_Luminosity, LW_Luminosity.

**Evidence before.** The fields are absent from snapshots.

**Change.** 171 source lines. `OUTPUT_SGCHEM_RADIATION_FIELDS` (needs band fluxes, sets `OUTPUT_SHIELD_FAC`): names as in the reference, computed at output from the pixel fields, one float stored. Files: file_io/io.cc, sgchem_cell.cc.

**Cost.** 4 B per cell; none on quiet steps.

**Evidence after** (older base; to be re-measured on <TIP>). Against the reference, cell by cell, n_H = 100, 5 kyr: G0_FUV 1.0002 (p5-p95 0.9998-1.0005), G0_LW 1.0001, G0_FUV_atten and ShieldFac_Dust 1.023 (0.99-1.33), UV/LW luminosity 1.6e-4 dex (to be re-measured on <TIP>). Bar 5 % met for the median.
Not comparable: CR_IonizationRate is SGChemCosmicRayIonRate on this base (2.06 x the reference's attenuated rate); the line is replaced by `sgchem_cr_ion_rate` when folded into cr/01. G0_NUV and G0_OPT are zero in the reference (written, never set); the port writes the real band field. Lyman_PhotonRate belongs to the pi series.

**Flag off.** See the end of this file.

**Limits.** CR_IonizationRate not comparable on this base; G0_NUV and G0_OPT are zero in the reference.

**Depends on / followed by.** tc/05; par/07.

---
## par/07-snapshot-chemistry-rates

**What and why.** Snapshot output of the chemistry rates, names as in the reference.

**Evidence before.** The base writes no chemistry rates; analysis of cooling channels needs them.

**Change.** `OUTPUT_INDIVIDUAL_COOLRATES`, `OUTPUT_CHEM_PHOTORATES`: SGChemCoolRates (28), SGChemChemCoolRates (6), ChemPhotoRates (6).

**Cost.** 136 + 24 B per cell under the flags.

**Evidence after.** Not compared with the reference.

**Flag off.** See the end of this file.

**Limits.** Not compared with the reference.

**Depends on / followed by.** chem/07; par/08.

---
## par/08-stella-defaults-configs

**What and why.** The Stella default flag set and the configs that use it. SGCHEM, CHEMISTRYNETWORK, METALS, the pixel flag and the band-flux flag must stay in Config.sh: the chemistry sources read GIZMO_config.h only; with them set by the convenience flag alone the network ran with no columns and logged CALC_PHOTO errors. STELLA_DEFAULTS stops with `#error` if they are missing.

**Evidence before.** No Stella config exists on the base; the flag set has to be assembled by hand.

**Change.** STELLA_DEFAULTS and scripts/stella: Config_stella_{default,cloud,tallbox}.sh, stella_{cloud,tallbox}.params in the port's names, compile_suite row. STELLA_DEFAULTS is the default set; PER_DIRECTION and COUNT_6_8EV_TWICE stay opt-ins set by the configs. The configs list flags of pi/, cr/, tb/ that do nothing on this base: GIZMO does not reject unknown flags.

**Cost.** None: flag logic and configs only.

**Evidence after.** Compile: STELLA_DEFAULTS and Config_stella_cloud.sh build (narrow).

**Flag off.** See the end of this file.

**Limits.** The configs name flags of later series that do nothing until those PRs are in.

**Depends on / followed by.** gap/03, par/01 to par/07. Last of the series.

---
## Flag off (all PRs together, tip against gap/03 ff2993cf8)

gmc_cooling and FIRE3 configs: objcmp rc 0, 147/147 objects instruction-identical, 140 byte-identical; the other 7 differ only in line-number immediates and strings (cooling, begrun, init, main, timestep, io, merge_split) (older base; to be re-measured on <TIP>). Flag off: to be re-run on <TIP>.

## Tip checks (older base)

- Tests at the tip: single_stars_radpressure_cells 5/5, single_stars_merge_split 2/2, wakeup_falling_sedov 2/2, single_stars_gas_timestep 1/1, single_stars_radiation_pressure 3/3. The radiation-pressure matrix repeated on the tip gives the same ratios (0.995, 0.995, 1.002, 1.002, 1.049, 1.061).
- Cloud smoke (gmc_Z1_m1.0, 12k cells, 5 Myr, Config_stella_cloud.sh, no photoionisation on this base): 8 ranks reaches TimeMax, 381 stars (to be re-measured on <TIP>; the run kept on disk is a later one), radiation pressure firing, no chemistry errors. 32 ranks stopped at 3.9 Myr in let_finalize_unredirected_foreign_topleaves; with the empty-rank Father mirror fix (now in the base) it reaches TimeMax. Not a parity defect of this series.
