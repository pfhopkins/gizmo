# Audit guide: Stella on gpu_kokkos_unified

For Phil Hopkins. One branch to read before any PR is opened. Nothing has been pushed or posted.

## 1. What the branch is

- Branch `gku_stella_dev`: the code state to audit is 7165593ba, on gpu_kokkos_unified c846fe769 (your head of 2026-11-12, merged in). The branch follows your head; commits after 7165593ba add new features and are listed in section 8. Suite on the tip: 739 tests, 678 passed, 37 failed, 24 skipped. 19 of the failures fail on your head as well; one (hernquist with a randomised tree) varies from run to run with code identical to your head's for that build; the other 17 are ours, each with a known cause (section 5); one of them (ejecta cell) is fixed since and passes at 7165593ba.
- The Stella single-star model: stars drawn from an IMF, with lifetimes, winds, AGB deaths, SNe and Ia from installed stellar tables; an SF criterion; SGChem chemistry and cooling as a host branch with its own flag; per-direction columns and band fluxes inside the existing TreeCol walk; band-resolved radiation pressure; per-direction photoionisation; CR rate and CR-fluid exchange; an energy, mass and metal ledger; 27-element yields and ISM dust; M1 and TreeRay variants; tall periodic box; KETJU chains.
- Each capability sits behind its own compile flag, off by default. `STELLA_DEFAULTS` switches the usual set. Without these flags the code is meant to equal the base (section 4).
- A staging line to read, not to merge whole: it is the planned PR series applied in order.

## 2. Organisation

192 planned PRs: 60 with a broad physics test, 131 with a narrow one, one pure refactor. One text per series in this folder, each with: what and why, evidence before, change, cost, evidence after, flag off, limits, dependencies.

Order: fixes in your code that need no Stella (`small_fixes_B4_B5_C1_C2`, `B3_trimmed`, `buildsystem`); the slice (`slice`: tables, IMF, star record, restart layout, spawn, gas collection, SF, SN, winds, star death, AGB, Ia); chemistry in parallel (`chem`); shared upgrades (`treecol`, `gaps`, `cr`, `metals`); photoionisation, M1, TreeRay (`pi`, `m1`, `treeray`); then `extra`, `more`, `healpix`, `tallbox`, `parity`, `final` (mostly diagnostics and test hooks); KETJU with Mike (`ketju`).

Texts added for this audit: B3 in its current form; star death at the first step past the lifetime (as gizmo2017); restart layout version 4; star-side radiation pressure direction; photoionisation step clamp; KETJU via `KETJU_PATH`; tree refresh when a star switches or its luminosity drifts; wind composition; Mike's causal wake for single-star SNe (his `SNE_BROADCAST_TIMESTEP` from #107, which you did not port; see section 6); final/08, final/36.

Exact metal exchange: Stella now uses your `METALS_EXACT_PAIRWISE_EXCHANGE` (#35 design) by default; our own exact-conservation option is removed from the branch.

Last fixes on the branch, each folded into its PR: a supernova with too little gas becomes a new ejecta cell before any kernel growth (growth is the fallback); radial-shell photon cap and SF converging-flow and dust divergence correct in comoving runs, KETJU unit conversion with h != 1, KETJU in periodic and comoving runs (limiter grid, region radii as physical lengths, region centre of mass across the box edge), SGChem in-cell column with M1 in comoving runs, and two build fixes for flag sets without per-direction photoionisation or single stars.

## 3. Build and run

Nothing is fetched at compile time.
- Stellar tables: `scripts/get_stella_tables.sh <PREFIX>` installs and hash-checks two HDF5 files (about 114 MB); parameter `StellarTablesFile`. Start-up stops with a message naming the script if they are missing or of another version.
- SGChem: `SUNDIALS_DIR` (6.x or 7.x; 7.x passes `SUN_COMM_NULL`); one SUNDIALS per build. Chemistry tables via `SGChemTablesPath`, hash-checked at start-up.
- KETJU: separate install found by `KETJU_PATH` (needs `include/mstar/mstar.hpp`, `include/ketju_integrator/ketju_integrator.h`, `lib/libmstar.a`); the build stops with a message naming the missing file. No submodules.
- Configs in `scripts/stella/`: `Config_stella_{default,cloud,tallbox,galaxy,zoom}.sh`, params for cloud, tallbox, galaxy, zoom. `SGCHEM`, `CHEMISTRYNETWORK`, metals and pixel flags stay in `Config.sh`, since the chemistry sources read `GIZMO_config.h` only; `STELLA_DEFAULTS` stops with `#error` if they are missing.
- Nine configs compile on the tip (cloud, default, tallbox, net 17, c3, KETJU, galaxy, zoom, double precision; make clean each). Cloud smoke test (12 Myr) at 24 and 32 ranks reaches the end with no error and no LET pack or finalize stop.

## 4. What was measured on the tip

- Cloud against the reference implementation (stella_dev), 10 seeds to 9.3 Myr, two-sided Mann-Whitney: median final drawn mass 1257 vs 1075 Msun (p = 0.13), star count 2174 vs 1872 (p = 0.18), HII mass at 3 to 9 Myr p = 0.017 to 0.91 (lowest at 6 Myr, more HII mass than the reference). Before the switch to your exact metal exchange the same seeds gave 1084 Msun (p = 0.65) and HII p >= 0.076. Rerun on the tip with the same 10 seeds: every star's birth identical to that run (and the smoke tests at 24 and 32 ranks likewise), so these numbers hold for the tip.
- Ledgers (same 10 runs): gas plus star mass from snapshots constant to at most 8.3e-9; feedback heat plus kinetic energy over the ejecta energy 1.0000001 to 1.0000034.
- Driven box (30 SNe) and a single-SN ladder (cells of 10, 1, 0.1 Msun; bars |G-1| <= 0.02 cumulative, 0.9 to 1.1 per event), with the causal wake: cumulative gain 1.011; single SN, largest gain after the explosion 1.008 (10 Msun cells), 1.019 (1 Msun), 1.079 (0.1 Msun). Without wake-up of inactive cells the driven box gains 2.82.
- Flag off, against your current head: 115 of 140 objects byte-identical (gmc_cooling) and 113 of 140 (FIRE-3). The rest have equal instructions except functions changed on purpose by general fixes: pending-momentum port, `restart()` (layout version 4), spawn as a function, `timestep`/`begrun` (B5, C1), `merge_split`, and with FIRE-3 the mechanical feedback functions (B3). Each PR carries its own run.

## 5. Known open

- Wind energy after your #107: weak feedback heat is now injected at once instead of as a rate (aa07435a5). Our per-cell energy closure for winds was set to 1e-3 under the rate form; it now gives 1.0011 and the wind momentum check 65 against 53 (test/single_stars_winds_agb_ia). Found by bisection; to be fixed in the winds PR, bars unchanged.
- Three tests are stale against the branch and fail for that reason only: winds in chunks expects a last wind delivery before death, while winds owed at death now go out with the ejecta (star death as gizmo2017); thermal-only winds read a ledger column removed after #107; treecol_nongas reads a diagnostic print you removed (823d63c95). To be updated in their PRs.
- SGChem box against the standalone solver: 6.2e-7 against a 1e-8 bar since the solar metal fraction changed to 0.014 (final/15); cause inside the run not found.
- Net 5 lacks cooling from photoionised metals; net 20 never fills its carbon column; with M1 and SGChem molecular gas stays transparent to ionising photons; two M1 tests open (radiation pressure once 1.35 against 1.1; slab attenuation).
- Tall box: slab gravity of the production walk 1.13e-2 against 4e-3 (tb/01); TreeCol columns in the tall box 0.019 against 0.01 (not traced).
- H nuclei per mass: 0.7065 in the code, 0.714 in the chemistry; TreeCol H2/CO columns use the chemistry's, about 1 % above the reference. CR-fluid normalisation 3.9 % below the reference.
- The per-direction PI exchange record was 12 kB per entry in the reference and overflowed a buffer once; the compact-histogram PR is meant to fix it, to confirm on the tip. G0 r^2 against Q at 4 to 5 Myr is 15 to 35 % below the reference at 5 to 20 pc (different star mix, not a same-state comparison).
- KETJU with a periodic box but non-periodic gravity (`GRAVITY_NOT_PERIODIC`): a binary across the box edge ends with a wrong orbit (a = 0.28 of 1000 AU); with periodic gravity it matches the open box. Not traced.
- KETJU: two collectives per step when candidates exist but none are close; a node bulk-velocity residual on 2 ranks; a restart inside a close-pair test differs from a straight run (max dv 1.5e-4 km/s); merged stars need table fields the reader lacks.
- Some test bars were moved after results were seen; each PR text says so.
- Not in this round: stochastic IMF re-cut, MHD energy options, conservative psi cleaning.

## 6. Where your rules may be stretched

- Over ~300 source lines, splits proposed in the texts: tables loader (431), gas collection (663), per-direction fronts (348), solve cadence (356), three KETJU PRs (324 to 335).
- SGChem import, about 17k lines (core and net 5 about 12.7k; nets 3/15/16/17 2.3k; nets 18/20 2.0k; about 6.2k of it generated rates and tables). Bitwise-checked against the reference; proposed per file group, with the generator and a hash instead of a line review for generated files.
- Test hooks: about 40 small diagnostics and `TEST_` flags in the final series, mostly with narrow tests. We suggest keeping them on staging unless you want them.
- Memory: star record, 16-bit per-pixel cell values and the PI front record are added under flags; bytes per PR in each text. Totals against your 230 B/particle and 2.6 kB/cell are not yet added up.
- Causal wake: you did not port `SNE_BROADCAST_TIMESTEP` on #107 because of an `MPI_Allgather` every step and an all-gas times all-SN loop. Our single-star form runs a neighbour loop from each supernova only on steps with one; whether a step has one comes with the existing feedback source count, so a step without a supernova has no collective from it (tested). It is on in `STELLA_DEFAULTS` and is what closes the single-star SN energy above. We ask before sending it (question 8).
- For discussion only: a second radiation walk, so gravity can use `ADAPTIVE_TREEFORCE_UPDATE` for all particles while columns, G0 and fluxes run on their own cadence. It conflicts with your reuse rule and costs a second walk. On clouds, skipping gas cells gave 1.16 to 1.31 times speed-up with G0 off by more than 3 % in 33 to 44 % of cell-steps; skipping only collisionless particles is under test.

## 7. Questions

1. Chemistry import: library-style PRs reviewed against the reference dump, or per file group?
2. Test hooks and narrow tests: which stay in the tree, which on staging?
3. Type-restricted adaptive tree force (collisionless skip, gas always walks): acceptable as a general option? Any plan for radiation on its own cadence?
4. M1: PE band 8 to 13.6 eV with LW inside, NUV bin 6 to 8 eV, H and H2 rates from the H0 bin, HI atomic, no H2 in `rt_kappa`, `SGCHEM_RT_ABSORBERS_FOLLOW_THE_NETWORK`; 6 to 8 eV counted once by default, double count opt-in. Acceptable?
5. TreeCol: team scratch (96 doubles per target) or a second walk; 16-bit storage; could the 12-pixel column feed your consumers?
6. Ledger: separate file or rows in your energy file; cadence?
7. Gas collection: acceptable rule for shared donor mass (design note first)?
8. Single-star SN energy: keep the causal wake under its own flag in `STELLA_DEFAULTS`, make it opt-in, or is there a form you would rather take?

## 8. Commits after 7165593ba

New features and fixes added on the branch after the audit state, each with its test:
- Photoionised cells form no star through the instant-SF density cutoff (the rate's ionised-cell veto now holds on that path too); test `photoion_per_direction::test_no_instant_sf_in_ionised_cells`.
- `GALSF_SINGLE_STARS_PISN_ENERGY_FROM_HELIUM_CORE` (opt-in): pair-instability SN energy from the He core (Heger & Woosley 2002, Wise+12a fit); default stays 1e51 erg; test `single_stars_supernovae::test_pisn_energy_from_helium_core`.
- `SGCHEM_PRIMORDIAL_HELIUM`: He/H = 0.079 and X_H = 0.76 for any network; unit test `test_primordial_helium`.
- `SGCHEM_DGR_REMY_RUYER14` (in `STELLA_DEFAULTS`): dust-to-gas ratio of Remy-Ruyer+14 below 0.26 Zsun, linear above, so solar runs are unchanged; unit test `test_dgr_rr14` and `treecol_pixels::test_dust_to_gas_remy_ruyer14`.
- `SGCHEM_HMINUS_PHOTODETACHMENT_FROM_STELLAR_BANDS`: H- photodetachment from the stellar TreeCol bands (optical, NUV, 6-13.6 eV) with the Shapiro & Kang cross-section; unit test `test_hminus_bands`.
- Young single stars refresh their neighbour density in comoving runs too (it was done only in isolated runs); SN/AGB logs gain the ambient n_H, T, star h, neighbour count, birth and birth-frozen n_H; test `photoion_per_direction::test_young_star_density_follows_the_gas_in_comoving_runs` (the star held its first density, 0.044, while its gas fell to 0.022; now 0.022) and the log columns in `single_stars_event_logs`.
- `SGCHEM_UVB_TREECOL_SHIELDING`: the UV-background floor of comoving SGChem runs attenuated per pixel in 10 FG20 bands by a fourth stored TreeCol column (gas mass, 2 B per pixel per cell, only with the flag); tests `sgchem_uvb_shielding` (factors against the FG20 spectrum with Verner+96 cross sections) and `treecol_pixels::test_gas_column_for_uvb_shielding`.
- `GALSF_SINGLE_STARS_POPIII`: Pop III stars in gas below 1e-4 Zsun (n_H, converging flow, H2 fraction, reservoir Jeans mass; Pop III IMF); the star record carries the label (snapshot block `SingleStarPopIII`); a Pop III star collects gas only once its whole remaining mass is in reach, otherwise the draw fails and its gas goes back; unit test `test_popIII_sf` (criterion, IMF by KS test) and the one-zone collapse `sgchem_onezone::test_popIII_onezone` (pristine gas qualifies first at n_H 1.4e6, the Stella Jeans test only at 2.2e9). `Config_stella_zoom.sh` with network 17, Pop III, primordial helium and H- from the stellar bands.
- `Config_stella_tallbox.sh` names `SGCHEM_DGR_REMY_RUYER14`, as the other Stella configs get it from `STELLA_DEFAULTS`.
- Network 17 and oxygen: why the network has no oxygen chemistry, with the one-zone measurement, in `src/cooling/sgchem/cxx/DIFFERENCES.md` (comment and doc only).
- Tree (from Mike's starforge_dev e1b1c2bbb): a particle whose position is set rather than drifted (a jet or wind cell spawned into the standing tree, a sink re-centred on its accreted mass or moved to the potential minimum, a KETJU member set directly or snapped to its end position) gets every node box above it widened to hold it. Node centres stay fixed and a raised length only errs wide; widened top-level boxes reach the other ranks with the motion-bound flush. Test `tree_node_boxes` with the test-only audit `TEST_TREE_NODE_BOX_AUDIT` (a sink accreting and spawning jet cells: 2 particles outside a box before, 0 after).
- Full-run Pop III test `sgchem_keep_ic::test_popIII_stars_in_pristine_gas`: pristine gas at n_H 1e7 forms only Pop III stars that hold their drawn mass (>= 1 Msun); the same gas at 1e-3 Zsun forms none.

The new switches, their parameters, configurations and tests, and the stella_dev name of each, are documented in the User Guide (`docs/gizmo_documentation.md`, section "Stella Single Stars and SGChem Additions"). They are marked in development there and in `Template_Config.sh`.
