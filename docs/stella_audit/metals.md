# Metals and dust for single stars: PR drafts (series met)

Covers: solar abundances from the stellar tables, 27-element yields in extra passive slots, dust-to-gas ratio options in SGChem, single-star dust yields into the ISM dust model, SGChem reading the live dust, and metal diffusion defaults with single stars.

Order and dependencies: PR order of the whole work is slice2 -> chem (chem/12 before tc) -> tc -> gap -> cr (cr/04 before pi/06) -> met -> pi -> ext -> tr -> kj. Within this series each PR sits on the one before. Needs the extra passive metal slots PR (`up/C4-metals-extra-passive-species`). met/06 also needs an exact metal-diffusion option: today our `up/C3` (`TURB_DIFF_METALS_EXACT_CONSERVATION`, not taken upstream in this form), to be replaced by `METALS_EXACT_PAIRWISE_EXCHANGE` (see PR 06). tc/06c lands before gap and cr.

Base: `<BASE>`. Tip: `<TIP>`. Full-suite result: `<SUITE RESULT>`.

Numbers below were measured on an older base and are to be re-measured on `<TIP>`.

| PR | branch | src +/- | own test | scope |
|---|---|---|---|---|
| 01 | met/01-single-star-solar-abundances | 38/0 | `single_stars_metals`: solar, 2 passed (2 failed before) | broad |
| 02 | met/02-single-star-27-element-yields | 21/0 | `single_stars_metals`: 6 passed (3 failed before) | broad |
| 03 | met/03-sgchem-dgr-independent-of-metallicity | 15/0 | `treecol_pixels` + `single_stars_metals`: 2 passed (2 failed before) | narrow |
| 04 | met/04-single-star-dust-yields | 38/0 | `single_stars_metals`: dust yields passed (failed before) | broad |
| 05 | met/05-sgchem-dust-from-ism-dust-model | 27/3 | `single_stars_metals` + `treecol_pixels`: 2 passed (2 failed before) | broad |
| 06 | met/06-single-star-metal-diffusion-defaults | 28/0 | `single_stars_metals::test_27_element_diffusion` (Stella switch pending) | broad |

All are under 300 source lines. Source lines = `git diff --numstat` of `src/` without `Template_Config.sh` against the previous PR (tests and docs excluded), recomputed on the current branches.

Test constant: `X_H_PRIMORDIAL` is 0.7065 in `test_solar_abundances`, the `HYDROGEN_MASSFRAC` of SGChem builds (the code and the reference implementation, the reference, use the same formula; the bar is unchanged). Open question to the reviewer: primordial He = 0.2935 is above solar He = 0.2703, so He falls with Z below solar. Is this X_H = 0.7065 the right primordial value for the single-star tables, or should the primordial He be set differently?

How "flag off" is checked, for every PR: the base and the PR tip built in the same worktree path with the base `gmc_cooling` configuration (`SINGLE_STAR_STARFORGE_DEFAULTS`, `MAGNETIC`, `COOLING`, `METALS`, `COOL_METAL_LINES_BY_SPECIES`, ...) and with `FIRE_PHYSICS_DEFAULTS=3`, every object compared byte by byte, `objdump -d` for those that differ. In every PR the only differences are `core/init.o` (line-number immediates) and `core/main.o` (build-info string). A first base built in another directory showed `cooling.o` "instructions differ": that is the source-directory path compiled into the code (`GIZMO_SOURCE_DIR`), not code; the base is therefore built in the same worktree. Flag-off results below were measured on an older base (147 objects, 144 byte-identical in every row) and are to be re-run on `<TIP>`.

Regression at the tip (met/05; older base, to be re-run on `<TIP>`): all 11 single-star, TreeCol and table test directories, 54 passed; PR 01 changes the solar reference of every tables build. `sgchem_box` is unchanged by the series (SGChem-only builds untouched, objcmp above). Full suite: `<SUITE RESULT>`.

Tests: the base harness, SGChem network 5 as the chemistry (production model), Kokkos OpenMP on login cores. The metal ledger test `test/ledger_mass_metals` needs cr/04, so it ran on a scratch merge of met/05 and cr/05 (no branch). Metal ledger, measured (older base; to be re-measured on `<TIP>`): the gas metal gain equals the logged event metals to 1.3e-14 (FIRE layout, 27 elements) and 6e-15 (slice test `run_elements`) of the gas metal mass; the 27-element closure (total Z = sum of the 25 metal slots) holds to 1.4e-15 per cell after winds, a core collapse, an AGB death and two Type Ia, with cell splits. Diffusion off in all of them.

What the slice already did (not repeated here): slice PR 10 maps SN, wind, AGB and Type Ia yields onto the FIRE-3 slots (slot 0 = all table metals C..Zn, slots 1-10 He..Fe, other slots only diluted). Its metal ledger is exact already (above); its test bar (1e-10) was looser than the ledger.

---

## 01. Single stars: solar abundances from the stellar tables

Branch `met/01-single-star-solar-abundances`. Commits: test, change, docs.

**What and why.** In a build with the stellar tables, `SolarAbundances` (and so `InitMetallicity` and every "Z / Z_sun") use the tables' solar composition: Z_sun = 0.014 and the Asplund+09 present-day element pattern, as in the reference (`SolarAbundances[0] = 0.014` for Stella builds, `init.cc`; element pattern of the tables). On the base a single-star run started with `InitMetallicity 1` had Z = 0.02 with the older GIZMO pattern, so the stars' table metallicity, the chemistry's Z/0.014 and the stated solar value disagreed by 43 %.

**Evidence before** (older base; to be re-measured on `<TIP>`). `test/single_stars_metals::test_solar_abundances` (fresh start, InitMetallicity 1 and 0.3): total Z 0.02 instead of 0.014; C, N, O 32 %, 82 %, 44 % above the tables' solar values.

**Change.** `galaxy_sf/stellar_solar_abundances.h` (new: the tables' 27 solar mass fractions, the FIRE slot map), `core/init.cc` (under `GALSF_SINGLE_STARS_EVOLUTION_TABLES`: overwrite `SolarAbundances` 0-10 after the GIZMO defaults). No new flag, no parameter.

**Cost.** None: start-up only.

**Evidence after** (older base; to be re-measured on `<TIP>`). Both cases pass: Z = 0.014 x InitMetallicity, each slot = the tables' X_init at the Z = 0.014 node x InitMetallicity, He primordial + scaled excess (1e-10; the constants carry 10 digits, the table agrees to 2.5e-11).

**Flag off.** gmc and FIRE3: 147/147, 144 byte-identical; `core/init.o` same instructions (line numbers), `core/main.o` build info only (older base; to be re-measured on `<TIP>`). To be re-run on `<TIP>`.

**Limits (difference to the reference).** the reference also sets 0.014 for SGCHEM-only builds. Done that way here, the SGChem box test (`test_sgchem_cooling`, `test_sgchem_snapshot_io`) fails its 1e-8 box-vs-standalone bar (3e-7): at the new composition the round-off motions of the static box grow 25x faster (|v| 6e-8 vs 2e-9 km/s, densities drift 1e-8), which the standalone cell does not see; the chemistry itself agrees (a 1e-10 density change moves the standalone result by 1e-9). Kept to the tables flag here; SGChem-only builds are an open item for the chemistry series.

**Depends on / followed by.** Base. Followed by 02 (uses the header).

---

## 02. Single stars: 27-element yields in the extra passive metal slots

Branch `met/02-single-star-27-element-yields`. Commits: test, change, docs.

**What and why.** Option `GALSF_SINGLE_STARS_27_ELEMENT_YIELDS`: the 16 table elements outside the FIRE slots (F, Na, Al, P, Cl, Ar, K, Sc, Ti, V, Cr, Mn, Co, Ni, Cu, Zn) get their yields in the first 16 extra passive slots of `METALS_EXTRA_PASSIVE_SPECIES` (the general extra-slot PR). With it all 27 elements H..Zn are tracked as in the reference (`GALSF_RESOLVEDISM_METALS_INDIVIDUAL`). Total Z (slot 0) counts all table metals with the option on or off, so the shared slots are identical either way.

**Evidence before** (older base; to be re-measured on `<TIP>`). With 16 extra slots but no yields into them (the extra-slot PR alone), `test_27_element_closure` finds total Z != the sum of the 25 metal slots by up to 11.9 % per cell (around the Type Ia) and 0.38 % over the box; `test_27_element_supernova_yields` finds the 16 elements' gain 0; a fresh start leaves them 0.

**Change.**
- `declarations/precompiler_logic.h`: the option sets `METALS_EXTRA_PASSIVE_SPECIES=16` (or requires >= 16); needs `GALSF_SINGLE_STARS_EVOLUTION_TABLES` and `COOL_METAL_LINES_BY_SPECIES`.
- `galaxy_sf/single_star_feedback.cc`: the event's element ejecta into the extra slots (one loop, next to the FIRE slots).
- `core/init.cc`: a fresh start sets the 16 elements to the tables' solar values x Z/Z_sun, as the base slots.
- `galaxy_sf/stellar_solar_abundances.h`: the element list of the extra slots.

**Cost.** 16 x 8 B per particle (measured: particle structure 576 -> 704 B; cell structure unchanged at 416 B in this configuration; with metal-gradient builds also 3 x 16 x 8 B per cell). The slots ride the existing loops (exchange, diffusion, merge/split, IO). No collective, no work on quiet steps.

**Evidence after** (older base; to be re-measured on `<TIP>`). `test/single_stars_metals` (four stars: 20 Msun core collapse, 3 Msun AGB death, a white dwarf's Type Ia, a 35 Msun star's winds; the AGB white dwarf explodes too): closure per cell 1.4e-15 (bar 1e-12; start 1.2e-16); every slot gains; SN element gains vs the table SN ejecta scaled to the ejecta mass 5.5e-13 (bar 1e-11, Ni none in the table and none in the gas); total Z gain - event metals = 1.3e-14 of the gas metal mass (bar 1e-12); the same run with the 11-slot build: event log, slots 0-10, masses, energies, velocities, positions, star masses bitwise identical; a fresh start with the option sets the 16 elements to solar (1e-10). Compile-suite row builds.

**Flag off.** As 01 (gmc, FIRE3) (older base). To be re-run on `<TIP>`.

**Limits (difference to the reference).** the reference stores H explicitly (28 slots); here H stays implicit (FIRE layout), so 27 - 1 + total Z = 27 slots. An earlier design had 15 extra slots (Zn dropped); this PR keeps Zn (16 slots, +8 B) to track all 27 elements H-Zn as the reference does. Open question to the reviewer: 16 or 15 extra slots.

**Depends on / followed by.** 01, the extra-slot PR (`up/C4-metals-extra-passive-species`).

---

## 03. SGChem: dust-to-gas ratio independent of metallicity (opt-out)

Branch `met/03-sgchem-dgr-independent-of-metallicity`. Commits: test, change, docs.

**What and why.** The default dust-to-gas ratio already scales with the cell metallicity through one shared function (`cooling/sgchem/sgchem_dust_to_gas.h`, read by the chemistry, the TreeCol dust column and the single-star radiation pressure; cap 10, zero above 1e5 K, as the reference `resolvedism_dgr.h`). This adds the opt-out of the reference, `SGCHEM_DGR_INDEPENDENT_OF_METALLICITY`: DGR = `SGChemDustToGasRatio` x Milky Way at every Z (gizmo2017 behaviour with 0.01), in the same function, and the start-up stop of the reference when that puts more dust than metals in the gas.

**Evidence before** (older base; to be re-measured on `<TIP>`). `test_treecol_pixels::test_dust_to_gas_independent_of_metallicity`: with the flag set, the cells' dust-weighted masses still follow Z (up to 3x off); `test_dust_above_metals_stops`: a fresh start at InitMetallicity 0.3 with ratio 1 runs on.

**Change.** `sgchem_dust_to_gas.h` (one branch), `sgchem.cc` (start-up stop on fresh starts, message, `endrun` 78004), `precompiler_logic.h` (needs SGCHEM).

**Cost.** None.

**Evidence after** (older base; to be re-measured on `<TIP>`). Every cell's dust-weighted mass = 0.5 x its mass at Z = 0.1-3 solar (1e-12); the start-up stop with its message; InitMetallicity 1 runs.

**Flag off.** As 01 (gmc, FIRE3). SGChem build without the flag (treecol test config): 164/167 objects byte-identical, `sgchem.o`/`io.o`/`main.o` same instructions (line numbers, build info). Older base. To be re-run on `<TIP>`.

**Limits (difference to the reference).** the reference checks InitMetallicity on every start; here only on fresh starts, where InitMetallicity sets the gas metals (a snapshot start ignores it; the reference stops there too if InitMetallicity is not set, default 0).

**Depends on / followed by.** Chemistry series (SGChem), tc/02 (TreeCol node masses).

---

## 04. Single stars: dust yields into the ISM dust model (GALSF_ISMDUSTCHEM_MODEL)

Branch `met/04-single-star-dust-yields`. Commits: test, change, docs.

**What and why.** With the live dust model on (`GALSF_ISMDUSTCHEM_MODEL`, Choban+22), single stars bring no dust: their source model fills only the metal yields, so the dust slots are only diluted. This makes the stars' ejecta carry dust by the condensation rule of the reference (`resolvedism_dust_condensation`, Choban+22 efficiencies as quoted there): SN (all classes) carbon 0.15 of C, silicate 3.5e-4 and metallic iron 1e-3 of the key element; Type Ia iron 5e-3; AGB deaths and slow winds (< 100 km/s) by C/O (carbon = C not in CO; else silicate 0.8 with the O not in CO). The species, stoichiometry and key element are the model's own (`ISMDustChem_get_species_key_elem`, `ISMDustChem_get_elem_yields_from_species_yields`), so its transport, SN shock destruction, accretion and sputtering act on it unchanged.

**Evidence before** (older base; to be re-measured on `<TIP>`). `test_single_star_dust_yields` (20 Msun core collapse, 3 Msun AGB death, a white dwarf's Type Ia, dust-free gas): the gas gains no dust.

**Change.** `galaxy_sf/single_star_feedback.cc` only: `single_star_dust_yields` (element dust, sources 1 Type Ia / 2 SN / 3 AGB, species) called from the single-star branch of `particle2in_addFB_fromstars`.

**Cost.** A few operations per event, under the dust flag. (Dust model itself, measured: particle 576 -> 704 B, cell 416 -> 592 B with species 1+2.)

**Evidence after** (older base; to be re-measured on `<TIP>`). With the model's accretion, thermal sputtering and SN shock destruction set to 0 by its own scaling parameters (only the stars' dust is measured): element dust, species and sources equal the condensation of the table ejecta to 1.4e-15 (bar 1e-10): 0.0809 Msun, of it carbon 0.0787, silicate 1.96e-4, iron 2.0e-3. Compile-suite row builds.

**Flag off.** As 01 (gmc, FIRE3) (older base). To be re-run on `<TIP>`.

**Limits (difference to the reference).** the reference's own 5-field dust uses FeMgSiO4; here the silicate is the model's default composition (O 3.63, Mg 1.06, Si 1, Fe 0.571 per Si), and its key element is the scarcest of the four (the reference: Si, then limited by O, Mg, Fe; the same result when Si limits). The reference puts SN iron into the silicate iron when that is larger; here iron is its own species (model bit 2) or absent (bit 1 only).

**Depends on / followed by.** Slice PR 10-14 (events). Followed by 05.

---

## 05. SGChem: dust-to-gas ratio and gas-phase metals from the ISM dust model

Branch `met/05-sgchem-dust-from-ism-dust-model`. Commits: test, change, docs.

**What and why.** With `GALSF_ISMDUSTCHEM_MODEL`, the chemistry and the TreeCol dust column read the live dust instead of Z: DGR = dust mass fraction / 0.01 (Milky Way, cap 10, zero above 1e5 K) in the shared function, and the gas-phase C, O, Si (network 18 also Mg, Fe) = metals minus dust, as in the reference with its dust model (`chem_interface.cc`, `resolvedism_dgr.h`). The fixed RTZ depletion of network 18 is skipped then, as on the reference.

**Evidence before** (older base; to be re-measured on `<TIP>`). `test_chemistry_gas_phase_from_live_dust` (n_H = 1e3, 100 K, all carbon in carbonaceous dust, 20 kyr): x(CO) reaches 2.1e-7 (the fixed 50 % depletion leaves half the carbon in the gas); `test_treecol_pixels::test_dust_to_gas_from_live_dust`: dust-weighted masses follow Z, not the dust.

**Change.** `sgchem_dust_to_gas.h` (`sgchem_cell_dust_to_gas_ratio`), `sgchem_cell.cc` (inputs), `gravtree_treecol_pixels.h` (node payload), `precompiler_logic.h` (`#error` with `SGCHEM_DGR_INDEPENDENT_OF_METALLICITY`).

**Cost.** None.

**Evidence after** (older base; to be re-measured on `<TIP>`). x(CO) = 0 exactly, carbon stays in dust; dust-weighted mass = dust / 0.01 x m (1e-12).

**Flag off.** As 01 (gmc, FIRE3); SGChem build without the dust model: as 03 (the chemistry cell object byte-identical). Older base. To be re-run on `<TIP>`.

**Limits (difference to the reference).** Radiation pressure at a star still uses the star's metallicity; the reference with live dust uses the kernel average of the dust around the star (`DGR_around`, a density-loop field): open item.

**Depends on / followed by.** 03, 04 / 06 is independent of the dust PRs in content (stack order only).

---

## 06. Single stars: exact metal diffusion and the 1 % pair cap on by default

Branch `met/06-single-star-metal-diffusion-defaults`. Commits: test, change, docs.

**Stella switch pending.** This PR switches on, for single stars, an exact-conservation option for turbulent metal diffusion that is not in the base: `TURB_DIFF_METALS_EXACT_CONSERVATION` (each side of a pair moves half the exchange over its own step; the neighbour's share waits in a per-cell `Dyield_pending` array). The base asks for a different design (issue #35): the shorter-step endpoint owns the pair and applies it to both sides, `Dyield` is reused, positivity is enforced on the pair. That option exists as `METALS_EXACT_PAIRWISE_EXCHANGE` (separate PR on the base, branch `up/turb-diff-metals-exact-pairs`). This PR is to be re-pointed to it and is not for review as final. A first Stella cloud test with the new option (four realisations of a small cloud, 12 Myr) formed 1.04, 1.37, 1.92 and 2.65 times the stellar mass of the current option (the 1.37 case compares with a run that stopped early, at 11.5 Myr); the cause is under investigation.

**What and why.** With single stars, metal diffusion should conserve each element's mass to round-off with individual timesteps, as the reference implementation (stella_dev) does, and in the 27-element layout total Z must stay the sum of the 25 metal slots. This PR makes exact conservation and a cap on what one pair may move (1 % of the donor's species mass per pair, as the reference) the defaults when `TURB_DIFF_METALS` and `GALSF_SINGLE_STARS_FROM_IMF` are on, and closes total Z on the element slots inside the pair exchange.

**Evidence before** (older base). New test `single_stars_metals::test_27_element_diffusion`: winds, deaths and diffusion in the 27-element layout, against the same run without diffusion. Without this PR: total Z drifts from the slot sum by up to 1.27e-3 per cell, and the slot gains differ from the run without diffusion by 5.4e-6 (bars 1e-10 and 1e-6). FAIL.

**Change.** `precompiler_logic.h`: with `TURB_DIFF_METALS` and single stars, `TURB_DIFF_METALS_EXACT_CONSERVATION` and `TURB_DIFF_METALS_MAX_DONOR_FRACTION_PER_PAIR 0.01` are defined unless set. `turbulent_diffusion_functions.h`: the pair cap (donor loses at most that fraction of its species mass per pair, the same value from either side); with the 27-element yields, slot 0 (total Z) takes the sum of the exchanged metal slots (He excluded) before the exchange is applied. `Template_Config.sh` line, test README. 28 source lines added.

**Cost.** Through the option it switches on: `NUM_METAL_SPECIES` doubles per gas cell (`Dyield_pending`) and their ghost write-back; one atomic per species per pair on the neighbour side. The cap is a few flops per pair. No collectives added by this PR. The cost of the replacement option is measured in its own PR (+3 to +4 % per step on CPU, no per-cell array).

**Evidence after** (older base; the same values in a later run of the series). `test_27_element_diffusion`: closure of total Z on the slots at most 1.24e-15 per cell; slot gains equal those without diffusion to 5.9e-12. PASS.

**Flag off.** Only code under `TURB_DIFF_METALS` with single stars. To be re-run on `<TIP>`.

**Limits.** Depends on an option the base does not take as written (it also prints a per-step metal budget, which the #35 design rules out); the replacement gives a different stellar mass in small clouds (above). H stays implicit, as in the FIRE layout. The 1 % cap is a choice taken from the reference implementation.

**Depends on / followed by.** `up/C3` (merged into the base of this series; not in the upstream base), met/05 / pi series. Re-point to `METALS_EXACT_PAIRWISE_EXCHANGE` once the Stella cloud difference is understood.

---

## Dust: where Stella's model should live

The reference implementation (stella_dev) production runs without its dust model (`GALSF_RESOLVEDISM_DUST` off in its default config); the default is DGR from Z (shared function, PR 03). For live dust there are two homes:

| | A: Stella's own 5-field dust (the reference `GALSF_RESOLVEDISM_DUST`) | B: `GALSF_ISMDUSTCHEM_MODEL` (this series, PR 04-05) |
|---|---|---|
| model | C + silicate O/Mg/Si/Fe (FeMgSiO4); Choban+22 condensation; Nozawa+06 thermal sputtering; McKee 1989 SN destruction (0.4); logistic growth; seed 0.01 Z/0.014 (30 % C / 70 % olivine) | Choban+22 (the reference's efficiencies are quoted from it); species silicate, carbon, optional iron / O reservoir / inclusions; optional grain sizes; SN destruction 0.4 in the shocked mass; its own accretion and sputtering |
| what exists on the base | nothing | transport (diffusion, merge/split, exchange), IO, restart, SN shock destruction in the shared mechanical loop, accretion, sputtering, device kernels |
| new code | per-cell field (40 B), transport hooks, IO/restart, its process loop, SN destruction in the loop, seeding, the couplings: on the reference `resolvedism_dust.cc` (440 lines) + kernels (189) + `GALSF_RESOLVEDISM_DUST` hooks in 22 files; several PRs | 65 lines (PR 04 star dust, PR 05 chemistry and column) |
| memory | +40 B/cell | +176 B/cell, +128 B/particle (species 1+2) |

Proposal: B, with these points for the reviewer, because they change results:
1. The dust processes of the base take the temperature from GIZMO's own thermal model (`ThermalProperties`) and the dense-molecular fraction from its own fit, not SGChem's `Temp` and H2/CO; with SGChem the accretion and sputtering rates are then not consistent with the chemistry. A hook in `update_dust_processes` to read SGChem's state would be needed.
2. SN condensation: the population yields of the base (`ISMDustChem_get_SNe_dust_yields`) use 0.2 x scaling for silicate, carbon and iron in core-collapse SNe; the reference uses 0.15 / 3.5e-4 / 1e-3 (quoted there as Choban+22 "species"). For single stars this series uses the reference values; the silicate differs by ~570x. Which values are right is an open question to the reviewer and Choban.
3. Seeding: The reference seeds DGR 0.01 Z/0.014 (30 % carbon); the base `Initial_ISMDustChem_Depletion` seeds by Si depletion and `Initial_ISMDustChem_SiltoCarbRatio`: close with parameters, not identical.
4. Silicate composition: FeMgSiO4 vs the default mix of the base (element split of silicate dust differs).
5. Radiation pressure with live dust (`DGR_around`) is not ported (PR 05).

## Open items

- Exact metal diffusion: Stella switch pending (PR 06, `METALS_EXACT_PAIRWISE_EXCHANGE`, issue #35); small clouds formed 1.04 to 2.65 times more stellar mass with it in a first test.
- Primordial He / X_H = 0.7065 (see above): open question to the reviewer.
- SGChem-only builds (no tables) keep the GIZMO Z_sun = 0.02 while the chemistry uses 0.014 (PR 01): the SGChem box test must first be made robust to round-off motions (chemistry series).
- 16 vs 15 extra slots (Zn) (PR 02).
- Live-dust radiation pressure at the star (`DGR_around`), the SGChem state for the dust processes, the SN condensation efficiencies, the seed.
- The Hu dust variant of the reference (`EXPERIMENTAL_DUST_HU_MODEL`) is not ported.
- The wind channel's slow-wind dust (< 100 km/s) is in the code (PR 04) but no test star has slow winds.
