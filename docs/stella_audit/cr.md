# Cosmic rays into SGChem and the mass/metal/energy ledger: PR drafts (series cr)

Covers: the SGChem cosmic-ray ionisation rate (scaled with the stars' field, attenuated by the gas column, taken from the CR fluid, scaled with the recent SFR), ledger rows for mass, metals and feedback energy, and the exact energy exchange between the CR fluid and SGChem.

Order and dependencies: PR order of the whole work is slice2 -> chem (chem/12 before tc) -> tc -> gap -> cr (cr/04 before pi/06) -> met -> pi -> ext -> tr -> kj. This series stacks on `tc/06c-snapshot-radiation-fields` (after gap). The ledger row of pi/06 (photoionisation heating) goes into the row list of cr/04 when the PI series is stacked.

Base: `<BASE>`. Tip: `<TIP>`. Full-suite result: `<SUITE RESULT>`.

Numbers below were measured on an older base and are to be re-measured on `<TIP>`. Where a line says "earlier run on the audit line", the value comes from a run of the merged line before its tip; it too is to be re-measured on `<TIP>`.

| # | branch | src +/- | test | scope |
|---|---|---|---|---|
| 1 | `cr/01-sgchem-cr-rate-fluid-off` | 64/9 | `treecol_pixels::test_cosmic_ray_rate[CR_SCALE_WITH_G0, CR_BACKGROUND_IONRATE_ATTENUATED]` | narrow |
| 2 | `cr/02-sgchem-cr-rate-from-cr-fluid` | 22/4 | `treecol_pixels::test_cosmic_ray_fluid_rate` | narrow |
| 3 | `cr/03-isrf-cr-scale-with-recent-sfr` | 90/6 | `sgchem_recent_sfr::test_recent_sfr_scale`, `::test_quiet_calls_reduce_nothing` | narrow |
| 4 | `cr/04-ledger-mass-metals` | 71/6 | `ledger_mass_metals::test_ledger_mass_metals` | broad (conservation law) |
| 5 | `cr/05-ledger-feedback-energy` | 16/4 | `ledger_mass_metals::test_ledger_feedback_energy` | broad (conservation law) |
| 5b | `cr/05b-ledger-delivered-heat` | 6/3 | `ledger_mass_metals::test_ledger_books_what_reached_the_gas` | narrow |
| 6 | `cr/06-sgchem-cr-fluid-energy-exchange` | 51/4 | `sgchem_cr_fluid_energy::test_cr_energy_closes`, `::test_cr_loss_rate_as_stella_dev` | narrow |

All seven are under 300 source lines; no split needed. Source lines are recomputed against the previous PR (tests, docs, Template_Config excluded). cr/05b sits on cr/05 as its own branch; cr/06 was cut from cr/05, so 05b goes between them on the re-cut.

Which CR modes the reference implementation (stella_dev) uses in production, from its configs:
- Default config: constant `CosmicRayIonRate`; `CR_SCALE_WITH_G0`, `GHOSDT_G0_CR`, `CR_BACKGROUND_IONRATE_ATTENUATED` and `COSMIC_RAY_FLUID` are all off.
- Galaxy (tall box) config: `GHOSDT_G0_CR` + `CR_SCALE_WITH_G0`, no stars' field (no `G0_VARIABLE`).
- Cloud runs: `CR_BACKGROUND_IONRATE_ATTENUATED` with `GALSF_RESOLVEDISM_G0_VARIABLE`; two of the clouds use `COSMIC_RAY_FLUID` instead. `ShieldingLength` 50 pc in the galaxy runs.

---

## PR 1: SGChem cosmic-ray rate scaled with the stars' field, or attenuated by the gas column (`cr/01-sgchem-cr-rate-fluid-off`)

**What and why.** Without the CR fluid, SGChem uses the constant `SGChemCosmicRayIonRate`. The reference implementation (stella_dev) has two more modes that its production runs use: `CR_SCALE_WITH_G0` (rate scaled with the local FUV field) and `CR_BACKGROUND_IONRATE_ATTENUATED` (the background rate attenuated by the gas column, as in STARFORGE). This PR ports both. The column comes from the TreeCol pixel walk: the walk adds the gas mass to 12 per-pixel sums within `TreeColShieldingLength` and, at walk end, stores one number per cell, the pixel-mean attenuation factor.

**Evidence before.** On the base both flags are silently ignored: every cell gets the constant rate. The new test cannot find the rate in the cell dump (62 columns, test expects 63/64).

**Change.**
- `sgchem_cell.cc`: `sgchem_cr_ion_rate(i, G0)`, the cell's rate as in the reference `chem_interface.cc`:
  - `CR_SCALE_WITH_G0`: (G0/1.7) x rate, clamped to [1e-21, 2e-16];
  - `CR_BACKGROUND_IONRATE_ATTENUATED`: rate x `TreeColCRAttenuation` (replaces the scaling, as the reference).
- `gravtree_treecol_pixels.h`: under `CR_BACKGROUND_IONRATE_ATTENUATED`, a per-pixel gas column in the walk sums (node `gasmass`, already in the nodes); at walk end, the factor `f = mean_pix[1 if Sigma < 2.23e-3 g/cm^2 else exp(-Sigma/100) 2.23e-3/Sigma]`. `forcetree.cc`, `gpu_gravtree.cc`: pass `gasmass` to the accumulate call (host and device walk).
- `cell_data.h`: `float TreeColCRAttenuation` under the flag.
- `precompiler_logic.h`: `CR_SCALE_WITH_G0` needs the stars' field; the attenuated mode needs `GALSF_RESOLVEDISM_TREECOL_PIXELS`; both need `SGCHEM`.
- `treecol_pixels_test_dump.cc` (test only): writes the rate (and the factor).
- New flags: `CR_SCALE_WITH_G0`, `CR_BACKGROUND_IONRATE_ATTENUATED` (names of the reference, default off). No new parameters.

**Cost.** Attenuated mode: +4 B per gas cell; +12 doubles of per-target walk scratch (the gas column per pixel), the same deposit as the other pixel columns; nothing on quiet steps beyond the walk that runs anyway. Scaled mode: none. No collectives.

**Evidence after** (older base; to be re-measured on `<TIP>`). `test_cosmic_ray_rate` (1 rank, 12^3 cells, six stars):
- `CR_SCALE_WITH_G0`: every cell's rate equals the reference formula on the cell's G0 to 1e-12; cells on both sides of the 2e-16 clamp.
- `CR_BACKGROUND_IONRATE_ATTENUATED`: rate = rate0 x f (float storage, 1e-8); f against the reference formula on direct-sum 12-pixel gas columns within the shielding length: median 4.9e-3, p99 3.9e-2 (bars 1e-2/5e-2, the bars of the other pixel columns); f = 0.24-0.83, columns on both sides of 1e21 cm^-2.

**Flag off.** See "Flag off" at the end.

**Limits.** No difference to the reference in the formulas. Why not the existing `ColumnDensityBins`: they have 6 cube-face bins with no shielding-length cut, and are only on under `SINGLE_STAR_STARFORGE_DEFAULTS`; in a galaxy box the uncut in-plane columns would attenuate far more than the 50 pc cut of the reference. So the gas column rides the pixel walk.

**Depends on / followed by.** tc/06c / PR 2.

---

## PR 2: SGChem cosmic-ray rate and heating from the CR fluid (`cr/02-sgchem-cr-rate-from-cr-fluid`)

**What and why.** With `COSMIC_RAY_FLUID`, SGChem takes the ionisation rate from the fluid. Reuse before adding: the existing `Get_CosmicRayIonizationRate_cgs` (eos/cosmic_ray_fluid/cosmic_ray_functions.h), used by the base cooling. The fluid's energy density also goes to the chemistry for hadronic and Coulomb heating (`cr_energy_density`, already in the backend).

**Evidence before.** The base stops the build: `#error "SGCHEM: the coupling to RADTRANSFER and COSMIC_RAY_FLUID is not in this tree yet"`. Two Stella cloud runs use the fluid.

**Change.**
- `sgchem_cell.cc`: with the fluid, `sgchem_cr_ion_rate` returns the base function; `in.cr_energy_density` = `Get_CosmicRayEnergyDensity_cgs`.
- `chem_gizmo_backend.cc`: pass `cr_energy_density` whenever the fluid is on.
- `precompiler_logic.h`: the fluid is allowed with SGCHEM; `CR_SCALE_WITH_G0` and the attenuated mode are rejected with the fluid.
- No new flags or parameters.

**Cost.** None (one inline function per cell solve). No collectives.

**Evidence after** (older base; to be re-measured on `<TIP>`). `test_cosmic_ray_fluid_rate` (12^3 cells, e_CR 0.08-11 eV cm^-3, Z 0.1-3 solar): every cell's rate equals 1e-5 e_CR + 1e-21 Z/Zsun + 1e-19 Fe/Fe_sun (the base solar abundances) to 1e-12. Against the reference's 3e-17 (e_CR/1.8 eV cm^-3) + 1e-21 Z/0.014: the e_CR term is 0.961200 times the reference's (exactly the stated factor); the total rate is 0.961-1.092 x the reference's (the Fe term raises cells with little e_CR); an earlier run on the audit line gives 0.961-1.130.

**Flag off.** See "Flag off" at the end.

**Limits (deliberate differences to the reference, reuse of the base function).**
1. Normalisation 1.60e-17 per eV cm^-3 instead of 1.67e-17 (-3.9 %).
2. The base radioactive terms: 1e-21 Z/Zsun with `All.SolarAbundances[0]` (0.02 without FIRE-3 stellar evolution, the reference 0.014), plus 1e-19 Fe/Fe_sun (the reference none). With N_CR bins > 2 the Bethe-Bloch sum.
3. Hadronic/Coulomb heating is passed with any field (the reference only with `G0_VARIABLE`, `GHOSDT_G0_CR` or M1).
4. The reference's exact energy exchange (the solve's CR ionisation heating taken out of e_CR; the CR loss term reduced to Coulomb on free electrons) is PR 6. Until then the fluid loses the base loss terms and the gas gains the chemistry's heating, as in the base cooling.
5. `#error` for the fluid with `CR_BACKGROUND_IONRATE_ATTENUATED` (the reference adds the attenuated background on top; no production config combines them) and with `CR_SCALE_WITH_G0` (the reference: the fluid wins silently).

**Depends on / followed by.** PR 1 / PR 3.

---

## PR 3: SGChem field and cosmic-ray rate scaled with the recent SFR, a running sum (`cr/03-isrf-cr-scale-with-recent-sfr`)

**What and why.** The galaxy runs of the reference scale the uniform SGChem field and the CR rate with the star formation rate of the last 30 Myr (Hu+21; `GHOSDT_G0_CR` + `CR_SCALE_WITH_G0`): F = [M(stars < 30 Myr)/30 Myr] / [M_gas,start/4 Gyr], G0 = G0_LW = G0_dust = max(F SGChemG0, 0.324e-2), zeta = F SGChemCosmicRayIonRate. The reference loops over all particles with an `MPI_Allreduce` at every star-formation call. Here the young-star mass is a running sum, which avoids both.

**Evidence before.** the reference `sfr_eff.cc` has one O(NumPart) loop and one `MPI_Allreduce` per SF call. On the base the flag does not exist (the test build lacks the `RECENT_SFR` lines).

**Change.**
- `sfr_eff.cc`: a rank-local `std::map` birth time -> mass, not tied to particles (domain exchange does not matter: only the total over ranks is used). Births add this call's new star mass; mass changes of young single stars add (winds, deaths); entries leave at 30 Myr (the oldest first, O(1) on quiet calls). One scan of all particles at start-up (fresh start, snapshot or restart). A "changed" count rides the existing SF count reduction (2 -> 3 long long); the double sum is reduced only when some rank changed, or stars were born. F is set at every SF call, as the reference.
- `single_star_feedback.cc`: after the feedback loop, each source's mass change goes to the sum (young stars only).
- `init.cc`: `All.Mgas0` (gas mass at start), as the reference. `global_data_all_struct.h`: `Mgas0`, `FactorG0`.
- `sgchem_cell.cc`: the uniform field from F; with `CR_SCALE_WITH_G0`, zeta = F x rate (unclamped, as the reference).
- New flags: `ISRF_AND_CR_RATE_SCALE_WITH_RECENT_SFR` (default off; needs SGCHEM and GALSF, excludes the stars' field), `TEST_RECENT_SFR_SCALE_PRINT_FULL_SCAN_EVERY_CALL` (test only). No new parameters.

**Cost.** Per rank one map entry per distinct birth time in the last 30 Myr (16 B + node). Quiet SF call: one map lookup, no collective beyond the existing count reduction (now 3 instead of 2 long long). One `MPI_Allreduce` of one double on calls where a young star was born, changed mass, or passed 30 Myr. Flag off: the count reduction and all code unchanged.

**Evidence after** (older base; to be re-measured on `<TIP>`). `test/sgchem_recent_sfr` (8^3 box, SGChem, single stars, 2 ranks):
- `test_recent_sfr_scale` (two births, three 1 Msun stars passing 30 Myr, a young 8 Msun star losing wind mass; 65 SF calls): F equals the full scan (the reference's sum) at every call; in this run bitwise. G0 and zeta follow F. F drops at each 30 Myr crossing (3.78 -> 3.52 -> 3.25 -> 2.99) and with the wind mass loss.
- `test_quiet_calls_reduce_nothing` (two births, one 1 Msun star passing 30 Myr, an 8 Msun star of 31.5 Myr losing wind mass; 33 SF calls): 2 reductions, exactly on the two calls where F changed (births, expiry); 31 quiet calls add no collective; F = full scan throughout.

**Flag off.** See "Flag off" at the end.

**Limits (differences to the reference).**
1. Mass changes of young stars that are not single stars (no star record, e.g. stars from ICs without one) are not followed; the reference sums the current mass of every type-4 particle. In Stella runs all stars carry a record.
2. Only SGChem's field is scaled; the reference also scales `RT_ISRF_BACKGROUND` and the LEBRON CR background in builds without SGCHEM (not Stella production).
3. `#error` with the stars' field (the reference ignores the factor silently when `G0_VARIABLE` is on).

**Depends on / followed by.** PR 2 / PR 4.

---

## PR 4: ledger rows for mass and metals (`cr/04-ledger-mass-metals`)

**What and why.** The ledger of chem/10 (`OUTPUT_MASS_METAL_ENERGY_LEDGER`, `ledger.txt`, exact per-thread sums reduced once per snapshot) gets the mass and metal rows of star formation and stellar feedback. Reused, not duplicated: the same exact sums, fold, single `MPI_Reduce` and restart record. Rows are now an X-list (`LEDGER_ROWS`) in `proto.h`, each compiled with its module, with its own output unit (erg or Msun).

**Evidence before.** On the base the ledger has only the two chemistry rows (`ledger.cc` names array), and it is initialised only by `sgchem_init`; nothing books star formation or feedback, so a mass or metal leak in the shared feedback loop or in the single-star source model is invisible. Also: on a start from a snapshot (start flag 2) the file had no header line (`fresh = (RestartFlag == 0)`).

**Change.**
- `proto.h`: rows as `X(id, name, unit, state)`; SGChem rows under `SGCHEM`; state rows (gas and star mass and metal mass) and stars formed (mass, metals) under `GALSF`; mass and metals delivered to the gas under `GALSF_FB_MECHANICAL`; single-star ejecta mass and metals taken from the stars, remnant reset, star metal mass lost under the single-star feedback flags.
- `ledger.cc`: state rows summed over the local particles at output (exact sums, same reduction); per-row units; header whenever the file is new.
- Booking sites (one or two `ledger_add` each, during the module's own work): `sfr_collect_gas.cc` (LYRA birth), `sfr_eff.cc` (other births), `mechanical_fb.cc` (`verify_and_assign_local_mechfb_integrals`, the host apply loop: mass and metal mass each cell receives), `single_star_feedback.cc` (finalize: mass the loop took from each star, its metals from the yields, the reset to the table remnant).
- `begrun.cc`: `ledger_init` for every build with the flag (before restart files are read).
- `precompiler_logic.h`: the star rows need METALS.
- No new flags or parameters.

**Cost.** Flag on: +12 rows x 72 int64 per thread and rank; one exact addition per booking (a birth, a source, a receiving cell); one loop over local particles per snapshot (state rows), inside `savepositions`, which is collective already. No collective added, nothing on quiet steps, nothing with the flag off.

**Evidence after** (older base; to be re-measured on `<TIP>`). `test_ledger_mass_metals` (8^3 box, SGChem, single stars, 2 ranks, 0.29 Myr, 6 rows; 3.5 Msun formed, 2.31 Msun ejecta incl. an AGB death, 0.054 Msun metals ejected): relative to the total mass, residuals of star mass 1.7e-18, gas mass 5.2e-17, ejecta delivered vs taken 2.5e-18; relative to the total metal mass: 1.9e-18, 2.8e-16, 4.7e-18 (bar 1e-12). State rows = snapshot sums (1e-12).
Regression on the series tip: `sgchem_box::test_sgchem_ledger` and `::test_sgchem_restart` pass (the no-SGCHEM variant of that test builds with zero rows); `treecol_pixels::test_host_equals_device[2-2,1-4]`, `::test_chemistry_inputs` and the three CR tests pass.

**Flag off.** See "Flag off" at the end.

**Limits (differences to the reference).** the reference books its total-metal ledger with a full scan and collectives every step (`resolvedism_inject_fb_energy`); here the bookings ride the module loops and the reduction happens at snapshots. Not ported as rows: per-FIRE-slot metals, diffusion-pending mass (exact diffusion is a separate opt-in PR), LYRA collected/returned quanta (the collection is exact by construction; failed draws keep zero mass).

**Depends on / followed by.** PR 3 (stack order only; independent in content) / PR 5. The photoionisation heating row of pi/06 goes into the same row list (`LEDGER_ROWS`) when the PI series is stacked.

---

## PR 5: ledger rows for feedback energy (`cr/05-ledger-feedback-energy`)

**What and why.** Energy per channel for stellar feedback: what the source model asks the loop to couple (1/2 dm v_ej^2 per event, v_ej as the loop's input), and what the apply loop gives each cell: heat and kinetic energy as applied. Heat is always injected in the step (mechanical feedback runs in `run.cc`), so there is one heat row.

**Evidence before.** No energy budget for feedback.

**Change.** `proto.h` (feedback energy rows), `mechanical_fb.cc` (`ledger_add` in the apply loop for heat and kinetic energy; the kinetic row from the momenta as applied), `single_star_feedback.cc` (one `ledger_add` in finalize for the energy asked). No new flags or parameters.

**Cost.** As PR 4: a few more rows, one exact addition per receiving cell or source; nothing between snapshots.

**Evidence after** (earlier run on the audit line, heat always injected). `test_ledger_feedback_energy` (same run as PR 4): 1.08e47 erg booked by t = 0.29 Myr (winds of the 8 Msun star, the AGB death); the gas receives it as heat plus 4.04e45 erg kinetic; residual (gas - booked)/booked at most 5.6e-13 in all six rows (bar 5 %). On the older base, with the heat then sent to a deferred rate, the same test gave +9.7e-5 before the AGB death and +2.8e-3 after.

**Flag off.** See "Flag off" at the end.

**Limits (differences to the reference).** the reference books SN booked/coupled and KE-residual rows inside its own injector; here the rows sit in the shared loop. Not yet booked: CR energy injected and clipped, MHD numerical term and kick floor, chemistry rows for the hydro half-steps. Open items.

**Depends on / followed by.** PR 4 / PR 5b.

---

## PR 5b: ledger heat row from what reached the gas (`cr/05b-ledger-delivered-heat`)

**What and why.** PR 5 books the heat row from the energy the loop hands each cell (`TE_injected`). What the cell keeps can differ: dilution of the ejecta mass, cold ejecta and the injection rule decide the final internal energy. This PR books the heat row as the change of the cell's thermal energy over the apply step, m_f u_after - m_0 u_before, so the row is the heat the gas actually received. Feedback heat is always injected in the step on the base; there is one heat row, no deferred row.

**Evidence before** (older base). New test `test_ledger_books_what_reached_the_gas`: a 20 Msun star, 1 Myr old, with a steady wind in 1e7 K ionised gas (n_H = 10, 12^3 cells of 6 Msun, 2 ranks), against a twin with no wind owed (cooling cancels). On that base, where heat then went to a deferred rate, the heat row read 0 while the ledger had 9.85e45 erg of ejecta energy and the snapshots showed a gain of -1.06e43 erg: heat row plus kinetic row did not match what the gas gained, and the test failed.

**Change.** `mechanical_fb.cc` (`verify_and_assign_local_mechfb_integrals`): the cell's thermal energy before the apply step is kept, and `ledger_add(LEDGER_E_FB_THERMAL_NOW, mf u - TE_0)` after it, replacing the booking of `dTE`. Test and README. No flags or parameters. 6 source lines added, 3 removed.

**Cost.** One double per receiving cell inside the existing apply loop, with the ledger flag on. No collectives, nothing on quiet steps, nothing with the flag off.

**Evidence after** (earlier run on the audit line, heat always injected). `test_ledger_books_what_reached_the_gas` (narrow): gain from snapshots 9.832e45 erg; heat row 9.847e45 plus kinetic 2.9e40; difference 0.15 % of the ejecta energy (bar 2 %). PASS. `test_ledger_feedback_energy` and `test_ledger_mass_metals` pass in the same run (PR 4, PR 5).

**Flag off.** Only code under `OUTPUT_MASS_METAL_ENERGY_LEDGER`. To be re-run on `<TIP>`.

**Limits.** The row includes everything that changes u inside the apply step for that cell (it is the cell's change, not the source's offer); it does not see heat added later in the step. Not booked: CR energy injected and clipped, MHD term, kick floor (open items). On the current base the "before" of this test is to be re-measured: the run above predates the base's change to always-injected heat.

**Depends on / followed by.** PR 5 / PR 6.

---

## PR 6: exact energy exchange between the CR fluid and SGChem (`cr/06-sgchem-cr-fluid-energy-exchange`)

**What and why.** With the CR fluid, the chemistry heats the gas by CR ionisation (20 eV per ionisation, zeta from the fluid), but the fluid paid for it only through the generic loss term of the base (Coulomb + ionisation, 0.57 n_H0). As in the reference (`sgchem.cc`, `cosmic_ray_utilities.cc`), the solve's ionisation heating (the fluid's share of zeta, without the radioactive terms) now comes out of e_CR, and the fluid's own loss is hadronic plus Coulomb on free electrons only. Ledger rows book both channels.

**Evidence before.** `test_sgchem_cr_fluid_energy` on PR 5 plus the rows (older base; to be re-measured on `<TIP>`): the fluid loses 1.44 x the reference rate (the base ionisation-loss term) and nothing for the ionisation heating the gas receives. A port of the reference removal alone takes nothing out: the backend's T_CMB floor call (`cxx_cool_func(..., iflag_cmb = 1)`, called after the main call in every network RHS) resets `c.cr_ion_heat = 0`, so the integrated heating the driver reports is always 0 (ionisation row 0). The reference has the same code (`chem_cool_func.cc:430`, `chem_rhs_net5.cc:223`), so on the reference the exact exchange is dead: its CR-fluid cloud runs drop the ionisation term from the fluid loss and remove nothing instead, i.e. the gas gets the ionisation heating for free.

**Change.**
- `chem_cool_func.cc`: the T_CMB floor call leaves `cr_ion_heat` alone (the network RHS and rates are unchanged; the chemistry is bit-identical, only the reported integral is now the applied heating).
- `sgchem_cell.cc`: after the solve, f x (CR ionisation heating) x volume comes out of all CR bins (energy and flux, evolved and predicted, scaled together; capped at e_CR, first cap printed), f = 1 - zeta_radioactive / zeta with the base radioactive terms.
- `cosmic_ray_functions.h` (the non-spectral `CR_cooling_and_losses`): under SGCHEM, Coulomb on free electrons only.
- Ledger (with `OUTPUT_MASS_METAL_ENERGY_LEDGER`): `E_CR_erg` (state), `E_cr_losses_erg` (booked around the base call in `cooling.cc`), `E_cr_ionisation_from_fluid_erg`, `E_chem_cr_ionisation_heating_erg`.
- No new flags or parameters.

**Cost.** Per cell solve: a few flops and one pass over the CR bins. No memory, no collectives.

**Evidence after** (older base; to be re-measured on `<TIP>`). `test/sgchem_cr_fluid_energy` (8^3 box, n_H ~ 30 cm^-3, x_H2 = 0.4, 2 eV cm^-3 CRs, 2 ranks):
- `test_cr_energy_closes`: by 0.29 Myr E_CR = 0.747 E_CR,0; fluid losses 0.179, ionisation energy taken by the chemistry 0.074 (fluid share of the chemistry's CR ionisation heating 0.9971-0.9974); the CR closure residual is 3.2e-4 E_CR,0, the unbooked CR-gas exchange at the kicks (the same 3.25e-4 in the run before the fix).
- `test_cr_loss_rate_as_stella_dev`: fluid losses per interval / the reference's rate = 0.984-1.007 (bar 2 %). The base loss term without this PR (+0.57 n_H0 ionisation Coulomb) would be +12.8 % at these conditions (computed from the formula, not run).
- Chemistry unchanged by the backend fix: `E_chem_heating` row identical to the last digit in both runs.
- Before the fix: ionisation row 0, closure test fails.

**Flag off.** Only `#ifdef SGCHEM` / `COSMIC_RAY_FLUID` code, plus `chem_cool_func.cc` (SGChem only). Objcmp 147/147 same instructions in both configs (older base; to be re-measured on `<TIP>`). To be re-run on `<TIP>`.

**Limits (differences to the reference).**
1. The T_CMB fix above (a the reference bug; to be ported back to the reference).
2. Free electrons: the base call passes `Ne` (all free electrons from the chemistry); the reference passes x(H+) (`TracAbund[1]`). The Coulomb term is < 1e-4 of the hadronic term in molecular gas either way.
3. The radioactive share uses the base terms (PR 2), the reference its own 1e-21 Z/0.014.
4. Spectral CRs (`CRFLUID_EVOLVE_SPECTRUM`): the multibin loss routine is unchanged (the reference also only changed the single-bin routine).
5. Not booked: the CR-gas exchange at the kicks (adiabatic work, streaming), as on the reference; it is the residual of the CR closure (reported).

**Depends on / followed by.** PR 5b / -.

---

## Tests on the series tip

All tests rerun on the series tip after the last rebase: 13/13 pass (older base; to be re-measured on `<TIP>`): `treecol_pixels` CR tests (3) plus `test_slab_shielding` and `test_chemistry_inputs`; `ledger_mass_metals` (2); `sgchem_recent_sfr` (2); `sgchem_cr_fluid_energy` (2); `sgchem_box` ledger and restart (2). Full suite: `<SUITE RESULT>`.

## Flag off

Method: the base and each branch tip built in same-length directories with the base `test/gmc_cooling/Config.sh` and `test/fire/Config.sh` (FIRE-3 with mechanical feedback), compared with objcmp. Result (older base; to be re-measured on `<TIP>`). To be re-run on `<TIP>`.

| branch | gmc_cooling | fire |
|---|---|---|
| cr/01 | 147/147 same instructions (145 byte-identical; `cooling.o`, `main.o` strings only) | same |
| cr/02 | as cr/01 | same |
| cr/03 | 147/147 same instructions; `__LINE__` shifts in `init.o`, `sfr_eff.o` | same |
| cr/04 | 147/147 same instructions; `__LINE__` shifts in `begrun.o`, `init.o`, `sfr_eff.o` | + `mechanical_fb.o` |
| cr/05 | as cr/04 | as cr/04 |
| cr/06 | as cr/05 (`cooling.o` same instructions) | as cr/05 |

The walk objects (`forcetree.o`, `gpu_gravtree.o`) are byte-identical in both configs.

## Open items

- The the reference exact CR exchange is dead (T_CMB call zeroes `cr_ion_heat`): The reference needs the one-line fix; its CR-fluid cloud runs gave the ionisation heating for free.
- CR ledger rows (injected, clipped), MHD numerical term and kick floor, chemistry rows for the hydro half-steps.
- `test_cosmic_ray_rate[ATTENUATED]` runs 1 rank, 1 thread; the host/device walk equality for the new gas column is not tested separately (`test_host_equals_device` covers the walk without the CR flag).
- Ledger: restart is covered for the chemistry rows by `sgchem_box::test_sgchem_restart`; the new rows ride the same record but have no restart test of their own. 1 vs 4 ranks/threads not run for the new rows.
- Questions to the reviewer: should the ledger live in its own file or in the base energy file; may the young-star "changed" count ride the SF count reduction; is reusing the base CR rate including the Fe term acceptable.
- pi/06's photoionisation heating row: add it to the row list when the PI series is stacked on this one.
