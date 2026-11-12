# final/ series: test hooks, diagnostics and small single-star features

**Covers.** Single-star feedback and star-formation test hooks, ledger and log diagnostics, SGChem checks, hot-gas timestep handling, and a few small features of the reference implementation (stella_dev). Most PRs are test hooks or diagnostics and are short; each has one narrow test unless marked broad.

**Order.** Series order: slice2 -> chem (chem/12 before tc) -> tc -> gap -> cr (cr/04 before pi/06) -> met -> pi -> ext -> tr -> kj. Inside this file every branch is stacked on final/05 and the branches are independent of each other unless "Depends on" says otherwise. Base `<BASE>`, tip `<TIP>`.

**Numbers.** Results below were measured on older bases (4d8a5e9e0 and others) and are to be re-measured on `<TIP>`. The texts for the tree luminosity refresh, the wind composition fix and the causal wake (at the end) were measured on a later base (39df537c4, e57b6001b) and are likewise to be re-measured on `<TIP>`. Flag-off object comparisons are collected in the last section; every "Flag off" line below is to be re-run on `<TIP>`. Full-suite result: `<SUITE RESULT>`.

**Missing text.** final/01 (TREECOOL reader) is now upstream (f4f9bed98). Only the comoving SGChem UV background floor of final/01 remains and has no text yet (see the MISSING list).

**Open question for the reviewer.** final/43 and final/51 each add a hot-gas timestep clamp (see their texts). Should both stay, or only one?

---

## final/03-thermalproperties-from-chemistry

Source 23 lines. Test: broad.

**What and why.** With SGChem, `ThermalProperties` read the cached composition: temperature from u, n(H0) counting H2 as neutral, electrons including metals. The reference implementation returns the stored chemistry state, and its mean weight uses the chemistry's own H/He mix.

**Evidence before.** `test/sgchem_thermal_state` on the older base: 3 of 3 fail (n(H0) 0.993 against 1 - 2 x(H2) - x(H+) = 0.693; mean weight 1.33 against 1.52; temperature differs from the stored one).

**Change.** `eos/eos_functions.h`: SGChem branch in `ThermalProperties` (stored Temp, atomic H, n_e = n(H+) = x(H+), same fallback before the first solve as the reference) and in `Get_Gas_Mean_Molecular_Weight_mu` (X = 1/(1 + 4 ABHE), metals not counted; off under `GALSF_SGCHEM_VARIABLE_XH_AND_ABHE`). The reference applies the mix only with the 27-element layout; here it applies to every SGChem build. New test flag `TEST_SGCHEM_PRINT_THERMAL_PROPERTIES` (default off).

**Cost.** No bytes. No collectives. No work on quiet steps.

**Evidence after.** `test/sgchem_thermal_state`: 3 passed. Bars are the expressions of the reference functions, not a run of the reference.

**Flag off.** Builds without SGCHEM unchanged in source (branches inside `#if defined(SGCHEM)`); SGChem builds change on purpose. To be re-run on `<TIP>`.

**Depends on / followed by.** `<BASE>`.

## final/04-tidal-timestep-prefac-test

Source 0 lines (test only). Test: narrow.

**What and why.** The reference has tidal-step prefactors for gas and non-gas (default 1, no production parameter sets them). The base has one Config factor, `TIDAL_TIMESTEP_PREFAC` (c84215640). This branch adds no code, only a test that the base factor does what the reference's gas prefactor does.

**Evidence before.** `test/tidal_timestep_prefac` on the base without c84215640: fails, the factor is ignored (shortest gas step 5e-4 with and without it).

**Change.** Test only. Not covered by the base factor: a separate factor for non-gas particles, and values above 1; the reference's prefactor also scaled the tree-force cadence, the base factor does not. Default 1 in both.

**Cost.** None.

**Evidence after.** With c84215640: passed; f = 1e-3 gives 1.25e-4 against the bar (1.25e-4, 2.5e-4] at r = 11.9 pc; without the factor 5e-4.

**Flag off.** No source change.

**Depends on / followed by.** c84215640 in the base.

## final/05-mass-balance-breakdown

Source 92 lines. Test: broad.

**What and why.** The reference locates mass and metal leaks by bracketing each operation. The base has the checkpoint file at four tags (`OUTPUT_MASS_BALANCE_STEP_CHECKPOINTS`); missing were the finer tags, the metals of the diffusive exchange not yet applied, the per-element lines and the pair-flux against floor breakdown.

**Evidence before.** `test/ledger_step_breakdown` with base code: 3 failed (no feedback, gas-collection or diffusion tags, no `diff_unpack_breakdown.txt`, no MCBAL lines).

**Change.** `system/ledger.cc`, `core/proto.h`, `core/accel.cc`, `domain/domain.cc`, `galaxy_sf/sfr_eff.cc`, `hydro/hydro_force_loop.cc`, `hydro/hydro_toplevel.cc`, `declarations/precompiler_logic.h`. New flag `OUTPUT_MASS_BALANCE_BREAKDOWN` (default off, needs the checkpoints). `M_Z_ledger` gains `Dyield_pending`, with a column `M_Z_inflight` at the end of the file; the checkpoint now loops over local cells, as ghosts are live at the diffusion apply.

**Cost.** No bytes. Under the flag one Allreduce and four Reduces per checkpoint plus one Allreduce per hydro step: diagnostic only.

**Evidence after.** `test/ledger_step_breakdown`: 3 passed; max |dM_bary|/M 4.35e-16, max |dM_Z_ledger|/M_Z 1.32e-15, pair sum / M_Z 4.7e-25 with gross exchange 5.6e-8, floor 0. `test/ledger_step_checkpoints` passes unchanged.

**Flag off.** The `M_Z_ledger` change is inside `OUTPUT_MASS_BALANCE_STEP_CHECKPOINTS`. To be re-run on `<TIP>`.

**Depends on / followed by.** `<BASE>`. All later branches of this file stack on it.

## final/06-star-merger-products

Source 107 lines. Test: narrow.

**What and why.** Colliding table stars kept the survivor's age and birth Z and took their contact radius from a main-sequence fit. The reference gives the product the state of a rejuvenated main-sequence star (Hurley, Pols & Tout 2002 eq. 80) and takes radii from the tables. Used by the KETJU clouds.

**Evidence before.** `test/ketju_star_merger_product`: 2 failed. Product birth Z 0.0142, bar 0.010375; age 5.66e7 yr, bar 1.68e6 yr. A 3 Msun giant (28.5 Rsun, pericentre 0.096 AU, fit contact 0.013 AU) does not collide.

**Change.** `galaxy_sf/stellar_evolution_tables.{h,cc}`: `t_MS_start`, `t_MS_end`, `logR_cm` read under `KETJU_MERGE_STARS`. `gravity/ketju/ketju_coupling.cc`: contact radius from the tables with single-star tables. `gravity/ketju/ketju_star_mergers.{h,cc}`: product state (drawn mass, birth Z, formation time, wind clock in the star record). No new flag; on with `KETJU_MERGE_STARS`, `GALSF_SINGLE_STARS_FROM_IMF`, `GALSF_SINGLE_STARS_EVOLUTION_TABLES`, `GALSF_FB_MECHANICAL`. As in the reference, a star without a record then has contact radius 0.

**Cost.** Two 1650-double tables per rank and one 760 kB float table per node, under `KETJU_MERGE_STARS`. Three more doubles per merged victim in the existing point-to-point message. No collectives. No work on quiet steps.

**Evidence after.** `test/ketju_star_merger_product`: 2 passed; drawn mass 8, Z 0.010375, age 1684139 yr against 1684080 yr at the merger (60 yr of run time afterwards; bar 1e-3). `test/ketju_star_collision` (no tables) 3 passed.

**Flag off.** All inside the KETJU flags. To be re-run on `<TIP>`.

**Depends on / followed by.** final/05.

## final/07-timeline-restart-gas-test

Source 0 lines (test only). Test: narrow.

**What and why.** Regression test for a restart with a raised TimeMax (issue 23): gas on 2 ranks, no rebuild at the restart. No code change: the failure of issue 23 does not appear on this branch in this test.

**Evidence before.** `test/timeline_restart_tree` passes on the unchanged code: halved run against plain restart max|dx| 0, max|dv| 0, no endrun 90000103.

**Change.** Test only.

**Cost.** None.

**Evidence after.** Same run. A candidate fix (halve node `Ti_current`, `Ti_lastkicked`, `Previous_Ti_Current`, `PhotoionizedUntil`) gives the same result, so no fix is included: there is no failing run.

**Flag off.** No source change.

**Depends on / followed by.** final/05.

## final/08-energy-statistics-with-ledger

Source 7 lines. Test: narrow. Cut on an older base; merges onto the current line with one `Template_Config` line to combine (with final/16).

**What and why.** `TimeBetStatistics` is read only with DEVELOPER_MODE and reset to 1e10 at start-up without it, so energy.txt has one row. The reference reads the cadence and keeps it when its energy ledger is on; this PR does the same.

**Evidence before.** `test/ledger_mass_metals::test_energy_statistics_at_their_cadence` (built with OUTPUT_ADDITIONAL_RUNINFO, TimeBetStatistics 0.05, TimeMax 0.3): 1 row in energy.txt, bar 6.

**Change.** `core/begrun.cc`: with `OUTPUT_MASS_METAL_ENERGY_LEDGER` the tag is registered (default 1e10 when not set) and the reset is skipped. The ledger test's Config gains OUTPUT_ADDITIONAL_RUNINFO, which only adds output files.

**Cost.** None beyond the statistics the user asks for.

**Evidence after.** 6 rows (t = 0 to 0.25). The first bar said 7 rows; that counted a row at TimeMax, which the run never writes. The docstring says so.

**Flag off.** Unchanged without the ledger flag. With it (Stella cloud config) the statistics cadence follows the parameter file, on purpose. To be re-run on `<TIP>`.

**Limits.** Only with `OUTPUT_MASS_METAL_ENERGY_LEDGER`; without it `TimeBetStatistics` stays a DEVELOPER_MODE parameter, as in the base. The content of energy.txt is unchanged.

**Depends on / followed by.** final/05 (originally cut on final/07).

## final/10-test-hooks-births-sn-times

Source 54 lines. Test: narrow.

**What and why.** Three test-only switches for the single-star feedback: a star born with a given age, supernovae at forced times, and a time before which the feedback finds no gas. The last one tests the retry of an event that found no gas (the star keeps the event pending and fires it in a later step).

**Evidence before.** `test/single_stars_birth_age` and `test/single_stars_forced_events` on final/05: 3 of 3 fail. With `TestStarInitAgeYr` 2e6 yr the formation time is unchanged (difference 0, bar 2.045 code time); in the forced-time run no supernova happens in 8 kyr (stars of 20 Msun, 1 Myr old, table death at about 10 Myr); with no gas until 0.004 there is no event.

**Change.** `core/begrun.cc`, `declarations/global_data_all_struct.h`, `galaxy_sf/sfr_collect_gas.cc`, `galaxy_sf/single_star_feedback.cc`, `galaxy_sf/mechanical_fb.cc`. New flags, all default off and for tests only:
- `TEST_SINGLE_STARS_BORN_WITH_GIVEN_AGE` (`TestStarInitAgeYr`: formation time back-dated at birth).
- `TEST_SINGLE_STARS_SN_AT_FORCED_TIMES` (`TestSNTime0`, `TestSNSpacing`, `TestNumStars`: star ID k of the first `TestNumStars`, first generation, explodes at `TestSNTime0 + (k-1) TestSNSpacing`; winds still owed come with the death).
- `TEST_SINGLE_STARS_NO_GAS_IN_REACH_UNTIL_TIME` (`TestFBNoGasUntilTime`: before it the feedback loop runs with no sources, so every event stays pending).

Difference to the reference: when the reference finds no gas in the kernel it widens the kernel by 1.4 and measures again in the same step (up to 3 tries); here the event waits for the next step (see final/36 for the widening).

**Cost.** A few doubles in `All` under the flags. No collectives. No work on quiet steps.

**Evidence after.** `test/single_stars_birth_age`: formation time earlier by 2e6 yr in code time to 1e-9. `test/single_stars_forced_events`: 2 passed; supernovae at 0.002 and 0.005 (bar: forced time to within 1.5e-4); with no gas until 0.004 the first star fires at 0.004 after 32 steps without deposit, the second at 0.005 without delay, mass residual below 3e-14 Msun, mass conserved to 1e-12. Bars are the definitions of the parameters in the reference, written before the runs.

**Flag off.** Changes inside the flags, except the local `dead` variable in `single_star_eventrates`, which is computed as before. To be re-run on `<TIP>`.

**Depends on / followed by.** final/05.

## final/11-test-hook-single-star-wind-pinned

Source 21 lines. Test: narrow.

**What and why.** A test switch that fixes the wind of single stars to a constant rate and speed (reference `TestFixedWindMdot`, `TestFixedWindVel`), so that the wind power 1/2 Mdot v^2 is exactly known (analytic wind bubble tests).

**Evidence before.** `test/single_stars_wind_pinned` on final/10: fails; with `TestFixedWindMdot` 1e-4 the star loses no mass in the run (0.0 Msun; bar: more than 0.5 Msun, equal to Mdot times the time to the last delivery).

**Change.** `galaxy_sf/single_star_feedback.cc` (wind age without pre-main-sequence delay, mass `Mdot (t - t_last)`, the table's surface composition, speed `max(v, 10 km/s)`), `core/begrun.cc`, `declarations/global_data_all_struct.h`. New flag `TEST_SINGLE_STARS_WIND_PINNED_RATE_AND_SPEED` (default off) with parameters `TestFixedWindMdot` [Msun/yr], `TestFixedWindVel` [km/s]. Valid before the death of the star; the winds still owed at death come from the tables.

**Cost.** Two doubles in `All` under the flag. No collectives. No work on quiet steps.

**Evidence after.** `test/single_stars_wind_pinned`: 1 passed. The star loses 0.96275 Msun, equal to Mdot times the time to the last delivery to 1e-14; the last delivery is within 3e-4 code time of the star's age; the energy gained by gas and star is 0.9974 of 1/2 M v^2 (bar 1 %). Two earlier bars (1e-3, then 2 % against a twin without wind) were wrong: the hydro does not conserve the total energy of the box better than about 4e3 code units, varying from run to run, which is 2 % of the energy of a wind of 1e-5 Msun/yr at 2000 km/s; the test wind is 1000 times stronger in energy.

**Flag off.** Inside the flag. To be re-run on `<TIP>`.

**Depends on / followed by.** final/10.

## final/12-test-hooks-single-star-luminosity

Source 22 lines. Test: narrow.

**What and why.** Two test switches of the single-star radiation (reference `TestFixedLbol`, `TestFUVZero`): every star shines at a constant bolometric luminosity split into fixed band shares (radiation-pressure and attenuation tests), or no star emits (quiet medium experiments).

**Evidence before.** `test/single_stars_radiation_pinned` on final/10: 2 of 2 fail. With `TestFixedLbol` 1e39 erg/s the radiation-pressure loop still offers the table's momentum (2487 against 3089 in code units, 19 % off); with no radiation switch it offers momentum in all 33 rows (sum 4.97, bar 0).

**Change.** `galaxy_sf/single_star_luminosity.cc`, `galaxy_sf/single_star_feedback.cc` (ionising age), `core/begrun.cc`, `declarations/global_data_all_struct.h`. New flags, default off, for tests only: `TEST_SINGLE_STARS_LUMINOSITY_PINNED` (`TestFixedLbol` [erg/s]; from birth, FUV 0.5 (8-13.6 eV band), NUV 0.2, optical 0.3, Lyman-Werner 0.15, no ionising photons; takes precedence), `TEST_SINGLE_STARS_RADIATION_OFF` (all bands, bolometric and ionising rates zero).

**Cost.** One double in `All` under the flag. No collectives. No work on quiet steps.

**Evidence after.** `test/single_stars_radiation_pinned`: 2 passed. Pinned: momentum offered per time equals the reference estimator for the pinned split (absorbed luminosity times (1 + tau_IR) over c) to 2 % at all 31 steps and is constant in time; off: nothing offered.

**Flag off.** Inside the flags. To be re-run on `<TIP>`.

**Depends on / followed by.** final/10 (stack only).

## final/13-test-hooks-adiabatic-twin-wind-dump

Source 36 lines. Test: narrow.

**What and why.** Two test switches: leaving out the cooling and chemistry update (the adiabatic twin of a cooling run; reference `RESOLVEDISM_ADIABATIC_NOCOOL`), and a dump of the particle state before and after the feedback deposits of a wind (`TEST_DUMP_STATE_AROUND_WIND_DEPOSIT`), used to check conservation at the deposit itself.

**Evidence before.** `test/sgchem_adiabatic_twin` on final/10: the temperature of the box changes by a factor 1.16 in 31 kyr (bar: unchanged to 1e-6); `test/single_stars_wind_dump`: no dump file is written.

**Change.** `core/run.cc` (the cooling call is skipped; `#error` unless `COOLING_OPERATOR_SPLIT` and no `TRANSPORT_SUBCYCLE_COOLING`, else the hydro work would be lost with it), `galaxy_sf/mechanical_fb.cc`. New flags, default off, for tests only: `TEST_SKIP_COOLING_AND_CHEMISTRY_UPDATE`, `TEST_DUMP_STATE_AROUND_WIND_DEPOSIT` (per-rank file `wind_dep_dump.<rank>.bin`, format in the test README).

**Cost.** No bytes. The dump: one integer reduction in feedback steps and a write of all particles twice in steps with a mass return; for tests only. No work on quiet steps.

**Evidence after.** `test/sgchem_adiabatic_twin` `test_adiabatic_twin`: passed, every cell keeps its temperature to 1e-6 and its tracked species bitwise (the same box cools to 32 K in `sgchem_box`). `test/single_stars_wind_dump`: passed; 128 deposits per rank as (before, after) pairs; at each the total mass is conserved and the gas gains the star's loss to 1e-14 Msun, the momentum changes by less than 5 % of (mass returned x 3000 km/s). The bar of 1e-12 of the mass returned, written first, was wrong (a rounding of one cell is larger for the late, small deposits of 6e-5 Msun).

**Flag off.** Inside the flags. To be re-run on `<TIP>`.

**Depends on / followed by.** final/12 (stack only).

## final/14-single-stars-instant-sn-before-time

Source 11 lines. Test: narrow.

**What and why.** The reference has a test mode in which stars of 8 Msun and up explode at once while the time is below `TimeInstantSN` (supernova-driven box experiments that do not wait for stars to age).

**Evidence before.** `test/single_stars_instant_sn` on final/10: `test_instant_sn` fails (no supernova in 4 kyr for two 20 Msun stars 1 Myr old); `test_no_effect_at_zero` passes.

**Change.** `galaxy_sf/single_star_feedback.cc` (in `single_star_eventrates`: such a star is dead, and its owed winds come with the death), `core/begrun.cc`, `declarations/global_data_all_struct.h`. New flag `GALSF_SINGLE_STARS_INSTANT_SN_BEFORE_TIME` (default off), parameter `TimeInstantSN` [code time]; 0 has no effect.

**Cost.** One double in `All` under the flag. No collectives. No work on quiet steps.

**Evidence after.** `test/single_stars_instant_sn`: 2 passed; both stars explode in the first steps (time below 3e-4), mass conserved to 1e-12, and with `TimeInstantSN` 0 there is no supernova in the run.

**Flag off.** Inside the flag. To be re-run on `<TIP>`.

**Depends on / followed by.** final/13 (stack only); the forced-time hook of final/10 shares the local variable in `single_star_eventrates`.

## final/15-sgchem-solar-metal-fraction

Source 4 lines. Test: narrow.

**What and why.** With SGChem the solar total metal mass fraction is 0.014 (Asplund+09 present-day, as in the reference); builds with SGChem but without the stellar tables still used 0.02, so `InitMetallicity` in solar units and every Z/Z_sun gave different values.

**Evidence before.** `test/sgchem_box` `test_sgchem_solar_metallicity` on final/10: all cells have Metallicity[0] = 0.02 (bar 0.014).

**Change.** `core/init.cc`: `SolarAbundances[0] = 0.014` under `SGCHEM`. Builds with the stellar tables keep the table composition as before; builds without SGChem are unchanged. `Template_Config.sh` note.

**Cost.** None.

**Evidence after.** `test_sgchem_solar_metallicity`: passed (every cell 0.014 to 1e-12).

**Flag off.** Builds without SGCHEM unchanged in source; SGChem builds change on purpose. To be re-run on `<TIP>`.

**Depends on / followed by.** final/14 (stack only).

## final/16-ledger-kick-floor-mhd-rows

Source 42 lines. Test: narrow.

**What and why.** Two rows of the energy ledger: the gas energy the kick limiter and the temperature floors add (hydro kick and CR kick), and the energy the MHD source terms create (Powell and Dedner terms and the missing -B^2/2 dV/dt). With them the energy books can be closed in runs with floors and magnetic fields.

**Evidence before.** `test/sgchem_adiabatic_twin` `test_kick_floor_row` on final/10: no column `E_kick_floor_erg` in `ledger.txt`.

**Change.** `core/proto.h` (rows `E_kick_floor_erg` with SGCHEM, `E_mhd_numerical_erg` with SGCHEM and MAGNETIC, under `OUTPUT_MASS_METAL_ENERGY_LEDGER`), `core/kicks.cc`, `hydro/hydro_toplevel.cc`, `eos/cosmic_ray_fluid/cosmic_ray_functions.h`, `system/ledger.cc`, `declarations/cell_data.h`. No new flag.

**Cost.** One double (8 B) per gas cell with MAGNETIC, SGCHEM and the ledger (the MHD rate between the hydro step and the kick); nothing otherwise. No new collectives (the rows join the existing ones of the ledger at snapshots). No work on quiet steps.

**Evidence after.** `test/sgchem_adiabatic_twin` `test_kick_floor_row`: passed (2 of 2 with the twin test). In a box with a velocity wave and the floor at the start energy the floor row grows by 4.5e46 erg (bar: above 1e-2 of the first-order estimate of 1.3e47 erg), is exactly 0 with the floor at 2.73 K, and the MHD row stays below 1e-6 of the energy gained (2e31 against 1.07e47 erg; B = 1e-12, so this checks that build, booking and a zero hydro creation work, not the magnetic terms). The two rows do not close the gas energy in this floor-dominated case: the gain is 2.4 times the floor row. The row counts the clamps at the kicks, as in the reference; the rest is added where the predicted energy is clamped, outside the kicks. Bars that the rows close the energy to 2 %, and one of 0.75 to 1.25 of the estimate, were written first and were wrong. The magnetic terms themselves have no test.

**Flag off.** Rows exist only under the ledger flag with SGCHEM. To be re-run on `<TIP>`.

**Depends on / followed by.** final/15 (stack only); the test uses the adiabatic twin of final/13.

## final/17-event-logs-imf-donor-log

Source 32 lines. Test: narrow.

**What and why.** The per-donor log of the reference, `IMFinfo.txt`: for each new single star, which gas cells gave it how much mass. Needed to trace where the metals and mass of a star come from.

**Evidence before.** `test/single_stars_imf_donor_log` on final/10: both tests fail (no `IMFinfo.txt`).

**Change.** `galaxy_sf/sfr_collect_gas.cc`, from the grant records of the gas collection (`SscGrantRecord`), merged over the passes, failed draws left out. Under `GALSF_SINGLE_STARS_EVENT_LOGS`.

**Cost.** No bytes per particle. In steps that spawned a star: one gather and one gatherv of the rows to rank 0, after the existing birth-log gather. No work on other steps.

**Evidence after.** `test/single_stars_imf_donor_log`: 2 passed; per star the donor masses add up to the collected mass of the birth log (1e-12), each pair once, identical rows on 1 and 2 ranks.

**Flag off.** Inside `GALSF_SINGLE_STARS_EVENT_LOGS`. To be re-run on `<TIP>`.

**Depends on / followed by.** final/16 (stack only).

## final/18-event-logs-sfr-rism-windows

Source 30 lines. Test: narrow.

**What and why.** `sfr_rism.txt` of the reference: the star-formation rate over the last 50, 40, 30, 20, 5 and 1 Myr and the number of stars, written only in steps that build the tree, with one reduction.

**Evidence before.** `test/single_stars_sfr_windows` on final/10: fails (no `sfr_rism.txt`).

**Change.** `galaxy_sf/single_star_event_logs.cc` (`single_star_sfr_windows_write`), a call in `core/run.cc` inside the tree-build branch, `core/proto.h`. Under `GALSF_SINGLE_STARS_EVENT_LOGS`.

**Cost.** No bytes. One reduction of 7 doubles in steps that build the tree; none on other steps (the loop over the stars is a local pass on those steps).

**Evidence after.** `test/single_stars_sfr_windows`: passed; two 20 Msun stars 3 Myr old give N = 2 and the rates 40 Msun / window for 50 to 5 Myr and 0 for 1 Myr in all rows (to the 7 digits the file prints). In this small box the tree is rebuilt every step, so the number of rows equals the number of steps; that other steps do nothing follows from the call site.

**Flag off.** Inside `GALSF_SINGLE_STARS_EVENT_LOGS`. To be re-run on `<TIP>`.

**Depends on / followed by.** final/17 (stack only).

## final/19-sgchem-step-cooling-diagnostics

Source 18 lines. Test: narrow.

**What and why.** The per-step diagnostic line of the reference cooling (`SGCHEM_DIAG`): number of active cells, energy the update removed, running sum, temperature and density ranges.

**Evidence before.** `test/sgchem_step_diagnostics` on final/10: fails (no line).

**Change.** `cooling/cooling.cc`. New flag `SGCHEM_PRINT_STEP_COOLING_DIAGNOSTICS` (default off). The line is rank 0's own cells and the energy lacks the reference's adjustment for photoionisation heating (there is no such booking in the cooling here).

**Cost.** One double per active cell for the duration of the call under the flag. No collectives. No work on quiet steps.

**Evidence after.** `test/sgchem_step_diagnostics`: passed; 9 lines (the call at the start and 8 steps), 512 cells each, the energies of the first 8 add up to the thermal energy lost between the snapshots to 1e-5.

**Flag off.** Inside the flag. To be re-run on `<TIP>`.

**Depends on / followed by.** final/18 (stack only).

## final/20-single-stars-sf-converging-flow

Source 6 lines. Test: narrow.

**What and why.** The reference can require a converging flow for star formation: no star in a cell whose velocity divergence is zero or positive. The single-star criterion here has no such option.

**Evidence before.** `test/single_stars_sf_converging_flow` with the flag set on the code before the change: 1 failed; 2139 cells have a star-formation rate and some of them have div v >= 0 (the flag is ignored).

**Change.** `galaxy_sf/sfr_eff.cc` (3 lines inside the single-star block), `declarations/precompiler_logic.h` (error without `GALSF_SINGLE_STARS_SF_CRITERION`). New flag `GALSF_SINGLE_STARS_SF_CONVERGING_FLOW_ONLY` (default off). Uses the existing `Particle_DivVel`.

**Cost.** No bytes. No collectives. No work on quiet steps.

**Evidence after.** `test/single_stars_sf_converging_flow`: 1 passed; no forming cell has div v >= 0, both signs of div v are present, the forming set is the cells with a passing Jeans test and div v < 0.

**Flag off.** Inside `#ifdef GALSF_SINGLE_STARS_SF_CONVERGING_FLOW_ONLY`. To be re-run on `<TIP>`.

**Depends on / followed by.** final/05.

## final/21-single-stars-sf-instant-above-density

Source 34 lines. Test: narrow.

**What and why.** The reference can form stars at once above a hydrogen density (`nHcutoffSF`), whatever the Jeans test and the rate say, with the mass drawn from the number-weighted Kroupa IMF (the mass-weighted draw accepted with M_min / M). There is no such option here.

**Evidence before.** `test/single_stars_sf_instant_above_density` with the flag set on the code before the change: 1 failed, 1 passed. With a cutoff of 1e3 cm^-3 on a lattice at 7e3 cm^-3 (efficiency 1e-12) no star forms (0 of the bar of 3891 births); the passing check is the negative control (cutoff above the density: no star).

**Change.** `galaxy_sf/sfr_eff.cc` (7 lines in the single-star block), `galaxy_sf/single_star_imf.h` (`single_star_imf_draw_numberweighted`, keyed like the existing draw), `declarations/global_data_all_struct.h`, `core/begrun.cc`, `declarations/precompiler_logic.h`. New flag `GALSF_SINGLE_STARS_SF_INSTANT_ABOVE_DENSITY` (default off, needs `GALSF_SINGLE_STARS_SF_CRITERION`), parameter `nHcutoffSF` [cm^-3]. The reference draws the number-weighted mass with its own random stream; here the stream is the keyed generator, the distribution is the same.

**Cost.** One double in `All` under the flag. No collectives. No work on quiet steps.

**Evidence after.** `test/single_stars_sf_instant_above_density`: 2 passed. Cutoff 1e3: at least 95 % of the 4096 cells form in the first step, 65-85 % of the drawn masses are below 0.5 Msun (number-weighted Kroupa 76 %, mass-weighted 26 %); cutoff 1e9: no star.

**Flag off.** All inside the flag (the new draw function is an unused inline function). To be re-run on `<TIP>`.

**Depends on / followed by.** final/20 (stack only).

## final/22-treecol-pixel-columns-output

Source 41 lines. Test: broad.

**What and why.** The reference can write the per-pixel gas column of the gravity-tree walk to the snapshot (`IO_TREE_RAD_PROJECTION`). Here the pixel columns exist (dust-weighted gas, H2, CO per cell, 12 pixels) but only a test dump shows them.

**Evidence before.** `test/treecol_pixel_columns_output` with the flag set on the code before the change: 2 failed (no `TreeColumnPixels` dataset in the snapshot).

**Change.** `file_io/io.cc` (the new block in the seven places an output field needs, same pattern as `OUTPUT_CHEM_PHOTORATES`), `declarations/allvars.h`, `declarations/precompiler_logic.h` (error without `GALSF_RESOLVEDISM_TREECOL_PIXELS`). New flag `OUTPUT_TREECOL_PIXEL_COLUMNS` (default off). Field `TreeColumnPixels`, (N, 3 x pixels) in g/cm^2, order [column][pixel]. The reference writes only the total gas column; here the three columns the chemistry uses are written.

**Cost.** No bytes in memory; 144 bytes per cell in the snapshot at 12 pixels, under the flag. No collectives. No work on quiet steps.

**Evidence after.** `test/treecol_pixel_columns_output`: 2 passed (1 and 2 ranks). Dust-weighted column summed over the pixels against the direct sum over the cells: median 6.5e-3, 99th percentile 3.3e-2 (bars 1e-2 and 5e-2, the tree-error bars of `treecol_pixels`).

**Flag off.** All inside `#ifdef OUTPUT_TREECOL_PIXEL_COLUMNS`. To be re-run on `<TIP>`.

**Depends on / followed by.** final/21 (stack only).

## final/23-frozen-test-stars

Source 3 lines. Test: narrow.

**What and why.** For tests of the tree radiation and columns the star sources must stay fixed while the gravity tree (which carries the radiation walk) runs with self-gravity on. The reference pins every star in the kick for this.

**Evidence before.** `test/frozen_test_stars` with the flag set on the code before the change: 1 failed; two 1e3 Msun stars 1 pc apart with 3 km/s each move by up to 1.34 pc in 0.5 time units.

**Change.** `core/kicks.cc` (3 lines at the top of `do_the_kick`: star particles get zero `GravAccel` and `Vel` and the function returns). New flag `TEST_FROZEN_STARS_ZERO_VELOCITY_AND_ACCELERATION` (test only, default off).

**Cost.** No bytes, no collectives, no work on quiet steps.

**Evidence after.** `test/frozen_test_stars`: 1 passed; maximum position change 0.0 pc, maximum velocity 0.

**Flag off.** All inside the flag. To be re-run on `<TIP>`.

**Depends on / followed by.** final/22 (stack only).

## final/24-static-plummer-test-potential

Source 30 lines. Test: broad (energy in an analytic potential).

**What and why.** The reference has a fixed Plummer potential at the origin (`TEST_GRAVITY_STATIC_PLUMMER`) as a stand-in for a static gas potential in energy-conservation tests of direct-sum star integrators.

**Evidence before.** `test/static_plummer_test_potential` with the flag set on the code before the change: 1 failed; three massless collisionless particles feel no force, their energies change by up to 100 % and they end 7 pc from the reference orbits.

**Change.** `gravity/analytic_gravity.h`: `GravAccel_TestStaticPlummer` (with the existing `analytic_gravity_adds_to` guard, also the tidal tensor under `COMPUTE_TIDAL_TENSOR_IN_GRAVTREE`). New flag `TEST_GRAVITY_STATIC_PLUMMER` (test only, with `GRAVITY_ANALYTIC`), mass and scale from `TEST_GRAVITY_STATIC_PLUMMER_MSUN` (1000) and `_PC` (0.5).

**Cost.** No bytes, no collectives, no work on quiet steps.

**Evidence after.** `test/static_plummer_test_potential`: 1 passed. Energy changes 8.5e-4, 4.9e-5, 2.0e-3 of |E|, angular momentum 5e-15 or less, circular orbit radius 1.00015 pc, final positions 0.0005, 0.0018, 0.0004 pc from a scipy reference integration (T = 10). The first bars (1e-3, 0.02 pc) were written before any run and were too tight for a second-order leapfrog with the time-step criterion; the test docstring gives the final ones (1e-2, 0.1 pc), which still separate the force law from no force by two orders of magnitude.

**Flag off.** All inside the flag. To be re-run on `<TIP>`.

**Depends on / followed by.** final/23 (stack only).

## final/25-sgchem-print-input-pool-violations

Source 15 lines. Test: narrow.

**What and why.** Print, before the chemistry solve, every cell whose species hold more of an element than its pool (CO above min(C, O), hydrogen above 1, ...), with the full context. These compositions come from upstream operations (feedback injection, diffusion, dust) and make the solver stall; the clamp that follows hides them.

**Evidence before.** `test/sgchem_input_pool_violations` with the flag set on the code before the change: 1 failed; 0 violating cells reported out of the 3/4 of the 4096 cells whose IC violates a pool.

**Change.** `cooling/sgchem/sgchem_cell.cc` (15 lines before the clamp). New flag `TEST_SGCHEM_PRINT_INPUT_POOL_VIOLATIONS` (test only, default off; networks 5, 17, 20). It prints `SGCHEM_INPUT_VIOLATION: ID=.. viol=..` with bits 1 CO, 2 hydrogen, 4 helium, 8 deuterium (the last two for net 17) and the pools and species; the clamp is unchanged. The flag name differs from the reference's `GALSF_SGCHEM_DEBUG_INPUT`, which is a test hook, not a model option.

**Cost.** No bytes, no collectives; under the flag one printf per violating cell per solve.

**Evidence after.** `test/sgchem_input_pool_violations`: 1 passed; the cells reported are those of ID % 4 = 1, 2, 3, each with the bit of its violation, none of the consistent cells.

**Flag off.** All inside the flag. To be re-run on `<TIP>`.

**Depends on / followed by.** final/24 (stack only).

## final/28-habing-band-energy-script

Source 0 lines (script and test). Test: narrow.

**What and why.** The G0 bands of the chemistry take the Draine (1978) field divided by 1.7 as the unit; the four band energy densities in `cooling/sgchem/chem_interface.h` come from a script of the reference that is not here, so the constants cannot be checked or re-derived.

**Evidence before.** `test/habing_band_energy_densities` without the script: 5 failed (script missing).

**Change.** `scripts/habing_band_energy_densities.py` (copied from the reference, 50 lines): Habing and Draine band integrals (Draine 2011 eqs. 12.6-12.7) and the Mathis, Mezger and Panagia (1983) grain-heating shares. No source change.

**Cost.** None.

**Evidence after.** `test/habing_band_energy_densities`: 5 passed; Habing 6-13.6 eV is 5.29e-14 erg/cm^3 within 1 %, and the four constants in `chem_interface.h` equal Draine / 1.7 of the script to 5e-4.

**Flag off.** Not applicable (script and test only).

**Depends on / followed by.** final/05.

## final/29-read-ic-double-treecol-nongas-tests

Source 0 lines (tests only). Test: narrow.

**What and why.** Two regression tests of the reference that were missing here: HDF5 ICs keep the precision they are stored in (positions, velocities and masses of a double IC read back exactly, a float IC reads exactly its float values), and the TreeCol column densities of a gas cell do not change when a gasless star sits among the gas (the reference once kept the previous leaf's gas mass for non-gas leaves). Both hold here already; no source change.

**Evidence before.** Not applicable, the tests pass on the code as it is. Every HDF5 float block is staged as double (`get_input_float_bytes`), so velocities and masses are exact without `INPUT_IN_DOUBLEPRECISION` or `INPUT_POSITIONS_IN_DOUBLE`.

**Change.** `test/read_ic_double` (64 stars, float64 and float32 ICs, the t = 0 snapshot against the IC, bit for bit) and `test/treecol_nongas` (8^3 gas cells with and without a star of 1e-10 of a cell's mass; the six column bins of ten cells from the `GIZMO_TREECOL_DIAG` print at step 0 through `RT_USE_TREECOL_FOR_NH=6`, no bin changing by more than 1e-5 of the largest). The reference's treecol_nongas test links its own driver, which the harness here does not have; this one reads the existing print.

**Cost.** None.

**Evidence after.** `test/read_ic_double`: 1 passed; `test/treecol_nongas`: 1 passed.

**Flag off.** Not applicable (tests only).

**Depends on / followed by.** final/05.

## final/30-cooling-threaded-serial-crosscheck

Source 18 lines. Test: narrow.

**What and why.** The chemistry cells are solved by an OpenMP loop over the active cells. The reference can check, in the run itself, that this threaded pass gives the same result as a serial pass over the same inputs, byte for byte (shared state touched by the solve, thread-local storage, order dependence).

**Evidence before.** `test/sgchem_threaded_cooling_crosscheck` with the flag set on the code before the change: 1 failed; the run reports no comparison (0 of the at least 4 expected lines).

**Change.** `cooling/cooling.cc`, host cooling path only (3 + 12 lines around the threaded loop): copies of the active cells, a serial pass, restore, the threaded pass, `memcmp` of particle and cell structs; a difference stops the run (`endrun(8859)`) with the first ID and energies. New flag `TEST_COOLING_COMPARE_THREADED_PASS_WITH_SERIAL_PASS` (test only, default off). Global counters the solve updates (the ledger) see the cells twice with the flag. The reference flag is `SGCHEM_OMP_CROSSCHECK`.

**Cost.** Under the flag four copies of the active particle and cell structs per step (about 3 kB each per cell) and a second solve of every cell; no collectives; nothing without the flag.

**Evidence after.** `test/sgchem_threaded_cooling_crosscheck` (SGChem net 5, 2 ranks x 2 threads, 4096 cells, 4 steps): 1 passed; every step reports that the threaded pass equals the serial pass.

**Flag off.** All inside the flag. To be re-run on `<TIP>`.

**Depends on / followed by.** final/05.

## final/31-sgchem-restart-temperature

Source 28 lines. Test: narrow.

**What and why.** A snapshot carries no `Temp` (the chemistry temperature); after a snapshot restart every gas cell starts at 100 K until its first chemistry solve, and the first force pass reads that value. The reference has an experimental option that derives it from the energy and abundances before the first force pass.

**Evidence before.** `test/sgchem_restart_temperature` with the flag set on the code before the change: 1 failed; no cell reports a start temperature (the cells keep 100 K).

**Change.** `cooling/sgchem/sgchem_cell.cc` (`sgchem_startup_temperature`: the temperature of the cell step without the solve, 15 lines), `core/init.cc` (one loop over the gas for start flag 2 after the first density pass), `core/proto.h`. New flag `SGCHEM_TEMPERATURE_FROM_CHEMISTRY_AT_SNAPSHOT_RESTART` (experimental, default off); with `TEST_SGCHEM_PRINT_THERMAL_PROPERTIES` (final/03) it also prints the start temperature (`RESTARTTEMP ID T`).

**Cost.** No bytes; one pass over the gas at start; no collectives; no work on quiet steps.

**Evidence after.** `test/sgchem_restart_temperature` (SGChem net 5, atomic gas of 940 to 7500 K, 2 ranks): 1 passed; every cell reports (2/3) u m_p (1 + 4 ABHE) / (k_B (1 + ABHE + x(H+))) to 8e-6 (bar 1e-3) and none is at 100 K.

**Flag off.** All inside the flag. To be re-run on `<TIP>`.

**Depends on / followed by.** final/05 (uses the print flag of final/03).

## final/32-element-abundance-block

Source 75 lines. Test: narrow.

**What and why.** The ICs of the Stella runs (cloud, galaxy and zoom) carry the metals as a block `ElementAbundance` of the 27 element mass fractions H..Zn instead of `Metallicity`. The code reads and writes `Metallicity` slots in its own layout (FIRE slots 1-10 and, with `GALSF_SINGLE_STARS_27_ELEMENT_YIELDS`, 16 extra slots), so such an IC gives zero metals.

**Evidence before.** `test/element_abundance_block` with the flag set on the code before the change: 1 failed; the gas lattice and star IC carry a 27-element block and zero `Metallicity`, the restart snapshot has no `ElementAbundance` and the metal slots stay zero.

**Change.** `file_io/io.cc` (the seven places an output block needs, same pattern as `OUTPUT_CHEM_PHOTORATES`), `file_io/read_ic.cc` (fills the slots), `galaxy_sf/stellar_solar_abundances.h` (the two slot maps, 24 lines), `declarations/allvars.h`, `declarations/precompiler_logic.h` (error without `GALSF_SINGLE_STARS_27_ELEMENT_YIELDS`). New flag `GALSF_SINGLE_STARS_ELEMENT_ABUNDANCE_BLOCK` (default off). Gas and stars (types 0 and 4); the reference also lists type 5, which sinks here do not need. H is derived (1 minus the rest) on output and ignored on input; total Z is the sum over C..Zn. The block is read at snapshot restart (start flag 2) only, as every block after `IO_U`, and follows `Metallicity` in the read order, so it wins.

**Cost.** 216 bytes per particle in the snapshot with double output (27 values), none in memory; no collectives; no work on quiet steps.

**Evidence after.** `test/element_abundance_block`: 1 passed (2 ranks); metal slots of gas and star equal the block through the slot map to 1e-12, and the written block equals the one read to 1e-12.

**Flag off.** All inside the flag (the two inline slot maps are compiled only with it). To be re-run on `<TIP>`.

**Depends on / followed by.** final/05.

## final/33-orion-kokkos-systype

Source 26 lines (plus 7 lines of `.gitignore`). Test: narrow.

**What and why.** The Stella runs on Orion build with a private make include through `HOME_GIZMO`, because `src/Makefile` has no Orion entry; an unknown SYSTYPE falls through to the compiler defaults without a message (plain `mpiCC`, no Kokkos).

**Evidence before.** `test/makefile_systype_orion` (the commands `make -n` would run, nothing built) with SYSTYPE `Orion_Kokkos` on the old Makefile: 1 failed; the compile line is `mpiCC -Wall -g`, no Kokkos.

**Change.** `src/Makefile`: block `Orion_Kokkos` (25 lines: `mpicxx -std=c++20`, OpenMP, Kokkos 5.0.0 OpenMP install at `KOKKOS_HOME_ORION`, overridable), `src/Makefile.systype`: one comment line. Raven and Viper are not added: they need the Kokkos CUDA and HIP installs and module lists of those machines.

**Cost.** None.

**Evidence after.** `test/makefile_systype_orion`: 1 passed (`mpicxx -std=c++20`, `-fopenmp`, `-I$KOKKOS_HOME_ORION/include`, `-lkokkoscore`, `-L.../lib64`).

**Flag off.** Not applicable (a new Makefile block selected only by that SYSTYPE; other systypes unchanged).

**Depends on / followed by.** final/05.

## final/34-photoion-recombination-at-cell-temperature

Source 25 lines. Test: broad (Stromgren photon balance at two temperatures).

**What and why.** Scale the recombination cost of a cell above 1e4 K with the chemistry's case-B rate at its temperature, as the reference can. Off in production.

**Evidence before.** `test/photoion_recomb_temperature` without the flag: 1 failed; N <alpha_B> 695.5 (100 K gas) against 484.0 (gas at 1.6e4 K).

**Change.** `galaxy_sf/photoion_per_direction.{h,cc}`, `cooling/sgchem/cxx/chem_backend_cxx.cc`, `declarations/precompiler_logic.h`. New flag `GALSF_FB_PHOTOIONIZATION_RECOMBINATION_AT_CELL_TEMPERATURE` (default off, needs SGCHEM and the per-direction solve). A 51-entry table of alpha_B(T)/alpha_B(1e4 K), log T 4 to 9, travels in the call scalars and is read in the pair function; the reference reads the chemistry table per cell.

**Cost.** 204 bytes in the call scalars under the flag, a table refill per solve. No collectives. No work on quiet steps. PI cadence and search untouched.

**Evidence after.** `test/photoion_recomb_temperature`: 1 passed; N <alpha_B> 721.6 against 717.5 (0.6 %, bar 5 %), 749 and 1056 cells.

**Flag off.** All inside the flag. To be re-run on `<TIP>`.

**Depends on / followed by.** final/05.

## final/36-single-stars-no-gas-kernel-growth

Source 39 lines. Test: narrow. Cut on an older base; merges onto the current line with adjacent declarations in `core/proto.h` to combine.

**What and why.** The reference retries a single-star event that reaches no gas in the same step, widening the kernel x1.4, up to 3 times (bubble retry). Here the event waited for the next step.

**Evidence before.** `test/single_stars_no_gas_kernel_growth`: two 20 Msun stars with supernovae forced at 0.002 and 0.005; the first sits in an empty sphere of 3.5 pc, with its feedback kernel set to 3 pc by the new test hook (a star in a bubble). 1 failed; star 1 never couples within the run (8 kyr), star 2 couples at 0.005.

**Change.** `galaxy_sf/mechanical_fb.cc`, `galaxy_sf/single_star_feedback.cc`, `core/proto.h`: after the loop, the single stars whose event coupled nothing widen their kernel x1.4 and the loop runs again for them, at most 3 times. Whether another round runs is set by an Allreduce over all ranks, inside the block that runs only when some rank has an active feedback source. No retries while `TEST_SINGLE_STARS_NO_GAS_IN_REACH_UNTIL_TIME` (final/10) holds the events back, as in the reference. New test-only flag `TEST_SINGLE_STARS_FEEDBACK_KERNEL_AT_MOST` (parameter `TestFBKernelAtMost`). Coupling sources otherwise get kernels up to 2 kpc, so a hole alone never leaves one empty.

**Cost.** No bytes. One Allreduce of one integer per retry round. The first round's Allreduce runs on every step on which some rank has an active feedback source, also when every event coupled; with single-star winds that is most steps of a Stella run. None on steps without feedback sources.

**Evidence after.** 3 passed (with `test/single_stars_forced_events`): both supernovae couple in their own step (0.002, 0.005), with zero steps without deposit, mass conserved.

**Flag off.** The retry is on with the single-star tables, on purpose. FIRE and other builds unchanged (inside the single-star flags). To be re-run on `<TIP>`.

**Limits.** Not behind a flag of its own. At most three rounds (kernel up to 1.4^3 = 2.74 times); an event still without gas then waits for the next step, as before. The extra Allreduce could ride the existing per-step reduction of the feedback loop; not done.

**Depends on / followed by.** final/10 (uses its forced-event hooks); originally cut on final/19.

## final/40-winds-thermal-only

Source 20 lines. Test: broad.

**What and why.** Turn the momentum of winds and AGB ejecta wholly into heat: no kick, the full 1/2 M v^2 is deposited with the mass and metals. It brackets the production wind from the side opposite to a momentum-only wind (GRIFFIN, Lahen et al. 2023, fig. 3). Off in every config.

**Evidence before.** `test/single_stars_winds_thermal_only` without the change, 2 of 2 fail: radial gas momentum 1.10 (35 Msun wind) and 0.997 (3 Msun AGB death) of the booked momentum, the bar is below 1/4; heat row 0.872 and 0.788 of the ejecta energy.

**Change.** `galaxy_sf/mechanical_fb_functions.h`: mode 1 of the pair kernel takes the branch pure-thermal supernovae already use (momentum to couple 0, the ejecta carry the star's velocity, the cell closes the energy as heat). `declarations/precompiler_logic.h`: the requirements, and the radiation-pressure kick is off under this flag. New flag `GALSF_SINGLE_STARS_WINDS_THERMAL_ONLY` (default off).

Two points on the reference: its thermal-only branch leaves the cell velocity untouched, so the added mass arrives with dM v_j of momentum nobody paid for. Its kick branch and supernova branch were corrected for that; here the ejecta carry the star's velocity as in those branches, and the heat is the star-frame closure. For gas at rest relative to the star the two agree; the difference is of order dM/M. Its radiation-pressure pass shares the same branch and so gives no kick; here the separate loop is left out by `STELLA_DEFAULTS` and refused if set explicitly.

**Cost.** No bytes. No collectives. No work on quiet steps.

**Evidence after.** `test/single_stars_winds_thermal_only`: 2 passed. Radial momentum 0.169 and 0.0017 of the booked one (hydro response of the heated gas, as estimated before the run), heat row 1.00000, kinetic row below 1e-8 of it, mass to gas equal to mass from the star to 1e-14.

**Flag off.** Unchanged in source (preprocessor branches only). To be re-run on `<TIP>`.

**Depends on / followed by.** final/05. Followed by final/41.

## final/41-winds-in-chunks

Source 7 lines. Test: narrow.

**What and why.** Without its continuous-wind flag the reference holds a star's wind until the mass owed passes a fraction of the drawn mass, then returns it at once at the speed of that moment. Here the wind is returned every star step (the reference production choice) and there was no other mode.

**Evidence before.** `test/single_stars_winds_in_chunks` without the change, 2 of 2 fail: 16 deliveries from the first step (first 0.87497 Msun at 6 yr, then 4e-6 Msun a step), bar one; the mass owed at death leaves at 6 yr instead of after the death at 80 yr.

**Change.** `galaxy_sf/single_star_feedback.cc`: `single_star_eventrates` keeps the wind while the mass owed is at most 1 % (8 Msun) rising to 20 % (350 Msun) of the drawn mass, the reference formula. At the end of life the rest still goes out whatever its size, before the death. New flag `GALSF_SINGLE_STARS_WINDS_IN_CHUNKS` (default off). The reference serialises these events, one neighbour pass per star; here they go through the shared feedback loop like every other event. Effect on stars sharing neighbours in one step only: their weights use the masses before that step's deposits.

**Cost.** No bytes. No collectives. No work on quiet steps (one comparison where the wind was already evaluated).

**Evidence after.** `test/single_stars_winds_in_chunks`: 2 passed. One delivery at 55 yr, 1.000004 of the threshold, 2857.4 km/s against table 2857.4, star down by just that. Death case: nothing before the death, 0.4375 Msun (half the threshold) at 85.6 yr, the supernova at 91.7 yr.

**Flag off.** Unchanged in source (preprocessor branches and a comment). To be re-run on `<TIP>`.

**Depends on / followed by.** final/40. Combined with final/10 the chunk hold applies to a live star (`!dead`).

## final/43-hot-gas-timebin-clamp

Source 47 lines. Test: narrow.

**What and why.** Hydro applies each side of a pair flux on that side's own step, so pairs on different steps across a blast front do not conserve energy. This option keeps everything around hot gas on the step of the hot gas. Off in every config; the reference's own note says production adoption is not settled.

**Evidence before.** `test/hot_gas_timebin_clamp` without the flag: 3.0 % of the gas inside the sphere around the hot gas has a step within 2 of the shortest hot step, the bar is 98 %.

**Change.** `core/timestep.cc`: new `hotsync_clamp_prepare` before the steps are assigned; in the assignment, a particle inside the sphere takes at most a quarter of the shortest step of gas above 1e6 K. Sphere = hot-cell centroid + farthest hot cell + 3 largest kernel radii, as in the reference. New flag `RESOLVEDISM_HOTSYNC_CLAMP` (default off).

Three things a reviewer should know. `get_timestep` writes to the particle (dilation factor, transport substeps, tidal history), and the reference calls it on inactive hot cells; here the cell is saved and restored around that call. The clamp wakes nobody, so gas entering the growing sphere lags by up to one of its own steps: 75 % in step 1.1e-3 past a coarse sync, against 99 % just after one. The sphere is not wrapped at box edges and all hot gas shares one sphere, both as in the reference, whose comment already marks the first for periodic production use.

**Cost.** No bytes. Under the flag: a scan of the local gas and one integer reduction at every `find_timesteps`, also on steps without hot gas, because a rank cannot otherwise know that another rank has hot gas; three more reductions and a timestep evaluation per hot cell while gas is hot. This is a real per-step collective, only under the flag. Flag off: nothing.

**Evidence after.** `test/hot_gas_timebin_clamp`: passed. 99.63, 99.36, 99.99 % in step at 1.93, 3.86, 5.79e-3 (3.03 % at the first without the flag); shortest hot step a quarter of the unclamped one (2.375e-4 to 5.94e-5). The test shows the steps. The energy gain is not shown: no energy run of this flag has been made here.

**Flag off.** Unchanged in source (preprocessor branches only). To be re-run on `<TIP>`.

**Depends on / followed by.** final/05. Open question: final/43 and final/51 are two separate hot-gas clamps (this one around all hot gas; final/51 around new ejecta cells). Merged together, both are kept side by side in `core/timestep.cc` and `declarations/precompiler_logic.h`. Should both stay, or only one?

## final/44-chemistry-48-192-pixels

Source 10 lines. Test: narrow.

**What and why.** The gravity walk can give 12, 48 or 192 HEALPix pixels (NSIDE 1, 2, 4), but the chemistry's per-cell input arrays held 12, so a build with 48 or 192 pixels stopped at compile time. The reference chemistry takes any pixel count the walk gives. Off in every config (production uses 12).

**Evidence before.** `test/treecol_pixels`, `test_slab_shielding_finer_pixels[48]` and `[192]`: 2 of 2 fail, the build stops at `chem_gizmo_backend.cc:31`, "static assertion failed: ChemCell pixel arrays are sized for RT_USE_TREECOL_FOR_NH=12".

**Change.** `cooling/sgchem/cxx/chem_types.h`: the chemistry's pixel arrays are sized by the pixel count of the build (12 when there is none, for the one-zone drivers). `cxx/chem_gizmo_backend.cc`: the check becomes equality. `sgchem_config.h`: comment. No new flag; `GALSF_RESOLVEDISM_TREECOL_PIXELS=48` or `=192` now builds and runs through the chemistry. The walk keeps its own limit of 12, 48, 192 (the reference allowed NSIDE up to 8).

**Cost.** With 12 pixels: nothing. With N pixels the per-cell chemistry input is a stack temporary of 7 x 8 x N bytes (10.8 kB at 192); the stored cell fields were already sized by N (8 bytes per pixel plus 16). No collectives. No work on quiet steps.

**Evidence after.** `test/treecol_pixels -k test_slab_shielding`: 3 passed. Slab lit by one 40 Msun star, 20 steps of chemistry (0.15 Myr), cells on the star's axis: deep x_H2 0.0261 (12 pixels), 0.0262 (48), 0.0214 (192), bar within 1.5 of 12 pixels; lit face 1.4e-4, 1.7e-5, 5.1e-4, bar deep >= 30 x face. The face value moves by a factor 3 to 8 with the pixel count (the face cells see the star through pixels of different size); the bars do not test it.

**Flag off.** At 12 pixels the array size is the same constant and only a compile-time check changes. To be re-run on `<TIP>`.

**Depends on / followed by.** final/05.

## final/45-sn-hydro-recompute

Source 67 lines (35 without its first commit, which is af507c293, now in the base). Test: narrow, with a measurement. Cut on an older base; to be re-cut on `<BASE>`, where af507c293 drops out.

**What and why.** The reference's `FB_G17_BOUNDARY_INJECTION` moves the feedback deposit after the closing half-kick and repeats the hydro pass on a step in which feedback reached any cell (gizmo2017 order). The base now deposits mechanical feedback after the closing half-kick itself (af507c293), so the option adds only the repeat. Off in production.

**Evidence before.** `test/sn_hydro_recompute` on final/44 plus af507c293: 1 failed (no repeat with the flag set).

**Change.** `core/run.cc`, `core/accel.cc`, `core/proto.h`, `galaxy_sf/mechanical_fb.cc`, `declarations/precompiler_logic.h`. New flag `GALSF_FB_HYDRO_RECOMPUTE_AFTER_FEEDBACK` (default off, needs GALSF_FB_MECHANICAL). The repeat skips the thermal, photoionisation and radiation-pressure calls of the hydro pass, as they ran this step. The branch carries af507c293 and a ledger build fix as its first two commits; the first drops out once the base has af507c293.

**Cost.** No bytes. Under the flag: one Allreduce per step that has active feedback sources (none on quiet steps); a full hydro pass on each step with a deposit.

**Evidence after.** 2 passed. Adiabatic box, one 10 Msun star, 5 kyr, energy excess over start + 1e51 erg (max, end):
- 1 Msun cells: default -0.021 %, -0.119 %; with the repeat (9 repeats) the same to the printed digits.
- 0.1 Msun cells: default +1.798 %, +1.797 %; with the repeat +1.869 %, +1.869 %.

With the base deposit order, the repeat does not reduce the excess on this ladder. The reference's +2.0 % to +1.2 % at 1 Msun (stella_dev) was measured with its old mid-step deposit; that gain belongs to the order change, which is now the default.

**Flag off.** Compared with final/44 plus af507c293: FIRE-3 144/149 and gmc 145/149 byte-identical, the rest same instructions except cooling.o (only `InitCool`, see the last section). To be re-run on `<TIP>`.

**Limits.** On this ladder the repeat changes nothing material with the base's deposit order; it is offered as the reference's option, off in production. A full hydro pass per step with a deposit is expensive in runs with many events.

**Depends on / followed by.** final/44, af507c293 (in the base).

## final/50-ejecta-spawn

Source 169 lines. Test: broad.

**What and why.** A supernova in a cavity has almost no gas in its kernel, so a kernel deposit couples across the cavity to a distant shell. With `GALSF_SINGLE_STARS_SN_EJECTA_CELL=f`, a supernova whose kernel holds less gas than f times its ejecta mass becomes one new gas cell instead: ejecta mass, explosion energy as heat, yields, at the star with the star's velocity. The split routine then refines it and hydro drives the blast. The timestep sync that goes with it is final/51.

**Evidence before.** `test/single_stars_sn_ejecta_cell` on final/05: 1 failed, 2 passed. `test_new_cells_hold_the_ejecta` fails: the child cells around the 20 and 320 Msun stars hold 317.0 Msun against the 338.5 Msun of ejecta, because the kernel deposit splits lattice cells there and no ejecta cell exists. The books and the energy bar pass before too (a deposit keeps them), so they guard the change and do not tell the two apart.

**Change.** `galaxy_sf/mechanical_fb_functions.h`, `mechfb_loop.cc`, `declarations/particle_data.h`: one more weight slot, the gas mass in the star's kernel, filled in the loop's existing weighting pass, no new loop. `galaxy_sf/single_star_feedback.cc`: the decision; the event waits as `SN_DONE` with `M_at_SN_trigger` kept, so it needs no new star field, follows the star through domain exchange and restarts, and the mass stays on the star until the cell exists. `mesh/merge_split.cc`: the cell is made at the merge/split point, a copy of the nearest local cell with its pending kick, B flux, diffusion, radiation and cosmic-ray content cleared; the event then closes through the normal finalize, so logs, ledgers, the remnant mass and the black hole promotion are the same code as for a deposit. New flag `GALSF_SINGLE_STARS_SN_EJECTA_CELL=f` (default off). Failed SN, direct collapse, AGB deaths and Type Ia stay in the loop, as in the reference. `#error` with `GALSF_ISMDUSTCHEM_GRAINSIZEEVO`, whose size bins the new cell would not set. Differs from the reference on purpose: the radiation and cosmic-ray fields of the copy are cleared (the reference kept them, which creates energy in such builds); the temperature stored for SGChem is set from the new energy.

**Cost.** One float per particle under the flag. No new collective on a step without splits; one integer reduction on a step with splits, to write the event rows. The search for the nearest cell runs only for a pending event. No work on quiet steps.

**Evidence after.** `test/single_stars_sn_ejecta_cell` (SGChem network 5, tables, 2 ranks, f = 5, kernels hold 72 to 84 Msun): 3 passed. The 12 Msun star deposits, the 20 and 320 Msun stars make cells one step later; the new cells hold their 338.46 Msun to 1e-10; each remnant is the table's; gas around the 20 Msun cell gains 1.003e51 erg. Finding for the reviewer: at 1 Msun in 100 yr with cooling on, a heavy cell radiates: 320 Msun at Z = 0.96 kept 2e48 of 1e51 erg, 150 Msun 6.2e50. The option does what it says; the heavy ejecta collapse to n up to 5e5.

**Flag off.** Source unchanged with the flag off (all under `#ifdef`); to be re-run on `<TIP>`, including one FIRE build.

**Depends on / followed by.** final/05. Followed by final/51.

## final/51-hot-gas-timestep-sync

Source 52 lines. Test: broad (energy conservation, adiabatic).

**What and why.** Hydro fluxes are applied one-sided: each active cell applies its own half of a pair flux, so a pair conserves energy only when both partners step together. Around a young blast one timebin of difference makes or destroys energy. `HOT_GAS_TIMESTEP_SYNC` keeps everything near hot gas on one fine step while the hot gas lasts. The reference calls this SPAWN-SYNC / `RESOLVEDISM_HOTSYNC_CLAMP`.

**Evidence before.** `test/hot_gas_timestep_sync` (adiabatic, the one case where total energy is the bar; 20 Msun star as an ejecta cell in 1 Msun gas, 2 ranks, 3 kyr) on final/50, from that run's snapshots: energy gain 1.130, 1.159, 1.160, 1.155, 1.157, 1.158e51 erg against 1e51, worst +16.0 %. The reference (stella_dev) at 1 Msun: worst -5.07 % without the clamp.

**Change.** `core/timestep.cc`: while gas above 1e6 K exists, active particles within the hot centroid + the largest hot distance + 3 largest hot kernels step at no more than a quarter of the hot gas's Courant step, as in the reference. New flag `HOT_GAS_TIMESTEP_SYNC` (default off, on with `GALSF_SINGLE_STARS_SN_EJECTA_CELL`). `#error` with `SINGLE_STAR_AND_SSP_NUCLEAR_ZOOM`. Differs from the reference on purpose: distances use the nearest periodic image; and the search is armed by a feedback event or a new ejecta cell and disarms when no hot gas is left, where the reference scans all gas with three reductions at every sync point. Hot gas made by anything else, after the first step of a run, does not arm it.

**Cost.** No bytes. While armed: one scan of the local gas and three reductions per sync point, and the steps the sync asks for (135 against 36 sync points in the test). Not armed: nothing, so no work and no collectives on quiet steps.

**Evidence after.** `test/hot_gas_timestep_sync`: 2 passed, energy error per snapshot +0.49, +0.88, +0.91, +0.82, +0.86, +0.89 % (bar 1 %, written before the first run). The reference (stella_dev) at 1 Msun: +0.47 %. Hot gas still present at the end (bar), so the sync ran on a live blast.

**Flag off.** Source unchanged with the flag off (all under `#ifdef`); to be re-run on `<TIP>`, including one FIRE build.

**Depends on / followed by.** final/50. See the open question under final/43.

## Star death at the first step past the lifetime

See the slice series (star death at the first step past the lifetime). The three PRs below follow it.

## final/tree-light-1: tree luminosities follow the single stars

Branch final/tree-light-wind-telescope, commits 2da45a977 and 6620feb2e. Source 64 lines (27 + 37). Test: none in the repository; measured with a trace in cloud runs (see Limits). Sits on the star-death step.

**What and why.** The radiation fields of the gravity walk (G0, band fluxes) read star luminosities from the tree nodes. The node moments are refreshed when the tree is built or when something sets the stale flag. Single stars change their light without either: a star starts to shine at the end of its pre-main sequence, starts to ionise, goes dark at death, and fades or brightens between. On steps that reuse the tree, the nodes then carry old luminosities.

**Evidence before.** M2e4 cloud, seeds 0 to 2, root-node band luminosity against the direct sum over the stars at their current age, every gravity call: a band off by more than 1e-2 on 6.1, 1.4 and 2.5 % of calls, worst 7.2, 1.1 and 0.98 (relative). At 11, 9 and 13 steps on which a star switched on, started to ionise or went dark the root missed it.

**Change.**
- `galaxy_sf/single_star_luminosity.cc`, `core/accel.cc`, `core/proto.h`: before each gravity call, `single_star_light_switch_check` sets the existing stale flag (`TreeMomentsStaleFlag`) if a single star crossed one of its three switch times since the last call (2da45a977), or if one of its band luminosities moved by more than 1e-2 of the root's since the nodes last saw it (6620feb2e). After the call, `single_star_tree_lum_store` stores each star's band luminosities when the call rebuilt or refreshed the nodes.
- `galaxy_sf/single_star_record.h`: the stored luminosities, one float per band, in the compact star record.
- No new flag: on with `RT_USE_GRAVTREE`, `GALSF_SINGLE_STARS_FROM_IMF` and `GALSF_SINGLE_STARS_EVOLUTION_TABLES`. No new parameter; the 1e-2 tolerance is a compile-time constant.

**Cost.** `N_RT_FREQ_BINS` floats per single star in the star record (5 bands in the cloud configs: 20 bytes). No new collective: the stale flag is reduced by the Allreduce gravity already does. Work: before every gravity call, a pass over the local particles with a band-luminosity evaluation per single star, also on steps with few active particles; after a call that changed the root, a second such pass. Each refresh is the existing node-moment refresh. The pass could run over the star record only; not done.

**Evidence after.** Same runs with the switch refresh (2da45a977): a band off by more than 1e-2 on 0.6, 0.8 and 0.9 % of calls, worst 0.33, 0.31 and 0.30; at every switch the root equals the star sum (0 misses). The field at the cells against a direct sum over the stars, 16 dumps of rebuilt and reused trees: median at most 3.3e-4, 99th percentile at most 1.9e-3. What was left has no switch: bright stars fading near the end of life, up to 7 % per step, over at most 6.8 kyr of tree reuse. The drift refresh (6620feb2e) targets that rest; it has not been run yet: to be measured on `<TIP>`. The stellar mass formed differs between the run sets (838/565/1192 against 443/556/476 Msun), within the factor 2 this cloud shows under small perturbations, so it is not attributed to the change.

**Flag off.** Builds without the three flags are unchanged in source. Object comparison to be run on `<TIP>`.

**Limits.** No repository test: the measurement used a trace of root and direct sums added to a throw-away build. The bar set before the run (every call within 1e-2) was missed by the switch refresh alone; whether the drift refresh meets it is open. The per-call pass is O(local particles) on every step.

**Depends on / followed by.** Star-death step (slice series), tc/01 (single-star band luminosities as tree sources).

## final/tree-light-2: winds owed element by element

Branch final/tree-light-wind-telescope, commits 3ccf7e271 (test), 783e5062e (change), 0b395e0d5 (README). Source 12 lines. Test: broad (element totals against the tables and mass conservation).

**What and why.** A star's wind between two deliveries is the difference of the table's cumulative wind ejecta, element by element. When interpolation rounding made one element's difference slightly negative, the code replaced the whole delivery's composition by the surface composition. How much of each element a star returned then depended on how often it delivered.

**Evidence before.** `test/single_stars_winds_agb_ia::test_wind_elements_independent_of_cadence` (new): a 40 Msun star from 4.40 to 4.84 Myr in metal-free gas, delivering every 1 kyr and every 20 kyr; from 4.59 Myr one element's cumulative table entry is flat while the total still rises. Fails: metal slots off the table cumulative by up to 2.4e-2.

**Change.** `galaxy_sf/single_star_feedback.cc`, `single_star_wind_owed`: each element's owed mass is its cumulative difference, floored at 0 (the table is non-decreasing, so a negative difference is rounding); the delivery mass is their sum. The swap to the surface composition is removed. No new flag.

**Cost.** None.

**Evidence after.** The test passes: in both runs each metal slot gains the table cumulative of its elements between the start and the last delivery age to 1e-12, and the star loses what the gas gains to 1e-12. M2e4 cloud: the one star that dies in both runs (29.5 Msun, 1323 against 830 deliveries) returns metals equal to 2.8e-9 and mass to 2.8e-12 between the two runs; without the change the same comparison gave -7.8e-3 and -8.7e-4.

**Flag off.** Single-star builds only. Object comparison to be run on `<TIP>`.

**Limits.** The floor drops a negative rounding of an element instead of carrying it to the next delivery; the loss is at the rounding level of the interpolation.

**Depends on / followed by.** slice2/12 (winds), star-death step (slice series).

## final/sne-causal-wake-single-stars: supernova causal wake for single stars

Branch final/sne-causal-wake-single-stars, commits f90e3509b (test), 9f6627982 (change), 40aeaea60 (Template_Config, `STELLA_DEFAULTS`, README). Source 68 lines. Test: broad (energy of 30 supernovae and of one supernova at three resolutions).

**What and why.** The scheme is Mike Grudic's (`SNE_BROADCAST_TIMESTEP`, pull request #107, after Durier & Dalla Vecchia 2012): each gas cell caps its next step at 0.5 CourantFac of the time the fastest signal of this step's supernovae needs to reach it, and an inactive cell whose step would outlast that is woken now, so it closes its step with its pre-supernova forces. The base took the other part of #107 (deposit after the closing half-kick, af507c293, and direct heat injection) and did not take the wake, because of its per-step collective and its all-gas-times-all-supernova loop. This PR ports the wake to the single-star supernovae, because with the base's deposit order alone the single-star supernova energy is not conserved in the runs below.

**Evidence before.** Base deposit order and the star-death step, no wake. Energy gained by the gas from the supernovae, with cooling, gravity, driving and other feedback booked out, in units of 1e51 erg per supernova, at full synchronisations (G):
- Driven box, 50 pc, n_H 10, 42875 cells of 0.96 Msun, 30 stars of 11 to 24 Msun dying over 2 Myr: G = 0.958 over all 30; worst window of the hydro term -0.344 per supernova.
- One 20 Msun supernova in uniform gas, cells of 10, 1 and 0.1 Msun, four times of death before a 50 kyr step boundary: the largest G after the first full synchronisation is 0.61 to 3.24 (bar 0.9 to 1.1, set before the runs): missed at every resolution.

**Change.**
- `core/timestep.cc`: `sne_broadcast_record` collects this step's supernovae (position, signal speed); `sne_causal_wake` gathers them on all ranks after the feedback and gives each gas cell its arrival time; `get_timestep` caps the step at 0.5 CourantFac of it; `process_wake_ups` starts a woken cell on the step its arrival time allows.
- `galaxy_sf/single_star_feedback.cc`: at a supernova's finalize, the signal speed is the ejecta speed, or that of a mean gas cell given all of the energy, whichever is lower (as #107).
- `core/run.cc` (call after the mechanical feedback), `core/init.cc` (start values; mean gas cell mass), `declarations/particle_data.h`, `global_data_all_struct.h`, `precompiler_logic.h`, `core/proto.h`.
- Flag `SNE_BROADCAST_TIMESTEP` (Mike's name), default off, set by `STELLA_DEFAULTS`; `#error` without `GALSF_FB_MECHANICAL` and single stars from the tables (the FIRE path is not ported).

**Cost.** One `MyFloat` per particle (the arrival time) under the flag. Collectives: one `MPI_Allgather` of one integer per rank on every step that runs the mechanical feedback, which is every step with gas, also without supernovae; on steps with supernovae an `MPI_Allgatherv` of four doubles per supernova and a pass over the local gas times the number of supernovae. This is the cost the base declined for #107. The integer could ride the existing per-step reduction of the feedback loop; not done. Driven box: 8093 against 7644 steps (+6 %), wall 908 against 824 s (one run each).

**Evidence after.** Same runs with the wake:
- Driven box: G = 1.015; per-window hydro term +0.008 +- 0.006 per supernova, worst +0.029, lowest -0.001. Per event, energy in the gas against offered within 5.1e-4; ledger residual at most 1.3e-13 of 1e51 erg.
- Single supernova: largest G 1.005 to 1.027 (10 Msun cells), 1.014 to 1.020 (1 Msun), 1.055 to 1.089 (0.1 Msun). Bar met; 0.1 Msun is close to it.
- `test/sn_driven_box` gains `test_energy_conserved_over_the_events` (12 supernovae, G within 2 %) and `test_single_sn_energy` (10, 1 and 0.1 Msun, G within 0.9 to 1.1); their run on the branch is not recorded: to be run on `<TIP>`.

**Flag off.** Without the flag the code is unchanged in source; Stella builds change on purpose (on in `STELLA_DEFAULTS`). Object comparison to be run on `<TIP>`, including one FIRE build.

**Limits.** The per-step Allgather and the all-gas loop above. As Mike noted on #107, the cap is blunt: it shortens steps also in cold dense gas that the shock reaches later. The 0.1 Msun ladder is within 0.011 of its bar. Questions for the reviewer: keep it in `STELLA_DEFAULTS` with this cost, or first move the supernova count into the existing reduction and replace the all-gas loop by a neighbour search? Or is another fix of the single-star case preferred?

**Depends on / followed by.** slice2/10 (supernovae from the tables), star-death step (slice series), the base's deposit after the closing half-kick (af507c293).

---


## Not part of this audit round

These five branches are not included in the audit branch. Each is described here so the reviewer knows what exists.

- **final/26-mhd-cleaning-off-options** (4 lines, narrow). Two test switches, `TEST_MHD_NO_DEDNER_CLEANING` and `TEST_MHD_NO_POWELL_SOURCE_TERMS`, to isolate the effect of div B treatment. `test/mhd_cleaning_off_options` (Orszag-Tang, 32^2): fails before (2 failed), 2 passed after; without the Powell terms B differs from the default by 6.4e-2 of its maximum.
- **final/27-readsnap-ids-as-stored** (Python only). `analysis/readsnap.py` added 2^31 to every ID above 1e9 or below 0; with 64-bit IDs this shifts valid IDs. The fix reads IDs in the file's dtype. `test/readsnap_ids`: 2 failed before, 2 passed after.
- **final/46-mhd-kernel-volume-energy** (276 lines, broad). Two experimental closures for the energy gain of meshless MHD at low beta (`MHD_CONSERVE_ENERGY_ON_KERNEL_VOLUME_CHANGE`, `MHD_RESCALE_B_TO_BOOKED_MAGNETIC_ENERGY`, both default off). `test/mhd_kernel_volume_energy`: end drift +0.02752 without the options, -0.00334 and -0.00417 with them (reference implementation: -0.0020, -0.0029), 3 passed. Cut on an older base; to be re-cut.
- **final/47-mhd-energy-ledger** (84 lines, narrow). An MHD energy ledger. Unfinished. Cut on an older base.
- **final/52-conservative-psi-cleaning** (459 lines against final/03, narrow). Conservative psi cleaning. The change is committed; its run after the change has not been made. Not on the audit branch. Would need a split.

---

## Flag off, test kinds, integration (older base)

**Flag off (object comparison), measured on an older base (4d8a5e9e0); to be re-run on `<TIP>`.**
- All branches of this file except final/45 merged, with FIRE-3 and gmc configs: FIRE-3 149 objects, 139 byte-identical, 9 with the same instructions (only `__LINE__` immediates or strings), 1 different: cooling.o. gmc: 149 objects, 137 byte-identical, 11 same instructions, 1 different: cooling.o.
- cooling.o: only `InitCool` differs, in register allocation. Taking cooling.cc with every new gated block blanked out (line numbers kept) gives exactly the tip's object, so nothing changes semantically: `__LINE__` constants move and the register choice moves with them.
- final/45 against its own base (final/44 + af507c293 + ledger build fix): FIRE-3 144/149 and gmc 145/149 byte-identical; the rest have the same instructions, except cooling.o (again only `InitCool`).
- Stella cloud config, which switches several of these paths on: 124 of 171 objects byte-identical; 35 differ, on purpose: final/03 changes 5 objects on its own; final/15 (solar Z 0.014 with SGChem); final/16 (ledger rows plus an 8-byte cell field with MAGNETIC and the ledger; the field changes the cell struct, hence most of the objects).
- With final/08, final/36 and final/45 added: gmc 137/149 byte-identical, the rest same instructions except cooling.o; FIRE-3 135/149, with accel.o, run.o and mechanical_fb.o also different (the three files of af507c293, which final/45 carries; they drop out once the base has it). Stella cloud config: 123 of 171 byte-identical.
- The tree luminosity refresh, the wind composition fix and the causal wake have no object comparison yet; to be run on `<TIP>`.

**Test kind.** Broad (physics validation): sn_driven_box (causal wake), single_stars_winds_agb_ia (cadence case), ledger_step_breakdown, static_plummer_test_potential, photoion_recomb_temperature, treecol_pixel_columns_output, hot_gas_timestep_sync, and, as marked above, sgchem_thermal_state (final/03), single_stars_winds_thermal_only and single_stars_sn_ejecta_cell. Narrow: all other tests in this file.

**Integration notes.**
- Needs c84215640 (`TIDAL_TIMESTEP_PREFAC`, for the final/04 test) and af507c293 (deposit after the closing half-kick, for final/45).
- Merge resolutions when the branches are combined: final/10 with final/41 in `single_star_feedback.cc` (chunk hold applies to a live star); final/43 with final/51 in `timestep.cc` and `precompiler_logic.h` (both clamps side by side, see the open question); final/08 with final/16 on the ledger line of `Template_Config`; final/36 with final/50/51 in `proto.h`, and final/45 with final/13 in `mechanical_fb.cc` (both blocks kept).
- KETJU is a separate install found by `KETJU_PATH`; this file adds no submodule.
