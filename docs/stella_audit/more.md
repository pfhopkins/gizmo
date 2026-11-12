# More series (more/01 to more/08): CR fluid with single stars, and the step ledger

Series name: more. Base: <BASE>. Tip: <TIP>. Branch names are series names.
Covers: cosmic-ray fluid behaviour with single stars (CR exchange, initial CR energy, CR injection by supernovae and winds, CR energy in gas collection, CR speed default, Alfven-speed cap, neutral fraction with H2) and the mass-balance step checkpoints.
Order: slice2 -> chem (chem/12 before tc) -> tc -> gap -> cr (cr/04 before pi/06) -> met -> pi -> ext -> tr -> kj. The more series is stacked on cr/06-sgchem-cr-fluid-energy-exchange. Serial order: 01, 02, 03, 03b, 04, 05, 06, 07, 08.

Dropped as not a difference to the reference implementation (stella_dev): tidal prefactors (the port runs 1.0/1.0 as the reference), and the run-time `DomainSegmentsPerRank` form of MULTIPLEDOMAINS. The wake-up target bin is in the parity series (par/03). HEALPix balancing is in the healpix series. TreeRay helium is in the treeray series. `TREE_RAY_IR` has nothing to port: in the reference it is only an `#error` check (precompiler_logic.h:1871) and a "future" comment (gravtree.cc:773).

Test setup: Kokkos OpenMP on CPU, 2 ranks x 1 thread, not on a GPU. Bars were set in the test files before the first run and not changed; where a bar missed, the text says so. Numbers below were measured on an older base (the cr/06 tip of the time, 2 ranks x 1 thread) and are to be re-measured on <TIP>. Flag off: objects of base tip vs PR tip with the gmc_cooling and FIRE3 configs; no object with different instructions in any PR (an earlier objcmp of more/01 showed io.o and read_ic.o differing; it did not recur after a rebase, cause not found). Flag off: to be re-run on <TIP>. Full suite: <SUITE RESULT>. Tests at the series tip: 24 passed (the eight tests of this series plus the single-star and ledger tests they touch). Test breadth: all tests of this series are narrow (one flag, one default or one bookkeeping path); more/01, more/04 and more/07 compare with runs of the reference implementation on the same ICs.

| PR | branch | src +/- (no Template) | before | after | reference implementation |
|---|---|---|---|---|---|
| 01 | more/01-crfluid-exchange-pdv-only | 5/0 | ran unsplit, E_CR ratio 0.9999 | production path: closure -8e-5, -2.3e-4; E_CR 2.0045 vs 2.1245 | 2.0048, 2.1239 |
| 02 | more/02-crfluid-initial-energy | 19/0 | CR 0 | pass | not run side by side alone |
| 03 | more/03-ledger-step-checkpoints | 57/0 | tag bar could not pass | tag bar moved, reason in the test; closure 4.4e-16 (metals 1.4e-15) | none |
| 03b | more/03b-cr-ledger-rows | 90/5 | no rows | closes 1.7e-16 (box), 1.8e-16 (c8) | none |
| 04 | more/04-single-star-cr-injection | 40/14 | CR 1.55e47, gas 3.04e51 | CR 3.0000e50 = f E, gas 2.7026e51, wind 3.7671e48 = f KE | same SN box not run |
| 05 | more/05-collect-gas-cr-energy | 14/2 | 5.5e-6 | < 1e-6, collected row exact | the reference does not scale donors |
| 06 | more/06-single-star-cr-speed-default | 3/0 | did not discriminate | D 1.5 pc vs 11.5 pc | none |
| 07 | more/07-cr-alfven-cap-and-neutral-fraction | 18/1 | no cap | loss 0.00642 vs 0.631 uncapped; kappa 7143 vs 9331 unchanged code | 0.00642; 6761 |
| 08 | more/08-cr-shock-efficiency-read-ic | 17/0 | absent | shock row 0.2 / 0.1 = 2.00; IC CR energy kept | none |

All PRs are under 300 source lines.

Open: more/07's neutral-fraction part has no repo test; the supernova box was not run side by side with the reference; a cloud with M1 and the CR fluid together fails on the port (gas above 1e9 K within the first few kyr, no stars; to be re-measured on <TIP>).

---
## more/01: CRFLUID_EXCHANGE_PDV_ONLY

**What and why.** New flag `CRFLUID_EXCHANGE_PDV_ONLY`, default on with `COSMIC_RAY_FLUID` and single stars (as the reference, precompiler_logic.h:369). The CRs take only their own P dV work from the gas, not their pressure fraction of the shock heating.

**Evidence before.** Lattice blast (16^3, E = 1) with CR energy 2, two builds. The first bar (CR energy lower with the flag) was wrong: CR energy is the same to 1e-4 (ratio 0.9999). The difference is in the gas: the default exchange loses 1.7 % of the total energy in this problem (CR+gas total 3.931 against initial 3.998; 3.995, -0.07 %, with the flag). That may be a defect of the default exchange (energy removed from the gas and not credited to the CRs); it needs its own look.

**Change.** Flag and its default (precompiler_logic.h); in `CR_calculate_adiabatic_gasCR_exchange_term` the floor at the CR pressure fraction of the total term is skipped under the flag. 5 source lines.

**Cost.** None: a compile-time branch.

**Evidence after.** Magnetised lattice blast on the production path (operator-split cooling): total energy closes to -8e-5 (default) and -2.3e-4 (flag); E_CR 2.1245 (default) and 2.0045 (flag), reference implementation (stella_dev) 2.1239 and 2.0048. Narrow test.

**Limits.** Whether the default exchange loses energy (the 1.7 % above) is not resolved here.

**Flag off.** See the series note.

**Depends on / followed by.** cr/06; more/02.

---
## more/02: Initial CR energy on fresh starts

**What and why.** Parameters `CosmicRayInitialAmbient_eVcm3`, `CosmicRayInitialCloud_eVcm3`, `CosmicRayInitialCloudThreshold_nH` (defaults 0, 0, 10). Fresh starts (start flag 0) give the CRs an initial energy density split by n_H. Replaces nothing in the base (a fresh start zeroes the CRs; only start flag 2 keeps them).

**Evidence before.** CR energy 0 at a fresh start. A first test version put the density threshold in wrong units (n_H is 285 / 2850 cm^-3 in the test's units, not 0.29 / 2.9); the threshold was moved to 1000 before the pass.

**Change.** The three parameters (begrun, global data) and the start values in init; 19 source lines.

**Cost.** Three doubles in the global parameter struct; one pass over the gas at a fresh start only.

**Evidence after.** test/crfluid_initial_energy passes: CR energy density 1.0 eV cm^-3 on the ambient side and 0.1 on the dense side, as set. Narrow test. Not run side by side with the reference alone.

**Flag off.** See the series note.

**Depends on / followed by.** more/01; more/03.

---
## more/03: Mass-balance step checkpoints

**What and why.** `OUTPUT_MASS_BALANCE_STEP_CHECKPOINTS` (needs the ledger). `mass_balance_steps.txt`: M_bary, gas metals, ledger-corrected metals at pre/post hydro+feedback and pre/post star formation, a line when one changes by more than 1e-14.

**Evidence before.** No file. The first bar ("four tags present") could not pass: the writer logs only changes (35 lines: post_hydro_fb 33, pre_star_formation 1, post_star_formation 1; no pre_hydro_fb because nothing changed there). The tag bar was moved, reason in the test.

**Change.** New output and its flag; 57 source lines.

**Cost.** One MPI_Allreduce per checkpoint, nothing when off. Reads host P (CPU build; on a GPU the arrays must be current there, not checked).

**Evidence after.** Closure bars hold: max dM_bary/M 4.4e-16, max dM_Z_ledger/M_Z 1.4e-15 (bar 1e-12), 35 checkpoint lines. Narrow test.

**Limits.** Reads host arrays; on a GPU build the arrays must be current on the host at the checkpoints, not checked.

**Flag off.** See the series note.

**Depends on / followed by.** more/02; more/03b.

---
## more/03b: CR energy rows in the ledger

**What and why.** With `OUTPUT_MASS_METAL_ENERGY_LEDGER` and `COSMIC_RAY_FLUID`, every change of the CR energy gets its own cumulative row, so the CR energy closes against its channels: E_CR - E_CR(0) = injected by supernovae + injected by winds + shock + clip + exchange + transport - streaming - collected with gas - losses - ionisation. Without it the CR energy has a state row but nothing to close against, and a CR energy error cannot be located.

**Evidence before.** The ledger has only E_CR and (with SGChem) the loss and ionisation rows. The new test cannot run: its rows do not exist.

**Change.** Rows in `LEDGER_ROWS_CR` (proto.h): offered, injected_sn, injected_wind, shock, clip, exchange, transport, streaming, collected; the E_CR state row no longer needs SGChem. Bookings: `ledger_cr_kick` in the CR kick (transport, streaming loss, and what the cap and zero floor change), `ledger_cr_exchange` and `ledger_cr_shock` (cosmic_ray_functions.h, host only); supernova vs wind split in the mechanical loop (`CR_energy_from_sn` in the gas delta); offered energy in single_star_feedback.cc; collected CR energy around the gas-collection apply. The per-bin streaming-loss term (`DtCosmicRayEnergy_StreamingLoss`), so far kept only with TRANSPORT_SUBCYCLE, is also kept with the ledger. 90 source lines added, 5 removed.

**Cost.** Under the ledger flag only: one float per CR bin per gas cell (the streaming-loss term, unless TRANSPORT_SUBCYCLE already has it); one double in the mechanical-feedback gas delta; two sums of the local CR energy around each gas-collection call (only on steps that collect gas). Bookings are per-thread exact sums, reduced only at snapshot output; no collective on other steps.

**Evidence after.** test/ledger_cr_rows (the magnetised lattice blast of more/01 with the ledger): residual at most 1.7e-16 of E_CR(0) at every ledger line (bar 1e-5). On a cloud with single stars, winds and CRs (24,754 cells, about 6 Myr): residual 1.8e-16 of the sum of the channels. Narrow test.

**Flag off.** See the series note. With the ledger off nothing changes.

**Limits.** The loss rows need SGChem. The bookings are host only (on a GPU build they are compiled out of the device kernels, so the rows would not close there). The Template_Config line of the ledger still mentions a deferred heat branch that no longer exists; the line needs a rewrite.

**Depends on / followed by.** more/03; more/04.

---
## more/04: CR injection by single-star supernovae and fast winds

**What and why.** The loop's CR injection is turned on for single stars (`MECHFB_CR_INJECTION`, `CR_DYNAMICAL_INJECTION_IN_SNE`). The pure-thermal supernova gives heat what the CRs leave. The old `#error` is removed. `CosmicRay_SNeFraction` exists and defaults to 0.1 without FIRE. Winds go through the same block (not separately tested).

**Evidence before.** The build stops at the old `#error`. In the first test version: CR gained 1.55e47 erg against 1.0e50 expected, gas gained 3.04e51 against 2.90e51. The first CR number came from the reduced-speed correction (`evaluate_cr_transport_reductionfactor`, about 1.5e-3 in this box (to be re-measured on <TIP>), also in the reference) which the first bar ignored.

**Change.** precompiler_logic.h (defaults with single stars), mechanical_fb_functions.h (CR share for single-star events, heat reduced by 1 - f), begrun and global data (`CosmicRay_SNeFraction`), single_star_record.cc (`#error` removed). 40 source lines added, 14 removed.

**Cost.** One double in the global parameter struct. Per feedback event the existing CR injection block of the loop; nothing on steps without events.

**Evidence after.** One supernova: CR offered and deposited 3.0000e50 erg = f E; gas gained 2.7026e51 of 2.7000e51. Fast wind: CR 3.7671e48 = f x 3.7671e49 erg of wind kinetic energy. Narrow test. The same supernova box was not run in the reference.

**Flag off.** See the series note.

**Depends on / followed by.** more/03b; more/05.

---
## more/05: Gas collection keeps the donors' CR energy

**What and why.** Gas collection keeps the donors' specific CR energy (CR energy, flux, spectrum number, wave energies scaled with the mass fraction). The `#error` is reduced to radiation and sinks.

**Evidence before.** Build stops at the old `#error`; after removing it, a first scaling left gas mass 0.996528 and CR energy 0.996522, a mismatch of 5.5e-6 (bar 1e-6; without scaling 3.5e-3) (to be re-measured on <TIP>).

**Change.** sfr_collect_gas_loop.h: each donor's CR energy, flux (and spectrum numbers and wave energies when evolved) scaled with its remaining mass fraction; the `#error` keeps radiation and sinks. 14 source lines added, 2 removed.

**Cost.** One loop over the CR bins per donor cell, only when gas is collected.

**Evidence after.** Gas mass 0.996528 and CR energy 0.996527 of the initial: mismatch below 1e-6; the ledger's collected row is exact. Narrow test. The reference does not scale donors.

**Flag off.** See the series note.

**Depends on / followed by.** more/04; more/06.

---
## more/06: Single-star CR speed default

**What and why.** 300 km/s CR speed default with single stars, together with the mechanical-feedback defaults of more/04.

**Evidence before.** The observable (steps) did not discriminate: 9 steps in both cases. The default is confirmed instead by the preprocessor on the test config: `CRFLUID_SPEEDOFLIGHT_REDUCTION` (3.e7/C_LIGHT_CGS), `CRFLUID_EXCHANGE_PDV_ONLY`, `MECHFB_CR_INJECTION`, `CR_DYNAMICAL_INJECTION_IN_SNE`.

**Change.** 3 source lines.

**Cost.** None.

**Evidence after.** Radius holding 99 % of the CR excess after the test time: 1.5 pc with the default, 11.5 pc at the full speed of light. Narrow test.

**Flag off.** See the series note.

**Depends on / followed by.** more/05; more/07.

---
## more/07: CR streaming loss capped at the CR speed; ion fraction from SGChem

**What and why.** Two changes, as in the reference. (1) New flag `CRFLUID_STREAMING_LOSS_CAPPED_AT_CR_SPEED`, on by default with `COSMIC_RAY_FLUID` and single stars: in `CR_get_streaming_loss_rate_coefficient` the Alfven speed is capped at the (reduced) CR speed, so streaming losses cannot outrun the CR transport. Without it, gas with an ion Alfven speed far above 300 km/s drains the CR energy in a few steps. (2) With SGChem, the ion fraction for the Alfven speed and the ion-neutral damping comes from the network's electrons (`Get_Gas_Ionized_Fraction`; net 5 subtracts the carbon in CO), so H2 counts as neutral.

**Evidence before.** Periodic box, ion Alfven speed 3000 km/s, CR speed 300 km/s, CR gradient along the field: 63 % of the CR energy lost (0.631). Molecular box (H2 fraction 0.49, 24,754 cells): median CR diffusivity 9331 on the unchanged code against 6761 in the reference at the same time.

**Change.** precompiler_logic.h (default), cosmic_ray_functions.h (cap), cosmic_ray_utilities.cc and eos_functions.h (SGChem ion fraction). 18 source lines added, 1 removed.

**Cost.** One min per streaming-loss call; no field, no collective.

**Evidence after.** Box: CR energy lost 0.00642 with the cap, 0.631 without, reference 0.00642 (bars: at most half of uncapped, within 5 % of the reference). Molecular box: median diffusivity 7143, reference 6761 (bar 10 %). The streaming-cap test is narrow; the molecular box is a comparison run, not a repo test.

**Flag off.** See the series note.

**Limits.** The ion-fraction part has no repo test. One line of this commit (`saturation_fraction_for_craccel` reading `All.CosmicRay_ShockAccelEfficiency`) belongs to more/08 and must move there in the re-cut: on its own this commit does not compile with `CRFLUID_INJECTION_AT_SHOCKS`.

**Depends on / followed by.** more/06; more/08.

---
## more/08: CR shock efficiency as a parameter; CR energy from the ICs

**What and why.** Two small options, as in the reference. (1) Parameter `CosmicRay_ShockAccelEfficiency` (with `CRFLUID_INJECTION_AT_SHOCKS`): the fraction of the shock energy flux put into CRs, default the compile-time value, so it can change between runs and restarts without a rebuild. (2) Flag `INPUT_READ_COSMICRAYENERGY`: a fresh start keeps the `CosmicRayEnergy` of the ICs instead of zeroing it (cloud ICs with a prescribed CR field).

**Evidence before.** The efficiency is fixed at compile time; a fresh start always zeroes the CR energy.

**Change.** begrun.cc and global data (parameter, default from the flag value), cosmic_ray_diffusion_functions.h (reads the parameter), init.cc and read_ic.cc (keep the IC block). 17 source lines.

**Cost.** One double in the global parameter struct. None per cell, no collective.

**Evidence after.** test/ledger_cr_rows, shock part: shock row 8.59e54 erg unset and at 0.1, 1.72e55 at 0.2 (ratio 2.00). test/crfluid_read_ic_energy: CR energy of every cell in the first snapshot equals the ICs to 1e-5. Both narrow.

**Flag off.** See the series note.

**Limits.** None known.

**Depends on / followed by.** more/07. Last of the series.

---
## CR loss row, like for like

The reference with one line of the cooling function removed (the cosmic-ray ionisation heat term set to zero in the CMB branch of `chem_cool_func.cc`) against this series, molecular IC, 32 ranks. Cumulative row including ionisation, port / reference: 0.906 at 5e-5, 0.952 at 1e-4, 0.966 at 1.5e-4 (last reference row) (to be re-measured on <TIP>). Bar 5 % at the end of the run: met (-3.4 %); missed at 5e-5. Counters at 1.5e-4: hadronic 1.002, Coulomb 1.017, ionisation 0.963, sum 0.995.
