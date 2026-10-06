# GIZMO RT and microphysics issues found during the jaco integration

Legacy-code findings that stand on their own, independent of jaco. Evidence is from code reads (line numbers at
origin/starforge_dev d45941ca unless stated) and from the runs cited. Experiment branch for the RT energy fixes:
`rt_energy_ab` (worktree `~/code/gizmo_rt_ab`, tip 83f6b1cc), with the sensitive test `test/rt_closed_box`, the
switches `RT_FIX_IR_DOUBLE_COUNT` / `RT_FIX_DUST_GAS_ABSORPTION`, the GIZMO_config.h include guard and the
`RT_ISRF_BACKGROUND=0` override. Runs and analyses: `/mnt/ceph/users/mgrudic/starforge/M10_core/rt_energy_ab/`.
Decisions that affect the jaco branches are cross-referenced from JACO_PENDING_DECISIONS.md.

## Review status (independent code read, 2026-10-05)
Every mechanism below was re-read in the source by the coordinator, separately from the agents that found it:
- Verified by reading: 1.1 D1 (donation + E_abs_tot_toIR, IR processed last), 1.1 D2 (total-opacity absorption into
  rt_eqm_dust_temp while the gas share is also deposited), 1.1c photoionization/photoelectric heating at true c with
  the full absorbed energy also donated (rt_utilities.cc:956 `+= de_abs`, cooling.cc Heat_Ion_from_RHD at C_LIGHT_CGS),
  1.1d opening-kick deposit discarded (rt_utilities.cc:932 rate over dt_entr -> hydro_toplevel.cc:971 zeroes it;
  run.cc order hydro:125, kick:152, hydro:256, kick:266, cooling:548), closing-kick deposit applied once for split cells
  (kicks.cc:269, dt_hydrokick = dt_entr) and over the full step for unsplit cells (cooling.cc:1477 into Q, integrated
  over dtime), split cells' DtInternalEnergy in code units used as cgs at cooling.cc:244 (only unsplit cells are
  converted at :166), 1.2 limiter under RSOL, 2.1 x-tolerance-only root-find, 3.1 Rad_Je zeroing only when
  RT_SOURCES & 1 (startup-only zeroing at rt_utilities.cc:1082), Iliev photon count, ISRF edge reset (initial guess
  only in legacy), the GIZMO_config.h re-include (eos.h:4, aneos.h:4), 4.1 init.cc:1074-1078 at function scope, and
  eos.cc set_eos_pressure taking gamma_eos_index before the temperature refresh for SoundSpeed (:49, :185) while
  cell.Gamma has no reader anywhere in the tree.
- Measured, not re-derived: all run numbers (A/B tables, closed-box ledgers, subsuite statistics).
- Unverified: the T_rad weighting "stuck at the lowest temperature" claim (3.1).

## 1. Radiation-transport energy accounting

### 1.1 RT kick energy accounting (confirmed by reading; A/B done 2026-10-05)
- D1: donor-band energy absorbed by dust is added to the IR band twice per kick (direct donation and through
  `E_abs_tot_toIR` in the IR update), `radiation/rt_utilities.cc` ~:743/:824/:843. Measured exactly 2.000x on the
  first kick.
- D2: the gas-opacity share of IR absorption heats the dust balance (:734-736) while also heating the gas (~:845),
  and the gas share is deposited at c~ rather than c.
Legacy-only A/B: branch `rt_energy_ab` (worktree `~/code/gizmo_rt_ab`, cfae0811 off origin/starforge_dev d45941ca),
switches `RT_FIX_IR_DOUBLE_COUNT` and `RT_FIX_DUST_GAS_ABSORPTION`, arms base/d1/d2/d12; results and plots in
`/mnt/ceph/users/mgrudic/starforge/M10_core/rt_energy_ab/{shu_M120/analysis,analysis}/summary.txt`.
- shu_M120 (616 yr): D1 lowers urad_FIR 16% at the centre and up to 32% in the outer envelope; T/T_dust/T_rad move
  2-5%; the test still passes (0.052/0.037/0.052 vs 0.1). D2 within run-to-run scatter. Double count = 8.6% of the
  injected energy.
- M10 8-seed ensemble (98 kyr): double-counted energy 0.25-2.8x the sinks' luminosity and 0.5-1.1x the final box
  IR energy (continuous ISRF reprocessing), but box E_IR only 3-7% lower when fixed (boundary-dominated field).
  Sink growth: d12 null (total ratio 0.997, 4/8 higher, p = 0.95); d1 marginal (+5% total, 7/8, p = 0.11; per-seed
  median +16%, p = 0.055, driven by seed 2: 0.46 -> 0.89 Msun); d2 null. Peak T_dust near sinks within 10%.
  Does not explain the dense-gas 70-80 K vs 28 K question.
Sensitive test (2026-10-05, commits 39201f4f/7d445bcf on rt_energy_ab): `test/rt_closed_box` -- periodic 32^3 glass,
n_H = 1e3, solar Z, FREEZE_HYDRO, no sources, no ISRF, bands pre-filled from a snapshot (flag 2). Unfixed: total band
energy +66%, IR gain / donor loss = 2.000000 (strict xfail); fixed: 2.6e-10 and 1.000000. Donor decay rates match
rt_kappa to 1%. D2 has NO ledger-sensitive test and cannot: the T_dust it changes is not used in the kick's energy
split (only the T_rad target and opacities); in production regimes f_gas < 1e-2, and where f_gas ~ 0.8 (dense ionized,
cold IR) gas-dust collisions dominate dust heating 300x. Leave D2 as correct-by-construction.
RT subsuite with both switches on (ccalin030, 8x2): HII_region_simple PASS, HII_region PASS, gmc_cooling_rt FAILS on
urad_FIR (max rel 0.215 vs 0.1 tolerance; on/off 0.76-0.94 by n_H bin; D1 alone causes it, D2 alone < 0.1%). The
benchmark was made with the double count, so adopting D1 means regenerating gmc_cooling_rt's reference. Remaining
long tests (HII_region_subcycle, shu_M120, gmc_cooling, SN_singlestar, gmc_cooling_rt, HII_region; off then on) are in
`sbatch /mnt/ceph/users/mgrudic/starforge/M10_core/rt_energy_ab/tests/run_rt_subsuite_genoa.sbatch` (not submitted).
Decision: fix D1 on starforge_dev (now backed by a sensitive test) and regenerate the gmc_cooling_rt benchmark; D2
is harmless either way (keep for consistency or drop). Not merged; your call.

### 1.1a RT subsuite on a genoa node, switches off vs on (job 7177500, export 7d445bcf, 2026-10-05)
15 test ids, 48 ranks x 2 threads, 1h11m. Every test passes in both arms except `gmc_cooling_rt[baseline]`, which
fails with the fixes on urad_FIR only (max rel 0.216 vs 0.10; on/off per n_H bin 0.93 -> 0.76, T within 3.4%,
T_dust within 0.1%, T_rad within 1%). shu_M120 passes with the fixes (T 0.032, T_dust 0.034, T_rad 0.050) with
urad_FIR 0.31 off the benchmark (not asserted; 0.12 before). HII_region: r_IF and T_peak identical, box FIR energy
2.68e4 -> 1.75e4. SN_singlestar (all three variants), gmc_cooling, HII_region_subcycle and the snapshot-restart test
unchanged. Switch lines verified present/absent in every run log.
Also on rt_energy_ab (e1f2ea45, 963fa578, 83f6b1cc): an include guard on the generated GIZMO_config.h so
precompiler_logic.h's #undefs are no longer cancelled by eos.h's re-include (bit-identical for the three production
configs; it changes the outcome only for MHD_CONSTRAINED_GRADIENT=2+MODIFIED, a user-set NUM_ADDITIONAL_PASSIVESCALAR
value with ISMDUSTCHEM, and a bare SINGLE_STAR_AND_SSP_NUCLEAR_ZOOM, all to the originally intended values), and
`RT_ISRF_BACKGROUND=0` honoured as "no background" even under the STARFORGE umbrella (every ISRF site compiled out;
the cosmic-ray background then falls back to the unattenuated Milky Way value rather than zero).

### 1.1b IR-thick cells under COOLING: band energy drains into the gas (new, 2026-10-05)
rt_closed_box now runs at n_H = 1e3, 1e4, 1e5, 1e7 (rt_energy_ab 116d5b5a). Without COOLING the fixed build conserves
band energy to round-off up to tau_IR = 6.8e3 per cell. With COOLING at n_H = 1e7 (tau_IR ~ 2.5 per cell) the
bands + gas ledger loses 26% over the run (0.37% at 1e5), independent of D1/D2; strict xfail documents it. Half of
it is the gas share of IR absorption being deposited at c~ instead of c (an experiment at c halves the drain); the
other half is gas emission that the cooling step does not return to the bands, untraced. This is the RSOL
band/gas coupling convention, not a kick bug, and it is the regime of IR-trapped cores.
Decision: trace the second half (one agent-day) before deciding whether the RSOL convention needs a consistent
treatment on starforge_dev.

### 1.1c Full-STARFORGE closed box: D1 exact, and the RSOL coupling drains quantified (rt_energy_ab 0d109a0c)
rt_closed_box now also runs SINGLE_STAR_STARFORGE_DEFAULTS + SINGLE_STAR_FB_RAD + COOLING (all five bands, STARFORGE
cooling/dust/chemistry, gas cube inset in a larger box to dodge the ISRF edge reset), without sources and with a static
30 Msun star. D1 ledger exact at n = 1e3 and 1e7: 1.00000000 fixed / 2.00000000 unfixed. Band ledger closes to 1e-4
with every flow tallied (new per-band diagnostic, c8b1eab9). Not asserted, reported: under RSOL the bands + gas
budget does not close by construction. Named drains, cumulative: (1) IR gas-share heating at c~ (1:1 in band units;
25% of the band energy at 1e7), inconsistent with GIZMO's own RSOL ledger in kicks.cc:337-341 which values radiation
at c/c~ of gas energy; (2) cooling return at c~/c of the gas emission (limiters clip 9-27% of what is offered at 1e3);
(3) photoheating at c against ionizing-band loss at c~ while 100% of the absorbed ionizing energy is also donated to
OPT/NIR: ~16% x c/c~ of the absorbed energy is created; photoheating drives 83-99.8% of the cooling return.
Build-system defect found on the way: eos.h re-includes GIZMO_config.h after precompiler_logic.h's #undefs, so
RT_ISRF_BACKGROUND=-1 (and the #undefs of MHD_CONSTRAINED_GRADIENT, NUM_ADDITIONAL_PASSIVESCALAR_SPECIES_FOR_YIELDS_
AND_DIFFUSION, SINGLE_STAR_AND_SSP_NUCLEAR_ZOOM) cannot take effect from Config.sh. Not fixed (changes production builds).
Decision: whether the RSOL band/gas convention gets a consistent treatment (value every band/gas exchange at the
same c, and stop donating photoheated ionizing energy to OPT/NIR) on starforge_dev; it is larger than D1 in IR-thick
and photoionized gas. Needs its own design pass and benchmark regeneration.
Decision (MYG, 2026-10-06): deferred; not in the option-2 baseline. The jaco side already uses the consistent
convention (band rows at rsol), and `starforge_RT` will check it with the closed-box ledger.

### 1.1d The IR-thick drain, closed (rt_energy_ab 5f2d395b diagnostic; ledgers close to 5e-12)
The "untraced half" is not gas emission: the cooling step returns every erg it is offered (no limiter clipped in
any fixed_cooling run). Both halves are the gas share of IR absorption:
- Opening half-kick: `rt_utilities.cc:~890` writes the share into `DtInternalEnergy` as a rate over dt/2; the next
  `hydro_force_initial_operations_preloop` (hydro_toplevel.cc:~971) zeroes `DtInternalEnergy` for every active cell
  before anything applies it (run.cc order: hydro :125 -> first kick :152 -> hydro :256 -> second kick :266 ->
  cooling :548). Destroyed every step, every build. 53% of the drain at 1e7 and 52% at 1e5.
- Closing half-kick: applied once (split cooling, kicks.cc:263) or twice (unsplit, cooling.cc:1431 uses the dt/2 rate
  over dt; STARFORGE cells go unsplit after step 1, trigger not isolated) but at 1:1 in band units where the kicks.cc
  convention (E_band/(c~/c) + E_gas conserved) values it at c/c~. 53% / 51%.
- Mechanisms ruled out with numbers: return limiter (0 in split cells; inert there because cooling.cc:242 reads
  DtInternalEnergy in code units as cgs), gas-dust coupling (routed at exactly c~/c), kick-vs-cooling dust
  temperature (both balances close; T_dust differs 0.05-0.15 K, opacity only), band caps (0).
- STARFORGE-only extra: photoionization (cooling.cc:1246-1262) and photoelectric heating (:1286) are applied at c
  while the kick also donates the full absorbed band energy onward -> created energy; at 1e5 the bands GAIN 2.3%.
Experiment E2 (gas gains (c/c~) de_abs f_gas applied to u in each half-kick, not as a rate): bands +1.58% at 1e7
(= CR heating + IC relaxation re-radiated + BE residual) and +0.014% at 1e5; STARFORGE +2.84% (= photoion +
photoelectric double count + CR). All experiments reverted.
Consistent convention per site: IR gas share deposited at c/c~ once per half-kick to u; cooling return and dust
coupling keep c~/c (correct); limiter in one unit system, de_rad <= (c~/c) max(0, -(du - work)), drop de_u_radabs
(double-counts the deposit), fix the split-cell units; photoionization/photoelectric: donate absorbed minus (c~/c)
heat; CR background booked as injection. Predicted residual: BE tolerance only (~3e-4 at 1e7; ~0 if the return
uses the actual du). Ledger tool: scratchpad rtledger/ledger3.py.

### 1.2 Cooling-radiation return limiter under reduced c
`rt_cooling_radiation_to_bands`: `ratefact` carries c~/c but `de_u` does not, so the cap degenerates to "a band gains
cooling radiation only if the cell's internal energy fell this step". Measured effect in the tests < 0.7%.

### 1.3 Recombination: case-A rates with case-B photon routing (verified by reading, 2026-10-06)
`cooling/cooling.cc:1477-1479` tabulates Verner & Ferland (1996) total radiative recombination coefficients, i.e.
case A (H at 10^4 K: 4.19e-13 vs case B 2.59e-13). They drive the H/He balance (:813, :896-898) and the
recombination cooling (:996-1000). The photon routing assumes case B instead: :1316 ("we're usually assuming case B
recombination (UV emitted photons re-absorbed)") and :1317 send `LambdaRec` to the NUV band, and nothing returns
ground-state recombination photons to the ionizing band. Those photons are neither transported (case A) nor reabsorbed
on the spot (case B); they are lost. In photoionization equilibrium an HII region then needs alpha_A/alpha_B ~ 1.6x
the ionizing photons: Strömgren radius ~0.85x and ionized mass ~0.62x the case-B values at 10^4 K.
`radiation/rt_chem.cc:167` uses alpha_B but runs only without COOLING (`core/run.cc:483`), so every STARFORGE run
takes the case-A path. `test/iliev_test1` checks against the case-A R_S: consistent with the code, not with
Iliev et al. (2006), who use case B.
Fix options: (a) case-B rates and recombination cooling, keeping the cascade routing (a table change; HII regions
grow ~17% in radius, and the RT subsuite benchmarks move: HII_region*, gmc_cooling_rt, shu_M120, iliev_test1);
(b) keep case A and emit the alpha_A - alpha_B share into the ionizing band, which must then carry ~14.5 eV photons
without creating energy. Sensitive test: iliev_test1 against the case-B R_S (it measures 0.976-1.020 of the case-A
R_S, so ~0.85 of the case-B one). jaco: `starforge_legacy_RT` reproduces the case-A rate; `starforge_RT` will default
to case B with a case-A switch.
Decision (MYG, 2026-10-06): deferred; not in the option-2 baseline. `starforge_RT` takes case B on its own.

## 2. Cooling solver

### 2.1 First cooling call from a warm neutral start
`DoCooling`'s bracketed root-find stops on an x-tolerance only (`system/bracketed_rootfind.h`), so a discontinuity in
the root function with a sign change at the jump is accepted as a root: 89-98% of du unresolved on the first call
(measured by instrumentation). Harmless without RT (no H memory); with time-dependent H+ the wrong first-step
ionization persists. This is what the gmc_cooling_rt benchmark encodes.
Decision: regenerate that benchmark from ICs whose ionization matches their temperature (recommended), or hold the
jaco variant to a same-IC legacy run. Until then its jaco variants are xfail with this reason.
Decision (MYG, 2026-10-06): regenerate the gmc_cooling_rt benchmark from a cold neutral start (InitGasTemp ~100 K)
with the option-2 fixes. The root-find fix itself (x-tolerance only) is deferred to a later batch.

## 3. Idealized-test and build-system defects

### 3.1 Idealized-test defects
Continuous photon injection without gas sources never zeroes `Rad_Je` (accumulates every step; STARFORGE injects
discretely, unaffected). The `RT_ILIEV_TEST1` source emitted ~half the intended photons (fixed under that flag only).
The ISRF boundary resets T_dust to 20 K after the cooling solve: harmless in legacy (initial guess only).
Unverified: the claim that T_rad weighting sticks at the lowest temperature in optically thick cells.

## 3b. Found while closing the option-2 blast radius (2026-10-06)

### 3b.1 Uninitialized `dt_hydrostep_i` in the hydro flux loop (verified in source)
`hydro/hydro_evaluate.h:25` declares `dt_hydrostep_i`; lines 152 (`dt_hydrostep = DMAX(dt_hydrostep_i, dt_hydrostep_j)`,
the flux-limiter timestep for turbulent metal diffusion, conduction, viscosity) and 357 (MFV mass-flux holder) read it,
and nothing assigns it. Upstream 41f0a224 (Hopkins, 2025-06-18) removed `dt_hydrostep_i = local.Timestep * ...` while
introducing `local.dt_hydrostep_i` (set at hydro_toplevel.cc:406). So the limiters use stack garbage, layout-dependent:
a change elsewhere (F6's stack frame) perturbed a FIRE-3 run at 1e-3 in H2 fractions. With `dt_hydrostep_i =
local.dt_hydrostep_i;` in both arms the F6 A/B on test/fire is bit-identical over 16 snapshots. One-line fix; affects
every MFM/MFV build with diffusion/conduction/viscosity, and every A/B that relies on bit-identity.

### 3b.2 fire_rtsources cannot run at all (both base and fixed)
Aborts in init() at the first tree build (node arena exhausted after 18 growths): FIRE-3 reads IC positions in float32,
leaving 22 exactly coincident pairs; the tree only splits coincident particles below 1e-4 of the softening, and gas
KernelRadius is zero at flag 0 before the first build, so the pair subdivides forever (test/fire uses the same IC with
flag 2 and survives). With positions read in double the run dies at the first step: the IR band starts as a 20 K
blackbody filling the cell volume (rt_utilities.cc:1081; 3.7e8 vs thermal 1.3e-5 code units), the unbounded IR
diffusion coefficient in thin gas gives dt ~ 3e-23, and get_timestep's FIRE-RT clamp (timestep.cc ~829-833) never
raises it to the RT Courant step. This standing "known failure" therefore tests nothing; F1/F2/F3/F6 have no FIRE+M1 RT
coverage until it is repaired.

### 3b.3 F6 on FIRE-3 is arithmetically inert; F5 is live in test/fire
FIRE-3 never changes gamma across ThermalProperties (probe: 0 of 16.8M updates), so F6 is a no-op there. F5 is live
because test/fire restarts with flag 2: snapshot_001 T differs in 6073 cells, end-of-run H2 mass +21%, stellar mass
-0.13%, inside the test's tolerance.

## 4. Non-RT defects

### 4.1 Snapshot restart and sound speed
- `core/init.cc:1072-1078`: the snapshot-restart rebuild of MolecularMassFraction sits at function scope with
  i == N_gas and never touches a real cell. Fix = move it into the gas loop; changes non-JACO restart behaviour.
- d4e01a38 cached `Gamma` from the fresh temperature, but nothing reads `cell.Gamma`; `SoundSpeed` still uses the
  gamma computed before the temperature refresh, feeding the Courant step, the Riemann solver and the slope limiters.
  Fixing it changes every non-JACO result.
Recommendation: both on their own branches with a sensitive test each.

