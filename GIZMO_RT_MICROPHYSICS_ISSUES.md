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

### 1.1e The drain is per step, and the gas share counts the dust's scattering (rt_closed_box, 2026-10-09)
Found with the jaco closed-box harness (gizmo_jaco_dev branch `rt_closed_box_jaco`, e694dcec); legacy code at b01265d8.
Runs, A/B arms and scripts are in /mnt/ceph/users/mgrudic/jaco_push_2026-10/b_rt_energy_and_hii/jaco_cbox/.
- **Per-step loss (extends 1.1d).** Where a cell is IR-thick over one step (x = c~ kappa_abs rho dt >> 1; the RT
  Courant step gives x ~ 0.4 tau_cell), each half-kick absorbs nearly the whole band and hands the gas f_gas of it at
  1:1 (rt_utilities.cc:931). The cooling return re-radiates it at c~/c. So the E_gas + E_b c/c~ ledger loses ~2 f_gas
  of the band per step, whatever dt is: the drain per unit time scales as 1/dt.
  Closed box (STARFORGE, c~/c = 1e-4, 64 steps at 12 K): -26% / -48% / -34% of the ledger at n_H = 1e7 / 1e8 / 1e9.
  From LTE at n_H = 1e9: -2.5% at 200 K and -6.4% at 500 K.
  In shu_M120's n_H > 1e9 cells it is ~3e-4 per step, i.e. k ~ 2e3 per code time and k t ~ 1.4 over the run. That is a
  leading, timestep-dependent sink in the core. jaco's `starforge_legacy_RT` reproduces it.
- **Gas-share denominator.** f_gas = kappa(-1,-1)/kappa(0,0) (:843) divides by the total opacity. Its dust part lacks
  the absorption factor A(T_rad) = 1 - 0.5/(1 + 725^2/(1 + T_rad^2)) (:1860) that the absorption opacities carry. So
  the gas absorbs at A(T_rad) times its own kappa_gas: 1.000 at 12 K, 0.975 at 167 K, 0.84 at 500 K, 0.77 at 680 K.
  jaco's `starforge_legacy_RT` divides by the dust absorption opacity, so it drains 1/A more. Closed box: 1.0364
  measured vs 1.0367 predicted at 200 K, 1.1896 vs 1.1919 at 500 K.
  shu_M120 A/B (legacy with jaco's denominator; the base arm reproduces the reference to 1e-4), median ratios to the
  reference at snapshot 5, T / u_IR:

  | cut | A/B | jaco starforge_legacy_RT |
  |---|---|---|
  | n_H > 1e9 | 0.969 / 0.881 | 0.971 / 0.886 |
  | n_H > 1e8 | 0.980 / 0.917 | 0.982 / 0.914 |

  This fully explains the shu_M120 thick-gas gap between the two codes. It grows as T_rad rises: the mass-weighted
  1/A goes from 1.009 to 1.074.
- **Cosmic rays.** With `RT_ISRF_BACKGROUND=0`, legacy keeps unattenuated cosmic-ray heating (+23% of the ledger at
  1e9); jaco attenuates it by column.
- **Loose end.** Around a star the IR kick sources do not sum to zero: 0.6-0.8% of the injected energy, in both codes.
  Untraced.

Classification: the A(T_rad) denominator is a legacy defect: gas absorption is undercounted by A. jaco departing from it
is a fidelity defect against `starforge_legacy_RT`'s brief. The per-step drain belongs to the deferred convention
decision (1.1c-d); `starforge_RT` removes it by construction.
Decision (MYG, 2026-10-09):
- Gas-share denominator: `starforge_legacy_RT` takes GIZMO's A(T_rad), so it tracks legacy term for term. The legacy
  defect stays recorded for a later baseline batch.
- Per-step drain: fix it in baseline GIZMO now, with 1.1d's per-site convention: the IR gas share deposited at c/c~
  once per half-kick to u; the opening half-kick's deposit no longer destroyed; the return limiter in one unit system.
  rt_closed_box asserts the full ledger. The RT benchmarks are re-baselined, and `starforge_legacy_RT` tracks the
  fixed convention.

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

### 1.4 Photoionization over-counts injected photons; injection reprocessing hides it (unresolved-Strömgren probe, 2026-10-09)
Found with `test/stromgren_unresolved` (branch `stromgren_probe` e6f9f258). Setup: a fixed 4e4 K source in a uniform box,
Q swept so that R_S,B/Δx = 0.1-10, at n_H = 1e2 and 1e5. Runs are in /mnt/ceph/users/mgrudic/jaco_push_2026-10/b_rt_energy_and_hii/jaco_stromgren/.

- **Defect.** The cooling module's photoionization rate c σ n_γ (`cooling.cc` ~912-925) is applied to the discretely
  injected lump without limiting it to the photons present. Each injected photon then ionizes ~c~ σ n_HI Δt atoms:
  63 measured vs 66 predicted at n_H = 1e2, and 820 vs ~930 at 1e5.
  - With `RT_REPROCESS_INJECTED_PHOTONS` off, the ionized mass is 9-2300x the Strömgren mass in unresolved cases.
  - With it on (production), resolved fronts still carry 16-100% excess ionized mass against the case-A reference:
    1.16 M_S at R_S = 10 Δx, 1.46 at 3 Δx, 2.0 at 3 Δx and n_H = 1e5.
- **What the construct does.** Its R_HII estimate is 0.72 R_S,B, and a cell is exempt only if max(r, Δx) < R_HII. So for
  R_S,B ≲ 1.4 Δx every injected ionizing photon is turned into optical light at injection: no ionization, no
  photoheating, no thermal pressure.
  - At R_S,B ~ Δx the ionized mass is therefore a step function of resolution (0 → 1.51 M_S at 8x mass resolution).
  - Radiation pressure can flip it by lowering the kernel density.
- **Classification:** legacy defects.
  - The construct is load-bearing for legacy photon conservation, so it must stay on in legacy until the
    photoionization rate is limited to the photons present.
  - jaco `starforge_RT` counts photons exactly (absorbed/injected EUV = 1.000) and drops the construct.

Status: recorded, no decision.

### 1.5 Under RT_INFRARED, CoolingRate's sub-cell IR self-absorption double-counts trapping (by reading, 2026-10-12)
`cooling/cooling.cc:1540-1546`, under `RT_INFRARED`, multiplies every heating and cooling term by
f = 1/(1+tau_self^2), with tau_self = kappa_gas(T,T) * rho * dx/2. The terms scaled are Heat, Lambda, LambdaMetal,
LambdaExc, LambdaRec, LambdaIon, LambdaPElec, LambdaFF, LambdaMol, LambdaDust and LambdaCompton. It is meant to correct
for trapping in unresolved optically thick cells. Two problems:
- **Trapping is counted twice.** The emission that survives f still enters the M1 bands through the cooling return,
  and the RT kick absorbs band energy in the same cell, partly through the same gas opacity (the gas share). The
  `RT_INFRARED` build already sets `COOL_LOWTEMP_THIN_ONLY` "so as not to double-count trapping"
  (precompiler_logic.h:811), but this factor escapes that guard.
- **It scales terms that are not radiation:** photoelectric and cosmic-ray heating (Heat) and gas-dust collisional
  exchange (LambdaDust).

**Size.**
- In molecular gas at 1e-3 Msun cells, GIZMO's gray kappa_gas (~7e-4 cm^2/g) gives tau_self <~ 1e-3, so f ~ 1 until
  n_H ~ 1e13.
- Not yet measured in warm, dense, partly ionized gas (2000-8000 K). There kappa_gas's H-/Kramers term is large, and
  tau_self may reach order unity.

jaco's `starforge_legacy_RT` reproduces f for parity (rule `GIZMO_IR_SELF_ABSORPTION`); `starforge_RT` has no f.
Fix options:
- (a) drop f under RT_INFRARED, consistent with the COOL_LOWTEMP_THIN_ONLY intent;
- (b) restrict f to the radiative terms.
Sensitive test: the LTE closed box, or a cell with tau_self ~ 1, comparing the net exchange with and without f.
Measured on the M2e3 shakedown (Rusty 7222668, 2026-10-13), at t ~ 0.79 when the first sink forms, with the factor
validated to round-off against GIZMO's compiled opacity:
- max 1 - f = 5e-9;
- the cooling and heating removed are ~1e-14 of the totals;
- a pre-sink core at n_H = 8e10 reached tau = 0.015.
At 1e-3 Msun cells, tau ~ 1 needs n_H >~ 5e10 even in warm, partly ionized gas. So f matters only in dense protostellar
envelopes, not in HII regions. Tool: `m2e3_jaco_shakedown/measure_selfabs.py`, also run by `watch_shakedown.py`.
Status: recorded; negligible at production resolution so far. Recheck after HII regions form.

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
Update (2026-10-13): the unconverged root-find is not confined to the first call, and it now has a fix; see 2.2. The
cold-start benchmark (6602963f, 95e37029 on starforge_dev) does not depend on it: old and fixed headers agree to 0.7%.

### 2.2 The root finder's early exit leaves DoCooling unconverged on nearly every call (fixed on `rootfind-brent-fix`)
`system/bracketed_rootfind.h` is the Brent block behind DoCooling (`cooling.cc:421`), the H2-fraction solve (`:2136`)
and `rt_ir_lambdadust`. Its use in `rt_eqm_dust_temp` sits inside `#if (0)`. Line numbers at 3b0923b5. Three defects:
- **A sub-tolerance step was treated as convergence, and the unevaluated interpolant was returned.** This came in
  with 8821246e (2024-07-02, "perf: ... quits if the interpolant iterate looks converged"). STARFORGE v1.3 (168cdb3d)
  has it; v1.2 (30459741) does not.
- The tolerance came from the proposed point and was not recomputed after a bisection, so the block could exit with
  a wide bracket.
- When the ends did not bracket the root, they were widened by /1.1 and *1.1, which walks away from the root for a
  negative variable and ends in endrun.

**Measured in gmc_cooling_rt** (logger after the DoCooling solve, no extra evaluations, about 36,000 solves per run):

| | old header | fixed header |
|---|---|---|
| solves returning an unevaluated point | 99.9% | 0% |
| final bracket width / u, median | 1.5% | 9e-5 |
| solves uncertain by >1% of u | 60% | 0% |
| solves uncertain by >10% of u | 8.7% (max 58%) | 0% (max 0.65%) |

The step's own |du|/u is about 1.2%, so the old answer was usually uncertain by the whole step.

**Effect on gmc_cooling_rt.** All of the measured simulation-level effect comes through the warm start (2.1), not the
equilibrium state.
- Warm start (InitGasTemp 1e4 K, 3b0923b5): the fix raises T by 3-21%, x_e 4-5x at n_H = 280-770 cm^-3, FUV by
  2-18% and FIR by 12-17%. Swapping the header one call site at a time puts 100% of this on DoCooling.
- Mechanism (tracer-logged): the first step (dt = 3820 yr, much longer than the cooling time) sets H+ through the
  `RT_CHEM_PHOTOION` implicit update nH0 = (HI + f*alpha)/(1 + f*(alpha + Gamma_coll + Gamma_ph)), f = dt*n_e, at the
  end-of-step T. With n_e ~ n(H+), this has a threshold at a = dt*n_H*Gamma_coll(T) = 1, about 13.8 kK at n_H ~ 150.
  - The converged first step ends at 13.8 kK (a ~ 1) and leaves H+ = 2e-5 to 1.6e-3.
  - The old header ends at 12.3 kK (final bracket 44% of u, a ~ 0.22) and leaves H+ = 2e-7. HII = a*x_m/(1-a)
    reproduces this to 2%.
  - Afterwards H+ only recombines, so the 450x first-step gap becomes the 4-5x x_e gap at 1 Myr.
  Neither answer is physical: it is an end-of-step ionization over a step much longer than the cooling time, at a
  threshold.
- Cold start (InitGasTemp 100 K, current starforge_dev eafcbd5b): old header, fixed header and the 2.3 fixes agree
  to 0.7% in every bin of T, Tdust, x_e, FUV and FIR, and all three pass the cold-start benchmark.

The only other evidence of wider impact is per call, not per run: from fresh states (dust-temperature harness), the old
header put 17 of 150 sampled ionised cells (5e3-3e4 K) 10-25% too cold, up to 47% from the root in u.

Not fixed by this: the stop is still on bracket width, so a sign change at a discontinuity of the residual is still
accepted as a root (2.1). The same thing happens in strongly irradiated envelope cells, where the dust equilibrium
switches branch as T changes. DoCooling lands on that jump in about half of 50 sampled envelope cells, whatever the
dust solver.

- Fix: commits 53efd51c and cf468206 on branch `rootfind-brent-fix`, both pushed.
- Sensitive test: `test/rootfind_unit`, 17 bracketed problems, no GIZMO build. The pristine header fails 10-11 of
  them; the fix passes all 17.
- Cost: about 15% more cooling-function calls per gas solve.

**Status:** merged onto origin/starforge_dev eafcbd5b (a05459db on `rootfind-brent-fix`, clean merge, not yet pushed). No
benchmark needs regenerating. Whether v1.2 vs v1.3/dev runs differ because of this is untested: v1.3 (168cdb3d)
has the early exit and v1.2 (30459741) does not, but the cold-start test shows no effect.

### 2.3 Equilibrium dust-temperature solvers miss their roots (fixes on `dust-solver-fullset`)
Found with a stress harness that links the production objects and brute-force scans each solver's residual
(`test/dust_temp_harness`, uncommitted, in `~/code/gizmo_starforge_dev`).

**`rt_ir_lambdadust`** (the cooling-side solve, called inside DoCooling):
- **It pins at MAX_DUST_TEMP = 1e4 K, returning LambdaDust = 0, although its residual has a root at 600-2000 K.**
  This affects 17% of reachable protostellar-envelope states (n_H 1e7-1e11, T_eq ~ 300-3000 K) and 12% of a
  20,000-state random scan.
- Cause: the residual has a stable lower root and an unstable upper one. The solver starts above the upper root,
  climbs to the ceiling and never searches downward. The upper root exists because non-IR absorption uses the dust
  fraction at the *previous* Dust_Temperature, so absorption does not fall with sublimation while emission does.
- **Its answer depends on the dust temperature it is entered with.** One cell (n_H = 1e10, T = 300 K,
  u_opt = 1e8 eV/cm^3) gives 1e4 / 618 / 317 / 127 K when entered from 300 / 680 / 2000 / 9000 K. Inside DoCooling,
  each call seeds the next, so the gas residual depends on the order of evaluation.

**`rt_eqm_dust_temp`** (RT kick side; its result is what snapshots record):
- 37% of random states end off the root.
- In HII gas the cause is a step-size exit, |d(T-Tdust)| < 1e-3 |T-Tdust|, which allows ~10 K errors when T >> Tdust.
- Below 3000 K the near bracket end is never tightened.
- Consequence: snapshot Dust_Temperature in HII regions is unreliable at the factor-2 level.

**Fixes on `dust-solver-fullset`** (62426f7f, with P5 dropped in 3495bca3; pushed):
- P1: don't start from capped fixed points; walk down from the start point to the lowest root before declaring
  sublimation.
- P3: `rt_eqm_dust_temp` converges on bracket width.
- P4: the residual uses the dust fraction at the trial Tdust. This is a physics change: 16% of previously converged
  cooling-side answers move, mostly cold-biased hysteresis cases.
- P6: cap the climb acceleration.

Rejected, P5 (Brent on Tdust instead of Tdust - T): it resolves Tdust but not T - Tdust, which is what LambdaDust
depends on. In tightly coupled gas T - Tdust is 1e-10 to 5e-3 K, and P5's errors were 0.015 to 3e5 times T - Tdust,
including the sign.

**Results on the 20,000-state scan:**
- Spurious ceiling pins: 2391 -> 72.
- `rt_eqm_dust_temp` off its root: 7431 -> 1.
- No dependence on the entry dust temperature.
- Cost: `rt_ir_lambdadust` goes from 13.4 to 17.8 us per call (measured with P5 in place).

**Sensitive test: `test/dust_temp_solvers`.**
- It runs 50 cell states against brute-force roots of each solver's own residual, so physics changes that move the
  root do not break it. It takes about 10 s, including the GIZMO build.
- It fails 36 cases on 3b0923b5, 34 with only the header fix, 8 with P5, and 0 on the branch.

**Still open:**
- The envelope discontinuity in 2.2.
- `rt_eqm_dust_temp`'s residual contains the previous Tdust through `fac_abs` = fdust(Tdust)/fdust(Tdprev). This
  rescales gas-floored absorption by up to 1.4e6 after a hot excursion: a cold ISRF cell entered from 1e4 K moves
  from 16.3 K to 59 K.
- The solvers' own NaN guards are dead under -ffast-math, like every isnan in the build.

**Status:** merged onto origin/starforge_dev eafcbd5b as 681ee28c on `dust-solver-fullset` (not yet pushed). The only
conflict was c1c7c9eb's floor and iteration cap in `rt_eqm_dust_temp`'s bracketing loops; both they and P3's bracket
tightening are kept. On eafcbd5b without these fixes `test/dust_temp_solvers` still fails 36 cases: the guards stop the
forgedinfire hang but not the misses. Not merged into starforge_dev.

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

### 4.2 C+ 158 µm cooling normalised ~3-4x low (possible; checked against the cited rates, 2026-10-08)
`cooling/cooling.cc:1163`: `Lambda_Cplus = Z_C * 4.7e-28 * (T^0.15 + 1.04e4*n_elec/sqrt_T) * exp(-91.2/T)`, per n_H^2,
times the neutral fraction (:1188). The comment cites Barinovs et al. (2005) for H and Wilson & Bell (2002) for e-,
"assuming factor of 0.5 depletion". The per-ion rates are right. T^0.15 fits the shape of Barinovs' Eq. 12,
L = e^(-91.2/T)(16 + 0.344 sqrt(T) - 47.7/T) 1e-24 erg cm^3 s^-1 per C+ per H, to +-4% over 20-2000 K. At the same
abundance the e- term matches GJ07's C+-e- rate per ion to 3%.

The abundance is the problem. Dividing by L gives x_C+ = 4.9e-5 at Z_C = 1 (+-4% over 20-2000 K). The stated half-solar
C/H is 1.94e-4 with the default `SolarAbundances[2]` = 3.26e-3 (`core/init.cc:198`, AG89), or 1.43e-4 with the
proto-solar 2.53e-3 (:209). So both channels, which share the prefactor, are 4x or 2.9x low. GIZMO's dense-gas CO/C
term, which the C+ term hands off to, was not checked.

A second, smaller point: the e- term is also multiplied by the neutral fraction, which suppresses C+-e- cooling in
partly ionized gas.

Effect: jaco's starforge model with the GJ07 C+ rate (6.3x per ion over its previous coefficient, added together with
O I/Si+/Fe+ lines) takes the gmc_cooling median T at n_H ~ 13 from 112 K to 42 K (legacy: 158 K); an e^(-92/T)
estimate attributes the drop to C+. Runs in /mnt/ceph/users/mgrudic/jaco_fs_gates.
Fix: rescale 4.7e-28 to the stated depletion. This changes every run with cold atomic gas. `starforge_legacy_RT` keeps
the current coefficient. Sensitive test: Lambda_Cplus against Barinovs' L(T) times 0.5 solar C/H.
Status: recorded, no decision.

### 4.3 SN ejecta overrunning long-timestep cells inject spurious energy (fullphysics_mini, 2026-10-11)
Found while attributing a jaco/legacy gap in `test/fullphysics_mini` (branch `sn_parity`; runs and analysis in
/mnt/ceph/users/mgrudic/jaco_push_2026-10/d_bugfixes/sn_parity/). That test is a 50 Msun cloud with a 30 Msun star going SN at t = 0.086.
- About 0.23 kyr after the SN, the ejecta reach coarse envelope cells on long timesteps (kernel ~0.2 box, n_H ~ 1e2,
  ~1e4 K, ~600 ejecta cells inside one kernel). Those cells jump to 1e9-1e11 K and ~1e4 km/s.
- Total gas energy jumps by up to 1.1-1.7 E_SN within one ~10 yr output interval. Legacy and jaco both do it, from
  identical pre-jump states.
- With `MaxSizeTimestep` 1e-5 instead of 1.25e-3 the jump disappears in both codes, and their energy histories match.
- Likely cause: the long-step cells are not shortened (WAKEUP) before the ejecta arrive, so they take fluxes over a
  step far longer than the signal-crossing time. Not isolated.
- Consequences:
  - It inflates post-SN dense-gas temperatures by 0.1-0.26 dex in that test. Legacy, as the reference, is wrong there.
  - It is a production concern wherever SN ejecta meet coarse, long-step gas.
Classification: legacy GIZMO energy-conservation defect (timestepping), independent of the cooling module.
Status (2026-10-13): **the jump is a snapshot-restart artefact.** Every run above was a flag-2 restart from snapshot_017 at
t = 0.085, 1.1 kyr before the SN, which re-synchronises every coarse cell onto a fresh maximum step (1.2 kyr) that the
ejecta then hit mid-step. A continuous baseline from t = 0 (ab_snwarn/control2, same tree and build) never exceeds
0.9987 E_SN and its peak temperature is 5e8 K, not 5.8e10 K. The 0.1-0.26 dex post-SN inflation attributed to this
needs re-examination: the test's own two legs resume with flag 1, which preserves the hierarchy.
What the continuous run does show is that the gas has no warning of the SN: the feedback timestep limiter advertises
the ejecta speed only at ProtoStellarStage 6, set as the ejecta are launched. With the speed advertised 3 maximum steps
before the (deterministic) death (`SINGLE_STAR_FB_SNE_WARNING_STEPS`, sn_parity), the remnant's energy history tracks a
MaxSizeTimestep = 1e-5 reference to 1-2% at every output, where the unwarned run lags by up to 30% at +200 yr
(0.92 vs 0.70 E_SN), for +2% steps in the window and fewer steps overall. Runs: ab_snwarn/ (control2, warn, dtref).
