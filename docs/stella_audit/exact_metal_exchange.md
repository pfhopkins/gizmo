# METALS_EXACT_PAIRWISE_EXCHANGE: metal mass exact with individual timesteps (opt-in)

Series: correctness option on the base, following the design in issue #35; branch `up/turb-diff-metals-exact-pairs`. Independent of the Stella series.
Stella switch pending: Stella still uses its own exact-conservation option (met/06). In a first Stella test with this option, four realisations of a small cloud (12 Myr) formed 1.04, 1.37, 1.92 and 2.65 times the stellar mass of the current option (the 1.37 case against a run that stopped early); under investigation. This PR does not depend on that.
Covers: turbulent metal diffusion and MFV metal advection with individual timesteps. Source +76 -9 (one PR, no split into diffusion and advection).
Order: none required.
Base: <BASE>. Tip: <TIP>. Numbers below were measured on 070c5b78f unless marked otherwise.
Test: `test/metal_exchange_timebins`, narrow (one bookkeeping property); a temporary check that is not proposed for the permanent suite and can be dropped from the PR. Full suite: <SUITE RESULT>.

## What and why
With individual timesteps each cell applies only its own step's share of a pair's turbulent metal exchange, and the MFV metal advection term is one-sided, so metal mass is conserved only to integration accuracy. As outlined in #35, this option handles both like the MFV mass flux: the endpoint with the shorter step owns the pair, computes the exchange over its own step, and applies it to itself and, with the opposite sign, to its neighbour. With equal steps each side applies half.

## Evidence before
`test/metal_exchange_timebins` IC: 20^3 periodic box, 1e3 K | 1e7 K at equal density, 5-6 occupied timebins, 5 metal species including a step into exactly zero. MFM, default exchange, 4 ranks: metal mass drifts by 1.6e-5 (relative, worst species) at step 48 and 2.4e-5 after 1046 steps; bar 1e-12. (On an older base, 43025c83, 3.66e-6 at step 48 in an earlier box; measured on an older base.)

## Change
New flag `METALS_EXACT_PAIRWISE_EXCHANGE`, default off. It needs METALS with TURB_DIFF_METALS or MFV. `#error` with `GALSF_ISMDUSTCHEM_MODEL` or `TRANSPORT_SUBCYCLE`, which are not implemented: the dust fields in Dyield are not all per unit mass, and the subcycle walker reruns the pair body.
- `hydro_pair_types.h`: the ownership rule, `PAIR_EXCHANGE_I_OWNS` / `PAIR_EXCHANGE_SPLIT`, shared by the MFV mass flux (both blocks in `hydro_functions.h`, same comparisons) and the metals.
- `turbulent_diffusion_functions.h`: with the flag a pair is computed only by its owner (or by both, half each, at equal steps), with limiters over the owner's step instead of max(dt_i, dt_j). Positivity on the pair: the donor gives at most 1/(2 DesNumNgb) of its metal mass to one partner per step. out.Dyield += share x exchange, and an atomic add of -share x exchange into CellP[j].Dyield. The 0.01 Z floor is unchanged.
- `hydro_functions.h`: MFV metal advection under the flag: the owner moves Z_upwind x dm (same dm and share as the mass flux) to both cells.
- `hydro_toplevel.cc`: Dyield reused, apply then clear. At the end of every hydro step every gas cell, active or not, applies its Dyield and clears it. Nothing stays pending, so merge, split and spawn need no bookkeeping.
- `kicks.cc`, `merge_split.cc` (MFV only): metal mass is MassTrue x Z. The kick keeps it when MassTrue changes, and a gas merge mixes Z by the parents' MassTrue.
- `hydro_force_loop.cc`: `GHOST_WRITEBACK_GAS_ADD_ARRAY(Dyield, NUM_METAL_SPECIES)` and the ghost pre-zero, as for dMass.
- `precompiler_logic.h`: the `#error` checks. Docs paragraph, `Template_Config.sh` line, two compile-suite rows.
- No budget printout.

## Cost
Measured first. CPU, Kokkos OpenMP, 4 ranks, 64000-cell box of the test, one largest step = 1185 steps, two repeats agree to 0.7 %. Mean per step:

| | steps | total off | total on | hydro off | hydro on | pair kernel off | pair kernel on |
|---|---|---|---|---|---|---|---|
| second order, tiny (<1% active) | 71 | 8.0 ms | 8.9 ms | 2.6 ms | 2.7 ms | 0 | 0 |
| second order, medium | 1091 | 475 ms | 491 ms (+3.4%) | 133 ms | 137 ms (+3%) | 166 ms | 181 ms (+9%) |
| second order, all active | 22 | 676 ms | 705 ms (+4.3%) | 171 ms | 180 ms (+5%) | 268 ms | 294 ms (+10%) |
| LOWORDER, tiny | 71 | 6.8 ms | 8.0 ms | 2.0 ms | 2.3 ms | 0 | 0 |
| LOWORDER, medium | 1091 | 445 ms | 460 ms (+3.5%) | 127 ms | 130 ms (+3%) | 141 ms | 155 ms (+10%) |
| LOWORDER, all active | 22 | 622 ms | 650 ms (+4.4%) | 164 ms | 169 ms (+3%) | 224 ms | 247 ms (+10%) |
| whole run (wall) | | 536 / 502 s | 554 / 519 s (+3.4% / +3.4%) | | | | |

Tiny steps: +0.9-1.2 ms, at the 10 ms timer resolution; it includes the pass over all gas cells that applies and clears Dyield. Counts relevant on GPU (from instrumented runs, not part of the PR): 31-32 owned pairs per active cell per step. The flag adds one j-side atomic per species per owned pair: 149-155 per active cell per step with 5 species for diffusion (MFM), 318-321 with MFV advection, where the default has one (dMass, MFV only). Ghost write-back: NUM_METAL_SPECIES doubles per ghost (40 B here against 8 B for dMass; about 7700 ghosts per rank per hydro call, about 0.3 MB). With `GALSF_ISMDUSTCHEM_MODEL=15` and 16 grain-size bins NUM_METAL_SPECIES is about 200 (1.6 KB per ghost), which is why that combination is refused for now. Not measured: any GPU time or GPU atomic contention (no GPU available); dust-chemistry and subcycling builds (refused by `#error`).

## Evidence after
Instrumented copies of the branch print, per step, the long-double sum of MassTrue x Z per species and of MassTrue + pending dMass (never committed). Worst case first, 20^3 deep box, to t = 0.0125:

| case | ranks | steps | timebins | worst species error | in eps | mass error | min Z | floor hits |
|---|---|---|---|---|---|---|---|---|
| MFM second order | 1 | 1046 | 5 | 2.6e-16 | 1.2 | 0 | 0 | 0 |
| MFM second order | 4 / 2 | 1046 | 5 | 2.2e-16 / 1.9e-16 | 1.0 / 0.9 | 1.3e-17 / 0 | 0 | 0 |
| MFV LOWORDER | 1 / 2 / 4 | 2224 | 6 | 2.1e-16 / 8.9e-17 / 1.2e-16 | 0.9 / 0.4 / 0.5 | 1.2e-16 | 0 | 0 |
| MFV second order, 2 ranks x 2 threads | 2 | 2224 | 6 | 1.8e-16 | 0.8 | 6.9e-17 | 0 | 0 |
| MFV second order | 1 / 2 / 4 | 2224 | 6 | 8.0e-17 / 1.7e-16 / 1.3e-16 | 0.4-0.8 | 1.0e-16 | 0 | 0 |
| MFV, diffusion x 1e4 | 4 | 2511 | 7 | 1.6e-16 | 0.7 | 5.0e-17 | 0 | 0 |
| MFM LOWORDER | 1 / 2 / 4 | 1046 | 5 | 1.1e-16 / 7.7e-17 / 7.8e-17 | 0.3-0.5 | 0 | 0 | 0 |
| MFM, diffusion x 1e4 (positivity limit active: 12406 capped pair exchanges) | 4 | 1877 | 8 | 3.0e-17 | 0.1 | 2.3e-17 | 0 | 0 |
| default exchange, MFM | 4 | 1046 | 5 | 2.4e-5 | | | | |

The residual is storage round-off: the same MFV run with Metallicity and Dyield stored in float gives 3.4e-8 (0.3 float eps), 2e8 times the double result. It does not grow with step count (0.4-1.2 eps up to 2511 steps). In MFV the positivity cap never binds (the existing limiter on the MFV mass-flux rate is tighter). In MFM it binds, and with the cap removed the floor still does not fire in this case (12382 would-be caps, 0 floor hits, 0.2 eps), so this case shows the cap is active but not that it is needed.

Committed check `pytest test/metal_exchange_timebins`: MFM at every snapshot on 1/2/4 ranks; MFV second order and LOWORDER at the synchronised end (mid-run MFV snapshots hold the drifted mass, not MassTrue); and the default-exchange control.

Further checks, flag on, worst case first, all pass (bars written before the runs):

| check | worst result | bar |
|---|---|---|
| comoving EdS box (a 0.163 -> 0.16608, 12.2 Myr), MFM / MFV | closure 2.4e-16 (1.1 eps) / 1.4e-16; metal into the cold half 0.988 / 0.989 of the physical run | 1e-12; 10 % |
| restart from restart files (stopped by TimeLimitCPU at step 437 / 899, resumed), MFM / MFV, closure against the IC across the restart | 2.2e-16 (1.0 eps) / 1.3e-16 | 1e-12 |
| restart from mid-run snapshot_002 (MFM) | 3.7e-17 against the snapshot, 2.2e-16 against the IC | 1e-12 |
| pending at restarts and outputs | max over cells of abs(Dyield) at every step end = 0 exactly (applied and cleared in the same hydro step), so nothing is pending at any restart, output, merge, split or star formation | 0 |
| 8 ranks MFM / MFV; 1 rank x 4 threads MFV | 2.2e-16 / 1.0e-16; 1.3e-16 | 1e-12 |
| merge and split (masses spread 50 %, thresholds in the header): MFV 912 splits, 624 merges; MFM 228 splits, 2 merges | 1.4e-16 (MFM) / 9.0e-17 (MFV) at every step | 1e-12 |

## Outline of issue #35, point by point
File:line at the branch tip.

| point | where | done |
|---|---|---|
| shorter step owns the pair and computes it with its own step; equal steps half each | `turb/turbulent_diffusion_functions.h:44-48` (skip unless owner, share 1 or 0.5, dt_pair = dt_i) | yes |
| Dyield reused, apply then clear | `hydro/hydro_toplevel.cc:426-439` | yes; departure: applied for every gas cell at the end of every hydro step, not only when the cell is next active (see below) |
| atomic add into a persistent per-cell accumulator with gas ghost write-back | `turb/turbulent_diffusion_functions.h:114`, `hydro/hydro_functions.h:401`; `hydro/hydro_force_loop.cc:79-80` (GAS_ADD_ARRAY) and `:203-204` (ghost pre-zero) | yes |
| one ownership helper shared with the MFV mass flux | `hydro/hydro_pair_types.h:44-47`, used in `hydro/hydro_functions.h:388,390,475,478` and the diffusion | yes, as two comparison macros, not a function, so the default build keeps identical instructions |
| positivity on the pair exchange, not a floor after the update | `turb/turbulent_diffusion_functions.h:108-112` | yes; the value 1/(2 DesNumNgb) is a choice made here |
| 0.01 Z floor unchanged | `hydro/hydro_toplevel.cc:438` (same DMAX(.., 0.01 Z)), default path `:301-302` untouched | yes; never fired in any run |
| MFV metal advection under the same option | `hydro/hydro_functions.h:397-401` | yes; as a full upwind flux (see below) |
| no permanent budget printout | none added | yes |
| measure cost: tiny, medium, all-active steps; LOWORDER and second order; CPU and GPU | cost table above | CPU yes; GPU: counts only (atomics per active cell, write-back bytes), no GPU timing |

## Additions beyond the outline, and why
- `core/kicks.cc:291-292` (MFV): when the kick changes MassTrue, Z is rescaled by MassTrue_old/MassTrue_new so MassTrue x Z is unchanged. MFV applies the mass change at the kicks and the metal change at the hydro step; with Z held fixed under mass change (the default) the implied metal Z_cell x dm uses Z at kick time, not at pair time, and closure fails at the integration-error level.
- MFV advection form: the owner moves Z_upwind x dm to both cells instead of the default one-sided correction (Z_j - Z_i) x dm. Same to first order; with the kick rescale it makes metal move only through antisymmetric pair terms.
- `mesh/merge_split.cc:1106-1107, 1360-1361` (MFV): a gas merge mixes Z by the parents' MassTrue, not the drifted Mass. Without it, merges alone gave 1.5e-8.
- `declarations/precompiler_logic.h:1939-1945`: `#error` with `GALSF_ISMDUSTCHEM_MODEL` (the dust slots of Dyield include absolute numbers and bin masses that are not per unit mass, so apply-then-clear and the kick rescale do not carry over) and with `TRANSPORT_SUBCYCLE` (its walker reruns the full pair body every subcycle, which would repeat the j-side writes); also an `#error` without METALS plus TURB_DIFF_METALS or MFV.
- Per-pair cap 1/(2 DesNumNgb) of the donor's metal mass: not a strict bound (a cell can have more than 2 DesNumNgb pair partners). It bound only in strong diffusion with MFM, and there the existing limiter alone also kept every cell positive (cap removed: 0 floor hits).
- dt_j inside the diffusion function is taken from Pj (`turbulent_diffusion_functions.h:45`) to keep the function signature, and so the default build's objects, unchanged.

## Departures from the outline, named
1. Apply-then-clear for every cell every hydro step. It keeps donors' Z current when several owners take from an inactive cell, so the pair cap acts on the real metal mass, and it leaves nothing pending at restarts, outputs, merges, splits and star formation (all checked above). Cost: one pass over the gas per step (+0.9-1.2 ms on tiny steps here).
2. The shared helper is a pair of macros.
3. MFV advection as a full upwind flux plus the kick rescale.
4. No GPU timing.

## Flag off
objcmp of every object against the base, configs MFV, MFM and MFV+LOWORDER of the test: 137 of 140 byte-identical in each. `hydro/hydro_force_loop.o` and `mesh/merge_split.o` differ only in `__LINE__` immediates of error calls (lines shifted by 3-7); `core/main.o` only in the build timestamp; no instruction differs. Flag off: to be re-run on <TIP>.

## Limits
CPU only: no GPU timing or atomic contention measured. Refused with the ISM dust model and with transport subcycling. The per-pair cap is not a strict bound. The test is a bookkeeping check, offered as a temporary gate.

## Depends on / followed by
None. One PR, +76 -9 source lines. The Stella metals PR that needs exact metal conservation waits for this option (Stella switch pending, see the first line).
