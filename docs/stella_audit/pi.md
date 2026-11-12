# Per-direction photoionisation (series pi)

Series name: pi. Covers: single stars as sources of the HII-region loop with the table photon rate (02, 03), and the opt-in per-direction photoionisation model `GALSF_FB_PHOTOIONIZATION_PER_DIRECTION` (04a to 11). pi/01 (HII loop without RT links) is the same change as up/B4 and is not repeated here.

Order and dependencies. Series order: slice2 -> chem (chem/12 before tc) -> tc -> gap -> cr (cr/04 before pi/06) -> met -> pi -> ext -> tr -> kj. Inside pi each branch sits on the one before. pi/02 needs slice2/15 and up/B4. The ledger row of pi/06 goes into cr/04's table of ledger rows. Base: <BASE>. Tip: <TIP>. Full suite on the tip: <SUITE RESULT>.

Numbers below (test results, flag-off object comparisons, timings, benchmark values) were measured on an older base (43025c83 and earlier) and are to be re-measured on <TIP>. Every flag-off line below is to be re-run on <TIP>. Nothing is pushed. Tests ran with Kokkos OpenMP on CPU, not yet on a GPU.

How "flag off" is checked: the base and the PR tip are built in two directories of the same path length with the same Config.sh. Every object file is compared and objdump is used for those that differ. Two configs: the gmc_cooling test config, and FIRE_PHYSICS_DEFAULTS=3 (feedback and HII code).

| PR | branch | src lines +/- | test on tip | scope of test | flag off |
|---|---|---|---|---|---|
| 02 | pi/02-hii-loop-table-rates | 88/10 | single_stars_radiation: 1 passed (build #error before) | narrow | 142/145 byte-identical, rest same instructions |
| 03 | pi/03-birth-density | 58/3 | radiation 2 + record 4 passed (onset: 1 of 4 stars before) | narrow | 141/145, rest same instructions |
| 04a | pi/04a-photoion-cadence | 75/5 | photoion_per_direction: 1 passed (stars step with gas before) | broad | see 05 |
| 04b | pi/04b-per-direction-fronts | 343/5 (over 300, split proposed) | 3 passed (no photoionisation log before) | broad | see 05 |
| 04c | pi/04c-photoion-stamp | 66/8 | 3 passed (ledger 0 before) | broad | see 05 |
| 05 | pi/05-overlap-rounds | 122/23 | 5 passed (pair 1.385x before) | broad | 136/145, rest same instructions (FIRE3, gmc) |
| 06 | pi/06-sgchem-coupling | 52/4 | 6 passed (ionised cells neutral before) | broad | see 07 |
| 07 | pi/07-quiet-steps | 40/11 | 7 + 2 + 4 passed (16 calls before, 12 after) | broad | 136/145, rest same instructions |
| 08 | pi/08-starbench-ladder | test only | m10, m1 pass; m0.1 see 08 | broad | n/a |
| 09 | pi/09-compact-histogram | 140/28 | 8 passed ('cap ineffective' before) | broad | see 10 |
| 10 | pi/10-solve-timing | 7/5 | 8 passed (no column 13 before) | narrow | 136/145, rest same instructions (FIRE3, gmc) |
| 11 | pi/11-solve-on-source-step | 294/62 (over 300 with removals, split proposed) | 22 passed (6 of 6 new tests fail before) | broad | 140/146, rest same instructions |
| 11c | (no branch; e57b6001b, 39df537c4) | 1/0 | 1 passed (0 of 3039 cells ionised past the end before, 3039 of 3039 after) | narrow | not run separately |

Each PR has three commits in the order test, change, docs. Flag off for 04a to 04c is covered by the 05 tip: their code is a subset of it, all under the new flag. 06 flag off is covered by 07, 09 by 10.

StarBench note. The reference implementation's StarBench ladder (-28.8 / -16.9 / -6.7 %) ran at HYDROGEN_MASSFRAC 0.76. Production SGChem builds use 0.7065. So 06 keeps 0.76 under `TEST_PHOTOIONIZATION_FIXED_RATE_TWO_TEMPERATURE_GAS` (constants.h) and only there.

---

## 02. Single stars: table ionising photon rate in the HII-region loop

**What and why.** IMF-sampled single stars become sources of the HII-region loop, with the photon rate Q of the stellar-evolution tables. This is the default path outside the per-direction model. A star ionises from the onset used in the reference: at least 8 Msun, and past both the pre-main sequence and the Kelvin-Helmholtz time (the free-fall part follows in 03). The radiation-pressure part of the earlier commit is dropped: it is unreachable under the slice's `#error` and is not photoionisation.

**Evidence before.** test/single_stars_radiation fails at build: the slice's `#error` forbids single stars as radiation sources.

**Change.** single_star_feedback.cc: `single_star_ionizing_onset_yr`, `single_star_ionizing_luminosity_cgs`, `single_star_ionizing_photon_rate`. radfb_local.cc: for a single star the loop takes Q / 3.05e10, its own photon-per-erg convention. rt_utilities.cc: the single-star branch. single_star_record.cc: the `#error` allows the HII loop with tables. stellar_evolution_tables.cc/.h: the photon-rate and ionising-luminosity lookups. 88 source lines added, 10 removed.

**Cost.** One table lookup per active star per HII call. No collectives. No new bytes.

**Evidence after** (measured on an older base; to be re-measured on the audit branch). single_stars_radiation, with three stars: 35 Msun at 1 Myr, a supernova remnant, and 35 Msun at 1000 yr (not yet ionising). Exactly one source in every step, and the logged photon rate equals the table's to the 6 digits printed. 1 passed. Narrow test.

**Flag off.** 142/145 objects byte-identical, the rest same instructions (FIRE3, gmc) (measured on an older base; to be re-measured on the audit branch). Flag off: to be re-run on <TIP>.

**Limits.** Single stars only become sources from the tables; the free-fall part of the onset follows in 03.

**Depends on / followed by.** slice2/15, up/B4; followed by 03.

---

## 03. Single stars: birth density in the star record, ionising onset with t_ff

**What and why.** The reference delays the ionising photons by min(t_KH, t_ff(birth density)), at most 1 Myr, and by the pre-main sequence. The free-fall time needs the parent cell's density at birth. It goes into the compact star record as a float. The reference holds it as MyFloat (float), so there is no precision difference.

**Evidence before.** test_ionising_onset uses eight 10 Msun stars at 0.995 and 1.005 of their reference onset, with one case each where the pre-main sequence, t_KH and t_ff decide. Before the change one star ionises instead of four.

**Change.** single_star_record.h: field `rho_birth` (float, under GALSF_SINGLE_STARS_EVOLUTION_TABLES), carried through exchange and restart files. Snapshot and IC field `SingleStarBirthDensity` (io.cc, read_ic.cc, allvars.h). It is set in `single_star_attach_to_new_star` (physical parent density). `single_star_ionizing_onset_yr` = max(t_PMS, min(t_KH, t_ff, 1 Myr)), with no second delay when the density is unknown. The restart record size goes into the restart layout table (RESTARTFILE_LAYOUT_VERSION 4).

**Cost.** 4 B per single star (record, under the tables flag). No collectives.

**Evidence after** (measured on an older base; to be re-measured on the audit branch). Four of eight stars ionise, exactly those past their reference onset. single_star_record has a new tables variant: the birth density follows 64 stars bitwise through 4-rank exchange, snapshots and a restart. Radiation 2 passed, record 4 passed. Narrow tests. Not covered: the value set at a live birth, because no single-star formation test builds with tables.

**Flag off.** 141/145 byte-identical; io.o and read_ic.o differ only in line-number immediates (measured on an older base; to be re-measured on the audit branch). Flag off: to be re-run on <TIP>.

**Limits.** Not covered by a run: the value set at a live birth, because no single-star formation test builds with tables.

**Depends on / followed by.** 02; followed by 04a.

---

## 04a. Photoionisation per direction: flag, cadence and timestep of sources

**What and why.** First piece of `GALSF_FB_PHOTOIONIZATION_PER_DIRECTION`. An ionising star is capped at the step PhotoionMaxTimestep (default 1e4 yr), shorter steps are allowed. A star that will ionise steps onto its onset age. Each source solves on its own active steps (11 adds the solve logic).

**Evidence before.** Two 35 Msun stars (1 Myr, and 1000 yr old with onset at 15 kyr) in static gas, MaxSizeTimestep 20 kyr, PhotoionMaxTimestep 1 kyr: both step with the gas (15 kyr).

**Change.** New galaxy_sf/photoion_per_direction.cc: `photoion_timebin`, `photoion_max_timestep`. timestep.cc applies the cap to single stars. Parameters: PhotoionMaxTimestep [yr, default 1e4]; HIIRegion_fLum_Coupled is also read under the new flag (default 1). An `#error` makes the flag exclusive with GALSF_FB_FIRE_RT_HIIHEATING and requires the single-star flags.

**Cost.** One table lookup per active single star per timestep. No collectives, no bytes.

**Evidence after** (measured on an older base; to be re-measured on the audit branch). In every step the ionising star sits at dt <= 1 kyr while the gas keeps its 15 kyr step. 1 passed. Broad test.

**Flag off.** See 05. Flag off: to be re-run on <TIP>.

**Limits.** Run with Kokkos OpenMP on CPU only.

**Depends on / followed by.** 03; followed by 04b.

---

## 04b. Photoionisation per direction: Stromgren fronts per HEALPix pixel

**Size.** 343 source lines added, 5 removed: over the ~300 limit. Proposed split: 04b-1 the Spec header (about 125 lines: `PiHistSpec`, HEALPix pixel), 04b-2 the host driver (about 190 lines: front analysis, rounds, log). A histogram without the front analysis, or the reverse, has nothing to test on its own, so the split needs a unit test of the histogram for 04b-1. Open question for the reviewer: split as proposed, or keep as one piece.

**What and why.** The front solve for each source alone. Each source splits Q over 12 HEALPix pixels (NSIDE 1). Per pixel, the Spec `PiHistSpec` sums the cells' recombination bills (alpha_B n_e n_H V, case B 2.6e-13, at each cell's density) into 64 radial bins as int64 fixed point (units of Q/12 / 2^40). It also keeps the innermost cell radius per bin. The host finds the crossing, rounds the marking front up through the straddling cell, and grows or refines the search sphere. The Spec runs where the cell lives (`needs_live_neighbours`), so cells on other ranks count. The sums are integers, so the fronts do not depend on order, threads or ranks.

**Evidence before.** The Stromgren test writes no photoionisation log.

**Change.** New galaxy_sf/photoion_per_direction.h (Spec, HEALPix NSIDE-1 pixel; checked against the reference's vec2pix_ring on 2e6 random directions, 0 mismatches). Driver in photoion_per_direction.cc: on steps where the source is active, the sources come from ActiveParticleList, followed by passes of histogram plus analysis. accel.cc calls it in place of the HII loop. One log row per solve goes into HIIheating.txt, with columns 1-6 as the existing loop, then ledger, rounds, passes, and the smallest and largest front. Parameter PhotoionMaxSearchRadius [physical kpc, default 0.05]. Runner instantiation.

**Cost.** On solve steps only: an AccumData of 9.2 KB per source; one sum of the source count; per pass one runner call and one reduction; one reduction of the log row. No per-cell bytes. With no source there is no work, but one reduction remains (removed in 07).

**Evidence after** (measured on an older base; to be re-measured on the audit branch). 35 Msun star in n = 10 gas (R_S = 12.7 pc, 9 cells): the smallest and largest of the 12 fronts lie within one cell of R_S. 1, 2 and 4 ranks and a permuted particle order give the same fronts and passes. 3 passed. Broad test.

**Flag off.** See 05. Flag off: to be re-run on <TIP>.

**Limits.** Determinism checked on CPU (1, 2, 4 ranks, permuted order); not on a GPU.

**Depends on / followed by.** 04a; followed by 04c.

---

## 04c. Photoionisation per direction: cells ionised until the next solve

**What and why.** The persistent state. `PhotoionizedUntil` (integertime, 8 B per cell) is set by AtomicMax to the end of the stamp. A cell is ionised while Ti_Current is below it. There is no global reset; cells not claimed again expire by themselves, which removes the flicker of the reference's earlier version. Star formation and the single-star gas collection skip ionised cells, as for DelayTimeHII. The stamp runs to the next multiple of the cap step, a solving source also releases cells it left, and log columns 4-6 count a source's cells by share (see 11).

**Evidence before.** Ledger of the ionised cells = 0.

**Change.** `PiStampSpec` (stamp plus ledger: the bills of the stamped cells, and the newly stamped cells, their mass and mean radius in log columns 4-7). Cell field and init; checks in sfr_eff.cc and sfr_collect_gas. Restart: the field goes into the restart layout table (version 4).

**Cost.** 8 B per gas cell (under the flag). One runner call per solve.

**Evidence after** (measured on an older base; to be re-measured on the audit branch). Recombinations of the ionised cells / Q = 0.9994-0.99999. Their volume radius is within one cell of R_S, and they are the same cells at every solve. Ranks and order give identical cell counts. 3 passed. Broad test. Not covered by a run: the SF and donor skips.

**Flag off.** See 05. Flag off: to be re-run on <TIP>.

**Limits.** Not covered by a run: the star-formation and donor skips of ionised cells.

**Depends on / followed by.** 04b; followed by 05.

---

## 05. Photoionisation per direction: overlapping sources share the bills

**What and why.** The overlap rule ("claim, sum, share"), replacing the reference's ID-ordered serialisation. After its own solve, every source claims its cells: A_c += q_sc, with integer weight q_sc = clamp(round(2^32 a_sc / C_c), 1, 2^54) and a_sc = (Q/12) h^2 / (4 (r^2 + h^2)). Then, in rounds, each source re-bills: a held cell costs C_c q_sc / A_c, and a cell it would join costs C_c q_sc / (A_c + q_sc). Bills only fall, so fronts only advance (max with the held front), and new cells are claimed. Rounds stop when no front moves, or after PhotoionMaxRounds. Unspent photons escape and are not handed on. A bright source pays most of a shared cell. There is no random draw and no ID order.

**Evidence before.** Two identical stars 0.5 R_S apart ionise 1.385x the cells of one star; both bill the shared cells in full. A bright and a faint star (Q ratio 7): union / single 1.000 for a Q ratio of 1.144.

**Change.** A side buffer A_c (int64 by local gas index, zero between solves); `PiClaimSpec`, `PiDrainSpec`; the shared branch of the histogram; the stamp books each source's share. Rounds in the driver. Parameter PhotoionMaxRounds (default 16).

**Cost.** 8 B per local gas cell in the side buffer (allocated at the first solve, not in the cell struct). Per round: one histogram call, one claim call, one integer reduction. A single source does no rounds.

**Evidence after** (measured on an older base; to be re-measured on the audit branch). Pair: 6068 / 3035 = 1.999x the cells of one star, ledger 0.999. 1, 2 and 4 ranks give the same cells and rounds. Bright and faint: cells x1.146 for a Q ratio of 1.144, ledger 1.001. 5 passed. Broad test.

Rounds cap (convergence study on the symmetric pair, cap 2/4/8/16/32/64): cells 5608/6026/6069/6069/6069/6069; ledger 0.923/0.992/0.99927 (the same from cap 8 on); smallest front 13.02/13.83/13.87 pc, largest 18.81/19.17/19.20 pc (the same from cap 8 on). The rule "stop when no front moves" ends after 9 rounds (8 that move, 1 that confirms). Wall-time differences between caps are within run-to-run noise (23-30 s per run). A tolerance stop would save at most the confirming round, so the rule stays and the default cap is 16 as a safety bound for dense clusters. Cost per round: one histogram and one claim neighbour-loop call, plus one integer reduction.

**Flag off.** FIRE3 and gmc_cooling: 136/145 byte-identical, the rest same instructions (line-number immediates and strings) (measured on an older base; to be re-measured on the audit branch). Flag off: to be re-run on <TIP>.

**Limits.** The bright-pays-most rule is checked on the union only, not per source.

**Depends on / followed by.** 04c; followed by 06.

---

## 06. SGChem: photoionised cells from the per-direction solve

**What and why.** In SGChem's DoCooling branch, a cell with Ti_Current < PhotoionizedUntil gets ionised abundances (`sgchem_set_ionized`: H+ 0.9998, H2 and CO destroyed, He and the low-IP metals singly ionised). Its energy is raised to at least that of 1e4 K ionised gas, and the network is not solved, as in the reference's ionised block. The raise is booked in a new ledger row. Test-only flag `TEST_PHOTOIONIZATION_FIXED_RATE_TWO_TEMPERATURE_GAS`: every ionising star emits TestPhotoionFixedRate, ionised gas is held at 1e4 K and neutral gas at TestPhotoionNeutralTemp, with no network. These are the benchmark conditions of Hu+17 App. A; as there, neutral cells keep their abundances. The StarBench ladder (08) uses it.

**Evidence before.** SGChem build, 35 Msun star: the cells inside the fronts stay neutral and cold (H+ 9e-6).

**Change.** sgchem_cell.cc (ionised branch, test pin, chemistry ledger rows only when the network ran); ledger row `E_photoionisation_heating_erg` (proto.h, ledger.cc); test parameters (allvars, begrun); photoion_source_rate takes the test rate.

**Cost.** None per cell; the network is skipped for ionised cells. One ledger row.

**Evidence after** (measured on an older base; to be re-measured on the audit branch). Inside the fronts H+ > 0.999 and the energy is at least that of 1e4 K. Outside, the cells are unchanged. The ledger row is within 5% of the thermal energy the ionised cells gained, and the chemistry heating row stays below 1% of it. A first version booked the raise twice (as chemistry heating too); this is fixed and tested. 6 passed. Broad test.

**Flag off.** See 07. Flag off: to be re-run on <TIP>.

**Limits.** The test flag holds hydrogen fraction 0.76 for the benchmark; production uses 0.7065.

**Depends on / followed by.** 05, cr/04 (ledger row table); followed by 07.

---

## 07. Photoionisation per direction: no work or collective on quiet steps

**What and why.** Performance rule: no collective on steps without ionising sources. The module keeps the global source count of its last solve (start-up: unknown, so one count). Ionising onsets, i.e. single stars whose onset falls in the step that ends now (made active there by 04a's step limit), ride the existing star-formation Allreduce as a third long long. A photoionisation step solves only if sources exist. A solve that finds none sets the count to zero. (11 replaces this: the source count and the entry in the star-formation reduction are gone; "any ionising star active on this step" rides the reduction of the highest active bin at the sync point.)

**Evidence before.** A 35 Msun star with onset at 15.3 kyr: on the 06 code the module issues one Allreduce on each of the 5 photoionisation steps before the onset. Its first solve row counts 16 reductions and runner calls (11 for the solve + 5), measured with the column-12 counter.

**Change.** photoion_per_direction.cc (count, onset count, guard, log column 12 = the module's reductions and runner calls since the previous row); sfr_eff.cc (third entry in the existing reduction).

**Cost.** One more long long in an existing reduction. Quiet steps: no solve and no collective of the module, but one read per active particle (`photoion_new_sources_local`, in the star-formation reduction).

**Evidence after** (measured on an older base; to be re-measured on the audit branch). The first solve comes at 15.50 kyr (onset 15.34 kyr, photoionisation step 0.98 kyr) and costs 11 + 1 (the start-up count); every later solve costs 7 + 2 per search pass. 7 + 2 + 4 passed. Broad test. (After 09 it is 7 + 5 per pass; after 11, 8 + 5 per pass on the cap step, 9 + 5 off it, and no start-up count.)

**Flag off.** 136/145 byte-identical, the rest same instructions (measured on an older base; to be re-measured on the audit branch). A first cut changed sfr_eff.o's reduction with the flag off; fixed and rechecked. Flag off: to be re-run on <TIP>.

**Limits.** Replaced in part by 11 (trigger at the sync point instead of the star-formation reduction).

**Depends on / followed by.** 06; followed by 08.

---

## 08. Photoionisation per direction: StarBench ladder test

**What and why.** The acceptance test: the StarBench D-type expansion (Bisbas+15; n_H = 100, Q = 1e49 /s, 4.5 Myr) at three resolutions. It must reproduce the reference's validated ladder (-28.8 / -16.9 / -6.7 % at m10 / m1 / m0.1) within 2 points, under the benchmark's thermal conditions (06's test flag; Hu+17 App. A). Test only.

**Evidence before.** On the same ICs and grader, the default path (the existing HII loop with table Q, 02) misses StarBench by -72 / -78 / -81 %.

**Change.** test/photoion_starbench: lattice ICs (40^3 / 86^3 / 185^3), identical to the ICs of that comparison (positions to 1e-17, masses bitwise); Config, parameters; `starbench.py`, the grader (StarBench combined solution on references measured in the run). It is ported from the reference's grader and gives the same number on the same run (-28.29 % both). m10 runs inside pytest (3 min on 16 threads). m1 and m0.1 are batch jobs (`run_ladder.sh`), graded by the test when `GIZMO_STARBENCH_DIR` points at finished runs.

**Cost.** Test only.

**Evidence after** (measured on an older base; to be re-measured on the audit branch). One node, 1 rank, 16/32/64 threads. On the 07 tip: m10 in pytest 1 passed (4 threads, 15 min); m1 graded by the test from the batch run, 1 passed. Broad test.

| rung | this series | reference implementation (stella_dev) | difference | wall time |
|---|---|---|---|---|
| m10 | -28.3 % | -28.8 % | +0.5 | 3 min |
| m1 | -16.2 % | -16.9 % | +0.7 | 26 min |
| m0.1 | -6.34 % | -6.7 % | +0.4 | ~4 h |

Cadence twin at m10: a 4x finer photoionisation cadence (1.2 kyr instead of 4.9 kyr) changes R_IF(t) by at most 0.29 % (bar: 2 %); median -28.6 % vs -28.3 %.

**Flag off.** Not applicable (test only).

**Limits.** m0.1 runs about 4 h and is graded from a batch run, not inside pytest.

**Depends on / followed by.** 07; followed by 09.

---

## 09. Photoionisation per direction: histograms out of the reply records

**What and why.** The reference carried the per-pixel histogram (12 x 64 bins of cost and innermost radius, 12 kB) in every exchange record. At 112 ranks that overflowed BufferSize in a production run. In this tree the neighbour-loop exchange cannot overflow: rounds are bounded by CommChunkSize, and an active whose records exceed one round ships in a solo round with the warning "cap ineffective". So the same record size shows up as rounds, memory and bandwidth instead: 12.6 kB per (source, peer, block of 8 tree nodes). The fix follows the reference's: the histogram leaves the record. The rank holding the cells fills a pool by global source index (integer atomic adds, atomic minima). The non-empty bins go to the source's rank in one Alltoall + Alltoallv per pass, and are summed there in any order (integer sums, minima), so the result is bitwise the same. The reply record shrinks from 12,288 B to 8 B (a counter).

**Evidence before.** Four separate stars on 4 ranks, CommChunkSize 0.02 MB: "single active's export set (2 envelopes, ~25216 bytes) exceeds CommChunkSize bunch (1 envelopes); shipping a solo oversized round — cap ineffective for this call".

**Change.** photoion_per_direction.h: `PiHistSpec` writes into the pool (`pi_pool_claim`, saturating atomic add, `Kokkos::atomic_min`); AccumData `PiCountOut`; `PiSourceIn.g`. photoion_per_direction.cc: the pool (grown, and the pass repeated, if a pass fills it; all-zero between passes), `pi_histograms` (pass, overflow check, exchange). Global indices come from one Allgather, which replaces the source-count Allreduce.

**Cost.** Per pass: one Allreduce (pool full), one Alltoall and one Alltoallv of the non-empty bins (24 B each), replacing 12.3 kB per reply record. Pool: 16 slots x 12.3 kB = 197 kB per rank, grown on demand; 4 B per source of the solve for the slot index. Quiet steps unchanged (none).

**Evidence after** (measured on an older base; to be re-measured on the audit branch). No "cap ineffective"; 4-rank cells and fronts equal 1 rank's. Every other photoion log (single source, 4 ranks, pair on 1 and 4 ranks, bright and faint, SGChem) is bitwise identical to before the change in columns 1-11. 8 passed; the quiet-step count is now 7 + 5 per pass. Broad test.

**Flag off.** See 10. Flag off: to be re-run on <TIP>.

**Limits.** Bitwise equality checked on CPU; GPU not run.

**Depends on / followed by.** 08; followed by 10.

---

## 10. Photoionisation per direction: wall time of each solve in the log

**What and why.** A performance check for the per-direction model: one source in 1e6 and 8e6 cells; the work per photoionisation step should not grow with the cell count. Until now nothing measured a solve on its own: cpu.txt books the runner calls under the generic neighbour-loop entries. The wall time of each solve now goes into the log as column 13 (slowest rank), and the test directory gets `run_performance.sh` (four batch jobs, flag on and off) for the scaling runs.

**Evidence before.** HIIheating.txt has 12 columns and no timing; the quiet-step test cannot check that a solve was timed.

**Change.** photoion_per_direction.cc: `MPI_Wtime` at the start of a solve; the elapsed time rides the existing MAX reduction of the log row as a third value and is written as column 13. begrun.cc: the column in the log header. test/photoion_per_direction: column 13 checked in the quiet-step test; `run_performance.sh` and the numbers in the README. 7 source lines added, 5 removed.

**Cost.** Two timer calls per solve and one more double in a reduction that already exists. No field, no new collective, nothing on quiet steps.

**Evidence after** (measured on an older base; to be re-measured on the audit branch). test/photoion_per_direction: 8 passed (column 13 present and positive). Narrow test. Scaling runs, 8 ranks x 8 threads, one source, 10 steps, all cells active:

| cells | solve | runner calls + exchanges | source scan | run total, flag on / off |
|---|---|---|---|---|
| 1e6 | 3.8 ms | 1.9 ms | 1.9 ms (with the first reduction) | 20.7 / 19.6 s |
| 8e6 | 18.8 ms | 1.5 ms (+2.4 first reduction) | 14.4 ms | 105.0 / 102.5 s |

The solve's own work is constant. The part that grows is the scan of ActiveParticleList for sources: one P[i] read per active particle, here 1e6 per rank, 0.3 % of the run. It scales with the active count, not with N_total. Memory, from sizeof on both builds: gas_cell_data 368 vs 360 B (+8 B, PhotoionizedUntil); particle_data unchanged (384 B); the claim buffer adds 8 B per local gas cell from the first solve on; histogram pool about 200 kB per rank; 4 B per single star (birth density, 03).

**Flag off.** FIRE3 and gmc_cooling: 136/145 byte-identical, the rest same instructions (measured on an older base; to be re-measured on the audit branch). Flag off: to be re-run on <TIP>.

**Limits.** Wall time of the slowest rank, so it includes waiting in the reductions. The scaling runs are CPU only; not on a GPU, not beyond 8e6 cells. A star-only active list (or a slot-to-particle index in the star record) would remove the source scan; open question for the reviewer whether that is wanted as a follow-up.

**Depends on / followed by.** 09; followed by 11.

---

## 11. Photoionisation per direction: each source solves on its own steps

**Size.** 294 source lines added, 62 removed: over the ~300 limit. Proposed split: 11a, the snapshot field PhotoionizedUntilTime plus the timeline-halving fix and the stamp rule (about 150 lines, useful alone); 11b, the own-step solves (about 200 lines). The integer-cast clamp of the cap step is a separate small PR right after this one (11c below). Open question for the reviewer: split as proposed, or keep as one PR.

Line count by file: photoion_per_direction.cc 155/39, .h 59/12, io.cc 38, begrun 13/2, restart 8, run 8, proto 6/2, allvars 3, global_data 4/1, Template_Config 1, sfr_eff 0/6.

**What and why.** Cadence as gizmo2017 (photoionize.c:100-116, timestep.c:428-433). Each ionising source solves on its own active steps; its step is capped at PhotoionMaxTimestep (default 1e4 yr), shorter allowed, no floor. gizmo2017's global clearing (photoionize.c:712) is not copied. Open question for the reviewer: should there be a floor on the source step.

**Evidence before.** Six new tests (a star on a step shorter than the bin solves on that step; an inactive star's region survives another star's solve; on 1 and 2 ranks): 6 of 6 fail on the 10 code. Commits: 1328fb24 (test: hook TEST_PHOTOIONIZATION_ONE_STAR_ON_QUARTER_STEP, snapshot field PhotoionizedUntilTime, 5 new tests), f98e7801 (change), e91dd6e4 (test expectation: no partner search while the table is empty), 53e9f0ec (docs). Hashes are from the older base.

**Change.**
- Trigger: "any ionising star active" rides the existing HighestActiveTimeBin reduction at the sync point (one read per active particle); the module returns at once otherwise. The entry in the SF reduction (07) is removed: SF runs after the solve within a step, so it could not tell the same step.
- Stamp: to the next multiple of the cap step. Steps are nested, so every live source is active there: an inactive source's region cannot drop out. A solving source releases cells inside its own last fronts that nobody claims now.
- Front table: per live source (160 B: ID, time, position, reach, Q/12, 12 fronts, ledger), sorted by ID, the same on every rank, one Allgatherv per solve. A row ends at the first multiple of the cap step after its solve. The table goes into the restart files; its record is added to the restart layout table (RESTARTFILE_LAYOUT_VERSION 4), together with the single-star records.
- Sharing with inactive sources: off the cap step one Allgatherv of the solving spheres; each rank lists the table rows that can reach each solving source. In the kernels A_c = solving claims + the partners' integer weights for cells inside their last fronts (computed, never stored). Partners' fronts stay fixed; each solving source pays its own Q exactly; an inactive partner's budget can be off until its next solve (at most one cap step); at each multiple all sources solve together. Integer sums, no ID order.
- Log: column 2 = sources solving now; 4-6 by share (a shared cell counts once over its sources); 14-15 smallest and largest per-source recombinations / Q.
- Also fixed (gap since 04c): readjust_timebase (TimeMax raised on restart) halved every integer time but PhotoionizedUntil, so cells could stay ionised up to ~TimeMax_new/TimeMax_old too long; now cells and table are halved.
- photoion_timebin multiplies by cf_hubble_a (cosmological runs; untested there).

**Cost.** Quiet steps: no solve, no module collective; one read per active particle at the sync point, one int in an existing reduction. Solve: +1 Allgatherv (table), +1 off the cap step (partner spheres); 8 + 5 per pass on the cap step, 9 + 5 off it. Memory: 160 B per live ionising source per rank; partner list ints per solve.

**Evidence after** (measured on an older base; to be re-measured on the audit branch). The checks were defined before the runs and not moved afterwards.

| # | check | result | verdict |
|---|---|---|---|
| 1 | StarBench ladder: m10, m1, m0.1 each within 2 points of 10 (10: m10 -28.29 %, m1 -16.22 %, m0.1 -6.34 %) | m10 -28.29 %, m1 -16.22 %, m0.1 -6.34 % (identical to 10; same 1024 solves: one source, always on the cap step) | pass |
| 2 | overlap test (`test_overlapping_sources_share`) 2.0x +- 0.05 on 1, 2, 4 ranks; the same pair with the stars on different steps also 2.0x +- 0.05 on 1 and 2 ranks | 2.0x on 1/2/4 ranks; pair on different steps 2.0x on the joint solves, 1.0x for the star solving alone, on 1, 2 ranks and permuted order | pass |
| 3 | photon budget per source in static uniform gas: recombinations / Q within 0.98-1.02 at every solve, also when a partner is inactive | columns 14-15 within 0.98-1.02 at every solve in the own-step and different-step tests | pass |
| 4 | small cloud, 4 ID seeds per branch, at least 9 ionising stars: per seed the median smallest and largest front on the solves at multiples of the cap step; difference of seed medians between 10 and 11 at most the max-min range of the 10 seeds | largest front: 10 27.6 pc (range 46.9) vs 11 50.0 pc: pass. Smallest front: 10 0.153 pc (range 0.248) vs 11 0.633 pc: miss. Precondition also unmet: 11 seeds peak at 4/8/6/3 ionising stars (10: 7/10/8/9) | miss |
| 5 | unit tests fail on 10, pass with the change, on 1 and 2 ranks | 6 of 6 fail on the 10 code; 22 passed on the final commit (photoion_per_direction 15, single_stars_radiation, single_star_record) | pass |
| 6 | flag off, FIRE3 and gmc_cooling, against 10: byte-identical or same instructions | 140/146 byte-identical, 6 same instructions (line numbers, strings) | pass |
| 7 | measured, no bar: solves per Myr and photoionisation time share | see below | measured |

Test scope: broad.

Cause of the miss in check 4. The seeded runs diverge before any photoionisation. At 1 Myr the 10 and 11 positions differ by 1e-9 and by 1e-5 at 3.9 Myr; two runs of 10 that differ only in thread count (16 vs 4) differ by the same amounts. By 3.9 Myr, before the first solve, the star counts already differ (3 vs 4, 3 vs 6), and the 11 seeds then formed fewer, more massive ionising stars (e.g. one 238 Msun star, Q 2e50, cleared all gas within the 50 pc search radius: fronts 0). So the seeded comparison measures different star populations, and 4 seeds of 10 do not sample that scatter (a second set of 10 at 4 threads gives smallest-front medians up to 0.43 pc range). Direct evidence on the fronts: both branches restarted from one snapshot of 10 at 6.8 Myr with 11 and with 9 ionising stars, to 7.8 Myr: first rows identical in every printed digit; over 129 solves the ratio 11 / 10 of smallest front, largest front and cells has median 1.000 (16-84 %: 0.998-1.005). The miss stays a miss. Open question for the reviewer: accept the paired restart as the front check, or ask for many more seeds.

**Check 7, measured** (small cloud; seeds = 16 threads, to 7.8 Myr; paired = from one snapshot, 4 threads, 1 Myr).

| run | solves per Myr 10 / 11 | solves by part of the sources | PI time share 10 / 11 | solve wall time, median |
|---|---|---|---|---|
| seeds (10: 7-10 sources, 11: 3-8) | 131.2 / 136-141 (one seed 275) | 11: 14-15 % of solves (53 % in the 238 Msun seed) | 0.15-0.24 / 0.03-0.16 (different populations) | 12-59 ms / 28-52 ms |
| paired, 11 sources | 131.9 / 131.9 | 0 | 0.266 / 0.254 | 126 / 128 ms |
| paired, 9 sources | 131.9 / 131.9 | 0 | 0.267 / 0.272 | 210 / 235 ms |

In the paired windows (at least 9 ionising stars) no star is on a step shorter than the cap, so 11 solves exactly as 10. Extra solves come from stars on shorter steps (onset, approach to death): +4-7 % solves per Myr in the seeds, 2.1x in the seed with the 238 Msun star, whose step shrinks near its death. Short steps cost nothing extra per solve beyond the partner search; no floor was added.

Per-source budget in the cloud (columns 14-15): 0.007 to 2.6e6, the round-up of 04b on faint sources in dense gas, and escape where the front reaches the 50 pc search limit; 10 shows the same values (1.3e5 on its first faint row).

**Flag off.** 140/146 byte-identical, 6 same instructions (measured on an older base; to be re-measured on the audit branch). Flag off: to be re-run on <TIP>.

**Interaction with the current base.**
- restart.cc: the restart header keeps a size table of the raw structures (layout version 4) and asks that every new one be added; add the front table (PiFrontRecord) and the single-star records.
- Mechanical feedback runs in run.cc after the closing half-kick; a star that went supernova has Q = 0 at its next sync, so nothing here depends on the order.
- The sink wake-up floor and the spawned-cell step only shorten or end steps; the stamp rule needs every ionising source's bin at most the cap bin, which still holds.
- run.cc: the reduction extended here (HighestActiveTimeBin in find_next_sync_point) is unchanged upstream.
- sfr_eff.cc: this PR removes 07's reduction entry; take this version there.

**Limits.** Check 4 (seeded cloud fronts) missed; see below. No floor on the source step.

**Depends on / followed by.** 10; followed by 11c.

---

## 11c. Photoionisation per direction: clamp the cap step before the integer cast

**Size.** 1 source line (fix e57b6001b), test 39df537c4 (20 lines). No branch of its own yet; commit pair test first, then fix, on top of 11.

**What and why.** `photoion_timebin` converts the cap step min(PhotoionMaxTimestep, MaxSizeTimestep) to integer time. When the run window TimeMax - TimeBegin is shorter than about PhotoionMaxTimestep / 8 (1.25 kyr at the default 1e4 yr), the value does not fit the integer timeline and the cast overflows: the cap bin comes out as bin 1 and the stamps end 2 ticks ahead, so cells ionised by the last solve are released before the end of the run. This is the same defect as the one up/B5 fixes for MaxSizeTimestep, here reached through the photoionisation cap. The fix clamps the step to half the timeline before the cast, as up/B5 does.

**Evidence before.** New test `test_run_shorter_than_the_photoionisation_step` (TimeMax 1e-4, PhotoionMaxTimestep 1e4 yr): the run ends, but in the last snapshot 0 of 3039 ionised cells have a stamp past the end of the run. Fails.

**Change.** photoion_per_direction.cc: one line in `photoion_timebin`, `dt = min(dt, 0.5 TIMEBASE Timebase_interval)` before the conversion. Test in test/photoion_per_direction.

**Cost.** None.

**Evidence after.** The same test passes: 3039 of 3039 cells stay ionised to the stamp of the last solve, the run takes 3 steps. Narrow test (one bug; offered as a temporary check).

**Flag off.** No change outside `GALSF_FB_PHOTOIONIZATION_PER_DIRECTION`. Flag off: to be re-run on <TIP>.

**Limits.** Only runs shorter than the cap step reach it; production runs never do.

**Depends on / followed by.** 11, up/B5 (same clamp in the main timestep). Last of the series.

---

## Open items

- Not done: determinism on a GPU and with Mode A (all Specs run where the cell lives, so Mode B on CPU only; Mode D, the device form of that path, is untested); 8 ranks; the performance test beyond 8e6 cells on a GPU; the asymmetric-pair "bright pays most" check per source (only the union is checked); the SF and donor skips of ionised cells in a run; the birth density set at a live birth.
- Threads: the overlapping pair on 1 rank with 1 and 4 threads writes identical log rows (cells, rounds, fronts, ledger). Ranks 1/2/4 and a permuted order are in the tests.
- Deviations from the design: no TREE_RAY_PI or alpha_B(T) sub-flag (not ported).

## Cloud smoke test (production model; measured on an older base; to be re-measured on the audit branch)

Config: SGChem net 5, single stars (IMF, tables, gas collection, SF criterion), mechanical feedback from the tables, per-direction photoionisation, ledger; gravity on. IC: a 12,377-cell 1 Msun GMC (70 pc box), the reference's cloud parameters; 4.9 Myr. Built from the 10 tip. Login node, 4 threads: 100 s.

- Ran to the end with no error or NaN; the only warning is the expected SF probability > 1 for one 0.09 Msun draw.
- 200 stars were born and all draws were filled: 181 Msun in stars (drawn mass 183.6 Msun, largest star 59.5 Msun).
- Photoionisation starts at 4.04 Myr (the first massive star's onset), with up to 3 sources: 90 solves, at most 10 rounds and 10 passes, at most 67 ms per solve. 1985 cells are ionised at the end.
- Ledger: photoionisation heating 1.8e49 erg; chemistry heating 4.9e47 and cooling 1.9e50 erg; 8 mechanical events.
- Finding: the photon ledger (recombinations of the ionised cells / Q) ranges from 0.36 to 126, median 1.8. The cause is the straddling-cell round-up (the reference's accepted overshoot, as in gizmo2017): in dense collapsing gas one cell's recombination bill can be far above a pixel's Q/12, and the round-up still ionises it. It is kept as the validated reference behaviour, but in dense gas it over-ionises by up to two orders of magnitude in photons. Open question for the reviewer: cap the round-up (e.g. only when the straddling cell's bill is at most the pixel budget)?
