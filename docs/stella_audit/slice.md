# Single stars on gpu_kokkos_unified: slice series (slice2/01 to slice2/15, with 12b and 12c)

Series name: slice2. Covers the single-star vertical slice: stellar-evolution tables, IMF draw, star record, star formation by gas collection, supernovae, winds, AGB deaths, Type Ia, step limit, star death on the star's own step; plus the restart layout entries of the records.
Base: <BASE>. Tip: <TIP>. Each PR is one branch, stacked on the one before, one commit each for test, change, docs.

Order: slice2 first, then chem (chem/12 before tc), tc, gap, cr (cr/04 before pi/06), met, pi, ext, tr, kj. Inside this series PR 01 to 15 land in numeric order; PR 02 follows 01, PR 05 follows 04, PR 07 follows 03, 04 and 06, and so on (see each PR).

| PR | branch | commits | src lines (+/-) | test kind |
|---|---|---|---|---|
| 01 | slice2/01-tables-install-check | 3 | 207/0 | narrow |
| 02 | slice2/02-tables-loader | 3 | 427/2 (over 300, split proposed) | narrow |
| 03 | slice2/03-imf-draw | 3 | 41/0 | narrow |
| 04 | slice2/04-star-record | 3 | 280/3 | narrow |
| 04b | restart layout version 4 (one commit on the audit branch, no PR branch yet) | 1 | 14/3 | narrow (existing restart tests) |
| 05 | slice2/05-never-merge | 6 | 16/1 | narrow |
| 06 | slice2/06-spawn-function | 1 | 52/40 | none (refactor) |
| 07 | slice2/07-collect-gas | 3 | 661/2 (over 300, split proposed) | broad |
| 08 | slice2/08-sf-birth | 3 | 30/0 | broad |
| 09 | slice2/09-sf-criterion | 3 | 46/0 | broad |
| 10 | slice2/10-sn-source-model | 6 | 265/5 | broad |
| 11 | slice2/11-sn-thermal-coupling | 6 | 24/0 | broad |
| 12 | slice2/12-winds | 7 | 127/28 | broad |
| 12b | slice2/12b-mechfb-apply-step | 3 | 19/9 | broad |
| 12c | slice2/12c-closure-moving-gas (on an older integrated line; follows the HEALPix-weights PRs) | 2 | 4/4 | broad |
| 13 | slice2/13-agb-deaths | 3 | 29/16 | broad |
| 14 | slice2/14-type-ia | 5 | 91/7 | broad |
| 15 | slice2/15-step-limit | 5 | 46/0 | narrow |
| 16 | star death on the star's own step (three commits on the audit branch, no PR branch yet) | 3 | 29/33 | broad |

Source lines are the added and removed lines under src/ against the previous PR (tests, scripts, docs and data excluded; PR 01 adds 94 lines of install and hash scripts on top). PR 13 sits on PR 12, not on 12b. "Broad" tests check a physical problem against a reference or a conservation law; "narrow" tests check one flag, default or bug.

About the measurements. Every test result, object comparison and number below was measured on an older base and is to be re-measured on the audit branch. The same holds for each "Flag off" line. Full-suite result: <SUITE RESULT>.

How "Flag off" is checked, for every PR: <BASE> and the PR tip are built with the same Config.sh in directories of the same path length, every object file is compared byte by byte, and `objdump -d` is run on those that differ ("same instructions": only immediates such as line numbers, or strings, differ). Configurations: `test/SN_singlestar/Config.sh` (default) and a FIRE build (`FIRE_PHYSICS_DEFAULTS=3`) for the PRs that touch feedback or star-formation code. From PR 06 on, `galaxy_sf/sfr_eff.o` differs in instructions, because the star spawn is now a function; with that function exercised, <BASE> and the branch write bitwise identical snapshots (PR 06).

Tests run with the repository's test harness (README, pytest module, `mpi_launch_command`; each run in its own `run_*` directory, which the test clears first). New flags have rows in `test/compile_suite`. Built and tested with Kokkos OpenMP on CPU; not yet run on a GPU.

Open points for the reviewer:
- The weighting pass of the mechanical feedback now takes the ejecta mass of every mode in the base. The single-star ejecta mass override (`single_star_ejecta_mass_this_step`) is kept after the base's weighting pass, because that pass calls `particle2in` with channel 0.
- The cell-mass energy overshoot of the upstream momentum solve (PR 12) is described there; whether a note belongs upstream is a question for the reviewer.
- Not tested in this series: booked versus coupled wind momentum, a Weaver bubble with a pinned wind, the ionised-cell case of the star-formation criterion, GPU.
- Radiation from single stars (table luminosities in the local radiation pressure and the HII loop) is not in this series: both loops need the gravity-tree radiation bands, which single stars do not feed here. Single stars stay a compile-time error with every radiation flag. Cosmic rays from single-star SNe, FoF/SUBFIND with records, and sinks/RT/cosmic rays with the gas collection are compile-time errors too.

---

## 01. Stellar evolution tables: install script, path parameter, start-up hash check

Branch slice2/01-tables-install-check. Commits: test, change, docs. 207 source lines. Test kind: narrow.

**What and why.** Single stars take lifetimes, ejecta, yields and remnants from the stella-tables tables (github.com/Ulli1991/stella-tables; two HDF5 files, about 110 MB). They are kept out of the repository and never fetched by GIZMO. This PR adds a way to install them and a start-up check that the installed files are the version the sources expect. Nothing reads the tables yet (PR 02).

**Evidence before.** The branch has no way to name or check external stellar-evolution tables; the single-star source model (PR 10 on) needs them.

**Change.**
- New flag `GALSF_SINGLE_STARS_EVOLUTION_TABLES` (default off).
- `scripts/get_stella_tables.sh [PREFIX]`: builds the tables from the stella-tables repository at the commit these sources expect (or copies an existing build, `STELLA_TABLES_FROM`), runs its tests, installs into PREFIX and checks the hashes. Run by hand or by the test harness, never by make. `scripts/stella_tables_hash.py` computes the same hash.
- New parameter `StellarTablesFile` (main table; the wind table is read from the same directory).
- Start-up check (`galaxy_sf/stellar_tables_check.cc`, `stellar_evolution_tables.cc`): rank 0 recomputes the content hash of both files and compares it with the one compiled in. The hash covers every dataset (path and data), not HDF5 metadata, which differs between two builds of the same commit. A missing file or other content is a controlled stop naming the install script. SHA-256 is written out in the file, so the check adds no library to the build.
- Harness helper `get_stellar_evolution_tables()` (`test/harness/gizmo/test.py`): installs once (into `GIZMO_STELLA_TABLES_DIR` or `~/.stella-tables/<commit>`) and returns the parameter value.
- Template_Config line, compile_suite row `GALSF GALSF_SINGLE_STARS_EVOLUTION_TABLES`.

**Cost.** No per-particle memory. About 2.7 s on rank 0 at every start and restart (hash of both files; to be re-measured on <TIP>). No collective outside start-up; nothing on quiet steps.

**Evidence after.** `test/stellar_evolution_tables` (16^3 box, 2 ranks), narrow: the installed files pass and the run reaches TimeMax; a missing file stops at start-up and names the install script; a file with the right name and other content stops at start-up and names its hash. 3 passed (measured on an older base; to be re-measured on the audit branch).

**Flag off.** Default config: 140/140 objects, 137 byte-identical, the rest same instructions (measured on an older base; to be re-measured on the audit branch). Flag off: to be re-run on <TIP>.

**Depends on / followed by.** First of the series. PR 02 loads the tables.

---

## 02. Stellar evolution tables: node-shared loader and lookups

Branch slice2/02-tables-loader. Commits: test, change, docs. 427 source lines (cc 348, h 66). Test kind: narrow.

**What and why.** Loads the tables checked in PR 01 and provides the lookups the single-star source model uses: lifetime, PMS time, remnant mass and class, wind speed, cumulative wind ejecta, surface abundances, SN and AGB element ejecta, Type Ia yields. Only lookups with a user in this series are here; the others come with their first user. Interpolation is as in the reference implementation (stella_dev `resolvedism_stellar_tables.cc`), including its two rules: across a change of remnant class the nearest corner is taken (this keeps pre-SN mass = remnant + ejecta), and a table corner flagged dead (-99) is left out, not blended in (the age-resolved wind speed uses it). The first user is PR 10.

**Evidence before.** There is no reader for the tables; nothing can look up a stellar lifetime or yield on this branch.

**Change.** `galaxy_sf/stellar_evolution_tables.{h,cc}`. Rank 0 reads. The small (mass x metallicity) tables are broadcast to every rank. The age-resolved tables used (wind speed, surface abundances, cumulative wind ejecta; about 142 MB, to be re-measured on <TIP>) are held once per node in MPI shared memory, released at the end (`main.cc`), and accounted in the memory ledger (new family "single stars", `proto.h`, `memory_ledger.cc`). Loaded in `begrun` after the hash check. Lookups are host-only: they run once per source on its own step and reach the loops in the source packs.

**Cost.** Per node about 142 MB (age-resolved tables, one copy per node), per rank about 0.5 MB (small tables); both to be re-measured on <TIP> from the memory ledger. Collectives only at start-up (broadcasts, one shared window). Nothing on quiet steps.

**Evidence after.** `test/stellar_evolution_tables` adds `test_lookups` (narrow): a standalone build of the loader (`unit/`, like the helmholtz unit test) evaluates every lookup on 64 masses x 8 metallicities x 7 ages (226,843 values) on 2 ranks.
- Exact at the grid nodes (compared with the HDF5 datasets).
- Pre-SN mass (dataset) = remnant + SN or AGB element ejecta at the nodes to 1e-12.
- Lifetime falls with mass above 2 Msun.
- A stored sample of 3719 values (every 61st) agrees with the reference implementation built the same way to 1e-12 (bitwise on Orion).
- Both ranks give equal values.

4 passed (measured on an older base; to be re-measured on the audit branch). The unit build's binary and outputs are git-ignored.

**Flag off.** Default config: 140/140 objects, 136 byte-identical, the rest same instructions (measured on an older base; to be re-measured on the audit branch). Flag off: to be re-run on <TIP>.

**Size.** 427 source lines, over the ~300 limit. The loader (node-shared segment, collectives, hash-checked datasets) and the interpolation are a one-to-one port of the reference implementation. Proposed split: 02a, file read, hash/version check against the install stamp, node broadcast (about 200 lines); 02b, lookups by mass/Z/age with the two stated rules (about 230 lines).

**Depends on / followed by.** PR 01. First user: PR 10.

---

## 03. Single stars: IMF draw

Branch slice2/03-imf-draw. Commits: test (unit), change, docs. 41 source lines. Test kind: narrow.

**What and why.** One initial mass from the mass-weighted Kroupa IMF (M dN/dM; slopes -0.3 below 0.5 Msun, -1.3 above; `IMFSampleMinMass` or 0.08 to 350 Msun) by the exact inverse of its cumulative distribution. One uniform comes from the keyed generator of `gpu_rng.h` (key = the particle's event key, counter = salt, integer time, draw index); closed form, no loop, host and device. A draw depends only on (key, event), not on rank, thread or call order. With the star-formation acceptance m_cell/M_star (PR 08) the formed stars follow the number-weighted Kroupa IMF (LYRA, Gutcke et al. 2021).

**Evidence before.** The branch has no IMF draw for individual stars (GALSF_SFR_IMF_SAMPLING samples numbers of massive stars of a population).

**Change.** New header `galaxy_sf/single_star_imf.h`. No flag; nothing includes it until PR 07/08.

**Cost.** None (header only).

**Evidence after.** `test/single_star_imf` (unit, host compiler, no GIZMO run), narrow:
- 10^6 draws pass a Kolmogorov-Smirnov test against the analytic mass-weighted Kroupa distribution for m_min = 0.08 and 4 Msun.
- Weighting by 1/M gives the number-weighted mean mass to 1%.
- The same (key, event) gives bitwise the same masses; another event gives others.
- Two-sample Kolmogorov-Smirnov test against the reference implementation's rejection sampler (stella_dev), both m_min.

4 passed (measured on an older base; to be re-measured on the audit branch).

**Flag off.** The header is not included by any object file; nothing changes. Flag off: to be re-run on <TIP>.

**Depends on / followed by.** None. Used by PR 07 (mass quantum) and PR 08 (draw at birth).

---

## 04. Single stars: compact star record through exchange, restart and IO

Branch slice2/04-star-record. Commits: test, change, docs. 280 source lines (between 250 and 300). Test kind: narrow.

**What and why.** An IMF-sampled star carries a few quantities no stellar population has: its drawn mass and birth metallicity now; its feedback state from PR 10. They live in a compact rank-local record reached through `P[i].StarIdx` (slot + 1, 0 = none), not in the particle struct.

**Evidence before.** There is no place to keep a star's drawn mass across domain exchange and restarts without growing every particle.

**Change.**
- New flag `GALSF_SINGLE_STARS_FROM_IMF` (default off).
- `galaxy_sf/single_star_record.{h,cc}`: struct-of-arrays slots, fields listed once (one macro; a field is added with its first user), free list, capacity doubles when full and is kept; memory ledger family "single stars".
- `P[i].StarIdx` (int) sits in the padding after ID_generation of the default particle layout.
- `domain_exchange`: records travel with their stars (one message per rank pair that exchanges stars with records; the receiver counts them among the particles it received; no collective; new tag TAG_SINGLESTAR_DATA).
- Restart files carry the slots: the capacity, each field array, then the used flags. The bytes per slot enter the base's restart layout table in PR 04b. A removed particle frees its record (`rearrange_particle_sequence`).
- Snapshots: `SingleStarDrawnMass`, `SingleStarBirthMetallicity`. Initial conditions or a snapshot start may carry them (a star gets a record only if its drawn mass is > 0).
- Compile-time errors: with FoF/SUBFIND (records do not follow that redistribution yet); with any stellar feedback or stellar radiation flag (they would give a sampled star population rates; the single-star feedback comes in PR 10); with the other stellar-mass models (GALSF_SFR_IMF_SAMPLING, GALSF_SFR_IMF_VARIATION, SINGLE_STAR_SINK_DYNAMICS, GALSF_MERGER_STARCLUSTER_PARTICLES).
- Template_Config line, compile_suite row `GALSF GALSF_SINGLE_STARS_FROM_IMF`.

**Cost.** Under the flag: 4 B per particle (in existing padding: struct size unchanged in the default layout), 21 B per record slot in memory (two doubles, used flag, free-list entry), 17 B per slot in restart files. Later PRs add fields with their first user (PR 10: 11 B, PR 12: 8 B); the total per slot is recorded in the restart layout (PR 04b). One point-to-point message per rank pair exchanging such stars; no collective; nothing on quiet steps.

**Evidence after.** `test/single_star_record`, narrow: 64 stars of 1-50 Msun (above the merge threshold) cross a 5.3 pc periodic box 1.5-4.5 times on 4 ranks. Every snapshot holds all 64 with mass and record bitwise as in the initial conditions. A run stopped halfway and continued from restart files ends with the same records. 1 rank gives the same. 3 passed (measured on an older base; to be re-measured on the audit branch). Negative control during development: with the record dropped on receipt, check 1 fails.

**Flag off.** Default config: 140/140 objects, 132 byte-identical, the rest same instructions (measured on an older base; to be re-measured on the audit branch). Flag off: to be re-run on <TIP>.

**Depends on / followed by.** None in code. Followed by PR 05 (never merged) and PR 07 (new stars get records).

---

## 04b. Restart files: record sizes in the restart layout table (layout version 4)

One commit on the audit branch (`file_io/restart.cc`), no PR branch yet. 14 source lines added, 3 removed. Test kind: narrow (the existing restart tests).

**What and why.** The base checks at restart that the restart files were written with the same layout: a header with a magic number, a layout version and the size of every raw structure (`restart_layout_header`, version 3). Three Stella structures are written raw but were not in that table: the single-star records (PR 04), the photoionisation front table (pi/04b) and the ledger rows (cr/04). A build with another record layout or ledger table would read them without noticing. This adds them to the table and raises `RESTARTFILE_LAYOUT_VERSION` from 3 to 4, since the header size changes.

**Evidence before.** On the base, a restart with a single-star record of another field set is read as raw bytes; the layout check does not cover it (read from the code: `restart()` writes the slots, `restart_layout_header_of_this_run()` has no entry for them). No failing run was made.

**Change.** Three entries in the item table, each set only under its flag: "single-star record" = bytes per slot (sum of the record field sizes plus the used flag); "photoionisation front" = bytes per front record (`photoion_front_record_bytes()`, 160 B on the audit branch); "ledger rows" = the number of ledger rows. Version 4. Without these flags the entries are 0.

**Cost.** 24 B more in each restart file header (three 64-bit sizes). Nothing at run time.

**Evidence after.** The restart tests of the series pass with the new header (`test/single_star_record` restart check, the photoionisation and ledger tests that restart); to be re-run on <TIP>. A restart file of version 3 is refused by the base's existing version check, as for any layout change.

**Flag off.** `restart()` and the header function change in every build (the item list and the version are not under a flag); the object comparison shows `restart()` with changed instructions and everything else identical (to be re-run on <TIP>). Restart files written by the base before this PR cannot be read after it.

**Limits.** Only sizes are checked, not field order; a reordering of record fields needs a version bump by hand, as the base's comment says. Restart files of version 3 are not converted.

**Depends on / followed by.** PR 04 (record), pi/04b (front table), cr/04 (ledger rows). Alternative: each entry lands with its first user, the version bump with the first of them.

---

## 05. Single stars: IMF-sampled stars are never merged

Branch slice2/05-never-merge. Commits: test, change, docs. 16 source lines. Test kind: narrow.

**What and why.** A star with a record is one physical star; merging it with another star loses its record and makes a star of two masses. Light stars (below the merge threshold of the gas resolution) are merged by the default particle merging.

**Evidence before.** The record test with stars from 0.1 Msun (below the threshold of 0.49 Msun set by 1 Msun cells), built and run on the test commit before the change: all 3 checks fail; star ID 532 is merged in the first 0.1 time units (the first snapshot holds 63 of 64 stars) (measured on an older base; to be re-measured on the audit branch).

**Change.** `mesh/merge_split.cc`: such a star is not a merge candidate (`does_particle_need_to_be_merged`), not a merge target, and `merge_particles_ij` refuses it (warns once) if a caller bypasses the first two. Template_Config text.

**Cost.** None (two integer tests in the merge search, under the flag).

**Evidence after.** Same test, same initial conditions (narrow): 3 passed (measured on an older base; to be re-measured on the audit branch).

**Flag off.** Default config: 140/140 objects, 132 byte-identical, the rest same instructions (measured on an older base; to be re-measured on the audit branch). Flag off: to be re-run on <TIP>.

**Depends on / followed by.** PR 04.

---

## 06. Star formation: spawn of a star from a gas cell as one function

Branch slice2/06-spawn-function. One commit (pure refactor). 52 source lines added, 40 removed. Test kind: none.

**What and why.** The star spawn inside `star_formation_parent_routine` (new index after the existing particles, timebin links, child ID, mass taken from the cell, tree insertion, the tree-slot deferral and the MaxPart stop) becomes one function, `spawn_star_from_gas(i, mass, stars_spawned)`, so the single-star path (PR 07, PR 08) spawns through the same code. No behaviour change.

**Evidence before.** Not applicable (refactor).

**Change.** `galaxy_sf/sfr_eff.cc` (function placed before `star_formation_parent_routine`; the call site is two lines), declaration in `core/proto.h`.

**Cost.** None.

**Evidence after.** Star formation with GALSF_GENERATIONS=2 on a 16^3 lattice, 1 rank: <BASE> and this tip write bitwise identical snapshots, 2670 stars, every dataset (measured on an older base; to be re-measured on the audit branch).

**Flag off.** Default and FIRE config: 140 objects, all byte-identical or same instructions except `galaxy_sf/sfr_eff.o` (the function call), which the snapshot comparison above covers (measured on an older base; to be re-measured on the audit branch). Flag off: to be re-run on <TIP>.

**Depends on / followed by.** None. Used by PR 07 and PR 08.

---

## 07. Single stars: collect the drawn mass from neighbouring gas

Branch slice2/07-collect-gas. Commits: test, change, docs. 661 source lines (over 300, split proposed below). Test kind: broad.

**What and why.** A new single star is spawned with zero mass and takes its drawn IMF mass from the gas around it, in passes with a growing radius (LYRA, Gutcke et al. 2021; radius rule, 50% per-pass cap, mass floor, 1% acceptance and 10-pass limit as in the reference implementation). New: donors shared by several stars that form in the same step are split in proportion to the requests, independent of order. All transfers are whole multiples of one power-of-two mass quantum, so every sum is an integer sum: mass is conserved exactly and the result does not depend on rank count, thread count, evaluation path or particle order. A star that does not reach its drawn mass within 10 passes is a failed draw: no star, and every quantum goes back to its donor. A design note for the gas collection is to be attached before review. In this PR stars are placed by a test-only flag; star formation uses it from PR 08.

**Evidence before.** The branch cannot form a star of a given mass without making a cell (or part of it) the star; with 1 Msun cells a 40 Msun star needs mass from many cells.

**Change.**
- New flag `GALSF_SINGLE_STARS_COLLECT_GAS_FROM_NEIGHBORS` (default off; implies GALSF_SINGLE_STARS_FROM_IMF). Test-only flag `TEST_SINGLE_STARS_BIRTH_AT_GIVEN_CELLS` with parameter `TestSingleStarBirths` ("ID:M_drawn,...").
- Five neighbour-loop Specs (count, request, grant, clear, apply) in `galaxy_sf/sfr_collect_gas_loop.h`, evaluated where the donor lives (host walker or Mode D); instantiated in `mesh/neighbor_loop_runner.cc`.
- Driver `galaxy_sf/sfr_collect_gas.cc` (in GPU_OBJS), called at the end of the star-formation step only when stars were spawned anywhere (known from the existing spawn-count reduction). Failed draws: grants are returned exactly from records kept on the donor's rank.
- Metals: a donor keeps its metallicity bitwise; the star gets the exact fixed-point metal sum, rounded once when stored as a mass fraction.
- Log `single_star_births.txt`: drawn and collected mass, radius, passes, failed draw, grants split at oversubscribed donors, grants from donors on other ranks, parent density.
- Compile-time error with radiation transport, cosmic-ray fluid or sinks (their extensive donor fields are not scaled yet).
- Template_Config lines; compile_suite rows (with the test flag; with MAGNETIC and the finite-volume solver).

**Cost.** No field in the cell or particle struct. Per-donor sums (3 x 8 B) in a side buffer by local gas index, allocated at the first collection and kept all-zero between collections; in the memory ledger. Collectives only in steps that spawned a star: per pass one integer reduction and the exchanges of four runner calls; one runner call to apply; a gather of the birth-log rows; only if a draw failed, a gather of the failed stars' identities. Nothing on quiet steps.

**Evidence after.** `test/single_stars_collect_gas` (broad; lattice of 1 Msun cells, metallicity gradient, bulk velocity and shear; stars placed by the test flag):
- Two and five stars with shared donors: each within 1% of its drawn mass, mass exactly conserved (exact rational sums), donors keep their metallicity bitwise, metal ledger within the storage rounding bound, momentum to 1e-12.
- Five-star case bitwise identical on 1, 2, 4 ranks, 2 ranks x 2 threads and permuted storage order.
- Stars at the box edges collect across the periodic wrap from donors on other ranks and give bitwise the 1-rank result.
- An unreachable 100 Msun draw leaves every gas mass and metallicity bitwise unchanged; a reachable star in the same box forms with exact totals.
- With the default merging of light cells all stars still form.

7 passed (measured on an older base; to be re-measured on the audit branch).

Comparison with the reference implementation (stella_dev), diagnostic, measured on an older base: one star with unshared donors (20 Msun) uses the same 146 donors, largest per-donor difference 9.4e-14 Msun. With shared donors the reference implementation depends on the decomposition (1 vs 4 ranks: donor masses differ by up to 0.52 Msun with 2 stars, 0.97 Msun with 5) and two of its five stars stop short within the 1% tolerance; here all reach their drawn mass and the result is bitwise rank-independent. Between the codes the difference is up to 1.8 Msun per donor (by design: proportional split instead of serving stars in order).

**Flag off.** Default config: 140 objects, all byte-identical or same instructions except `galaxy_sf/sfr_eff.o` (as PR 06) (measured on an older base; to be re-measured on the audit branch). Flag off: to be re-run on <TIP>.

**Size.** 661 source lines, over the ~300 limit. The five parts (request Specs, donor sharing, quantised transfer, apply step, failed-draw return) only work together: exact mass conservation holds only if all of them use the same quantum and the same donor list. Without the apply step nothing is observable; without the failed-draw return a failed star would keep or lose mass. `sfr_collect_gas_loop.h` (273 lines) is the loop body, `sfr_collect_gas.cc` (342 lines) the driver; the birth log is about 35 lines. Proposed split: 07a, loop header as a Spec with its test on fixed donors (about 270 lines); 07b, driver and donor selection (about 190); 07c, metal and momentum bookkeeping (about 180). The design note (order-independent constrained allocation of shared donor mass, no ID-order serialisation) goes first.

**Depends on / followed by.** PR 04, PR 06 (PR 03 for the mass quantum's upper IMF end). Followed by PR 08.

---

## 08. Single stars: IMF draw at birth, zero-mass spawn into the gas collection

Branch slice2/08-sf-birth. Commits: test, change, docs. 30 source lines. Test kind: broad.

**What and why.** Star formation makes single stars: the IMF mass is drawn first (PR 03, keyed), the probability is (m / M_drawn)(1 - e^-p), and an accepted cell spawns a zero-mass star through `spawn_star_from_gas` that collects its drawn mass (PR 07). As in the reference implementation, the probability can exceed 1 when M_drawn < m (the realised rate is then capped there); reported once per rank. The star-formation log counts the collected mass. Works with the existing star-formation criteria; PR 09 adds the single-star one.

**Evidence before.** With PR 07 alone, star formation still spawns population stars.

**Change.** `galaxy_sf/sfr_eff.cc` (under GALSF_SINGLE_STARS_COLLECT_GAS_FROM_NEIGHBORS). Parameters `IMFSampleMinMass` (default 0.08) and `SingleStarsRNGRealisation` (default 0; folded into the key of the draw, for independent realisations of a run).

**Cost.** None per particle; no collective added.

**Evidence after.** New `test/single_stars_formation` (broad; 16^3 lattice of 1 Msun cells at n_H about 1e4, density threshold the only criterion, high efficiency):
- The stars of the first star-forming step are bitwise identical on 1, 2 and 4 ranks.
- Over 10 steps: several hundred births, every star within 1% of its drawn mass, drawn masses pass a Kolmogorov-Smirnov test against the number-weighted Kroupa IMF.

2 passed (measured on an older base; to be re-measured on the audit branch).

**Flag off.** Default config: as PR 06 (measured on an older base; to be re-measured on the audit branch). Flag off: to be re-run on <TIP>.

**Depends on / followed by.** PR 03, PR 07. Followed by PR 09.

---

## 09. Single stars: star-formation criterion and cool-down after a birth

Branch slice2/09-sf-criterion. Commits: test, change, docs. 46 source lines. Test kind: broad.

**What and why.** The reference implementation's criterion for single stars: the density threshold; no star formation in a cell the HII loop marks as ionised; Jeans mass pi^(5/2)/6 c_s^3 / (G^(3/2) rho^(1/2)) below FacSfThreshMJ x DesNumNgb x m, with the physical density and the physical sound speed `effective_soundspeed()` (no magnetic term, as in the reference); rate m/t_sf. After a completed birth (not a failed draw) the parent cell forms no star until its density is recomputed, and then for SFCooldownFreeFallFraction x its free-fall time. Fix relative to the reference: it multiplies the sound speed by the scale factor a, which changes M_J by a^3 in cosmological runs.

**Evidence before.** No Jeans-mass criterion on the cell mass scale and no cool-down: a cell can form stars in consecutive steps (negative control below).

**Change.** New flag `GALSF_SINGLE_STARS_SF_CRITERION` (default off; implies the gas collection; replaces, and may not be combined with, `GALSF_SFR_CRITERION`). Parameters `FacSfThreshMJ`, `SFCooldownFreeFallFraction`. Files: `sfr_eff.cc` (criterion branch, cool-down), `sfr_collect_gas.cc` (sets the cool-down after a birth), `density_loop.cc` (clears the wait flag), `cell_data.h`. Template_Config line, compile_suite row.

**Cost.** Under the flag two fields per gas cell (a MyFloat and a byte). No collective; nothing on quiet steps.

**Evidence after.** `test/single_stars_sf_criterion` (broad; 16^3 magnetised lattice at n_H about 1e4, temperatures on either side of the limit):
- One step at negligible efficiency: non-zero SF rate exactly where the criterion recomputed from the snapshot says so (cells within 0.1% of the limit may differ); about half pass.
- With the density threshold above every cell, no cell forms stars.
- 16 steps at high efficiency: repeat births are at least the cool-down apart; without the cool-down a repeat birth comes sooner (negative control).

4 passed (measured on an older base; to be re-measured on the audit branch). Not tested: the ionised-cell case (needs the HII loop, not combined with single stars yet).

**Flag off.** Default config: as PR 06 (measured on an older base; to be re-measured on the audit branch). Flag off: to be re-run on <TIP>.

**Depends on / followed by.** PR 08.

---

## 10. Single stars: supernovae from the stellar-evolution tables in the feedback loop

Branch slice2/10-sn-source-model. Commits: test, change, docs. 265 source lines (between 250 and 300). Test kind: broad.

**What and why.** IMF-sampled stars become supernova sources of the shared mechanical feedback loop, with event times, classes, ejecta, yields and remnants from the tables (first user of PR 02). The coupling is the loop's; PR 11 adds the pure-thermal variant. A star never gains mass at death (`test_no_mass_gain_at_death` fails before this change and passes after).

**Evidence before.** With PR 04-09 a sampled star is a compile-time error together with any stellar feedback: the loop would give it stellar-population rates.

**Change.**
- `galaxy_sf/single_star_feedback.cc`, hooked into `mechanical_fb_calculate_eventrates` and `particle2in_addFB_fromstars` (`stellar_evolution.cc`) and around the loop in `mechanical_fb_calc_toplevel`.
- Death: age > PMS time + table lifetime, once. The mass at that moment is frozen in the record; every pack of the event (all loop modes, local and exported) recomputes ejecta, energy and yields from it. Classes from the table remnant type: core-collapse, electron-capture, pulsational and pair-instability SNe 1e51 erg; failed SN and direct black hole no energy. Ejecta = frozen mass - table remnant, composition from the table SN element ejecta scaled to that mass. Metals in the FIRE layout: slot 0 = all table metals C..Zn, slots 1-10 He, C, N, O, Ne, Mg, Si, S, Ca, Fe; other slots are only diluted.
- After the loop the star keeps exactly the table remnant mass (0 for pair instability: the particle is removed); the difference to what the loop removed is booked in `single_star_events.txt`. An event whose ejecta found no neighbour weight stays pending (counted; stop after 1000 steps).
- In a single-star build a star without a record is no feedback source (it would get the population rates).
- The event log is gathered only inside active feedback steps (one integer reduction there; a gather only if a row exists).
- The record gains the feedback state (frozen mass, zero-weight counter, event bits; 11 B per slot); snapshots `SingleStarEventBits`, so a start from a snapshot does not explode a remnant again.
- Compile-time error with COSMIC_RAY_FLUID (no cosmic rays from single-star SNe yet). Record guard: mechanical feedback allowed with the tables; thermal and FIRE feedback still error.
- Template_Config text, compile_suite rows.
- The weighting pass takes its ejecta mass from the base's rule; the single-star value is set in 12b.

**Cost.** 11 B per record slot. In feedback steps one integer reduction, and a gather only when an event happened. Nothing on quiet steps; table lookups only for stars on their own step.

**Evidence after.** `test/single_stars_supernovae` (broad; adiabatic, 32^3 lattice of 1 Msun cells at n_H = 10, four stars at 12 km/s: 20 Msun core collapse, 40 Msun failed SN, 150 Msun pulsational pair instability, 320 Msun pair instability, Z = 0.014):
- One event per star of the table class; remnant exactly the table value, pair-instability star gone.
- Removed mass = trigger mass - remnant to 1e-12; gas gains the ejecta (1e-12) and the ejected metals (1e-10).
- Events identical on 1, 2, 4 ranks and with 2 threads.
- In the FIRE metal layout (second build with COOLING + COOL_METAL_LINES_BY_SPECIES) every slot gains exactly the table SN element ejecta (1e-9).

3 passed; the lookup test too (measured on an older base; to be re-measured on the audit branch).

Comparison with the reference implementation (stella_dev), diagnostic, measured on an older base; same initial conditions, 48^3 lattice at n_H = 100: remnants 1.54157 / 37.8977 / 15.43405 Msun for 20 / 150 / 40 Msun in both codes; gas mass gained 155.12668 Msun in both; metal mass 12.66258 vs 12.66248 Msun (equal within the reference's float32 output); He, C, N, O, Ne, Mg, Si, S, Ca, Fe identical to all printed digits.

**Flag off.** Default config: 140/140 objects compared; 129 byte-identical, the others same instructions, except `galaxy_sf/sfr_eff.o` (the spawn function of PR 06). FIRE config (FIRE_PHYSICS_DEFAULTS=3): 127 byte-identical, the same otherwise (measured on an older base; to be re-measured on the audit branch). Flag off: to be re-run on <TIP>.

**Depends on / followed by.** PR 02, PR 04 (records), PR 08/09 for stars formed in a run. Followed by PR 11.

---

## 11. Mechanical feedback: pure-thermal supernova coupling, default for single stars

Branch slice2/11-sn-thermal-coupling. Commits: test, change, docs. 24 source lines. Test kind: broad.

**What and why.** For single-star supernovae the reference implementation's validated coupling is pure thermal: the whole explosion energy as heat over the same weights as the ejecta mass, with the post-deposit mass, no radial ejecta momentum; the ejecta carry the star's velocity. On a resolved lattice the loop's mechanical coupling adds more energy than the explosion has. The added ejecta mass is cold: it carries no ambient u, because the internal energy is diluted at once in the deposit.

**Evidence before.** The supernova test with an energy and a momentum budget, run on the test commit (PR 10 code, mechanical coupling): the total energy gained is 1.10 times the explosion energies (3.30e51 erg for 3e51); `abs(dE/E_SN - 1) = 0.101`, bar 1e-3 (measured on an older base; to be re-measured on the audit branch).

**Change.** New flag `GALSF_FB_MECHANICAL_SNE_THERMAL_ONLY` (off for every existing configuration; on by default for single stars from the tables; `GALSF_SINGLE_STARS_SNE_MECHANICAL_COUPLING` keeps the mechanical coupling). In `mechanical_fb_functions.h` two `#ifdef` places: the per-source setup (no momentum term, the whole energy as thermal residual) and the pair kernel (the ejecta velocity term also when there is no radial momentum). The apply step is unchanged. Template_Config line; compile_suite rows (the flag on FIRE-style mechanical feedback; single stars with the mechanical coupling).

**Cost.** None.

**Evidence after.** The same test (broad): total energy rises by the explosion energies (relative error -1.9e-5, bar 1e-3; the ejecta's kinetic energy at the stars' speed, 2.3e-4 of the total, is partly lost when they mix with gas at rest); momentum conserved to 1e-10. A mass check in hot gas gives 0.99999999 right after the deposit and 1.00004 at the end of the run, 1e-3 bar unchanged (these two values to be re-measured on <TIP>). 3 passed (measured on an older base; to be re-measured on the audit branch).

Comparison with the reference implementation (stella_dev), measured on an older base: energy within 11 pc of the core-collapse SN 1.0015e51 erg (here) vs 1.0052e51 erg.

**Flag off.** Default config: 140/140 objects compared; 129 byte-identical, the others same instructions, except `galaxy_sf/sfr_eff.o` (the spawn function of PR 06). FIRE config: 127 byte-identical, the same otherwise (measured on an older base; to be re-measured on the audit branch). Flag off: to be re-run on <TIP>.

**Depends on / followed by.** PR 10.

---

## 12. Single stars: winds from the tables through the loop's mass-return mode

Branch slice2/12-winds. Commits: test (wind test; SN test counts winds as returned), change, docs. 127 source lines added, 28 removed. Test kind: broad.

**What and why.** Massive single stars return their winds during their life, and what is not returned by death leaves with the death ejecta. The loop's mass-return mode (mode 1) was compiled only with the FIRE stellar evolution. This PR also closes the energy per cell in the star frame.

**Evidence before.** Without wind return a star keeps its whole-life wind mass until death; the reference implementation (stella_dev) returns it continuously.

**Change.**
- `GALSF_FB_MASS_RETURN_EVENTS` (`precompiler_logic.h`) = `GALSF_FB_FIRE_STELLAREVOLUTION` or single stars with the tables. It replaces that flag where it gates the mode-1 fields and checks (`particle_data.h`, `init.cc`, `mechanical_fb.cc`, `mechanical_fb_functions.h`, `mechfb_loop.cc`, `density_loop.cc`). No change for any existing configuration (FIRE flag-off below).
- Winds (drawn mass >= 8 Msun, after the PMS): every star step returns the table's cumulative wind ejecta between the age of the last delivery and now, at the table wind speed, with that interval's composition (surface composition where the interval is flat). The last delivery age (record, 8 B; snapshot `SingleStarLastWindLogAge`) advances only when the mass coupled, so nothing is lost and the delivered mass telescopes to the table. At death any wind owed up to the end of life goes first (one step), then the death; winds still not returned go with the death ejecta (their composition added).
- Per-cell energy closure in the star frame (pair kernel): each cell takes its share of the ejecta energy less the kinetic energy it gained in the star's frame.

**Cost.** 8 B per record slot. Table lookups once per massive star per own step. No collective added.

**Evidence after.** `test/single_stars_winds_agb_ia` (broad; adiabatic, 24^3 lattice at n_H = 10; a 35 Msun star 1 Myr old, 10 kyr):
- Wind mass returned = the table's cumulative wind ejecta at the last delivery age (1e-9); that age = the star's age to one step.
- Mass conserved to 1e-12.
- Momentum change below 5% of the first step's wind momentum.
- Same wind mass on 1 and 4 ranks.
- The supernova test runs with initial conditions that count the winds as returned.
- Wind energy test `test_wind_energy`: the energy the gas and the star gained (kinetic + thermal) over the booked 1/2 M v^2 of the mass returned at the table speed. The bar was 0.99 to 1.08 (1.045 before the closure); with the closure it is 1e-3 (0.9995). Both values to be re-measured on <TIP>; the pass/fail before and after is in the test logs.

2 + 3 passed (measured on an older base; to be re-measured on the audit branch).

Before the closure, the ratio was the unchanged upstream momentum solve (`mechanical_fb_functions.h`), not the single-star code, and it grows with the ejecta mass per cell mass: one 24.6 Msun dump at 4830 km/s into cells of 1 / 8 / 64 Msun gives 1.31 / 1.09 / 1.005 of 1/2 M v^2. The 150 and 40 Msun dumps (100 and 25 Msun into 1 Msun cells) give 1.18 and 1.24; the per-step winds of a star formed in the run are far below that regime (measured on an older base; to be re-measured on the audit branch).

Comparison with the reference implementation (stella_dev), measured on an older base; table-driven wind on both sides, 35 Msun, n_H = 10, 150 kyr: wind mass returned 0.04129 Msun in both at 147 kyr; energy 4.80e48 vs 4.94e48 erg (-3%); shell radius from the density peak 5.12 vs 5.88 pc (0.25 pc bins, a few cells at 1 Msun); Weaver radius from each run's own energy input 5.28 vs 5.31 pc.

**Flag off.** Default config: 140/140 objects compared; 128 byte-identical, the others same instructions, except `galaxy_sf/sfr_eff.o` (the spawn function of PR 06). FIRE config (where the renamed gate is on): 126 byte-identical, the same otherwise (measured on an older base; to be re-measured on the audit branch). Flag off: to be re-run on <TIP>.

**Depends on / followed by.** PR 11.

---

## 12b. Single stars: their own ejecta mass in the feedback weighting pass

Branch slice2/12b-mechfb-apply-step, on slice2/12-winds. Commits: test, change, docs. 19 source lines added, 9 removed. Test kind: broad.

**What and why.** The weighting pass of the mechanical feedback sums neighbour weights with an ejecta mass. The base now takes the ejecta mass of every active mode (#106 and the general fix up/B3-trimmed), but it gets it through `particle2in` with channel 0, which for single stars returns only the SN and Type Ia ejecta: a wind or AGB step would fall back to the star's mass (a 35 Msun star returning 1e-3 Msun of wind would be weighted as 35 Msun). This PR sets the single-star ejecta mass after the base's block, from the record and the tables (`single_star_ejecta_mass_this_step`). Two small parts in the same apply path come with it: single-star cells are diluted (cold ejecta) before the injected energy is read, and pure-thermal SNe (PR 11) take the whole energy as heat through the per-cell closure, with `retain_thermal_flag` checked before the closure.

**Evidence before.** No test fails before this change. Energy is read from the gas state in double-precision snapshots: gain = E(run) - E(twin) at the same time, where the twin is the same initial conditions without the event, so cooling, gravity and radiation cancel without any ledger; offered = 1e51 erg per SN, 1/2 sum dM v_w^2 for winds; bar 2 %. Slice-level case (the adiabatic configuration of `test/single_stars_winds_agb_ia`; a 150 Msun star's wind in 1e7 K gas moving radially out at 0.75 v_w, n_H = 1): 1.0003 before, 1.0075 after. A first version of the test without the twin gave 1.32, which was the hydro drift of the flow. The PR is offered for consistency with the base's rule, not for a measured error; it can be dropped if the reviewer prefers.

**Change.** `mechfb_loop.cc` (override after the base's weighting-pass block, single-star builds only), `single_star_feedback.cc` (`single_star_ejecta_mass_this_step`), `mechanical_fb.cc` (dilution before the energy read), `mechanical_fb_functions.h` (pure-thermal SN through the cell closure), `proto.h`. Test case added to `test/single_stars_winds_agb_ia`.

**Cost.** One table evaluation per single-star source in the weighting pass. No memory, no collective, nothing on quiet steps.

**Evidence after.** Measured on an older integrated line (to be re-measured on <TIP>), cloud configuration with double-precision output (chemistry net 5, gravity, MHD, photoionisation, radiation pressure), lattice of 1 Msun cells at 1e7 K, ionised:
- SN, 20 Msun, n_H 10: 0.9806 before and after.
- Winds, 150 Msun at n_H 1: 1.0018; 35 Msun at n_H 10: 0.9925; before and after the same.
- Wind in outflowing gas: 1.1162 before, 1.1456 after. 12c item 6 shows this is the time integration of long steps, not the coupling.
- Weak wind, 20 Msun at n_H 10: run minus twin is not reliable past t of about 4e-5 at this cooling (0.83 early, 0.054 at the end).
- The weighting-pass change moves no static case in any printed digit.
- These values were read from snapshots whose last output lands on TimeMax before the closing kick; on closed end states the SN gives 0.990 to 0.992 (12c item 4). The production-case values come from the analysis of those runs; the analysis output was not stored and is to be regenerated.
- Smoke run of the integrated line: reaches TimeMax, mass drift -2.1e-9, 1004 stars, 469 Msun at 4.9 Myr.

**Flag off.** On the older line: gmc_cooling 143/145 and 145/147, FIRE 142/145 and 144/147 objects byte-identical, none with different instructions. To be re-run on <TIP>.

**Limits.** No failing case before the change. Cooling runs past about 4e-5 code time cannot be judged by run minus twin.

**Depends on / followed by.** PR 12 and up/B3-trimmed (same files; on the audit branch single stars keep their dilution before the apply step and use dU = dTE/mf there, so u is not diluted twice). Followed by 12c, 13.

---

## 12c. HEALPix feedback weights: booked wind momentum and the mass x kernel deposit; energy with gas moving relative to the star

Branch slice2/12c-closure-moving-gas: two commits (test, fix) on an older integrated line that already holds the HEALPix and mass x kernel weights; on the audit branch both commits are present as they are. 4 source lines added, 4 removed (3 and 3 on the audit branch, where one of the four gates no longer exists). Test kind: broad (energy against what the source offers).

**What and why.** Two things: a measurement that SN and wind deposits close in energy when the gas moves relative to the star, and a fix for the one case where they do not, the HEALPix-weights variant (`GALSF_FB_MECHANICAL_HEALPIX_WEIGHTS`, off in every configuration). That variant solved the wind momentum and limited the apply step (`f0`), so a slow wind lost energy even at rest (82 km/s effective instead of the table's 103 km/s) and a moving star gave 2.1 times what it offered. The fix uses the booked wind momentum, the overlap heat of the mass x kernel weights and `f0 = 1` for HEALPix weights too.

**Evidence before.** Energy from the gas state in double-precision snapshots, in the star's frame: E = sum m (u + |v - v_star|^2 / 2), gain = E(run) - E(twin), twin = same initial conditions without the event, offered as in 12b, bar 2 %. Cases: star at rest or moving at 30, 100, 300 km/s; SN (20 Msun), fast wind (35 Msun at 1 Myr, about 3000 km/s), slow wind (20 Msun near the end of its life, 50 to 150 km/s); each in hot gas (1e7 K, ionised) and cold gas (100 K, neutral, short window), plus the outflow case of 12b. New test in `test/mechfb_healpix_weights`: moving star, fast and slow wind, energy over the same star at rest, bar 1e-9. Before the fix: fast wind 0.99999999928 (pass), slow wind 2.1158 (fail).

**Change.** `mechanical_fb.cc`, `mechanical_fb_functions.h`: the gates `#if defined(GALSF_FB_MECHANICAL_MASS_KERNEL_WEIGHTS)` for the booked wind momentum, the overlap heat and `f0 = 1` also take `GALSF_FB_MECHANICAL_HEALPIX_WEIGHTS`. Builds with neither flag are unchanged by construction.

**Cost.** None.

**Evidence after.** Measured on the older integrated line, cloud configuration with double-precision output, 2 ranks (to be re-measured on <TIP>). Gain from closed end snapshots only (item 4).
1. The energy clip does not fire on the default deposit. With the booked wind momentum and momentum shares equal to mass shares, a cell's heat is 1/2 dm w^2 - dKE = 1/2 mu |v - w|^2 >= 0 (the reference implementation's inelastic residual). A passive counter of clipped energy reads 0 in every case. Allowing negative heat gives the same numbers to every digit.
2. The deposit closes in every case: gain + (run - twin) chemistry cooling - heating, over offered: SN 0.991 to 1.001; fast winds 0.9995 to 1.0000; slow winds 0.983 to 0.996 (hot, warm). Raw gain alone fails only through differential cooling, for example the fast hot wind at 300 km/s: 1.579 raw, 0.9999 closed; the slow warm wind at 300 km/s: 0.462 raw, 0.996 closed.
3. Cold-gas (100 K) winds lose the heat to the photoionisation pin (cells inside the HII region set to the ionised state); with `HIIRegion_fLum_Coupled 0` they close (0.998 and 0.996 at 300 km/s).
4. The 2 % SN deficit of 12b was the snapshot, not the gas: snapshots write `P.Vel` kicked to mid-step with `InternalEnergyPred`, and the last output landed on TimeMax before the closing kick. On closed end states: hot SN 0.990 to 0.992 at 0, 30, 100, 300 km/s. Reference implementation (stella_dev): 0.980 to 0.986.
5. Long windows lose energy to the time integration, in both codes: cold SN at rest, 3.2e-6 window, 0.958 at default steps, 1.0020 with `MaxSizeTimestep` 1e-7 (with cooling); the reference implementation with the same step cap 0.9635 against 0.9652 here. Coupling after the closing kick (af507c293, ported in a scratch tree) does not remove it (0.956).
6. The outflow wind case of 12b (1.764 on this line) is the same integration error, not the clip: clip counter 0, ledger 0.9998, the excess does not scale with what is offered (a 35 Msun star in the same flow, 60 times less wind energy, has the same 1.85e49 erg excess), and with `MaxSizeTimestep` 1e-7 gain/offered = 1.0000.
7. HEALPix fix: slow wind 2.1158 before, 1.0000000061 after, which misses its 1e-9 bar by 6e-9; cause not found (the star's recoil term is -1.4e-9). Fast wind passes before and after. Mass x kernel weights test: 26 passed, 2 failed, the 2 being the HEALPix slow-wind cases above.
8. The reference implementation's slow-wind cases are not comparable (its star dies when the current mass reaches the pre-SN mass, inside the window); its fast winds: hot 0.965 to 1.002, warm 1.000, cold 0.963 to 0.973.

**Flag off.** Builds without `GALSF_FB_MECHANICAL_HEALPIX_WEIGHTS` are unchanged by construction (all edits inside that gate). To be re-run on <TIP>.

**Limits.** The HEALPix slow-wind case misses its 1e-9 bar by 6e-9. Long steps lose a few per cent of the deposit energy in both codes (items 5, 6). A run shorter than about 1.25 kyr hangs in the photoionisation step cast; that fix is in the pi series (pi/11).

**Depends on / followed by.** The HEALPix and mass x kernel weights PRs (hp/01, hp/02) and 12b. The earlier size of 338 lines counted the weights code of those PRs; the own change is 8 lines, so no split is needed.

---

## 13. Single stars: AGB deaths with white-dwarf remnants from the tables

Branch slice2/13-agb-deaths. Commits: test, change, docs. 29 source lines added, 16 removed. Test kind: broad.

**What and why.** Stars with a white-dwarf remnant die without a supernova: their envelope returns through the mass-return mode. Until here they lived forever.

**Evidence before.** A 3 Msun star past its lifetime keeps its mass (PR 10-12 skip white-dwarf progenitors).

**Change.** `single_star_feedback.cc`: AGB death = mass - table remnant at 30 km/s, AGB element ejecta plus the winds not returned during the life (below 8 Msun that is all of them; 26% of the ejecta at 1 Msun), remnant = table; event bit `SINGLE_STAR_EVENT_AGB_DONE`.

**Cost.** None beyond PR 12.

**Evidence after.** The wind test (broad) adds a 3 Msun star just past its lifetime: one AGB death, white-dwarf remnant of the table, ejecta = 3 Msun - remnant (1e-12); mass conserved; same events on 1 and 4 ranks. 2 + 3 passed (measured on an older base; to be re-measured on the audit branch).

**Flag off.** Default config: 140/140 objects compared; 128 byte-identical, the others same instructions, except `galaxy_sf/sfr_eff.o` (the spawn function of PR 06) (measured on an older base; to be re-measured on the audit branch). FIRE build not rerun on the older base: only single-star files change. Flag off: to be re-run on <TIP>.

**Depends on / followed by.** PR 12.

---

## 14. Single stars: Type Ia supernovae of white dwarfs

Branch slice2/14-type-ia. Commits: test, change, docs. 91 source lines added, 7 removed. Test kind: broad.

**What and why.** White-dwarf remnants explode as Type Ia SNe with the reference implementation's per-step hazard: t^-1 delay-time distribution from 40 Myr, normalised to the dead-progenitor fraction of the IMF (tabulated at start-up from the tables and the IMF lower end), keyed random number. The whole mass goes at 1e51 erg with the table Type Ia yields, no remnant (pure thermal, PR 11).

**Evidence before.** White dwarfs from PR 13 never explode.

**Change.** New flag `GALSF_SINGLE_STARS_TYPE_IA` (default off); test-only `TEST_SINGLE_STARS_TYPE_IA_RATE_MULTIPLIER`. Files: `single_star_feedback.cc` (hazard, event), `stellar_evolution_tables.cc` (dead-progenitor fraction, 15 x 96 doubles). Template_Config lines; the compile_suite row with the per-element build gains the flag.

**Cost.** 11 kB per rank (the tabulated fraction). One keyed random number per white dwarf per own step. No collective.

**Evidence after.**
- The wind test (broad) adds a 1 Gyr white dwarf and the test-only multiplier: Type Ia of both white dwarfs (the initial one at the first step, the one from the AGB death later): the whole mass at 1e51 erg, no remnant, ejecta almost all metals; mass conserved to 1e-12; same on 1 and 4 ranks.
- The dead-progenitor fraction (8 metallicities x 40 ages, past both ends of its table) agrees with the reference implementation's to 1e-12 (all 320 values stored in the lookup test).
- `test/single_stars_type_ia_rate` (broad): 3000 white dwarfs of drawn mass 3 Msun, 1 Gyr old, about 20 kyr with the test-only multiplier 1e6. The number of Type Ia events by half time and by the end agrees with 3000 (1 - exp(-H)), H the integral of the reference implementation's hazard (t^-1 distribution, its constants and dead-progenitor fraction), within 4 binomial sigma; no star explodes twice.

Passed: winds_agb_ia 3 (with the wind energy test), tables 4, rate 1 (measured on an older base; to be re-measured on the audit branch).

**Flag off.** Default config: 140/140 objects compared; 128 byte-identical, the others same instructions, except `galaxy_sf/sfr_eff.o` (the spawn function of PR 06) (measured on an older base; to be re-measured on the audit branch). FIRE build not rerun on the older base: only single-star files change. Flag off: to be re-run on <TIP>.

**Depends on / followed by.** PR 11, PR 13.

---

## 15. Single stars: step limit from the time to the next stellar-evolution event

Branch slice2/15-step-limit. Commits: test, change, docs. 46 source lines. Test kind: narrow.

**What and why.** A single star's step must not jump over its death. The reference implementation's rule: at most MaxStellarTimestep (default 50 kyr), 1/50 of the time to death within 100 such steps of it, never past it, floor 10 yr.

**Evidence before.** Stars 37 kyr from death, MaxSizeTimestep = the whole 60 kyr run: the stars step over their death. `test_deaths_on_time_with_long_steps` fails on the test commit: all four die at 60 kyr, the end of their one step, instead of 37-38 kyr (measured on an older base; to be re-measured on the audit branch).

**Change.** `single_star_max_timestep` (`single_star_feedback.cc`), applied in `get_timestep` for stars with a record. Parameter `MaxStellarTimestep` [yr] (default 50000).

**Cost.** One table lookup per single star per timestep calculation. No collective.

**Evidence after.** The same test (narrow): each star dies within 1 kyr of its table death time (37-38 kyr); the other supernova and wind tests are unchanged. 4 + 2 passed (measured on an older base; to be re-measured on the audit branch).

**Flag off.** Default config: 140/140 objects compared; 127 byte-identical, the others same instructions, except `galaxy_sf/sfr_eff.o` (the spawn function of PR 06). FIRE config: 125 byte-identical, the same otherwise (measured on an older base; to be re-measured on the audit branch). Flag off: to be re-run on <TIP>.

**Limits.** PR 16 replaces this rule by the plain step cap (death at the first step past the lifetime). The two can be folded: PR 15 then only adds `MaxStellarTimestep` and its test.

**Depends on / followed by.** PR 10. Followed by PR 16.

---

## 16. Single stars: death at the first own step past the lifetime, with the winds still owed in the ejecta

Three commits on the audit branch (b0af54fc2 change, 61eef1ceb test, b422e430e docs), no PR branch yet. 29 source lines added, 33 removed (`single_star_feedback.cc`, one comment in `timestep.cc`). Test kind: broad (death times and ejecta against the tables).

**What and why.** PR 15 shortens a star's steps as it nears death (1/50 of the time left within 100 steps of it, never past it). In a cloud every massive star does this, and the whole run takes the short steps: in a tall-box run of the reference implementation (stella_dev) a profile showed about 6 times more global steps from this rule (to be re-measured on <TIP>). The earlier resolved-ISM code that the single-star model follows (gizmo2017, Hu et al. 2017 lineage; test names use that label) has only a fixed step cap and lets a star die at its first step past its lifetime. This PR does the same: the step is at most `MaxStellarTimestep`, nothing else, and the death fires at the first own step past PMS time + table lifetime. A star that dies late by up to one step still returns all of its wind: the wind mass owed up to the end of its life leaves with the death ejecta, with its own composition, instead of a separate wind step before the death.

**Evidence before.** `test/single_stars_supernovae`, two new tests, run on the series before this change (older base):
- `test_death_step_as_gizmo2017` (stars 37 kyr from death, cap 10 or 50 kyr, longest step 120 kyr): 377 steps before the first death, against at most 10 (cap 10 kyr) or 4 (cap 50 kyr) allowed by the bar. Fails for both caps.
- `test_death_with_winds_owed` (stars past their death with the winds of the last 5 % of their life not returned): the 40 Msun failed SN, which has no table yields, ejects 0.335 Msun of metals against the 0.148 Msun its owed winds carry. Fails.

**Change.** `single_star_max_timestep` returns `MaxStellarTimestep` only. `single_star_eventrates`: winds while age <= end of life, death at the first step after it; the separate step that returned the owed winds before the death is removed. New `single_star_death_ejecta`: above 8 Msun the owed winds (mass and element yields, as a wind delivery would return them) plus the table ejecta scaled to the rest; below 8 Msun the winds not returned during the life added to the AGB or SN ejecta, as before. The test-only options of the event hooks (a forced death time, an instant SN, winds in chunks) keep working: forced and instant set the star dead, chunks hold only winds before death.

**Cost.** Fewer steps: the death needs no step of its own and no step shortening. No memory, no collective.

**Evidence after.** `test/single_stars_supernovae`: 8 passed, including both tests above (each star dies after 37 kyr and at most one cap later; no step before the first death below half the cap; the owed winds leave with the death ejecta to 1e-4 in metal mass; mass to 1e-12). Measured on an older base, to be re-measured on <TIP>. Not yet measured: step count and wall time of a cloud or tall-box run with this rule on the audit branch.

**Flag off.** Only single-star code changes (`single_star_feedback.cc` is compiled only with the tables and single stars; the `timestep.cc` edit is a comment). To be re-run on <TIP>.

**Limits.** A star dies up to one step (at most `MaxStellarTimestep`, default 50 kyr; the 2017 code used 10 kyr) after its table lifetime; the death energy and ejecta are those of the table at the death, the delay is not corrected. Step-count gain on a production run not yet measured.

**Depends on / followed by.** PR 12 (winds), PR 13 (AGB deaths), PR 15 (replaces its step rule; the two can be folded). Last of the series.

