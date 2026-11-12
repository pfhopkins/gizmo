# SGChem chemistry: PR drafts (series chem)

**Series:** chem/01 ... chem/12. SGChem non-equilibrium chemistry (Glover/Clark networks) as C++ on SUNDIALS CVODE, wired into the cooling module behind the build flag `SGCHEM`. The reference implementation (stella_dev) is used for comparison numbers.
**Base:** `<BASE>`. **Tip:** `<TIP>`.
**Order:** slice2 -> chem (chem/12 before tc) -> tc -> gap -> cr (cr/04 before pi/06) -> met -> pi -> ext -> tr -> kj.
**Inside the series:** 01 -> 02 -> 03 -> 04 ... -> 11 -> 12 (12 only touches start-up). Nothing in GIZMO calls the chemistry before PR 4.
**One SUNDIALS per build:** SGChem and CHIMES cannot be built together (`#error` in PR 4).
**Numbers:** numbers below were measured on older bases (43025c83, 701a78c5) and are to be re-measured on `<TIP>`. Source lines are recomputed against the previous PR (tests, docs, scripts, Markdown notes and Template_Config excluded). Full suite: `<SUITE RESULT>`.

| PR | branch | source lines (+/-) | test | breadth | depends on |
|---|---|---|---|---|---|
| 1 | `chem/01-sgchem-core-net5` | +12723 (imported; incl. 141 lines of the tables check shared with slice2/01) | `test/sgchem_onezone` net 5 | broad | tables PR (same three files) |
| 2 | `chem/02-sgchem-networks-3-15-16-17` | +2258 (imported) | `test/sgchem_onezone` nets 3/15/16/17 | broad | 1 |
| 3 | `chem/03-sgchem-networks-18-20` | +1973 -3 (imported) | `test/sgchem_onezone` nets 18/20 | broad | 2 |
| 4 | `chem/04-sgchem-build-and-startup` | +146 -3 | `test/sgchem_box::test_sgchem_startup` | narrow | 3 |
| 5 | `chem/05-sgchem-cell-state` | +39 | `test_sgchem_cell_state` | narrow | 4 |
| 6 | `chem/06-sgchem-cooling-hook` | +161 -3 | `test_sgchem_cooling` | narrow | 5 |
| 7 | `chem/07-sgchem-snapshot-io` | +196 | `test_sgchem_snapshot_io` | narrow | 6 |
| 8 | `chem/08-sgchem-restart` | 0 (test only) | `test_sgchem_restart` | narrow | 7 |
| 9 | `chem/09-sgchem-uniform-field` | +30 -2 | `test_sgchem_habing_field[1,10,100]` | narrow | 8 |
| 10 | `chem/10-sgchem-ledger` | +142 -3 | `test_sgchem_ledger` (+ restart test) | narrow | 9 |
| 11 | `chem/11-sgchem-net5-euv-h2` | +13 -2 | `test_sgchem_net5_euv_h2[10,100]` | narrow | 10 |
| 12 | `chem/12-sgchem-keep-ic-abundances` | +5 | `test/sgchem_keep_ic` | narrow | 11 |

## Import size: open question for the reviewer

PRs 1-3 are imported C++ code, checked bitwise against the reference implementation, and they are far above the ~300 line limit:

- chem/01: about 12,700 source lines.
- chem/02: about 2,260 lines.
- chem/03: about 1,970 lines.

The rest of the series (PRs 4-12) is 0 to ~200 lines each and stays within the limit.

A split by file group would look like this:

- **chem/01**, reviewed against the reference dump rather than line by line:
  - types, constants and network header (~1300);
  - spline and table build (~900);
  - generated rates (`rates_gen.cc` 4067, `rates_gen.h` 384, `rates_registry.inc` 349, `chem_tables_data.h` 1387): ship the generator and a hash, not a review of the output;
  - right-hand side common code and net 5 (~350);
  - CVODE driver (672);
  - photo, temperature, cool_func and util (~1900, 3 PRs);
  - GIZMO backend glue and interface (~600, 2 PRs);
  - Fortran reference dump tool (227, test only, could stay out).
- **chem/02:** one PR per network (net 3: 1017, net 16: 413, net 15: 302, net 17: 526). Net 3 is still over 300; split its right-hand side and cooling terms.
- **chem/03:** net 18 (959 + tables 98) and net 20 (646 + CIE 189), one PR each; net 18 tables as installed data.

Question: may the library import go as one PR reviewed as a library (with the bitwise comparison and the reference test as the evidence), or do you want the split above? Generated tables already move to installed data where feasible: the 12.6k-line network 18 header is done in PR 3. The next candidate is `chem_tables_data.h` (1385 lines, arrays extracted from the Fortran DATA statements, used by every network).

## Attribution and licence

Every C++ file of the chemistry (`cooling/sgchem/cxx/*.cc|*.h`) and the GIZMO glue files start with a two-line C comment naming the C++ author and stating "Based on the SGChem Fortran networks by Simon Glover (GNU GPL, as distributed in Arepo)". On the reference implementation the same comment-only change touches 34 files; objcmp of a net 5 production build (`Config_stella_default.sh`) shows 117/117 objects with identical instructions, the only difference being `__LINE__` +2 in `sgchem.o` (older base; to be re-measured on `<TIP>`).

PR 3 adds `cooling/sgchem/NOTICE`. It states no GPL version, because the version is still being confirmed with the original author. Open question: is that acceptable for the merge?

## Common evidence (PRs 1-3)

(Older base.) gcc 13.1, SUNDIALS 6.7.0, Kokkos 5.0.0 OpenMP.

- One-zone dumps of this tree are compared with the reference implementation (stella_dev, 56804448), built by its own `build_dump.sh` from a read-only copy with its production config `Config_stella_default.sh`. The comparison covers the chemistry tables, `cool_func` rates on a (T, n) grid and 18 one-zone runs. All seven networks are **bitwise identical**. Network 18 needs the same table loader on both sides (see PR 3).
- All seven networks also pass against the frozen reference: the SGChem Fortran for 3/5/15/16/17, the C++ backend at its introduction for 18/20. The reference data are checked with `data.sha256`. Bars: tables 1e-6, rates 1e-10, final one-zone T 1e-2. The largest one-zone T deviation is the Fortran DVODE vs CVODE step timing, e.g. net 5 zone 17 at 9.0e-3.
- Threads: eight cells are solved serially and 64 times concurrently on 4 OpenMP threads. The results are bit-identical for nets 5 and 17.

---

## PR 1: SGChem network core and network 5 (`chem/01-sgchem-core-net5`)

**What and why.** This adds the SGChem chemistry networks (Glover/Clark; Arepo SGChem is the reference) as C++ on SUNDIALS CVODE in `cooling/sgchem/`. This PR carries the shared core (rate tables, cooling functions, photo-rates, temperature lookup, CVODE driver) and network 5 (H2, H+, CO), the default network of the Stella model. Nothing in GIZMO calls it yet: the build flag and start-up follow in PR 4, the cooling hook in PR 6. Size: see "Import size" above.

**Evidence before.** The base has no SGChem, and CHIMES is the only non-equilibrium network. Stella runs need SGChem.

**Change.**
- `src/cooling/sgchem/cxx/`: the C++ backend as on the reference implementation (stella_dev, 56804448), without the Fortran backend and its C API. Changes from the reference:
  - the `SGCHEM_FORTRAN` switch is gone;
  - the tables directory comes from a parameter, else `CHEMCXX_DATADIR` (no default path into the source tree);
  - `chem_sundials.h` selects `SUN_COMM_NULL` for SUNDIALS 7 and later at both context sites;
  - the other networks are called only under their own `CHEMISTRYNETWORK` (one network per build), so the core links without them.
- `src/cooling/sgchem/{chem_interface.h, sgchem_consts.h, sgchem_network.h}`: the interface to GIZMO, Fortran-free.
- New `src/cooling/sgchem/sgchem_config.h`: chemistry-private switches. `WSS_CIE_COOL`, `CO_SHIELDING` and the cell-metal inputs are always on under `SGCHEM`. Without the TreeCol extension there are no per-pixel columns, so the chemistry is optically thin. `TEST_SGCHEM_ONEZONE_WITH_12_PIXEL_COLUMNS` gives the one-zone drivers the 12-pixel input layout of the frozen references; GIZMO builds reject it.
- `scripts/get_stella_tables.sh`, `scripts/stella_tables_hash.py`, `src/galaxy_sf/stella_tables_check.h`: byte-identical to the tables PR (1506cec0). Whichever lands second drops them.
- Test: `test/sgchem_onezone/` (pytest, standalone, no GIZMO run); `cxx/fortran_reference/{reference_dump.cc, reference.py, data.sha256}`.
- No new flags in this PR; no GIZMO file changed.

**Cost.** Nothing in any build: no GIZMO object compiles these files before PR 4.

**Evidence after** (older base; to be re-measured on `<TIP>`).
- `test_sgchem_onezone[5]` against the frozen Fortran reference: PASS (broad).
- Net 5 is bitwise identical to the reference implementation (tables, rates, one-zone).
- Threads: 4 threads, 64 concurrent solves, bit-identical to serial.

**Flag off.** No GIZMO source changed, so the objects are identical by construction. Flag off: to be re-run on `<TIP>`.

**Limits / open question.** The frozen reference data (about 10 MB) are not in the repository: `SGCHEM_REFERENCE_DATA`, checked with `data.sha256`. Proposed home: the stella-tables repository next to the tables. The test skips with a reason until it is set. Is that the right place?

**Depends on / followed by.** Tables PR (same three files) / PR 2.

---

## PR 2: SGChem networks 3, 15, 16 and 17 (`chem/02-sgchem-networks-3-15-16-17`)

**What and why.** The remaining right-hand sides of the classic networks, each selected by `CHEMISTRYNETWORK`:
- 3: Glover+10 with C/O chemistry;
- 15: NL99 extended (Glover & Clark 2012);
- 16: Gong+17;
- 17: net 5 plus He, D and HD.

Size: see "Import size" above (about 2,260 lines).

**Evidence before.** With PR 1 only network 5 links. Networks 3/15/16/17 are used in the PDR slab and Arepo SGChem cross-checks.

**Change.** `cxx/chem_rhs_net{3,15,16,17}.cc`, as on the reference implementation (stella_dev, 56804448). Test first: the parameter list of `test_sgchem_onezone` gains 3/15/16/17 (threads check for 17).

**Cost.** None unless `CHEMISTRYNETWORK` selects one of them; then only that network's code runs.

**Evidence after** (older base; to be re-measured on `<TIP>`).
- `test_sgchem_onezone[3,5,15,16,17]` PASS (broad).
- All four new networks are bitwise identical to the reference implementation.
- Net 17: 4 threads bit-identical to serial.

**Flag off.** No GIZMO source changed. Flag off: to be re-run on `<TIP>`.

**Limits.** One network per build (`CHEMISTRYNETWORK`). Net 3 alone is over 300 lines (split proposed above).

**Depends on / followed by.** PR 1 / PR 3.

---

## PR 3: SGChem networks 18 and 20; network 18 tables as installed data (`chem/03-sgchem-networks-18-20`)

**What and why.**
- Network 18 follows metal ions out of equilibrium: 113 species, opt-in.
- Network 20 is the TIGRESS-NCR network of Kim+23.

The network 18 rate tables (RTZ) were a generated 12.6k-line C header. They now come from an installed data file, `sgchem_net18_tables.bin` (602,568 bytes: the 44 arrays in declaration order, float64 little-endian). The file is hash-checked with the other tables, so the tables stay out of the repository. Size: see "Import size" above (about 1,970 lines).

**Evidence before.** With PR 2 networks 18/20 do not link. In the reference implementation the net 18 tables are compiled in (1.0 MB header).

**Change.**
- `cxx/chem_cool_net18.cc`, `chem_cool_net20.cc`, `chem_rhs_net20.cc`, `chem_net20_cie.h` as on the reference implementation.
- `chem_net18_tables.h` now declares the arrays and their read order (96 lines instead of 12,655).
- `n18_tables_load()` reads the file and rejects a wrong size; it is called after the table build for network 18.
- `stella_tables_check.h` gets one more entry (`sgchem_net18_tables.bin`, sha256 e3c41f2c...).
- `get_stella_tables.sh` installs and checks it when the source provides it. It is optional: stella-tables @a4830ab does not ship it yet.
- `cooling/sgchem/NOTICE` (see "Attribution and licence").
- Test first: the parameter list gains 18 and 20; the net 18 table hash is checked before the run.

**Cost.** None unless `CHEMISTRYNETWORK=18/20`. Network 18 reads 0.6 MB at start-up.

**Evidence after** (older base; to be re-measured on `<TIP>`).
- `test_sgchem_onezone[18,20]` against the frozen C++ reference: PASS (broad).
- Network 20 is bitwise identical to the reference implementation.
- Network 18 is bitwise identical to the reference implementation when that reads its tables the same way (read-only copy with the same loader: tables, rates and one-zone all bitwise).
- Against the unmodified reference, net 18 one-zone runs differ by up to 1.3e-3 in T, and trace species near the floors differ more. Cause: with compiled-in constant tables GCC folds `pow`/`log` of table entries at compile time, exactly rounded. Read at run time, they go through libm. The last-bit differences are amplified by the stiff solve. Tables and rates stay bitwise.
- The same loader was also put on the reference implementation (branch `chem/net18-tables-installed`, not merged: same loader, a hash entry, optional install, start-up check in `begrun.cc`, re-frozen reference hashes). Its net 18 dump is bitwise identical to this tree's (tables, rates, one-zone). The change against the unmodified reference is at most 1.31e-3 in T; the new run passes the old cxx_net18 reference (bar 1e-2).
- The cxx_net18 reference is re-frozen (`data.sha256` updated). This tree's net 18 test reproduces it exactly (18/18 zones, max T relative difference 0).

**Flag off.** No GIZMO source changed. Flag off: to be re-run on `<TIP>`.

**Limits (outside GIZMO).** stella-tables should build and ship `sgchem_net18_tables.bin`. A one-off converter from the old header exists; the real source is the reference implementation's `make_net18_tables.py` plus RTZ. After that the install script's optional entry can become required, with a new pinned commit.

**Depends on / followed by.** PR 2 / PR 4.

---

## PR 4: SGChem build flag, SUNDIALS link, parameters and start-up table check (`chem/04-sgchem-build-and-startup`)

**What and why.** This makes `SGCHEM` a build flag of the cooling module, as CHIMES is. It links SUNDIALS, reads the SGChem parameters and builds the chemistry tables in `InitCool()` after checking the installed table files against the compiled-in hashes. Cells still cool with the default module until PR 6.

**Evidence before.** A GIZMO build cannot select SGChem, and nothing checks that the installed chemistry tables are the version the sources expect.

**Change** (+146 -3 source lines).
- `Makefile`: objects under `SGCHEM`; `SUNDIALS_DIR` from the system file or environment (stops if unset); `-lsundials_core` for 7.x, `-lsundials_generic` for 6.x.
- `precompiler_logic.h`, under `SGCHEM`:
  - `#error` with CHIMES (one SUNDIALS per build), without COOLING or METALS, with COOL_GRACKLE;
  - `#error` with RADTRANSFER or COSMIC_RAY_FLUID (their coupling comes later).
- New flags (default off): `SGCHEM`, `CHEMISTRYNETWORK=N`.
- New parameters (reference-implementation name in brackets):
  - `SGChemTablesPath`;
  - `SGChemCosmicRayIonRate` (`CosmicRayIonRate`);
  - `SGChemDustToGasRatio` (`DGRnormalized`);
  - `SGChemDeuteriumAbundance` (`DeutAbund`).
- `cooling/sgchem/sgchem.cc`:
  - start-up: rank 0 hashes the four tables (five for network 18), one broadcast, then a clean stop naming the install script;
  - network table build;
  - backend fatal errors abort the job.
- `cooling.cc`: one call in `InitCool()`. Plus `proto.h`, `global_data_all_struct.h`, `begrun.cc`, and `Template_Config.sh` lines.

**Cost.**
- Per particle/cell: 0 B.
- Collectives: one `MPI_Bcast` at start-up, only with `SGCHEM`.
- Start-up: ~1 s of hashing on rank 0.
- Quiet steps: no work.

**Evidence after** (older base; to be re-measured on `<TIP>`). `test/sgchem_box::test_sgchem_startup` (8^3 periodic box, 2 ranks; narrow):
- with the installed tables the run reaches TimeMax ("SGCHEM: chemistry tables built");
- with `wss_z_collis.txt` changed by one byte the run stops before the first step: "wss_z_collis.txt has hash ..., expected ...; install with scripts/get_stella_tables.sh". PASS.

**Flag off.** See "Flag-off result (PRs 1-4)" below.

**Limits.** SUNDIALS 6.7 tested; 7.x only compiled (`SUN_COMM_NULL`). The fluid and RT couplings are refused by `#error` until their PRs.

**Depends on / followed by.** PR 3 / PR 5.

---

## Flag-off result (PRs 1-4 together)

(Older base.) objcmp: base 701a78c5 vs the chem/04 tip, same-length directories, flags off.
- `gmc_cooling` config: 140 objects, 137 byte-identical. `cooling/cooling.o` (+4) and `core/begrun.o` (+18) have identical instructions, differing only in `__LINE__` immediates. `core/main.o` differs in data/strings only.
- `fire` config: the same result, 137/140.

PRs 1-3 change no GIZMO object. Flag off: to be re-run on `<TIP>`.

---

Common to PRs 5-11: `test/sgchem_box` is an 8^3 periodic gas box at rest (n_H = 100 cm^-3, T0 = 8000 K, solar metals from `InitMetallicity` in GIZMO's element pattern; fresh starts do not read IC metals), network 5, 2 ranks, 125 kyr in 32 equal steps. At rest every cell is a one-zone problem. `cell_standalone.cc` drives the C++ backend directly (built in the box build's `src/`) with the cells' density and energy and the run's timesteps. It assembles the inputs from the reference implementation's recipe (`chem_interface.cc`), not from the GIZMO glue. The backend is bitwise identical to the reference (PRs 1-3), so the standalone run gives the reference's numbers for the same cell.

---

## PR 5: SGChem cell state and start values (`chem/05-sgchem-cell-state`)

**What and why.** The chemistry keeps per cell its tracked species (number per H), the dust temperature and its gas temperature. This PR adds them to the gas cell under `SGCHEM` and sets the fresh-start values of the reference implementation: no H2 or CO, x(H+) = 1e-4, dust at 10 K, gas at 100 K.

**Evidence before.** With PR 4 a `SGCHEM` build has nowhere to keep the chemical state between steps.

**Change** (+39).
- `cell_data.h`: `TracAbund[SGCHEM_NUM_TRACKED_SPECIES]`, `DustTemp`, `Temp` under `SGCHEM`.
- `precompiler_logic.h`: `SGCHEM_NUM_TRACKED_SPECIES` per `CHEMISTRYNETWORK` (3, 7, 9, 19 or 113), `#error` for any other network. `sgchem.cc` checks it equals the backend's `TRAC_NUM` (`static_assert`). The species index macros of `sgchem_consts.h` (`IO`, `IC`, ...) stay private to the chemistry files.
- `init.cc`: one call `sgchem_init_cell_state(i)` in the fresh-start block; snapshot restarts keep the read values (PR 7).

**Cost.** Per gas cell 8 x (N_species + 2) B: 40 B for network 5 (368 -> 408 B), 120 B for 17, 920 B for 18. No collectives, no work on quiet steps.

**Evidence after** (older base; to be re-measured on `<TIP>`). `test_sgchem_cell_state` (narrow): the same box built without `SGCHEM` prints a 368 B hydro cell, with network 5 408 B: +40 B. PASS. The start values are checked in snapshot 0 from PR 7 on (no output of them before).

**Flag off.** See "Flag-off result (PRs 5-11)".

**Limits.** Memory grows with the network: 920 B per gas cell for network 18.

**Depends on / followed by.** PR 4 / PR 6.

---

## PR 6: SGChem in the cooling dispatch (`chem/06-sgchem-cooling-hook`)

**What and why.** Gas cells now cool through SGChem. One call in `DoCooling`, before the CHIMES branch, as CHIMES does on the host path (`COOLING_SOLVE_IN_HOST_LIBRARY`, with `SGCHEM` added to the switch in `cooling.cc`). The call builds the cell's chemistry inputs, runs the CVODE solve over the step and writes the state back. It also sets the cached `Ne`, `HI`, `Gamma` and `MeanMolecularWeight`, so `Temperature` equals the chemistry temperature.

**Evidence before.** With PR 5, `SGCHEM` builds still cool with the default module; `TracAbund` and `Temp` never change.

**Change** (+161 -3).
- New `cooling/sgchem/sgchem_cell.cc`:
  - inputs as the reference implementation's `chem_interface.cc` for the production configuration without tree radiation:
    - element pools from the FIRE-3 slots;
    - gas-phase C/O/Si after Milky-Way depletion x `SGChemDustToGasRatio`;
    - N, and the WSS CIE element masses (Zn = 0, not in the layout);
    - solver inputs capped at 10 Zsun;
    - DGR = `SGChemDustToGasRatio` x Z/0.014, capped at 10 and 0 above 1e5 K;
    - constant zeta = `SGChemCosmicRayIonRate`; no radiation field (PR 9);
  - the solve as the reference `do_sgchem_step` (pool clamps, net 17 ionised-cell clamp, `MinGasTemp` floor);
  - the `DoCooling` branch: the hydro heating is split half before and half after the solve, as in the grackle branch; a non-finite energy keeps the old one;
  - the cache: `Ne` = the chemistry's electrons per H, `HI` = 1 - x(H+) (neutral H nuclei, atomic and molecular), `Gamma` = `gamma_eos_value()`, `MeanMolecularWeight` = T_chem / ((Gamma-1) U_TO_TEMP u), `Temperature` = T_chem, `u_anchor` = u.
- `cooling.cc`: the call before the CHIMES branch; stops if handed arrays other than `P`/`CellP`.
- `precompiler_logic.h`: `SGCHEM` needs `COOL_METAL_LINES_BY_SPECIES` (the element slots), `#error` otherwise.
- `sgchem.cc`: stops for cosmological runs (UV background coupling not in this tree yet).
- `sgchem_config.h`: `SGCHEM_SOLAR_Z` 0.014 (the reference implementation's metallicity unit; `SolarAbundances[0]` is 0.02 or 0.0142) and the depletion constants.
- `Makefile` (one object), `proto.h`, `Template_Config.sh` (metals note).

**Cost.** Per active gas cell: one CVODE solve on the host, as before in the host loop. Thread-local solver scratch, no new memory, no collectives, nothing on cells that are not active.

**Evidence after** (older base; to be re-measured on `<TIP>`). `test_sgchem_cooling` (narrow): the cells cool from 8000 K to 32 K (x(H2) = 6.5e-3).
- Final u, T, Ne and HI equal the standalone backend run to 1e-8 (measured max 1.0e-10 over 512 cells). Not bitwise: the cell densities move at 1e-10 between steps, and the standalone run uses snapshot 0's.
- Snapshots are bitwise identical with 1 and 4 OpenMP threads per rank.
- Found on the way: in the run loop the first sync point has a zero timestep, so the cells cool on 31 of the 32 steps. The test replays exactly that.

**Flag off.** See "Flag-off result (PRs 5-11)".

**Limits / open questions.**
1. `HI` is read as neutral H nuclei per H; here it is set to 1 - x(H+), counting H2 as neutral. Is that the meaning its readers (rt_chem, eos_functions) expect?
2. `HYDROGEN_MASSFRAC` stays 0.76; the chemistry uses its own X_H = 1/(1 + 4*0.1).
3. The arena on GPU builds after the host branch.

**Depends on / followed by.** PR 5 / PR 7.

---

## PR 7: SGChem snapshot fields (`chem/07-sgchem-snapshot-io`)

**What and why.** Snapshots carry the chemical state, so runs can be analysed and restarted from snapshots: `SGChemTracAbund` (N_species per gas cell) and `SGChemDustTemp`. The temperature is `Temperature`, which equals the chemistry's after PR 6, so there is no separate field. Old snapshots name them `ChemcoolTracAbund` and `ChemcoolDustTemp`; these are read through the legacy-name hook.

**Evidence before.** With PR 6 the chemical state is lost at every snapshot, and a snapshot restart starts the chemistry from scratch. Snapshot Temperature was not the chemistry's: molecular gas at the energy floor shows 71.4 K, against 60 K before the change.

**Change** (+196). `allvars.h` (two block ids, last in the list and under `SGCHEM`, so that other builds keep every block number), `io.cc` (write buffer, bytes, values, gas-only, present, label, name, legacy name), `read_ic.cc` (read buffer). Every new case label is under `SGCHEM`: with the ids unconditional, the switch tables in `io.o`/`read_ic.o` changed in flag-off builds (caught by objcmp, fixed). Read on snapshot restarts (flag 2). Fresh starts from ICs take the start values, as the reference implementation does by default. The opt-in `SGCHEM_KEEP_IC_ABUNDANCES` comes in PR 12.

**Cost.** Snapshot size: 8 x (N_species + 1) B per gas cell in double precision. No run-time cost.

**Evidence after** (older base; to be re-measured on `<TIP>`). `test_sgchem_snapshot_io` (narrow), PASS:
- snapshot 0 holds exactly the start values;
- the final x(H2), x(H+), x(CO) and T_dust equal the standalone run to at most 7.9e-10;
- a flag-2 restart from snapshot 1, with the two datasets renamed to the old names, logs "reading legacy-named 'ChemcoolTracAbund'" and continues as the standalone run from the snapshot state (at most 4.0e-10).

**Flag off.** See "Flag-off result (PRs 5-11)".

**Limits.** Fresh starts ignore IC abundances unless `SGCHEM_KEEP_IC_ABUNDANCES` (PR 12).

**Depends on / followed by.** PR 6 / PR 8.

---

## PR 8: SGChem across restart files (`chem/08-sgchem-restart`), test only

**What and why.** The restart files dump `CellP` and `All` whole, and SGChem has no other persistent state: the solver scratch is set per call and the tables are rebuilt at start-up. So no restart code is needed. This PR adds the test that shows it. Note: the restart layout is now a table (`RESTARTFILE_LAYOUT_VERSION` 4); this PR text and test are to be re-checked against it on `<TIP>`.

**Evidence before.** No test covered SGChem across a binary restart.

**Change.** None in `src/`. New test only.

**Cost.** None.

**Evidence after** (older base; to be re-measured on `<TIP>`). `test_sgchem_restart` (narrow): a run stopped at t = 0.064 (restart files are written at `TimeMax`) and continued to 0.128 with flag 1 ends bitwise identical to the uninterrupted run: u, T, Ne, HI, species and dust T. From PR 10 on it also checks the ledger totals.

**Flag off.** No source change.

**Limits.** The restart layout of the base is now a table (`RESTARTFILE_LAYOUT_VERSION` 4); the test is to be re-run against it on `<TIP>`.

**Depends on / followed by.** PR 7 / PR 9.

---

## PR 9: uniform radiation field from parameters (`chem/09-sgchem-uniform-field`)

**What and why.** Until the TreeCol extension delivers per-pixel fields, the chemistry's only radiation input is a uniform, unattenuated ISRF-shaped field from parameters. It goes through the shared `chem_fuv_field_from_band_energy`, with the SGChem band constants (Draine 1978 / 1.7: 5.257e-14 erg cm^-3 per G0 in 6-13.6 eV, 7.799e-15 in the LW band). Grain heating is band-resolved (6-13.6, 3.4-6, 0.4-3.4 eV). The background floors are 0.324e-2 x `SGChemBackgroundG0` (`G0` in the reference implementation), LW a tenth of that, and they scale only the floors.

**Evidence before.** With PR 8 the chemistry runs without any radiation field (G0 = 0), and the reference implementation's production inputs cannot be reproduced.

**Change** (+30 -2).
- New parameters `SGChemG0` (6-13.6 eV, Habing units), `SGChemG0_LW` (11.2-13.6 eV) and `SGChemBackgroundG0`.
- `sgchem_network.h`: `sgchem_uniform_field()`.
- `sgchem_cell.cc`: G0, G0_LW and G0_dust from it.
- One token in the imported backend (`chem_gizmo_backend.cc`): it copies `G0_dust` also under a new chemistry-private `SGCHEM_CELL_GRAIN_HEATING_FIELD`. Before, only the per-pixel flags passed it, so a uniform field had no grain heating. The one-zone builds already passed it, so their output is unchanged.

**Cost.** A few flops per cell call; no memory, no collectives.

**Evidence after** (older base; to be re-measured on `<TIP>`). `test_sgchem_habing_field[G0 = 1, 10, 100]` (narrow; SGChemG0 = SGChemG0_LW = G0, SGChemBackgroundG0 = 1.7):
- Inputs (G0, G0_LW, G0_dust): equal to the reference implementation's production path for an isotropic Habing field to 1 ulp (relative at most 4.4e-16). That path (`GALSF_RESOLVEDISM_G0_VARIABLE`, 12 pixels, per-pixel UV and LW floors, 3.4-6 eV and optical grain heating) is transcribed in `cell_standalone.cc`.
- One-zone results on the two input sets agree to 1e-12 (bitwise at G0 = 100).
- Box cells vs the standalone run: 1.8e-7, 6e-10, 1.4e-10 for G0 = 1, 10, 100. At G0 = 1 the warm lattice moves and its densities scatter by 1.2e-7; the bar is 1e-6.
- End states: T = 90 / 189 / 438 K, x(H2) = 7.5e-5 / 8.0e-6 / 6.3e-7, T_dust = 14 / 20 / 30 K.

**Flag off.** See "Flag-off result (PRs 5-11)".

**Limits.** At G0 = 0 the backend skips every photo-rate, the ionising ones too (SGChem Fortran behaviour, also in the reference implementation). The floors keep G0 > 0 in production; any future ionising input must keep it so.

**Depends on / followed by.** PR 8 / PR 10.

---

## PR 10: chemistry ledger (`chem/10-sgchem-ledger`)

**What and why.** A budget ledger, starting with its chemistry rows: the energy the chemistry solve adds (heating) and removes (cooling), without the hydro half-steps. Each module adds to per-thread sums during its own work. The sums are exact (integer limbs over the whole double range), so the result does not depend on the order of the additions: not on thread count, schedule or the dynamic loop. They are folded and reduced only when a snapshot is written, with one `MPI_Reduce`, and rank 0 appends a row to `ledger.txt`. The reference implementation sums with OpenMP atomics, whose sum depends on the order. The output cadence is the snapshot cadence, because `TimeBetStatistics` is fixed at 1e10 without `DEVELOPER_MODE`.

**Evidence before.** No energy budget for the chemistry. The reference implementation's version sums with atomics, which is order dependent, and runs collectives on every step. Negative ledger totals printed NaN.

**Change** (+142 -3).
- New flag `OUTPUT_MASS_METAL_ENERGY_LEDGER` (default off).
- New `system/ledger.cc`: exact sums, `ledger_init`/`ledger_add`/`ledger_output`/`ledger_restart_io`.
- `proto.h`: row enum.
- `io.cc`: one call at the start of `savepositions`.
- `restart.cc`: the rank totals with a rows x limbs layout check (to be re-checked against the restart layout table, see PR 8).
- `sgchem_cell.cc`: one `ledger_add` per solve.
- `sgchem.cc`: `ledger_init`.
- `Makefile`, `Template_Config.sh`.

**Cost.** Flag on: 2 rows x 72 x 8 B per thread and per rank. One `MPI_Reduce` of 144 int64 per snapshot. Per cell solve: one exact addition (~10 integer operations). Nothing between snapshots; nothing at all with the flag off.

**Evidence after** (older base; to be re-measured on `<TIP>`).
- `test_sgchem_ledger` (narrow; G0 = 10): one row per snapshot. The rows close the box's thermal-energy change to 7e-16. The file is byte-identical with 1 and 4 threads.
- `test_sgchem_restart`: the final ledger row is bitwise the same after a restart.
- The only collective is inside `savepositions`, which is collective already. This holds by construction; the test does not count MPI calls.

**Flag off.** See "Flag-off result (PRs 5-11)".

**Limits / open questions.** (1) `ledger.txt` or rows in the existing energy file? (2) Is snapshot cadence right, or should it be a ledger parameter?

**Depends on / followed by.** PR 9 / PR 11.

---

## PR 11: network 5, ionising photons destroy and heat H2 (`chem/11-sgchem-net5-euv-h2`)

**What and why.** Arepo SGChem's network 5 destroys H2 with EUV photons (H2 -> H2+ [+ e] -> H + H) and heats with them. The imported network did not, so fully molecular gas next to an ionising source never became ionised. This is the reference implementation's change 02c2efbc, as its own PR after the import: the H2 rate is r x Gamma_HI per H2, with r = sigma_H2/sigma_HI (1.34 at <h nu> = 18.6 eV), and the heating per ionisation is <h nu> - 15.2 eV.

**Evidence before.** `test_sgchem_net5_euv_h2` on PR 10 does not compile, because the backend has no H2 channel. The same physics without it (r = 0): a shielded molecular cell (G0 = 1e-4, Gamma_HI = 1e-9 s^-1) stays molecular after 1 Myr, x(H2) = 0.45, x(H+) = 0.10, T = 360-540 K.

**Change** (+13 -2; plus a 25-line note in the deviations list): `chem_rhs_net5.cc` (rate and heating terms), `chem_calc_photo.cc` (photo-rates 40/44 on the ionising path), `chem_types.h`, `chem_gizmo_backend.cc`, `chem_interface.h` (two input fields), as in the reference change 02c2efbc. A note for this tree in the reference's deviations list.

**Cost.** Two terms in the net 5 right-hand side.

**Evidence after** (older base; to be re-measured on `<TIP>`). `test_sgchem_net5_euv_h2[n_H = 10, 100]` (narrow), from the reference implementation's `golden/test_euv_h2.cc`, plus a shielded case:
- RHS: the H2 channel adds exactly -r Gamma x(H2) and its heating (max relative difference 2.2e-16).
- With the channel, the molecular cell loses its H2 and ionises: x(H+) = 0.997 / 0.974, T = 2.6 / 2.3e4 K, at G0 = 1.7 and at 1e-4.
- Without the channel, the shielded cell stays molecular, as before.
- The references are not re-frozen. In GIZMO runs no ionising rate reaches the chemistry yet, and with zero rates the terms add exact zeros. The net 5 dump (tables, rates, 18 one-zone runs) is bitwise identical to before. All 7 networks pass `test_sgchem_onezone` against the unchanged frozen references.

**Flag off.** See "Flag-off result (PRs 5-11)".

**Limits.** The reference implementation's photon-budget changes for TreeRay-PI and M1 (absorbers r x x_H2) come with those couplings.

**Depends on / followed by.** PR 10 / the radiation coupling (TreeCol extension, TreeRay-PI, M1), which will pass Gamma_H2 = r Gamma_HI.

---

## PR 12: keep the abundances of the IC on a fresh start (`chem/12-sgchem-keep-ic-abundances`)

**What and why.** A fresh start sets the chemistry state of every gas cell to a default (H2 and CO zero, x(H+) = 1e-4) and ignores any `SGChemTracAbund` in the IC file. A molecular cloud IC, or a run restarted from another code's snapshot, cannot start from its own abundances. The new compile flag `SGCHEM_KEEP_IC_ABUNDANCES` reads them from the IC and leaves them alone. It is the flag of the same name in the reference implementation (there `CHEMCOOL_KEEP_IC_ABUNDANCES` before the rename).

**Evidence before.** `test_keep_ic_abundances` (`test/sgchem_keep_ic`) on a build with the flag, before the change: the IC carries x(H2) = 0.05-0.35, x(H+) = 0.02, x(CO) = 1-1.5e-5 per cell; snapshot 0 holds (0, 1e-4, 0) in every cell and the test fails at "snapshot 0 holds the IC abundances".

**Change** (+5 source lines): `cooling/sgchem/sgchem.cc` (the fresh-start default is skipped under the flag), `file_io/read_ic.cc` (the block is read from the IC under the flag; it is the only SGChem block read). Under the flag the dust temperature (10 K) and the gas temperature are still set at the start. New flag `SGCHEM_KEEP_IC_ABUNDANCES`, default off; no parameters. `Template_Config.sh` line; compile-suite row (commented out like the other rows that need an external library, SUNDIALS); `test/sgchem_keep_ic/README.md`.

**Cost.** None: no new fields, no collectives, nothing on quiet steps. One read of one IC dataset at start-up, only under the flag.

**Evidence after** (older base; to be re-measured on `<TIP>`). `test/sgchem_keep_ic` (narrow), the `sgchem_box` setup shortened to 8 kyr, run without and with the flag:
- Without the flag, snapshot 0 is (0, 1e-4, 0) in every cell although the IC holds other values.
- With the flag, snapshot 0 equals the IC exactly, the dust temperature is the 10 K start value, and after 4 kyr x(H2) and x(CO) have changed by 3.8e-4 and 1.3e-2 (x(H+) recombines, 0.68 relative).
- 2 passed. Same behaviour as the reference implementation: the abundances are read, nothing else is changed.

**Flag off.** (Older base.) objcmp: chem/11 tip vs chem/12 tip, same-length directories, `gmc_cooling` and `fire` configs, flag off. 140 objects in both, 137 byte-identical, 0 with different instructions. The others: `main.o`, `cooling.o` (data/strings only) and `read_ic.o` (`__LINE__` immediate, 3 lines shifted). The FIRE build is included although no feedback or SF code is touched. Flag off: to be re-run on `<TIP>`.

**Limits.** Only the tracked species are read from the IC; dust and gas temperature still take start values.

**Depends on / followed by.** PR 11 (any chemistry PR would do; it touches only the start-up) / none. Lands before the tc series.

---

## Flag-off result (PRs 5-11)

(Older base.) objcmp: base 701a78c5 vs each tip chem/05 ... chem/11, same-length directories, `gmc_cooling` and `fire` configs, flags off.
- 140 objects at every tip, 0 with different instructions, in both configs. Byte-identical: 136 (PRs 5, 6), 134 (PRs 7-9), 133 (PRs 10, 11).
- The others differ only in `__LINE__` immediates (`cooling.o`, `begrun.o`, `init.o`, `io.o`, `read_ic.o`, `restart.o`) or in data/strings (`main.o`).
- A first run showed PR 7 changing the instructions of `io.o` and `read_ic.o`: the new IO block ids shifted the switch tables. They are now last in the list and under `SGCHEM`.

Flag off: to be re-run on `<TIP>`.

Tests of every tip (`test/sgchem_box` + `test/sgchem_onezone`, 4 threads, 4 cores; older base, to be re-measured on `<TIP>`): chem/05 9 passed ... chem/11 18 passed, no failure. Full suite: <SUITE RESULT>.

Not done here: a cloud run with production chemistry vs the reference implementation (T, H2, CO distributions). It needs the per-pixel columns (TreeCol extension), without which the chemistry here is optically thin. The box comparisons above stand in for it cell by cell.
