# sgchem_box

Uniform periodic gas box (8^3 cells, n_H = 100 cm^-3, T0 = 8000 K, at rest; IC from `make_sgchem_box_ics.py`; solar
metals from `InitMetallicity`, GIZMO's element pattern) built with `SGCHEM` (network 5), 2 ranks, 125 kyr in 32 equal steps.

- `test_sgchem_startup`: the four chemistry tables are hash-checked at start-up against `src/galaxy_sf/stella_tables_check.h`.
  With the installed tables the run reaches `TimeMax` and builds the network tables; with one table changed by a byte
  the run stops before any step, naming the table, its hash and `scripts/get_stella_tables.sh`.
  The start-up log states the hydrogen mass fraction (0.7065, as ABHE = 0.1) and the energy floor of MinGasTemp, checked to 1e-3.
- `test_sgchem_cell_state`: the same box built without `SGCHEM` (in `output_no_sgchem/`); the gas cell grows by
  40 B with network 5 (three tracked species, dust and gas temperature).
- `test_sgchem_cooling`: the cells cool from 8000 K to 32 K through SGChem. Final energy, temperature, electron and HI
  fractions equal a standalone run of the backend (`cell_standalone.cc`, built in the box build's `src/`) on the same inputs
  and timesteps to 1e-8 (measured 1.0e-10: the cell densities move at 1e-10 between steps). The snapshots are identical
  with 1 and 4 OpenMP threads per rank.
- `test_sgchem_snapshot_io`: snapshot 0 holds the start values (`SGChemTracAbund` = 0, 1e-4, 0; `SGChemDustTemp` = 10 K);
  the final H2, H+, CO and dust temperature equal the standalone run to 1e-8 (measured 8e-10). A snapshot restart
  (flag 2) from snapshot 1 with the datasets renamed to the old `ChemcoolTracAbund`/`ChemcoolDustTemp` reads them
  (log line) and continues as the standalone run from that state (measured 4e-10).

Inputs: `SUNDIALS_DIR`; the chemistry tables from `SGCHEM_TABLES_DIR`, else installed by `scripts/get_stella_tables.sh`.
