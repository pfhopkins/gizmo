# sgchem_box

Uniform periodic gas box (8^3 cells, n_H = 100 cm^-3, T0 = 8000 K, FIRE-3 solar metals, at rest; IC from
`make_sgchem_box_ics.py`) built with `SGCHEM` (network 5), 2 ranks.

- `test_sgchem_startup`: the four chemistry tables are hash-checked at start-up against `src/galaxy_sf/stella_tables_check.h`.
  With the installed tables the run reaches `TimeMax` and builds the network tables; with one table changed by a byte
  the run stops before any step, naming the table, its hash and `scripts/get_stella_tables.sh`.
  The start-up log states the hydrogen mass fraction (0.7065, as ABHE = 0.1) and the energy floor of MinGasTemp, checked to 1e-3.
- `test_sgchem_cell_state`: the same box built without `SGCHEM` (in `output_no_sgchem/`); the gas cell grows by
  40 B with network 5 (three tracked species, dust and gas temperature).

Inputs: `SUNDIALS_DIR`; the chemistry tables from `SGCHEM_TABLES_DIR`, else installed by `scripts/get_stella_tables.sh`.
