# sgchem_keep_ic

The uniform periodic box of `test/sgchem_box` (8^3 cells, n_H = 100 cm^-3, T0 = 8000 K, network 5, 2 ranks, 8 kyr in 4
steps) with an IC that also carries `SGChemTracAbund` (x(H2) = 0.05-0.35, x(H+) = 0.02, x(CO) = 1-1.5e-5, different in
every cell). Run twice: without and with `SGCHEM_KEEP_IC_ABUNDANCES`.

- `test_default_resets_abundances`: without the flag snapshot 0 holds the default start (0, 1e-4, 0) whatever the IC holds.
- `test_keep_ic_abundances`: with the flag snapshot 0 holds the IC values exactly (the dust temperature is still the 10 K
  start value) and after 4 kyr x(H2) and x(CO) have changed by less than 5 per cent (x(H+) recombines).

Inputs as for `test/sgchem_box`: `SUNDIALS_DIR`, the chemistry tables from `SGCHEM_TABLES_DIR` or `scripts/get_stella_tables.sh`.
