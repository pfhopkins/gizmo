# sgchem_step_diagnostics

`SGCHEM_PRINT_STEP_COOLING_DIAGNOSTICS`: on each step with a status line, rank 0 prints `SGCHEM_DIAG` with its own active cells: count, energy the
cooling update removed [code] and its running sum, temperature and hydrogen-density ranges (stella_dev `cooling.cc`). No communication.

Problem: the box of `sgchem_box` (8^3 cells, n_H = 100, 8000 K), 1 rank, 8 steps of 0.004.
Checks: 9 lines (the call at the start and 8 steps; the last follows the final snapshot), N_active = 512; the first 8 lines add up to the thermal
energy the snapshots lost (1e-5) and the running sum ends at the sum of all; step-line energies positive, the largest temperature at most
8080 K and not rising, the smallest above 2.73 K, density within 90..110 cm^-3.

Inputs: `SUNDIALS_DIR`; chemistry tables from `SGCHEM_TABLES_DIR` (else `scripts/get_stella_tables.sh`).
