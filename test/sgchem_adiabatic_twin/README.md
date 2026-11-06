# sgchem_adiabatic_twin

`TEST_SKIP_COOLING_AND_CHEMISTRY_UPDATE` (test only; needs `COOLING_OPERATOR_SPLIT`, no `TRANSPORT_SUBCYCLE_COOLING`): the cooling and
chemistry update is not called, so the gas energy follows the hydro work alone. The adiabatic twin of a cooling run, as stella_dev's
`RESOLVEDISM_ADIABATIC_NOCOOL`.

Problem: the box of `sgchem_box` (8^3 cells, n_H = 100 cm^-3, 8000 K, at rest, SGChem network 5), 8 steps, 2 ranks.
Check: every cell keeps its temperature (1e-6) and tracked species (bitwise); `sgchem_box` cools the same box to 32 K.

Inputs: `SUNDIALS_DIR`; chemistry tables from `SGCHEM_TABLES_DIR` (else `scripts/get_stella_tables.sh`).
