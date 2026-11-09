# sgchem_input_pool_violations

`TEST_SGCHEM_PRINT_INPUT_POOL_VIOLATIONS` (test only, SGChem networks 5, 17 and 20): before the solve, every cell whose species
hold more of an element than its gas-phase pool (CO above min(C, O); 2 x(H2) + x(H+) above 1, relative tolerance 1e-3; for net 17
also He+ + He++ above the helium pool and D+ + HD above the deuterium pool) prints `SGCHEM_INPUT_VIOLATION` lines with the cell,
the bits of the violation (1 CO, 2 hydrogen, 4 helium, 8 deuterium), the pools and the species. The clamp that follows is not
changed.

Problem: the uniform box of `test/sgchem_box` with x(H2) and x(CO) set by ID % 4 (0.05 and 1e-5; 0.15 and 5e-3; 0.7; 0.9),
kept by `SGCHEM_KEEP_IC_ABUNDANCES`, 2 ranks, 8 kyr.

Check: the cells reported are those of ID % 4 = 1, 2, 3, each with the bit of its violation; the consistent cells are not.
Inputs as for `test/sgchem_box`: `SUNDIALS_DIR`, the chemistry tables.
