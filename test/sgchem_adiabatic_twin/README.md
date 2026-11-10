# sgchem_adiabatic_twin

`TEST_SKIP_COOLING_AND_CHEMISTRY_UPDATE` (test only; needs `COOLING_OPERATOR_SPLIT`, no `TRANSPORT_SUBCYCLE_COOLING`): the cooling and
chemistry update is not called, so the gas energy follows the hydro work alone. The adiabatic twin of a cooling run.

Problem: the box of `sgchem_box` (8^3 cells, n_H = 100 cm^-3, 8000 K, at rest, SGChem network 5), 8 steps, 2 ranks.
Check: every cell keeps its temperature (1e-6) and tracked species (bitwise); `sgchem_box` cools the same box to 32 K.

`test_kick_floor_row` (`OUTPUT_MASS_METAL_ENERGY_LEDGER`, `MAGNETIC` with a field of 1e-12 so that the MHD row is built): the same box with a velocity wave
(5 km/s, wavelength 10 pc) and the floor at the start energy (`MinGasTemp` 9000 K): the expanding half cannot cool. The increase of `E_kick_floor_erg`
is above 1e-2 of the first-order estimate M u_0 (gamma - 1) A k t / pi; `E_mhd_numerical_erg` is below 1e-6 of the energy gained (the hydro conserves
energy); with `MinGasTemp` 2.73 K the floor row is exactly 0. The floor row counts the clamps of the kicks only: it is 0.42 of the energy the gas gains,
the rest comes from the clamps of the predicted energy outside the kicks .

Inputs: `SUNDIALS_DIR`; chemistry tables from `SGCHEM_TABLES_DIR` (else `scripts/get_stella_tables.sh`).
