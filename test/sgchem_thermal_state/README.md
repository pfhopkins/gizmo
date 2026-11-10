# sgchem_thermal_state

ThermalProperties under SGChem: the uniform box of `test/sgchem_box` with molecular ICs (x(H2) 0.05-0.35, x(H+) 0.02, kept by
`SGCHEM_KEEP_IC_ABUNDANCES`), 2 ranks, 8 kyr. `TEST_SGCHEM_PRINT_THERMAL_PROPERTIES` prints what ThermalProperties returns for
every cell after its cooling step next to the stored chemistry state.

- `test_temperature_is_the_stored_one`: T = the temperature of the last solve.
- `test_species_from_the_stored_abundances`: n(H0) = 1 - 2 x(H2) - x(H+) (atomic H only), n(H+) = n_e = x(H+).
- `test_mean_weight_of_the_chemistry_mix`: mean weight from X = 1/(1 + 4 ABHE), metals not counted.

The bars are the expressions of ThermalProperties and Get_Gas_Mean_Molecular_Weight_mu.
Inputs as for `test/sgchem_box`: `SUNDIALS_DIR`, the chemistry tables from `SGCHEM_TABLES_DIR` or `scripts/get_stella_tables.sh`.
