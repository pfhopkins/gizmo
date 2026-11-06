# sgchem_restart_temperature

`SGCHEM_TEMPERATURE_FROM_CHEMISTRY_AT_SNAPSHOT_RESTART` (EXPERIMENTAL, SGChem): when a snapshot is restarted (start flag 2) every
gas cell's chemistry temperature is derived from its energy, density and abundances (no solve) before the first force pass. A
snapshot carries no `Temp`; without the flag every cell starts at 100 K until its first solve. With
`TEST_SGCHEM_PRINT_THERMAL_PROPERTIES` the start temperature of every cell is printed (`RESTARTTEMP ID T`).

Problem: the uniform box of `test/sgchem_box` as a snapshot, atomic gas (x(H2) = 0, x(H+) = 0.1) with specific energies of 10 to 80
(km/s)^2 (940 to 7500 K), 2 ranks.

Check: every cell reports the atomic-gas temperature (2/3) u m_p (1 + 4 ABHE) / (k_B (1 + ABHE + x(H+))) to 1e-3, not 100 K.
Inputs as for `test/sgchem_box`: `SUNDIALS_DIR`, the chemistry tables.
