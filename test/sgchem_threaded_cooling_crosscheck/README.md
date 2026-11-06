# sgchem_threaded_cooling_crosscheck

`TEST_COOLING_COMPARE_THREADED_PASS_WITH_SERIAL_PASS` (test only, host cooling path, i.e. SGChem): every cooling step runs the
active cells once serially on copies, restores the inputs, runs them threaded as usual and compares particle and cell structs
byte for byte. A difference stops the run (`endrun(8859)`) with the first cell's ID and energies; otherwise rank 0 prints
`COOLING_CROSSCHECK: N cells, threaded pass equals the serial pass`. Global counters that the solve updates (the energy ledger)
see the cells twice with the flag.

Problem: the uniform box of `test/sgchem_box` with molecular ICs (as `test/sgchem_thermal_state`), 2 ranks x 2 threads, 8 kyr.

Check: the run reaches TimeMax and reports the comparison in every step (at least 4 steps, over 100 cells in total).
Inputs as for `test/sgchem_box`: `SUNDIALS_DIR`, the chemistry tables.
