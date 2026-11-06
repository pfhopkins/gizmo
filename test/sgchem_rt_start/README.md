# sgchem_rt_start

With the moment solver and chemistry photoionisation on, the first force pass reads the chemistry's start, not a fully neutral cell. The uniform box of `test/sgchem_box` (8^3 cells, 2 ranks, no stars) with IC abundances x(H+) = 0.02.

- `test_start_is_the_chemistry_state`: in snapshot 0, written before any chemistry solve, HII is the IC x(H+) and the electrons exceed it.
- `test_start_electrons_match_the_first_solve`: Ne - HII (the C and Si pools) is the same in snapshots 0 and 1, within 1e-9.

Why it matters: cosmic-ray transport divides by the ionised fraction (ion Alfven speed), so a start at 1e-56 gives an Alfven speed 2e5 times too high and gas at 1e9 K in the first steps.

Inputs as for `test/sgchem_rt_photons`: `SUNDIALS_DIR`, chemistry and stellar tables.
