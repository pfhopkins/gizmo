# sgchem_solver_failure

C++ SGChem solver failures and the dust temperature, networks 5 and 17, without a GIZMO run (stella_dev D-58 to D-61;
`src/cooling/sgchem/cxx/DIFFERENCES.md`). Ported from stella_dev `cooling/sgchem/cxx/golden/test_solver_failure.cc` and
phase F of `golden/test_conservation.cc`, C++ only: gku has no Fortran backend, so the DVODE comparisons are not here.

`solver_failure.cc`, default build:
- phase A: the WLM cells whose CVODE step hit the floor (`net<N>_wlm_floor_cells.txt`, 66 + 7 cells logged on stella_dev),
  replayed from the logged substep state; the cell inputs must reproduce the run's logged H2 photodissociation rate (1e-4);
- phase H: the TNOCHEM hand-off state (hot-CIE ions) at bypass T 4.5e5 / 4.8e5 / 4.99e5 K, n_H 1e-2 to 10, dt 1e10 to 3e12 s.
- Bars: rc 0, no CVODE floor hit, no substep rejected for a D+ excess, final T within 1% of `net<N>_converged_reference.txt`
  (column T_C: the C++ solve at rtol 1e-8, generated on stella_dev at d0c4370d3). Recoverable H+ rejections only in the
  known calls (Uli 2026-11-19/20): net 5 cells 6796375 (<= 1), 1080890 (<= 2) and hand-off n_H 10, T 4.99e5 K,
  dt 7.7044e11 / 3e12 s (<= 2 each); net 17 hand-off n_H 10, T 4.5e5 / 4.8e5 K, dt 7.7044e11 / 3e12 s (<= 1 each).

`solver_failure.cc`, build with `TEST_SGCHEM_FORCED_SOLVER_FAILURE` (phase R): forced code 1 / 100 per attempt; the retry
rescues and starts from the accepted state bit for bit (R1/R2); TNOCHEM finish after heating across TNOCHEM (R3a/R3b);
abort in a 100 K cell (R4/R5); every dump rebuilds a 0xff- and a 0x00-filled cell and replays bit for bit (R6); flagged
evaluations book no stale CR ionisation heating (R7a-f).

`dust_temperature.cc`: dense shielded sequences (n_H 1e6, G0 0, Z 3, MinGasTemp 10 / 5 K after every solve): rc 0, no
bisection, <= 3e4 dust iterations per solve (F1); one root from 3.8 K and 30 K, <= 1e-6 (F2); Newton over Z 0.1-3,
n_H 1e4-1e6, T 5 K-1e4 K, G0 0 / 1.7 / 100 without bisection, <= 30 iterations from a cold start and <= 20 from the
previous root (Uli 2026-11-23) (F3); Newton slope within 1e-6 of a central difference (F4).

Run: `build_and_run.sh <net> <out>` (needs `mpicxx`, `SUNDIALS_DIR`, `CHEMCXX_DATADIR`), or the pytest
`test_sgchem_solver_failure.py` (tables from `SGCHEM_TABLES_DIR` or `scripts/get_stella_tables.sh`).
