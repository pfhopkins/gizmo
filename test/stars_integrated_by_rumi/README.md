# stars_integrated_by_rumi

Type-4 stars with the RUMI++ library (`STARS_INTEGRATED_BY_RUMI`), no gas, on the binary (1 + 1 Msun, a = 1000 AU, e = 0.9) and triple of `test/stars_hermite_direct_gravity` against the Newtonian reference orbit of `../stars_hermite_direct_gravity/fewbody_reference.py`, started from the IC file (the first snapshot already carries the opening half kick). `rumi_config.txt` holds the RUMI++ integrator keys (collisional, Newtonian, no stellar evolution; `R_NGB` 0.02 pc puts all test stars into one regularised subsystem).

* `test_startup`: the library starts (its build id in the log).
* `test_binary`, `test_triple` (the hierarchical triple of the Hermite test, two outer orbits): RUMI++ integrates the stars over their common steps; energy and end positions against the reference (G as GIZMO sets it): binary dE/E 5.9e-13, offset 2.7e-10 a; triple 1.4e-9, 5.3e-7 a_in; bars 3x.
* `test_snapshots_inside_steps`: outputs every T/7.3, so most fall inside a common step: each snapshot carries RUMI++'s state there (energy and positions against the reference: measured 1.2e-10 and 2.6e-10 a, bars 3x).
* `test_restart` (`TEST_RESTART_AT_STEP`): the triple stopped at step 40 and restarted from its restart files (RUMI++'s state among them) ends bit-identical to the uninterrupted run.

Runs only when `RUMI_PATH` points to a RUMI++ library install built against the same Kokkos as GIZMO (skipped otherwise), e.g. `RUMI_PATH=/u/uli/sw/rumi/4979aaf1b283-omp-mpi-kokkos5.0.0-gcc13` on Orion.
