# stars_integrated_by_rumi

Type-4 stars with the RUMI++ library (`STARS_INTEGRATED_BY_RUMI`), no gas, on the binary of `test/stars_hermite_direct_gravity` (1 + 1 Msun, a = 1000 AU, e = 0.9) against the Newtonian reference orbit of `../stars_hermite_direct_gravity/fewbody_reference.py`, started from the IC file (the first snapshot already carries the opening half kick). `rumi_config.txt` holds the RUMI++ integrator keys (collisional, Newtonian, no stellar evolution; `R_NGB` 0.02 pc puts all test stars into one regularised subsystem).

* `test_startup_and_direct_sum`: the library starts (its build id in the log); the stars feel each other only through the exact direct sum (star-star pairs out of the tree), one orbit on leapfrog steps: measured dE/E 1.1e-3, offset 3.8e-3 a; bars 3x.

Runs only when `RUMI_PATH` points to a RUMI++ library install built against the same Kokkos as GIZMO (skipped otherwise), e.g. `RUMI_PATH=/u/uli/sw/rumi/4979aaf1b283-omp-mpi-kokkos5.0.0-gcc13` on Orion.
