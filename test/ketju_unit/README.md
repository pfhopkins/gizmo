# KETJU integrator unit tests

Runs the original googletest suite of the KETJU integrator (MSTAR; `ketju-integrator/unit-tests/`,
GPLv3, Mannerkoski et al.) against GIZMO's vendored, modified `ketju-integrator/` sources.
No GIZMO build or simulation is involved.

## What is tested

26 cases, each on 1, 2 and 4 MPI tasks:
- `Integration_*`: binary integration (Newtonian, PN, spin, three-body PN terms, unit systems)
  against stored results of the old integrator.
- `Integration_with_mergers`, `Instant_mergers`: BH mergers (head-on, from an orbit, multiple
  mergers with per-particle extra data).
- `MST/ConstructionTest`: Prim and divide-and-conquer MST construction, 2 to 1000 random particles.

The GIZMO additions (external tide, contact collisions of non-PN particles, pericentre guard) are
off by default and not exercised by these cases; the suite guards the original behaviour.

## Running

    module load gcc/13; module load openmpi; module load cmake/3.30
    pytest test/ketju_unit/test_ketju_unit.py

The test copies `ketju-integrator/` to a pytest temporary directory (its CMake writes `bin/` and
`lib/` into the source tree), builds with `-DUNIT_TESTS=ON -DBUILD_GMOCK=OFF` and the vendored
googletest subset in `ketju-integrator/thirdparty/googletest/`, and runs with
`mpiexec --oversubscribe`. Build ~1 min with 4 make jobs; the runs take seconds.
