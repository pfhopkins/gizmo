# Legacy example parameter files

These are standalone `.params` files for a range of classic test problems
(shocktube, sedov, evrard, Orszag-Tang, Kelvin-Helmholtz, MRI, etc.). They are
kept here for reference only.

For running tests, use the modern pytest-based test bench instead: each problem
lives in its own directory under `test/<problem>/` with a matched `Config.sh`,
parameter file, and `test_<problem>.py` that builds, runs, and validates it
against a reference solution. See `test/README.md` and
`test/how_to_run_tests.md`.
