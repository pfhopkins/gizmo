# static_plummer_test_potential

`TEST_GRAVITY_STATIC_PLUMMER` (test only, with `GRAVITY_ANALYTIC`): a fixed Plummer potential phi = -G M / sqrt(r^2 + a^2) at
the origin, M = `TEST_GRAVITY_STATIC_PLUMMER_MSUN` (default 1000 Msun) and a = `TEST_GRAVITY_STATIC_PLUMMER_PC` (default 0.5 pc),
added to every active particle's acceleration (and, with `COMPUTE_TIDAL_TENSOR_IN_GRAVTREE`, to its tidal tensor). A stand-in
for a static gas potential in energy-conservation tests of direct-sum star integrators.

Problem: three massless collisionless particles (self-gravity off) at 0.3, 1 and 2 pc on a bound eccentric, a circular and an
eccentric orbit, 10 time units (pc / km/s, about three circular periods).

Checks: specific energy and angular momentum conserved to 1e-3, the circular orbit keeps its radius to 1%, final positions
within 0.02 pc of a reference integration (scipy, rtol 1e-11).
