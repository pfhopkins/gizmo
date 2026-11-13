# stars_hermite_direct_gravity

Type-4 stars on the 4th-order Hermite integrator with the exact direct star-star sum (`STARS_HERMITE_DIRECT_GRAVITY`), no gas. Two few-body problems, each against Newtonian point-mass orbits integrated in `fewbody_reference.py` (DOP853, rtol 1e-13; the softening, 6.4 AU, stays inside every pericentre):

* binary: 1 + 1 Msun, a = 1000 AU, e = 0.9, eight orbits, snapshots after orbits 1..7;
* triple: inner 1 + 1 Msun, a = 100 AU, e = 0.5; outer 1 Msun, a = 1000 AU, e = 0.3, inclined 30 degrees; two outer orbits, snapshot after the first.

Convergence test in the Aarseth step parameter `STARS_HERMITE_AARSETH_ETA` = 0.00125, 3.125e-4, 7.8125e-5 (binary) and 0.005, 0.00125, 3.125e-4 (triple): the step goes as sqrt(eta), so each rung halves it and the block steps shift by one bin. `DEVELOPER_MODE` with `ErrTolIntAccuracy` 0.049 (the largest GIZMO accepts): along the reference orbits the acceleration step sqrt(2 Err h/|a|) stays at least 1.2x (binary) and 1.8x (triple) above the Aarseth step on these ladders, so eta alone sets the step. The centre of mass sits near the origin so coordinate roundoff stays far below the finest rung. Snapshots are taken 1e-9 of an interval after a full sync point: there the stored start-of-step state is the corrected one (a snapshot at a step end holds the predicted state, 3rd order in velocity).

Bars (from theory, set before the runs):
* energy error and position offset fall at 4th order: measured order log2(error ratio) per rung within [3.5, 4.5];
* |dE/E| below the truncation bound of the scheme integrated along the reference orbit without cancellation (corrector quadrature error dt^5/720 of the 4th and 3rd derivatives of the acceleration, predictor error through the tidal tensor, Aarseth steps from the exact derivatives; Makino & Aarseth 1992);
* `_symmetric` (`HERMITE_CORRECTOR_ITERATIONS=2`, `HERMITE_SYMMETRIC_TIMESTEP`, the first two rungs of each ladder): the same bound, offsets at 4th order, and no secular drift: the slope of dE/E over orbits 1..7 of the binary below sqrt(eta) (one order in dt) times the base scheme's slope at the binary's first rung, eta 0.00125.

`fewbody_reference.py` writes the ICs of `test/stars_integrated_by_rumi`; this test writes its own, shifted to the origin.
