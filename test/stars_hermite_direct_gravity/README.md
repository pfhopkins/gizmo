# stars_hermite_direct_gravity

Type-4 stars on the 4th-order Hermite integrator with the exact direct star-star sum (`STARS_HERMITE_DIRECT_GRAVITY`), no gas. Two few-body problems, each against Newtonian point-mass orbits integrated in `fewbody_reference.py` (DOP853, rtol 1e-13; the softening, 6.4 AU, stays inside every pericentre):

* binary: 1 + 1 Msun, a = 1000 AU, e = 0.9, ten orbits;
* triple: inner 1 + 1 Msun, a = 100 AU, e = 0.5; outer 1 Msun, a = 1000 AU, e = 0.3, inclined 30 degrees; two outer orbits (52 inner).

Measured: relative energy error at the end and the largest offset of the relative positions from the reference (in units of the inner semi-major axis). Snapshots carry `HermiteSync*` positions and velocities (`IO_HERMITE_SYNC`). Bars are 3x the measured errors: binary dE/E 1e-4, offset 1.2e-3 a; triple 1.5e-5, 6e-3 a_in.

`fewbody_reference.py` writes the ICs and is shared with `test/stars_integrated_by_rumi`.
