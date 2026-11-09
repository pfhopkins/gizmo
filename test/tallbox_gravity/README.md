# tallbox_gravity

Self-gravity periodic in x and y by explicit image sums and open in z, for tall boxes
(`GRAVITY_PERIODIC_XY_IMAGES` with `BOX_PERIODIC`, `BOX_LONG_Z=8`, `BOX_OUTFLOW_Z`; parameter
`GravityPeriodicXYImages` = N rings, so (2N+1)^2 - 1 images per interaction besides the nearest image). The cubic-box
solvers (Ewald, PM) are not used; the tree wraps separations in x and y only.

| test | checks |
|---|---|
| `test_slab_against_direct_sum` | 1 and 2 ranks. Slab of three layers plus a layer 0.03 from each z face (1280 particles, random masses), box 1 x 1 x 8. Tight walk (`ErrTolTheta=0.15`, `ErrTolForceAcc=1e-6`) against a numpy direct sum over the same images: median, p99 and maximum of `\|a - a_ref\| / median\|a_ref\|`; production walk against the tight one; net force `\|sum m a\| / sum m\|a\|`; mass and net momentum over the run. The face layers would pull on each other through a z wrap. |
| `test_sheet_against_analytic_field` | 1 and 2 ranks. Exact 16 x 16 lattice sheet (surface density 1) with probe particles 0.2, 0.5, 1.0 above and below: `a_z = 4 G Sigma arctan(W^2 / (h sqrt(2 W^2 + h^2)))`, W = (N + 1/2) L, the field of the square patch the images cover (tends to `2 pi G Sigma` for h << W); the sheet's own acceleration against `2 pi G Sigma`. |

Limits are in `LIMITS` in the test. The production-walk limits (4e-3 / 1.5e-2, set before the first run) are NOT met:
`test_slab_against_direct_sum` fails on them. Measured 1.1e-2 median / 3.3e-2 p99 (1.9e-3 / 3.8e-3 for the same walk in an
open box). Measurements on the cause: the absolute difference between production and tight walk is 2.7e-2 at N = 0
(nearest-image wrap, no further images), 1.9e-2 / 3.1e-2 / 3.5e-2 / 2.8e-2 / 1.6e-2 at N = 1 / 2 / 4 / 8 / 16, so it does
not come from the image ring (it would shrink with N); targets in the middle of the box (no half-box plane inside it) show
1.3e-2, the others 2.3-3.1e-2; the tight walk matches the direct sum, and `ErrTolForceAcc=1e-6` reproduces the tight walk.
The wrap hypothesis (nodes straddling the half-box) explains about half of it; the rest is open. A bar of 1.7e-2 / 5e-2
would cover the measured value; whether to adopt it is for review.

Recorded (1 and 2 ranks identical): tight walk vs direct sum median 5e-14, p99 4.8e-3, max 5.9e-3; probe `a_z`
within 1.7e-7 of the analytic patch field; sheet self-acceleration 2.3e-3 of `2 pi G Sigma`; net force 3e-5;
momentum 5e-14.

Not covered: the potential (the image sum is truncated, so it is not the periodic potential) and the target's own
images (no net force, they only enter potential and tidal tensor).
