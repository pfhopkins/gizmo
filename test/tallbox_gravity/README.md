# tallbox_gravity

Self-gravity periodic in x and y by explicit image sums and open in z, for tall boxes
(`GRAVITY_PERIODIC_XY_IMAGES` with `BOX_PERIODIC`, `BOX_LONG_Z=8`, `BOX_OUTFLOW_Z`; parameter
`GravityPeriodicXYImages` = N rings, so (2N+1)^2 - 1 images per interaction besides the nearest image). The cubic-box
solvers (Ewald, PM) are not used; the tree wraps separations in x and y only.

| test | checks |
|---|---|
| `test_slab_against_direct_sum` | 1 and 2 ranks. Slab of three layers plus a layer 0.03 from each z face (1280 particles, random masses), box 1 x 1 x 8. Tight walk (`ErrTolTheta=0.15`, `ErrTolForceAcc=1e-6`) against a numpy direct sum over the same images: median, p99 and maximum of `\|a - a_ref\| / median\|a_ref\|`; production walk against the tight one; net force `\|sum m a\| / sum m\|a\|`; mass and net momentum over the run. The face layers would pull on each other through a z wrap. |
| `test_sheet_against_analytic_field` | 1 and 2 ranks. Exact 16 x 16 lattice sheet (surface density 1) with probe particles 0.2, 0.5, 1.0 above and below: `a_z = 4 G Sigma arctan(W^2 / (h sqrt(2 W^2 + h^2)))`, W = (N + 1/2) L, the field of the square patch the images cover (tends to `2 pi G Sigma` for h << W); the sheet's own acceleration against `2 pi G Sigma`. |

Limits are in `LIMITS` in the test. The production-walk limits (4e-3 median / 1.5e-2 p99 against the tight walk, set before the first run) are not met:
`test_slab_against_direct_sum` fails on them. Part of the difference comes from nodes straddling the half-box plane; the rest is open. Relaxed limits would be a
review decision.
