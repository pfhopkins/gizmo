# tallbox_gravity

Self-gravity periodic in x and y and open in z, for tall boxes (`BOX_PERIODIC`, `BOX_LONG_Z=8`, `BOX_OUTFLOW_Z`). Two methods:
the 2D Ewald table (`GRAVITY_PERIODIC_XY`, `Config_ewald.sh`), which the Stella tall box uses, and explicit image sums
(`GRAVITY_PERIODIC_XY_IMAGES`, `Config.sh`; parameter `GravityPeriodicXYImages` = N rings, so (2N+1)^2 - 1 images per interaction
besides the nearest image). The cubic-box solvers (3D Ewald, PM) are not used; the tree wraps separations in x and y only.

2D Ewald (`test_tallbox_gravity_ewald.py`):

| test | checks |
|---|---|
| `test_ewald_sheet_field_at_any_height` | 1 and 2 ranks. Exact 16 x 16 lattice sheet, probes at 0.2 to 3.5 L above and below (inside and beyond the table): `a_z = 2 pi G Sigma` to 1e-3; the sheet's own acceleration; mass and net momentum. |
| `test_ewald_slab_against_periodic_sum` | 1 and 2 ranks. Slab of three layers plus a layer 0.03 from each z face (1280 particles, random masses; the face layers would pull on each other through a z wrap). Tight walk (`ErrTolTheta=0.15`, `ErrTolForceAcc=1e-6`) against the sum periodic in x/y (numpy, `slab_reference.py`: 16 rings of images plus the uniform sheet outside them; 8 rings must agree with 16 to 1e-3): median, p99 and maximum of `\|a - a_ref\| / median\|a_ref\|` below 1e-4, 5e-4, 1e-3; production walk against the tight one (4e-3 median, 1.5e-2 p99); net force; mass and net momentum. |

Recorded for the slab (1 rank, before the limits of the Ewald slab test were set): reference with 8 / 16 rings against 32 rings: median 1.7e-5 / 2.7e-6,
max 6.9e-4 / 1.3e-4; tight Ewald walk against 32 rings: median 1.5e-5, max 2.7e-5; production against tight: 3.9e-4, p99 1.5e-3.
The image sum over 8 rings is off the periodic sum by 1.7e-2 (median) and up to 0.87 (face layers).

Image sums (`test_tallbox_gravity.py`): the path computes the truncated sum it states, which is not the periodic field.

| test | checks |
|---|---|
| `test_slab_against_direct_sum` | 1 and 2 ranks. The slab; tight walk against a numpy direct sum over the same images: median, p99 and maximum (1e-3, 5e-3, 2e-2); net force; mass and net momentum. |
| `test_slab_production_walk` | expected failure (strict). Production walk against the tight one, limits 4e-3 median / 1.5e-2 p99 as set before the first run: measured 1.1e-2 / 3.3e-2, nearly all in x and y, and still 4.3e-3 at `ErrTolForceAcc=1e-5`. The truncated sum jumps by 0.15 to 0.33 G m / L^2 where a source changes its nearest image, and the walk wraps a node as a whole. |
| `test_sheet_against_analytic_field` | 1 and 2 ranks. Exact 16 x 16 lattice sheet (surface density 1) with probe particles 0.2, 0.5, 1.0 above and below: `a_z = 4 G Sigma arctan(W^2 / (h sqrt(2 W^2 + h^2)))`, W = (N + 1/2) L, the field of the square patch the images cover; the sheet's own acceleration against `2 pi G Sigma`. |

Limits are in `LIMITS` in the tests.
