# tallbox_gravity

Self-gravity periodic in x and y by the 2D Ewald table and open in z, for tall boxes (`GRAVITY_PERIODIC_XY` with `BOX_PERIODIC`,
`BOX_LONG_Z=8`, `BOX_OUTFLOW_Z`), as the Stella tall box. The cubic-box solvers (3D Ewald, PM) are not used; the tree wraps
separations in x and y only.

| test | checks |
|---|---|
| `test_ewald_sheet_field_at_any_height` | 1 and 2 ranks. Exact 16 x 16 lattice sheet, probes at 0.2 to 3.5 L above and below (inside and beyond the table): `a_z = 2 pi G Sigma` to 1e-3; the sheet's own acceleration; mass and net momentum. |
| `test_ewald_slab_against_periodic_sum` | 1 and 2 ranks. Slab of three layers plus a layer 0.03 from each z face (1280 particles, random masses; the face layers would pull on each other through a z wrap). Tight walk (`ErrTolTheta=0.15`, `ErrTolForceAcc=1e-6`) against the sum periodic in x/y (numpy, `slab_reference.py`: 16 rings of images plus the uniform sheet outside them; 8 rings must agree with 16 to 1e-3): median, p99 and maximum of `\|a - a_ref\| / median\|a_ref\|` below 1e-4, 5e-4, 1e-3; production walk against the tight one (4e-3 median, 1.5e-2 p99); net force; mass and net momentum. |

Recorded for the slab (1 rank, before the limits of the Ewald slab test were set): reference with 8 / 16 rings against 32 rings: median 1.7e-5 / 2.7e-6,
max 6.9e-4 / 1.3e-4; tight Ewald walk against 32 rings: median 1.5e-5, max 2.7e-5; production against tight: 3.9e-4, p99 1.5e-3.
The image sums the code had before (`GRAVITY_PERIODIC_XY_IMAGES`, 8 rings; removed) were off the periodic sum by 1.7e-2 (median) and up to
0.87 (face layers), and their production walk stayed 1.1e-2 from the tight one (4.3e-3 at `ErrTolForceAcc=1e-5`).

Limits are in `LIMITS` in the test.
