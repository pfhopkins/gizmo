# treecol_tallbox

The per-direction (HEALPix) columns and stellar band fluxes of the gravity walk (see `treecol_pixels`) in a tall box that is
periodic in x and y by image sums and open in z (`GRAVITY_PERIODIC_XY_IMAGES`, `BOX_PERIODIC`, `BOX_LONG_Z=3`,
`BOX_OUTFLOW_Z`). The columns and fluxes use the same nearest-image wrap as the gravity walk, so they wrap in x and y and
not in z; no code beyond `GRAVITY_PERIODIC_XY_IMAGES` is needed, this test shows it.

| test | checks |
|---|---|
| `test_columns_and_fluxes_wrap_in_xy_only` | 1 and 2 ranks. Gas lattice (20 pc wide, 12^3 cells) in the lower third of a 20 x 20 x 60 pc box, six young stars inside it and one 2 pc below the top face (it would sit 2 pc from the gas through a z wrap). Every cell's 12 pixel columns, 6-13.6 eV energy density per pixel and band totals against direct sums with the nearest image in x and y and none in z. Limits as `treecol_pixels` (columns 1e-2 / 5e-2, fluxes 2e-3 / 3e-2); the column limit is not met (below); cells within 8 pc of an x/y face must be within 1.5x of the other cells' median column error. |

Recorded (1 rank; column limit missed): columns median 1.9e-2, p99 5.9e-2 (cells far from the faces: 2.3e-2, the tree error of this geometry);
6-13.6 eV energy density 1.2e-4 / 3.2e-4; band totals 3e-8 / 3e-4. Without the wrap the cells within 8 pc of an x/y face
are off by 0.21 in the column (median), ten times the others.

Inputs as `treecol_pixels`: `SUNDIALS_DIR`, the stellar and chemistry tables.
