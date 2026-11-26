# treecol_tallbox

The per-direction (HEALPix) columns and stellar band fluxes of the gravity walk (see `treecol_pixels`) in a tall box that is
periodic in x and y by the 2D Ewald table and open in z (`GRAVITY_PERIODIC_XY`, `BOX_PERIODIC`, `BOX_LONG_Z=3`,
`BOX_OUTFLOW_Z`), as the Stella tall box. The columns and fluxes use the same nearest-image wrap as the gravity walk, so they
wrap in x and y and not in z; no code beyond `GRAVITY_PERIODIC_XY` is needed, this test shows it. The columns come from the
walk with the relative opening criterion (`ErrTolForceAcc`); `DEVELOPER_MODE` lets the test set it, and the parameter file
holds the values a build without it forces. Built with `GALSF_RESOLVEDISM_TREECOL_PIXELS_OPEN_NODES_ACROSS_SHIELDING_LENGTH`: the walk opens
every node that lies partly inside and partly outside `TreeColShieldingLength` of a gas target.

Set-up: gas lattice (20 pc wide, 12^3 cells) in the lower third of a 20 x 20 x 60 pc box, six young stars inside it and one
2 pc below the top face (it would sit 2 pc from the gas through a z wrap). Every cell's 12 pixel columns, 6-13.6 eV energy
density per pixel and band totals against direct sums with the nearest image in x and y and none in z. 1 and 2 ranks.

| test | checks |
|---|---|
| `test_columns_and_fluxes_wrap_in_xy_only` | `ErrTolForceAcc` 2.5e-4 (no tree error left): columns median 1e-3, p99 5e-3; fluxes 2e-3 / 3e-2; cells within 8 pc of an x/y face within 1.5x of the other cells' median column error. |
| `test_columns_at_production_opening` | `ErrTolForceAcc` 0.0025: limits as `treecol_pixels` (columns 1e-2 / 5e-2, fluxes 2e-3 / 3e-2). |

Recorded (1 rank). `ErrTolForceAcc` 2.5e-4, 2.5e-5, 2.5e-6: columns median 4.4e-5, p99 1.6e-4 (the stored precision); cells near
an x/y face 4.4e-5, others 5.0e-5. `ErrTolForceAcc` 0.0025 without the flag (column limit missed): columns median 1.9e-2, p99 5.7e-2, every
cell low (mean -1.9e-2); cells far from the faces 2.1e-2; the opening angle (0.7, 0.35, 0.175) changes nothing. 6-13.6 eV
energy density 1.1e-4 / 3.4e-4; band totals 4e-8 / 7e-4. The removed image-sum path gave the same: 1.9e-2 / 5.9e-2.
Without the wrap the cells within 8 pc of an x/y face are off by 0.21 in the column (median), ten times the others.

Inputs as `treecol_pixels`: `SUNDIALS_DIR`, the stellar and chemistry tables.
