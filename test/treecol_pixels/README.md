# treecol_pixels

Per-direction (HEALPix) columns and stellar band fluxes from the gravity walk (`GALSF_RESOLVEDISM_TREECOL_PIXELS...`,
design: per-pixel columns and fluxes inside the existing TreeCol accumulation). Every run stops after the first gravity
computation: `TEST_DUMP_TREECOL_PIXELS_AND_EXIT` (test only) writes the tree radiation inputs and results as text to
`output/treecol_dump_*.txt`.

| test | checks |
|---|---|
| `test_pixel_columns_and_fluxes` | every cell's 12 pixel columns (dust-weighted, H2, CO; within `TreeColShieldingLength`) and stellar band energy densities against direct sums over all particles: columns median <= 1e-2, p99 <= 5e-2 (tree error); fluxes median <= 2e-3, p99 <= 3e-2; 1 and 2 ranks |
| `test_columns_unsoftened` | softening of 1 pc on a 1.7 pc lattice: the columns follow the unsoftened direct sum, as stella_dev |
| `test_chemistry_inputs` | SGChem field per cell from the tree energy densities (6-13.6 eV pixels floored at 0.324e-2 SGChemBackgroundG0/12): G0, G0_LW, band-resolved G0_dust as stella_dev (1e-6) |
| `test_far_field_g0` | after five steps with chemistry, one star dying in between: tree 6-13.6 eV field = direct sum over current star luminosities (median 1e-2, p99 5e-2), dead star dark; 2 ranks |
| `test_slab_shielding` | n_H ~ 1e3 slab lit by one 40 Msun star 3 pc in front (G0 ~ 300 at the face), 20 steps of chemistry (0.15 Myr): on the star's axis the cells more than 2 pc deep (N_H >~ 5e21 toward it) reach x_H2 > 1e-2 and >= 30x the lit face (the chemistry uses the pixel columns for dust, H2 and CO shielding) |
| `test_host_equals_device` | device walk = host walk cell outputs (float storage, at most one rounding step) and device repeat bitwise; 2 ranks x 2 threads and 1 rank x 4 threads (parameter `GravityHostWalkBelowActive` selects the walk) |
| `test_node_species_masses` | node dust-weighted gas, H2 and CO masses: cells as the chemistry inputs, root node = sum of the cells (1e-12) after the first tree build and after three steps of chemistry and drift; 1 and 2 ranks |
| `test_dust_to_gas_independent_of_metallicity` | build with `SGCHEM_DGR_INDEPENDENT_OF_METALLICITY`, `SGChemDustToGasRatio` 0.5, cell Z from 0.1 to 3 solar: every cell's dust-weighted mass is 0.5 x its mass (1e-12), as stella_dev; the chemistry and radiation pressure read the same function. Before: scaled with Z (up to 3x off) |
| `test_dust_to_gas_from_live_dust` | build with `GALSF_ISMDUSTCHEM_MODEL=(1+2)`, live carbonaceous dust 0-2 % of the gas mass per cell: every cell's dust-weighted mass is its dust mass / 0.01 (1e-12), as stella_dev with its dust model. Before: from the cell metallicity |
| `test_star_band_luminosities` | single-star band luminosities (PE, LW, 6-8 eV, NUV 3.444-6 eV, OPT) equal the stella-tables interpolation at the current age (1e-10), disjoint sum identity (1e-12), LW <= PE, root node = sum of the stars; 1 and 2 ranks |

Parameter: `TreeColShieldingLength` [pc], pixel columns only from mass closer than this (0 = no limit).

Inputs: `SUNDIALS_DIR`; the stellar and chemistry tables from `GIZMO_STELLA_TABLES_DIR` (else installed by
`scripts/get_stella_tables.sh`); `SGCHEM_TABLES_DIR` overrides the chemistry-table directory.
