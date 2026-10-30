# treecol_pixels

Per-direction (HEALPix) columns and stellar band fluxes from the gravity walk (`GALSF_RESOLVEDISM_TREECOL_PIXELS...`,
design: per-pixel columns and fluxes inside the existing TreeCol accumulation). Every run stops after the first gravity
computation: `TEST_DUMP_TREECOL_PIXELS_AND_EXIT` (test only) writes the tree radiation inputs and results as text to
`output/treecol_dump_*.txt`.

| test | checks |
|---|---|
| `test_pixel_columns_and_fluxes` | every cell's 12 pixel columns (dust-weighted, H2, CO; within `TreeColShieldingLength`) and stellar band energy densities against direct sums over all particles: columns median <= 1e-2, p99 <= 5e-2 (tree error); fluxes median <= 2e-3, p99 <= 3e-2; 1 and 2 ranks |
| `test_node_species_masses` | node dust-weighted gas, H2 and CO masses: cells as the chemistry inputs, root node = sum of the cells (1e-12) after the first tree build and after three steps of chemistry and drift; 1 and 2 ranks |
| `test_star_band_luminosities` | single-star band luminosities (PE, LW, 6-8 eV, NUV 3.444-6 eV, OPT) equal the stella-tables interpolation at the current age (1e-10), disjoint sum identity (1e-12), LW <= PE, root node = sum of the stars; 1 and 2 ranks |

Parameter: `TreeColShieldingLength` [pc], pixel columns only from mass closer than this (0 = no limit).

Inputs: `SUNDIALS_DIR`; the stellar and chemistry tables from `GIZMO_STELLA_TABLES_DIR` (else installed by
`scripts/get_stella_tables.sh`); `SGCHEM_TABLES_DIR` overrides the chemistry-table directory.
