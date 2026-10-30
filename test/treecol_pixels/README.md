# treecol_pixels

Per-direction (HEALPix) columns and stellar band fluxes from the gravity walk (`GALSF_RESOLVEDISM_TREECOL_PIXELS...`,
design: per-pixel columns and fluxes inside the existing TreeCol accumulation). Every run stops after the first gravity
computation: `TEST_DUMP_TREECOL_PIXELS_AND_EXIT` (test only) writes the tree radiation inputs and results as text to
`output/treecol_dump_*.txt`.

| test | checks |
|---|---|
| `test_star_band_luminosities` | single-star band luminosities (PE, LW, 6-8 eV, NUV 3.444-6 eV, OPT) equal the stella-tables interpolation at the current age (1e-10), disjoint sum identity (1e-12), LW <= PE, root node = sum of the stars; 1 and 2 ranks |

Inputs: `SUNDIALS_DIR`; the stellar and chemistry tables from `GIZMO_STELLA_TABLES_DIR` (else installed by
`scripts/get_stella_tables.sh`); `SGCHEM_TABLES_DIR` overrides the chemistry-table directory.
