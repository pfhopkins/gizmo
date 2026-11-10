# single_stars_gas_timestep

With `GALSF_RESOLVEDISM_TREECOL_PIXELS_STELLAR_BAND_FLUXES` gas drops gravity from the acceleration criterion and keeps the tidal
criterion: the band fluxes add no force.

| test | checks |
|---|---|
| `test_gas_step_is_tidal` | cold gas around a heavy star, all cells many kernels away: the shortest gas step is at least 1.5 x the acceleration criterion with gravity |

Inputs: `SUNDIALS_DIR`; tables from `GIZMO_STELLA_TABLES_DIR` (else `scripts/get_stella_tables.sh`).
