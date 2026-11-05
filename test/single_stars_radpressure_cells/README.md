# single_stars_radpressure_cells

Direct radiation pressure per gas cell from the pixel fluxes (`GALSF_FB_RADPRESSURE_ABSORPTION_FROM_TREECOL`, set with band fluxes),
deposited in the absorbing cell, all deposits scaled to the emitted luminosity when they exceed it; the IR boost stays at the star.

| test | checks |
|---|---|
| `test_thin` | n_H = 1: per-step momentum = direct sum over the cells + IR term (3 %) |
| `test_thick[disjoint]`, `test_thick[twice]` | n_H = 1e4, photon cap: per-step momentum = L_emit / c + IR term (3 %), both band countings |
| `test_snapshot_fields` | `OUTPUT_SGCHEM_RADIATION_FIELDS`: fields present, G0_FUV = analytic (3 %), star lums = tables |
| `test_outward` | the gas gains outward momentum |

Inputs: `SUNDIALS_DIR`; tables from `GIZMO_STELLA_TABLES_DIR` (else `scripts/get_stella_tables.sh`).
