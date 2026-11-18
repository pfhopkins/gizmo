# single_stars_radpressure_cells

Direct radiation pressure per gas cell from the pixel fluxes (`GALSF_FB_RADPRESSURE_ABSORPTION_FROM_TREECOL`, set with band fluxes),
deposited in the absorbing cell, all deposits scaled to the emitted luminosity when they exceed it; the IR boost stays at the star.

| test | checks |
|---|---|
| `test_thin` | n_H = 1: per-step momentum = direct sum over the cells + IR term (3 %) |
| `test_thick[disjoint]`, `test_thick[twice]` | n_H = 1e4, photon cap: per-step momentum = L_emit / c + IR term (3 %), both band countings |
| `test_snapshot_fields` | `OUTPUT_SGCHEM_RADIATION_FIELDS`: fields present, G0_FUV = analytic (3 %), star UV, LW luminosities and Lyman photon rate = tables |
| `test_outward` | the gas gains outward momentum |
| `test_no_extra_collectives[quiet]`, `[shining]` | MPI collectives per run equal the direct term at the star, no star and one shining |
| `test_radial_shells_quiet_steps` | no star: `GALSF_RESOLVEDISM_TREECOL_PIXELS_RADIAL_SHELLS` adds no MPI collective |
| `test_cap_sum_over_all_cells` | `TEST_SINGLE_STARS_RADPRESSURE_PRINT_CAP_SUM`: the photon cap sums every gas cell now (stella_dev's recount: current mass and dust, last flux), also when 8 heavy cells moving at 500 km/s split between all-active steps (decomposition every step) and a second star dies; used = recount to 1e-5, last cap sum = an independent sum over the final cells from the tables (3 %), mass to 1e-12 |

Inputs: `SUNDIALS_DIR`; tables from `GIZMO_STELLA_TABLES_DIR` (else `scripts/get_stella_tables.sh`).
