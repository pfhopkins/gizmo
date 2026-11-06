# single_stars_radiation_pressure

Photon momentum of IMF-sampled single stars in the local radiation-pressure loop (`GALSF_FB_RADPRESSURE_SINGLE_STAR_BANDS`,
design_feedback_coupling.md 4.6): the stella_dev estimator at the star (band luminosities from the tables absorbed over the
kernel column, IR boost on the absorbed light), distributed with the loop's own weights.

| test | checks |
|---|---|
| `test_momentum_budget` | momentum offered per step (MomWinds.txt) = the estimator from the tables, disjoint bands (2 %); coupled = offered on average (5 %); 2 ranks |
| `test_count_6_8ev_twice` | `GALSF_FB_RADPRESSURE_COUNT_6_8EV_TWICE` (stella_dev production, 6-8 eV also in the NUV term): budget ratio = table ratio (1e-3) |

Inputs: `SUNDIALS_DIR`; tables from `GIZMO_STELLA_TABLES_DIR` (else `scripts/get_stella_tables.sh`).
