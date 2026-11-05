# single_stars_merge_split

Cell merges and the split threshold under single stars, chemistry on, as stella_dev.

| test | checks |
|---|---|
| `test_chemistry_state_conserved` | merged cells keep total m x of H2, H+, CO (1e-6): the survivor takes mass-weighted abundances; dust temperature and temperature mix the same way |
| `test_split_threshold` | no threshold in the IC: `MaxMassForParticleSplit` = 2.01 x the largest IC cell (other builds keep 3.01) |

Inputs: `SUNDIALS_DIR`; tables from `GIZMO_STELLA_TABLES_DIR` (else `scripts/get_stella_tables.sh`).
