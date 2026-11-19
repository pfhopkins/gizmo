# energy_step_trace_cloud

Energy and mass step traces in a production cloud that forms single stars (as stella_sfr_log: Stella cloud flags and parameters,
cloud IC from `STELLA_CLOUD_IC`, 2.7 Myr, 2 ranks). Checked by `gizmo.conservation_trace`; the star-formation stage is exact
(heat of the mass taken into stars, donors' heat and kinetic energy, the new star's kinetic energy, CR energy collected).

Inputs: `SUNDIALS_DIR`, `STELLA_CLOUD_IC`; tables from `GIZMO_STELLA_TABLES_DIR` (else `scripts/get_stella_tables.sh`).
