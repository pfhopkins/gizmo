# energy_step_trace_sn

Energy step trace (`OUTPUT_ENERGY_STEP_TRACE`) and mass step trace (`OUTPUT_MASS_BALANCE_STEP_CHECKPOINTS`) in a production run:
Stella cloud flags, one 10 Msun star dying 1 kyr in and a 60 Msun star of 0.1 Myr with winds, n_H = 10, 8^3 cells, 10 kyr, 2 ranks. Checked by
`gizmo.conservation_trace` (mass and metal ledger 1e-12, no STAGE line of an exact stage, integration residual 1e-2 of the
first sync's scale or of the booked SN energy); the SN books 1e51 erg and the winds book energy.

Inputs: `SUNDIALS_DIR`; tables from `GIZMO_STELLA_TABLES_DIR` (else `scripts/get_stella_tables.sh`).
