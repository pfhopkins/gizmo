# fb_after_closing_kick_radp

With `RESOLVED_ISM_FB_AFTER_CLOSING_KICK` radiation pressure is deposited after the closing kick; with
`RESOLVED_ISM_RECOMPUTE_HYDRO_POST_FB` a step in which it pushed any cell repeats the hydro pass (first flag alone: SN and Ia only,
see sn_hydro_recompute). Stella cloud production flags, one 5 Msun main-sequence star (no wind, no death), n_H = 10, 8^3 cells,
20 kyr, 2 ranks.

- first flag alone: no repeat; both flags: one repeat per step with radiation-pressure momentum (MomWinds.txt), at least one;
- mass constant to 1e-12.

Inputs: `SUNDIALS_DIR`; tables from `GIZMO_STELLA_TABLES_DIR` (else `scripts/get_stella_tables.sh`).
