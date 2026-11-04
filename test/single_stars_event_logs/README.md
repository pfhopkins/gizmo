# single_stars_event_logs

Per-event logs of single stars (`GALSF_SINGLE_STARS_EVENT_LOGS`): `SNinfo.txt`, `AGBinfo.txt` and `WINDinfo.txt`, in the columns of
stella_dev. A rank keeps its rows until the next snapshot or restart file, where rank 0 gathers them and writes them sorted by
time and ID; between outputs there is no collective.

Problems: the supernova stars of `single_stars_supernovae` and the wind, AGB and Type Ia stars of `single_stars_winds_agb_ia`, with
their initial conditions and parameters.

Checks:
1. one SNinfo row per star, with the ID, class, drawn mass, ejecta, metal mass and energy of the source model's own record
   (`single_star_events.txt`); a failed SN has status 1 and no energy; age at or past the table lifetime; the C, O, Si, Fe shares
   as in the table; ambient density and internal energy those of the lattice;
2. one AGBinfo row for the AGB death; the two Type Ia events in SNinfo as class 7 with 1e51 erg; the WINDinfo masses add up to
   what the 35 Msun star lost by the last snapshot, at the table speed (1%), in time order;
3. the same rows on 1, 2 and 4 ranks.

Reference: the source model's event record, the tables, stella_dev's column definitions (`resolvedism_fb.cc`, `core/begrun.cc`).
