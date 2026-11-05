# ledger_step_checkpoints

The `ledger_mass_metals` box (8^3 cells of 1 Msun, SGChem, single stars from the tables: two births, a young star losing wind mass,
a 3 Msun star dying on the AGB; 2 ranks, 0.29 Myr) with `OUTPUT_MASS_BALANCE_STEP_CHECKPOINTS`.

`mass_balance_steps.txt` gets a line at a checkpoint (`pre_hydro_fb`, `post_hydro_fb`, `pre_star_formation`, `post_star_formation`)
where baryon mass, gas metal mass or the ledger-corrected metal mass changed by more than 1e-14 since the previous checkpoint, and
always at the first call. Bars, set before the first run: the file and the four tags exist; no logged change of M_bary or of
M_Z_ledger exceeds 1e-12 of its value.
