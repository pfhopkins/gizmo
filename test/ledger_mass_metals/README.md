# ledger_mass_metals

Mass and metal rows of the ledger (`OUTPUT_MASS_METAL_ENERGY_LEDGER`, `ledger.txt`, one row per snapshot): state rows
(gas and star mass and metal mass, summed over the particles at output) and cumulative bookings (stars formed and their
metals; single-star ejecta and their metals taken from the stars; the same delivered to the gas by the feedback loop;
remnant reset, the star mass removed when a dying star is set to its table remnant; metal mass the stars lose).

A closed periodic box of 8^3 cells of 1 Msun with SGChem and single stars from the tables, no metal diffusion: two births
at the first step, a young 8 Msun star losing wind mass, a 3 Msun star dying on the AGB; 2 ranks, 0.29 Myr, 7 rows.

| test | checks |
|---|---|
| `test_ledger_books_what_reached_the_gas` | `E_fb_thermal_now_erg` + `E_fb_kinetic_erg` = what the gas gained, from snapshots, run minus a twin with no wind owed, 2 % of the ejecta energy. `E_fb_thermal_now_erg` is the change of the cell's energy over the apply step; `E_fb_thermal_deferred_erg` is heat only put into `DtInternalEnergy`, not delivered, so it is not added |
| `test_ledger_feedback_energy` | ejecta energy booked from the tables (1/2 dm v_ej^2 per coupled event) against heat now, heat to the deferred `DtInternalEnergy` branch and kinetic energy given to the gas; residual reported per row, bar 5 % |
| `test_ledger_mass_metals` | every row closes, relative to the total mass and metal mass (1e-12): star and gas mass and metal changes = their bookings; ejecta delivered to gas = ejecta taken from stars; state rows = snapshot sums |

Inputs: `SUNDIALS_DIR`; stellar and chemistry tables from `GIZMO_STELLA_TABLES_DIR` (else installed by
`scripts/get_stella_tables.sh`); `SGCHEM_TABLES_DIR` overrides the chemistry-table directory.
