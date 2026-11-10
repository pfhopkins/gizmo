# single_stars_sfr_windows

`sfr_rism.txt` (`GALSF_SINGLE_STARS_EVENT_LOGS`): in the steps that build the gravity tree (one reduction there, nothing on
other steps), the star-formation rate [Msun/yr] over the last 50, 40, 30, 20, 5 and 1 Myr from the drawn masses of the stars of the
record, and the number of stars.

Problem: two 20 Msun stars 3 Myr old in the lattice of `single_stars_forced_events`, 8 kyr, 2 ranks.
Checks: N_stars = 2 in every row; SFR = 40 Msun / window for 50 to 5 Myr and 0 for 1 Myr (1e-6, the digits printed); times increasing; at least
one row and not more rows than steps.

Reference: the window and unit definitions.
