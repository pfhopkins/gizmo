# single_star_record

The compact record of IMF-sampled single stars (`GALSF_SINGLE_STARS_FROM_IMF`): drawn IMF mass and birth metallicity,
reached through `P[i].StarIdx`. It travels with its star in the domain exchange, is written to restart files and
snapshots (`SingleStarDrawnMass`, `SingleStarBirthMetallicity`), and is read from initial conditions for stars with a
drawn mass > 0. A star with a record is one physical star and is never merged.

Problem: an 8^3 lattice of 1 Msun gas cells in a 5.28 pc periodic box (no gravity) and 64 stars of 0.1-50 Msun (some
below the merge threshold of 0.49 Msun) at random positions, moving at 20-60 km/s, so they cross the box 1.5-4.5 times
and change rank many times on 4 ranks.

Checks:
1. every snapshot of a 4-rank run holds all 64 stars, each with its mass, drawn mass and birth metallicity bitwise equal
   to the initial conditions;
2. a run stopped halfway and continued from its restart files ends with the same records;
3. a 1-rank run ends with the same records.

Reference: the initial conditions (records must be carried, not computed). Negative controls run during development:
with the record dropped on receipt check 1 fails (so stars do change rank); without the merge guard one star (ID 532)
is merged and check 1 fails.

`test_split_threshold`: with single stars a cell splits above 2.01 x the largest gas cell of the initial conditions, as stella_dev.
