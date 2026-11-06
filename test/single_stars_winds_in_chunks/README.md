# single_stars_winds_in_chunks

`GALSF_SINGLE_STARS_WINDS_IN_CHUNKS`: a massive star keeps its wind until the mass owed exceeds a fraction of its drawn mass, 1 % at
8 Msun rising linearly to 20 % at 350 Msun, and returns it in one delivery at the wind speed of that moment. As stella_dev without
GALSF_RESOLVEDISM_WINDS_CONTINUOUS. The default returns the wind every star step.

Problem: 12^3 cells of 1 Msun, SGChem net 5, table stars, 2 ranks, 100 yr, one 35 Msun star (threshold 2.5 %, 0.875 Msun); the last
delivery age in the initial conditions sets the mass owed.

Checks: (1) the owed mass crosses the threshold 50 yr into the run: one delivery, not before 25 yr, of the threshold mass to 1e-3, at
the table speed of that moment to 1 %, and the star lost just that. (2) The star dies 80 yr into the run owing half the threshold: no
delivery before the death, then the owed mass goes out below the threshold, then the supernova.
