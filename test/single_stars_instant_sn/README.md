# single_stars_instant_sn

`GALSF_SINGLE_STARS_INSTANT_SN_BEFORE_TIME` (test only): while the code time is below the parameter `TimeInstantSN`, every
single star of drawn mass of at least 8 Msun explodes at its first feedback step, whatever its age (lifetime 0).
`TimeInstantSN` 0 has no effect.

Problem: lattice of `single_stars_forced_events` (24^3 cells of 1 Msun, two 20 Msun stars 1 Myr old, table death ~10 Myr), 4 kyr.

Checks: with `TimeInstantSN` 0.002 each star has one supernova row at a time below 3e-4, mass conserved (1e-12);
with 0 no supernova in the run.

Reference: the parameter definitions.
