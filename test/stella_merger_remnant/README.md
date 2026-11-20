# stella_merger_remnant

Driver test of `star_mergers_apply` (galaxy_sf/stella_star_mergers.cc) on Config_stella_default.sh + KETJU star mergers: a live 3 Msun
star merged into a heavier 8 Msun compact remnant keeps the remnant state (MstarSampleIMF[0] = 0), mass-weighted metals, the summed
mass handed over by the caller, and an emptied victim.

Not covered: the KETJU collision path itself with a remnant (MSTAR's survivor choice, the collision energy booking for a live star
hitting a remnant). That needs a Stella + KETJU simulation with a remnant in the ICs; test/ketju_star_collision runs the collision path
with live stars and the fallback radius only (review round 2, N6, 2026-11-19).
