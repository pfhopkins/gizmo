# single_stars_wind_dump

`TEST_DUMP_STATE_AROUND_WIND_DEPOSIT` (test only): in each feedback step with a mass return (wind or AGB death) every rank appends the
mass, velocity and gas energy of all its particles before (phase 0) and after (phase 1) the feedback loop to `wind_dep_dump.<rank>.bin`.
One reduction in those steps; nothing in other steps. Record: `int phase, long long record, double time, long long n`, then `n` x
`unsigned long long ID, int type, 5 doubles (mass, v_x, v_y, v_z, u)`.

Problem: one 35 Msun star 1 Myr old in the lattice of `single_stars_winds_agb_ia`, 0.01 code time, 2 ranks.
Checks: at least 5 deposits as (0, 1) pairs of equal record, time and size on both ranks; per deposit total mass conserved
and gas gain = star loss (both to 1e-14 Msun absolute), total momentum change below 5 % of (mass returned x 3000 km/s).

Reference: stella_dev `resolvedism_fb.cc` (`fb_test_state_dump`).
