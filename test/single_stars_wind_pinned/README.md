# single_stars_wind_pinned

`TEST_SINGLE_STARS_WIND_PINNED_RATE_AND_SPEED` (test only): the wind of every single star is a constant rate
`TestFixedWindMdot` [Msun/yr] at the speed `TestFixedWindVel` [km/s], read from neither the tables nor the pre-main-sequence
delay, so that the wind power 1/2 Mdot v^2 is known exactly (analytic Weaver bubble). Composition: the table surface
abundance. Valid before the star's death (the winds owed at death still come from the tables).

Problem: lattice of `single_stars_winds_agb_ia`, one 35 Msun star 1 Myr old with winds returned up to that age,
Mdot = 1e-4 Msun/yr, v = 3000 km/s, 0.01 code time.

Checks: (1) mass lost = Mdot x (time to the last delivery), 1e-9; (2) last delivery at the star's age to 3e-4 code time;
(3) energy gained by gas and star = 1/2 M v^2 of the mass lost, 1 % (the hydro energy drift of the box is about 4e3 code units, 0.1 % of this wind's energy).

Reference: the definitions of the parameters; conservation laws.
