# single_stars_cr_speed

With `COSMIC_RAY_FLUID` and single stars the CR transport speed defaults to 300 km/s (`CRFLUID_SPEEDOFLIGHT_REDUCTION` = 3e7/c),
A 24^3 magnetised box (units pc, Msun, km/s) with a CR slab of half-width 2 pc and B along x: the CRs free-stream.
Bars: at t = 0.002 the distance from the mid-plane holding 99 % of the excess CR energy has grown by at most 2 x 300 km/s x t in the
default build, and by more than 5 x that with `CRFLUID_SPEEDOFLIGHT_REDUCTION=(1.0)`.
