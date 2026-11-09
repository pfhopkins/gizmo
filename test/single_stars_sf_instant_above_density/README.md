# single_stars_sf_instant_above_density

`GALSF_SINGLE_STARS_SF_INSTANT_ABOVE_DENSITY` (parameter `nHcutoffSF`, cm^-3): every cell above that hydrogen number
density that is eligible for star formation (density threshold, cool-down) forms a single star in its step, with
probability 1 and whatever the Jeans criterion and the rate say. The mass is drawn from the number-weighted Kroupa IMF
(the mass-weighted draw accepted with M_min / M), as stella_dev; the usual draw would give a 1/M-biased mass.

Problem: the lattice of `single_stars_sf_criterion` (16^3 cells of 1 Msun, n_H ~ 7e3 cm^-3, efficiency 1e-12), one step.

Checks: cutoff 1e3: at least 95% of the cells form, 65-85% of the drawn masses below 0.5 Msun (number-weighted 76%,
mass-weighted 26%); cutoff 1e9: no star.
