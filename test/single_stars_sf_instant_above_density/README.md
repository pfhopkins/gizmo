# single_stars_sf_instant_above_density

`GALSF_SINGLE_STARS_SF_INSTANT_ABOVE_DENSITY` (parameter `nHcutoffSF`, cm^-3): every cell above that hydrogen number
density that is eligible for star formation (density threshold, cool-down) forms a single star in its step, with
probability 1 and whatever the Jeans criterion and the rate say. The mass is drawn from the number-weighted Kroupa IMF
(the mass-weighted draw accepted with M_min / M); the usual draw would give a 1/M-biased mass.

Problem: the lattice of `single_stars_sf_criterion` (16^3 cells of 1 Msun, n_H ~ 7e3 cm^-3, efficiency 1e-12), one step.

Checks: cutoff 1e3: at least 95% of the cells form, 65-85% of the drawn masses below 0.5 Msun (number-weighted 76%,
mass-weighted 26%); cutoff 1e9: no star.

Split (2026-11-23, as stella_dev): an event with X = m pfac at least twice its first mass-weighted draw M0 takes the sub-event
split of X from a fresh draw instead of one star. Check: efficiency 1e6 (X ~ 1 Msun): at least 3% of the cells form two or more
stars, masses number-weighted (0.65-0.85 below 0.5 Msun), at least 85% of the cells form.
