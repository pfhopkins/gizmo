# single_stars_birth_age

`TEST_SINGLE_STARS_BORN_WITH_GIVEN_AGE` (test only): a single star born in the run takes the formation time `t_birth - TestStarInitAgeYr`,
so its evolution (winds, death) starts at that age. Lattice of `single_stars_collect_gas`, one 35 Msun star born at the first step.

Check: the snapshot's `StellarFormationTime` is earlier by 2e6 yr (in code time) than in a twin with `TestStarInitAgeYr 0`, to 1e-9.

Reference: the definition of the parameter (stella_dev `sfr_eff.cc`: `StellarAge = Time - TestStarInitAgeYr / yr`).
