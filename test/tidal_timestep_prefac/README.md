# tidal_timestep_prefac (narrow)

`TIDAL_TIMESTEP_PREFAC`  in the role of a tidal prefactor for gas. Setup of
`test/single_stars_gas_timestep`, 2 ranks. With a factor f = 1e-3 the shortest gas step lies in (1/2, 1] of f sqrt(eta r^3 / (G M))
at the nearest cell; built without it the step is well above that. A separate prefactor for non-gas particles has no
counterpart (one factor for all).
