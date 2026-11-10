# single_stars_radiation_pinned

Test hooks of the single-star radiation, observed through the momentum the local radiation-pressure loop offers:
- `TEST_SINGLE_STARS_LUMINOSITY_PINNED`: every star above the mass cut has the bolometric luminosity `TestFixedLbol` [erg/s] from
  birth, no age drift, split: FUV (8-13.6 eV band) 0.5, NUV 0.2, optical 0.3, Lyman-Werner 0.15, no ionising photons.
- `TEST_SINGLE_STARS_RADIATION_OFF`: every band, the bolometric luminosity and the ionising rates are zero.
`TestFixedLbol` takes precedence if both are set.

Problem: as `single_stars_radiation_pressure` (one 40 Msun star, n_H = 100, 1 Msun cells, chemistry on).
Checks: pinned: offered momentum per time = the estimator for the pinned split, 2 %, constant in time 2 %; off: none offered.

Reference: the definitions of the parameters.
