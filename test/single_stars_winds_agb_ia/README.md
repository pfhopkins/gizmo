# single_stars_winds_agb_ia

Stellar winds, AGB deaths and Type Ia supernovae of IMF-sampled single stars, from the stellar-evolution tables, through
the mass-return mode (winds, AGB deaths: momentum-conserving) and the supernova mode (Type Ia: pure thermal) of the shared
mechanical feedback loop.
- Winds (drawn mass >= 8 Msun, after the pre-main sequence): every star step returns the table's cumulative wind ejecta
  between the last delivery and now, at the table wind speed; the delivery age advances only when the mass coupled.
- AGB death (white-dwarf remnant): ejecta = mass - table remnant at 30 km/s, AGB element ejecta.
- Type Ia (`GALSF_SINGLE_STARS_TYPE_IA`): per-step hazard of a white dwarf (t^-1 delay-time distribution normalised to the
  dead-progenitor fraction, keyed random number); the whole mass at 1e51 erg with the table Type Ia yields.

Problem: a 24^3 lattice of 1 Msun cells at n_H = 10 cm^-3, 100 K, Z = 0.014, no gravity, no cooling, with a 35 Msun star
1 Myr old, a 3 Msun star just past its lifetime, and a 1 Gyr old white dwarf. The test-only flag
`TEST_SINGLE_STARS_TYPE_IA_RATE_MULTIPLIER` makes every eligible white dwarf explode at once (the rate itself is checked in `single_stars_type_ia_rate`). About 10 kyr.

Checks:
1. the 35 Msun star has returned exactly the table's cumulative wind mass up to its last delivery age (1e-9), and that
   age is its current age to one step;
2. one AGB death of the 3 Msun star, white-dwarf remnant of the table, ejecta = 3 Msun - remnant;
3. Type Ia of both white dwarfs (the initial one at the first step, the new one after its AGB death): the whole mass at
   1e51 erg, no remnant, ejecta almost all metals;
4. total mass (gas + stars) conserved to 1e-12, metal mass rises at least by the event metals; momentum change below 5%
   of the first step's wind momentum (measured 3%: the momentum-conserving coupling balances its kicks only approximately on
   this lattice);
5. 1 and 4 ranks give the same wind mass (1e-12) and the same events;
6. (`test_wind_energy`, the 35 Msun star alone, as the supernovae's 1e51 erg would hide the wind's 4e49) the energy gained by
   gas and star (kinetic + thermal) closes on the booked 1/2 M v^2 of the wind returned at the table speed to 1e-3 (measured
   0.9995; 1.045 before each cell closed its own share in the star frame).
7. (`test_wind_energy_in_outflowing_gas`) a 150 Msun star's steady wind in 1e7 K gas moving out of it at 0.75 v_w, n_H = 1: run minus a twin with no wind owed (the flow's hydro drift cancels) closes to 2 % on 1/2 M v^2. The weighting pass sums use the ejecta mass, not the star's.

Reference: the tables (cumulative wind ejecta read with h5py at the star's grid node) and conservation laws.
