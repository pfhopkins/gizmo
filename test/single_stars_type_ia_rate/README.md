# single_stars_type_ia_rate

Rate of Type Ia supernovae of single white dwarfs (`GALSF_SINGLE_STARS_TYPE_IA`): per star and step the hazard is
(IA_DTD_NORM / t) * M_drawn * dt * IA_F_WD / f_dead(t), with f_dead the table's dead white-dwarf-progenitor fraction at age t and
metallicity (the constants and the form of stella_dev's `resolvedism_fb.cc`), compared with a keyed random number.

Problem: a 24^3 lattice of 1 Msun cells at n_H = 10 cm^-3, 100 K, Z = 0.014, no gravity, no cooling, with 3000 white dwarfs of drawn
mass 3 Msun, 1 Gyr old, at random positions. The test-only flag `TEST_SINGLE_STARS_TYPE_IA_RATE_MULTIPLIER` (1e6 here) scales the
hazard so that about 7% of them explode in the run (about 20 kyr).

Check: the number of Type Ia events by half time and by the end agrees with 3000 * (1 - exp(-H)), H the time integral of the hazard
above with f_dead from stella_dev (`unit/lookup_reference_ia.txt` of the stellar_evolution_tables test), within 4 binomial sigma;
every white dwarf explodes at most once.

Reference: the hazard of stella_dev (formula and constants), its dead-progenitor fraction, and the binomial statistics of independent stars.
