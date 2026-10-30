# single_stars_supernovae

IMF-sampled single stars as supernova sources of the shared mechanical feedback loop (`GALSF_FB_MECHANICAL` with
`GALSF_SINGLE_STARS_FROM_IMF` and `GALSF_SINGLE_STARS_EVOLUTION_TABLES`): death time, class, ejecta, energy, yields and
remnant come from the stellar-evolution tables; the coupling is the loop's, with the pure-thermal supernova variant
(`GALSF_FB_MECHANICAL_SNE_THERMAL_ONLY`, on by default for single stars): the whole explosion energy as heat over the same
weights as the ejecta mass, no radial ejecta momentum, the ejecta carrying the star's velocity.

Problem: a 32^3 lattice of 1 Msun cells at n_H = 10 cm^-3 and 100 K, no gravity, no cooling (an adiabatic test: the
budgets do not depend on a cooling module), with four stars moving at 12 km/s, each alone in its region and just past its
table lifetime at Z = 0.014: a core-collapse SN (20 Msun), a failed SN (40 Msun, no energy, the wind mass as ejecta), a
pulsational pair-instability SN (150 Msun) and a pair-instability SN (320 Msun, no remnant). A few steps. The initial
conditions count each star's winds as returned up to the end of its life (`SingleStarLastWindLogAge`), so the deaths
come at once.

Checks:
1. one event per star, of the class the table gives (`single_star_events.txt`);
2. each star keeps exactly its table remnant mass; the PISN star is gone; the mass the loop removed equals trigger mass
   minus remnant to 1e-12;
3. the gas gains the ejecta mass (1e-12) and the ejected metal mass (1e-10);
4. the total energy rises by the explosion energies to 1e-3 (measured -1.9e-5: kinetic energy of the ejecta, 2.3e-4 of
   the total, is partly lost when they mix with gas at rest); with the mechanical coupling it rose by 1.10 times them;
5. total momentum is conserved (the ejecta carry the stars' velocity; nothing radial is added);
6. the events are identical on 1, 2 and 4 ranks and with 2 threads; gas totals agree to rounding;
7. FIRE metal layout (a second build with `COOLING` and `COOL_METAL_LINES_BY_SPECIES`): every metal slot of the gas gains
   exactly the table SN element ejecta scaled to the ejecta mass (1e-9); slot 0 all table metals C..Zn.
- stars started below their table remnant mass: no star grows at death and the total mass is conserved (1e-12).
- in 1e7 K gas the total energy still rises by the explosion energies to 1e-3: the ejecta carry no ambient `u`.

Reference: the tables (classes, remnants, element ejecta) and conservation laws.
