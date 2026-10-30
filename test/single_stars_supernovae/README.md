# single_stars_supernovae

IMF-sampled single stars as supernova sources of the shared mechanical feedback loop (`GALSF_FB_MECHANICAL` with
`GALSF_SINGLE_STARS_FROM_IMF` and `GALSF_SINGLE_STARS_EVOLUTION_TABLES`): death time, class, ejecta, energy, yields and
remnant come from the stellar-evolution tables; the coupling is the loop's.

Problem: a 32^3 lattice of 1 Msun cells at n_H = 10 cm^-3 and 100 K, no gravity, no cooling (an adiabatic test: the
budgets do not depend on a cooling module), with four stars moving at 12 km/s, each alone in its region and just past its
table lifetime at Z = 0.014: a core-collapse SN (20 Msun), a failed SN (40 Msun, no energy), a pulsational
pair-instability SN (150 Msun) and a pair-instability SN (320 Msun, no remnant). A few steps.

Checks:
1. one event per star, of the class the table gives (`single_star_events.txt`);
2. each star keeps exactly its table remnant mass; the PISN star is gone; the mass the loop removed equals trigger mass
   minus remnant to 1e-12;
3. the gas gains the ejecta mass (1e-12) and the ejected metal mass (1e-10);
4. the events are identical on 1, 2 and 4 ranks and with 2 threads; gas totals agree to rounding;
5. FIRE metal layout (a second build with `COOLING` and `COOL_METAL_LINES_BY_SPECIES`): every metal slot of the gas gains
   exactly the table SN element ejecta scaled to the ejecta mass (1e-9); slot 0 all table metals C..Zn.

Reference: the tables (classes, remnants, element ejecta) and conservation laws.
