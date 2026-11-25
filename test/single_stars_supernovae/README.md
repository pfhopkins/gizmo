# single_stars_supernovae

IMF-sampled single stars as supernova sources of the shared mechanical feedback loop (`GALSF_FB_MECHANICAL` with
`GALSF_SINGLE_STARS_FROM_IMF` and `GALSF_SINGLE_STARS_EVOLUTION_TABLES`): death time, class, ejecta, energy, yields and
remnant come from the stellar-evolution tables; the coupling is the loop's, with the pure-thermal supernova variant
(`GALSF_FB_MECHANICAL_SNE_THERMAL_ONLY`, on by default for single stars): the whole explosion energy as heat over the same
weights as the ejecta mass, no radial ejecta momentum, the ejecta carrying the star's velocity.

Problem: a 32^3 lattice of 1 Msun cells at n_H = 10 cm^-3 and 100 K, no gravity, no cooling (an adiabatic test: the
budgets do not depend on a cooling module), with four stars moving at 12 km/s, each alone in its region and just past its
table lifetime at Z = 0.014: a core-collapse SN (20 Msun), a failed SN (40 Msun, no energy, no ejecta: its table ejecta are 0), a
pulsational pair-instability SN (150 Msun) and a pair-instability SN (320 Msun, no remnant). A few steps. The initial
conditions count each star's winds as returned up to the end of its life (`SingleStarBirthMass` = its mass plus the table's wind), so the deaths
come at once.

Checks:
1. one event per star, of the class the table gives (`single_star_events.txt`);
2. each star keeps exactly its remnant mass (a fraction of its mass at death, item 11); the PISN star is gone; the mass the
   loop removed equals trigger mass minus remnant to 1e-12;
3. the gas gains the ejecta mass (1e-12) and the ejected metal mass (1e-10);
4. the total energy rises by the explosion energies to 1e-3 (measured -1.9e-5: kinetic energy of the ejecta, 2.3e-4 of
   the total, is partly lost when they mix with gas at rest); with the mechanical coupling it rose by 1.10 times them;
5. total momentum is conserved (the ejecta carry the stars' velocity; nothing radial is added);
6. the events are identical on 1, 2 and 4 ranks and with 2 threads, and the gas mass agrees to rounding. The gas energy agrees
   to 1e-5 only without cell splitting (`test_ranks_and_threads_without_merge_split`, built with `PREVENT_PARTICLE_MERGE_SPLIT`:
   2e-16 between 1 and 2 ranks at every output). With the default splitting `test_ranks_and_threads` is an expected failure
   on the energy bar alone (1e-4): a split places the daughters at a fraction of the distance to the nearest neighbour on the
   same task, so daughters of cells near a task boundary sit elsewhere on another number of tasks (equal split counts, kernel
   lengths of the daughters up to 8 % apart at the first split);
7. FIRE metal layout (a second build with `COOLING` and `COOL_METAL_LINES_BY_SPECIES`): every metal slot of the gas gains
   exactly the table SN element ejecta scaled to the ejecta mass (1e-9); slot 0 all table metals C..Zn.
8. the death step (`test_death_step_first_step_past_lifetime`), caps 10 and 50 kyr: stars 37 kyr from death, a long largest step. A single star's step is `MaxStellarTimestep` (default 50 kyr; 1e4 yr in the production params) and no less: every step before the first death stays in that bin, and each star dies at its first step past its table death time, at most one cap late. With the former rule of 1/50 of the time to death: 377 steps before the first death, down to 0.47 kyr.
   A death with winds owed (`test_death_with_winds_owed`, the last 5 % of life not returned): the owed winds leave in the first step as one wind delivery (mass against the table to 1e-9), the deaths fire on the next step with the table's fractions of the mass left; whole mass in the gas (1e-12). The failed SN gives nothing but its winds.
9. stars started below their table remnant mass: no star grows at death and the total mass is conserved (1e-12).
10. in 1e7 K gas the total energy still rises by the explosion energies to 1e-3: the ejecta carry no ambient `u`. Before the death ejecta became fractions (item 11) the failed SN gave mass with no energy, which checked that the dilution of `u` is applied at once (measured then 1 - 1.3e-8 after the deposit, 1 + 3.9e-5 at the end of the run); it now ejects nothing.

Reference: the tables (classes, remnants, element ejecta) and conservation laws.

11. `test_death_ejecta_fractions` (stars at 1.0 and 0.3 of their drawn mass, no winds owed): each grid star ejects
    M sum e / (rem + sum e) and metals M sum_{C..} e / (rem + sum e) with the table's SN element ejecta e and remnant rem
    (PISN: rem 0), to 1e-10; 0 <= remnant <= M; the failed SN (table ejecta 0) ejects nothing (stella_dev F1, 8837629e8).
12. `test_sn_energies` (the four stars plus the table's first ECSN mass): every SN 1e51 erg by default (failed SN 0); with
    `RESOLVED_ISM_VARIABLE_SN_ENERGY` PISN 1e52 erg and ECSN 5e50 erg (Uli 2026-11-19; stella_dev 97b488f26). Replaces the He-core
    PISN test (the fit is removed).
