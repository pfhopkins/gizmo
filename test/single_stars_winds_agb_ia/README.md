# single_stars_winds_agb_ia

Stellar winds and AGB deaths of IMF-sampled single stars, from the stellar-evolution tables, through the mass-return mode
(momentum-conserving) of the shared mechanical feedback loop (Type Ia supernovae follow in this directory with
`GALSF_SINGLE_STARS_TYPE_IA`).
- Winds (drawn mass >= 8 Msun, after the pre-main sequence): every star step returns the table's cumulative wind ejecta
  between the last delivery and now, at the table wind speed; the delivery age advances only when the mass coupled.
- AGB death (white-dwarf remnant): ejecta = mass - table remnant at 30 km/s, AGB element ejecta.

Problem: a 24^3 lattice of 1 Msun cells at n_H = 10 cm^-3, 100 K, Z = 0.014, no gravity, no cooling, with a 35 Msun star
1 Myr old and a 3 Msun star just past its lifetime. About 10 kyr.

Checks:
1. the 35 Msun star has returned exactly the table's cumulative wind mass up to its last delivery age (1e-9), and that
   age is its current age to one step;
2. one AGB death of the 3 Msun star, white-dwarf remnant of the table, ejecta = 3 Msun - remnant;
3. total mass (gas + stars) conserved to 1e-12, metal mass rises at least by the event metals; momentum change below 5%
   of the first step's wind momentum (measured 3%: the momentum-conserving coupling balances its kicks only approximately on
   this lattice);
4. the energy gained by gas and stars (kinetic + thermal) is 0.99 to 1.08 of the booked 1/2 M v^2 of the wind returned at
   the table speed plus that of the AGB ejecta (measured 1.045). The coupling is the unchanged upstream momentum solve; its
   overshoot grows with the ejecta mass per cell mass: a 24.6 Msun wind dump at 4830 km/s into cells of 1 / 8 / 64 Msun gives
   1.33 / 1.10 / 1.005 of 1/2 M v^2. It matters only when a source returns more mass at once than the cells around it hold;
   the first step of this test (0.35 Msun, the wind since birth) is such a dump, the later steps are far below it;
5. 1 and 4 ranks give the same wind mass (1e-12) and the same events.

Reference: the tables (cumulative wind ejecta read with h5py at the star's grid node) and conservation laws.
