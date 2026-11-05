# single_stars_winds_agb_ia

Stellar winds of IMF-sampled single stars, from the stellar-evolution tables, through the mass-return mode
(momentum-conserving) of the shared mechanical feedback loop (AGB deaths and Type Ia supernovae follow in this
directory).
- Winds (drawn mass >= 8 Msun, after the pre-main sequence): every star step returns the table's cumulative wind ejecta
  between the last delivery and now, at the table wind speed; the delivery age advances only when the mass coupled.

Problem: a 24^3 lattice of 1 Msun cells at n_H = 10 cm^-3, 100 K, Z = 0.014, no gravity, no cooling, with a 35 Msun star
1 Myr old. About 10 kyr.

Checks:
1. the 35 Msun star has returned exactly the table's cumulative wind mass up to its last delivery age (1e-9), and that
   age is its current age to one step;
2. total mass (gas + stars) conserved to 1e-12, metal mass rises; momentum change below 5%
   of the first step's wind momentum (measured 3%: the momentum-conserving coupling balances its kicks only approximately on
   this lattice);
3. 1 and 4 ranks give the same wind mass (1e-12).
- the energy gained by gas and star closes to 1e-3 on 1/2 M v^2 of the wind returned: each cell closes its share in the star frame.

Reference: the tables (cumulative wind ejecta read with h5py at the star's grid node) and conservation laws.
