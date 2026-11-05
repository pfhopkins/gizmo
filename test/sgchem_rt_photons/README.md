# sgchem_rt_photons

What the moment solver absorbs in the ionising bins is what the chemistry consumes. A 40 Msun star in uniform gas at n_H = 100 cm^-3, 0.25 Msun cells, four ionising bins, RSOL 0.1, over 0.3 recombination times, 4 ranks.

- `test_atomic`: (new H+ + recombinations) / (table Q dt) within 0.8-1.2, front inside the box.
- `test_molecular`: all H in H2. H+ forms at more than half the photon rate and the front stays inside the box.
- `test_h0_only_atomic_control`, `test_h2_ledger` (expected failures, open): at RSOL 3e-4 with 1e49 photons/s in the H0 bin alone, atomic and molecular gas stay neutral: the solver's kick empties the bin in cells thick per step before the chemistry reads it. With the softer bins (`test_atomic`) the front starts.
- `test_radiation_pressure_once` (expected failure, open): no IR bin, thick gas, outward momentum added by the forces (force build minus no-force build) against absorbed energy / c: 1.46, bar 0.7-1.1.

Reference: the table's photon rate, alpha_B = 2.6e-13.
