# sgchem_rt_photons

What the moment solver absorbs in the ionising bins is what the chemistry consumes. A 40 Msun star in uniform gas at n_H = 100 cm^-3, 0.25 Msun cells, four ionising bins, RSOL 0.1, over 0.3 recombination times, 4 ranks.

- `test_atomic`: (new H+ + recombinations) / (table Q dt) within 0.8-1.2, front inside the box.
- `test_molecular`: all H in H2. H+ forms at more than half the photon rate and the front stays inside the box.

Reference: the table's photon rate, alpha_B = 2.6e-13.
