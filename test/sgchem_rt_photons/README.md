# sgchem_rt_photons

What the moment solver absorbs in the ionising bins is what the chemistry consumes. A 40 Msun star in uniform gas at n_H = 100 cm^-3, 0.25 Msun cells, four ionising bins, RSOL 0.1, over 0.3 recombination times, 4 ranks.

- `test_atomic`: (new H+ + recombinations) / (table Q dt) within 0.8-1.2, front inside the box.
- `test_molecular`: all H in H2. H+ forms at more than half the photon rate and the front stays inside the box.
- `test_h0_only_atomic`, `test_h2_ledger`: RSOL 3e-4, 1e49 photons/s in the H0 bin alone, no background G0: ledger with the photons added to the bins, 0.8-1.2. H2 with `SGCHEM_RT_ABSORBERS_FOLLOW_THE_NETWORK`.
- `test_h2_as_stella_dev_reported`: production opacity on H2, printed.
- `test_radiation_pressure_once`: no IR bin, thick gas, force build minus no-force build against absorbed energy / c, 0.7-1.1. A visible failure until it holds.

Reference: the table's photon rate, alpha_B = 2.6e-13.
