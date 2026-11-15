# stella_merge_kinetic_veto

A gas merge keeps momentum, so the pair's centre-of-mass kinetic energy (1/2) mu |dv|^2 leaves the kinetic budget; the merge
returns it as heat (`merge_particles_ij`). Where that energy dominates, the merge acts as an unresolved shock whose heat
cooling then radiates. `MERGE_SPLIT_LIMIT_KINETIC_DISSIPATION` refuses a merge when KE_com / (KE_com + E_light) exceeds
`MERGE_SPLIT_MAX_KINETIC_DISSIPATION_FRACTION` (0.5), E_light the thermal plus magnetic energy of the lighter cell. In Stella
builds (`GALSF_SINGLE_STARS_FROM_IMF`) it is on for every gas merge, as on stella_dev (fix/stella-merge-energy 530136337);
elsewhere it acts on spawned cells only, as before. `MERGE_SPLIT_ALLOW_KINETIC_DISSIPATION` switches it off.

Problem: the Stella cloud flags (`scripts/stella/Config_stella_cloud.sh`) plus `TEST_GAS_ENERGY_LEDGER_OUTPUT`; 16^3 lattice of
1 Msun cells at n_H = 10 cm^-3, 100 K, at rest (the `single_stars_metals` lattice), merge threshold 0.49 Msun (IC header);
two cells of 0.2 Msun, one at rest, one moving at 10 km/s (KE_com / (KE_com + E_th) close to 1); 1e-4 time units, 2 ranks.

Checks (bars written before the runs):
- default: the cell at rest merges, the moving cell does not;
- `MERGE_SPLIT_ALLOW_KINETIC_DISSIPATION`: both merge; against the same run with the second light cell at rest, the gas
  energy (E_kin + E_th, energy_ledger.txt) at the first report is higher by that cell's initial kinetic energy (10 Msun
  (km/s)^2) to 5% of KE_com = 8.33 (a merge dropping KE_com would leave 1.67). The merge happens before the first ledger
  report, so the ledger's own merge_split row cannot show it.
