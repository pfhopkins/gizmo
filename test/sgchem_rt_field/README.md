# sgchem_rt_field

SGChem reads G0, G0_LW and G0_dust from the moment solver's bins. A uniform molecular box at n_H = 100 cm^-3 with its bins seeded in the initial snapshot through `PhotonEnergy`, against the same box without radiation transport and the same field through `SGChemG0`. RSOL 1e-5 keeps the absorption over the run negligible. H2 fraction and temperature agree within 1 %.

- `test_habing_gate`: G0 = 10 in a Draine-shaped field, split over the five disjoint bins.
- `test_floor`: empty bins and `SGChemBackgroundG0` give the uniform path's floor.

`TreeColShieldingLength` 1e-6 keeps the columns empty, so the comparison holds the field alone.
