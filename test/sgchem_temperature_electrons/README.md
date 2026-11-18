# sgchem_temperature_electrons

The stored SGChem temperature (snapshot `Temperature`, the 1e5 K dust gates) must count the particles the network counts:
n_tot = (1 + x_He + x_e - x_H2) n_H with x_e = x_H+ + x_He+ + 2 x_He++ + (x_C - x_CO) + x_D+ + x_Si per H at network 17
(`cxx/chem_rhs_net17.cc`), x_H+ + (x_C - x_CO) + x_Si at network 5. It counted x_H+ only: ionised He gas came out 7 % hot.

Problem: the `sgchem_box` lattice at n_H = 1 cm^-3, network 17, solar metals without depletion, starting fully ionised
with x_He+ = x_He++ = 0.05 (`SGCHEM_KEEP_IC_ABUNDANCES`), about 5e4 K, 1 kyr on 2 ranks.

- `test_temperature_counts_all_electrons`: Temperature = (2/3) e / (k_B n_tot) to 1e-3 at the snapshots after the start, and
  ElectronAbundance = x_e to 1e-6 (Ne counts the same electrons).
- `test_mass_and_energy`: gas mass constant to 1e-12; thermal energy change = ledger chemistry rows + floor to 1e-6.

Inputs: `SUNDIALS_DIR`; chemistry tables from `SGCHEM_TABLES_DIR` (else `scripts/get_stella_tables.sh`).
