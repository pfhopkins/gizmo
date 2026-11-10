# sgchem_uv_background

The sgchem_box lattice (n_H = 100 cm^-3 physical, 8000 K, solar metals, network 5) at a = 0.25 (z = 3), 125 kyr, open boundaries (a periodic box has to hold Omega_M of mass),
`ComovingIntegrationOn 1`, run twice: with the default TREECOOL and with every rate in the table set to 1e-40.

- `test_runs_comoving`: both runs reach `TimeMax` (SGChem used to stop at start-up on `ComovingIntegrationOn`).
- `test_uvb_ionises`: the cells hold more electrons with the background (final mean x_e above 1e-3 and above 3 x the run
  without).
- `test_uvb_rates_in_log`: the start-up log names the redshift of the table lookup, and the default TREECOOL (it begins
  with `##` comment lines) is read with its rows: the reader used to find none and the background was zero without a message.

In a comoving run SGChem takes the metagalactic rates of the current redshift from TREECOOL: they are a floor
under the local H, He photoionisation and photoheating rates (the moment solver's or the tree's, whichever is larger), G0
is raised to Gamma_HI / 2.29e-10 and G0_LW to a tenth of it, the redshift reaches the network (CMB term). Non-comoving runs
do not use the background. `ReadIonizeParams` skips blank and `#` lines and stops on a malformed row or an empty table, as starforge_dev.

Inputs: `SUNDIALS_DIR`; the chemistry tables from `SGCHEM_TABLES_DIR`, else installed by `scripts/get_stella_tables.sh`;
`SPCOOL_TABLES` to link an existing `spcool_tables` instead of downloading it.
