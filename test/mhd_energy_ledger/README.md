# mhd_energy_ledger (narrow: a test-only diagnostic)

`TEST_MHD_ENERGY_LEDGER_OUTPUT`: the MHD energy rows in `ledger.txt` (`OUTPUT_MASS_METAL_ENERGY_LEDGER`). The static magnetised box of
`test/mhd_kernel_volume_energy` (beta about 0.14, adiabatic, 2 ranks, 1.96 Myr), all cells on one timestep; one build with Dedner
cleaning, one with `DIVBCLEANING_CONSERVATIVE_PSI_OVER_CH`. Bars in the test docstring (some changed after the first runs; the
docstring says which and why).

Run on the port: 3 passed. E_B change -1.426567568e48 erg = volume -1.071988210 + kick -0.354521940 + pending -0.000057419.
Dedner: drift +0.498e48, sources -0.074e48, unbooked E_B +0.584e48, rest -0.011e48 (stella_dev on the same box: +0.573, +0.611,
-0.011). Conservative cleaning: drift +2.145e48, of it Powell +1.534e48, rest -0.011e48. With the adaptive steps of the other test
the Dedner rest is -0.036e48 (cells on different steps; outside the MHD terms).
