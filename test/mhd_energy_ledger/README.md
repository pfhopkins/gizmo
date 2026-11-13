# mhd_energy_ledger (narrow: a test-only diagnostic)

`TEST_MHD_ENERGY_LEDGER_OUTPUT`: the MHD energy rows in `ledger.txt` (`OUTPUT_MASS_METAL_ENERGY_LEDGER`). The static magnetised box of
`test/mhd_kernel_volume_energy` (beta about 0.14, adiabatic, 2 ranks, 1.96 Myr), all cells on one fixed step (TimeMax/256 and /512); one build with Dedner
cleaning, one with `DIVBCLEANING_CONSERVATIVE_PSI_OVER_CH`. Bars in the test docstring, set before the run from the ledger's algebra.
