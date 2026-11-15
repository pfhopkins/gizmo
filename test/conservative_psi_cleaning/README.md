# conservative_psi_cleaning (narrow: one cleaning option and its sub-options)

`DIVBCLEANING_CONSERVATIVE_PSI_OVER_CH`: constrained div B cleaning in place of Dedner, with its options `_SYMMETRIC_DIVB`, `_NO_DAMPING_HEAT`, `DIVBCLEANING_CONSERVATIVE_SIGMA` and the test hook `TEST_DIVBCLEANING_ALTERNATING_CLEANING_SPEED`. A blob of B with a divergence in static gas, one build per option. Bars are in the test docstring. Cleaning alone is what is tested, so this one has no chemistry or stars.

The pair checks (SIGMA 0) close the total energy against the MHD energy ledger (`TEST_MHD_ENERGY_LEDGER_OUTPUT`, booking only): what is left after the unbalanced terms (Powell, closure limiter, floor) and the unbooked kernel-volume E_B must be time-integration error, converging between CourantFac 0.1 and 0.05; bar set before the run, derivation in the docstring.
