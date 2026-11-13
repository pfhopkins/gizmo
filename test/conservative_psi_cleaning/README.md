# conservative_psi_cleaning (narrow: one cleaning option and its sub-options)

`DIVBCLEANING_CONSERVATIVE_PSI_OVER_CH`: constrained div B cleaning in place of Dedner, with its options `_SYMMETRIC_DIVB`, `_NO_DAMPING_HEAT`, `DIVBCLEANING_CONSERVATIVE_SIGMA` and the test hook `TEST_DIVBCLEANING_ALTERNATING_CLEANING_SPEED`. A blob of B with a divergence in static gas, one build per option. Bars are in the test docstring. Cleaning alone is what is tested, so this one has no chemistry or stars.

Run on the port (2 ranks, 5 builds): 6 passed. Total energy change at the end, in units of the initial E_B: damped +0.0037, pair +0.0021 (largest 0.0024), symmetric +0.0021, alternating +0.0043 (largest 0.0055); E_psi reaches 0.23 E_B. Without damping heat the total falls by 0.32 E_B. The pair bar was changed after the first run (see the test docstring).
