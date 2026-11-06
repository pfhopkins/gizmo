# treecol_shells_helium

Helium in the TreeRay photoionisation (`GALSF_RESOLVEDISM_TREECOL_PIXELS_RADIAL_SHELLS_PHOTOION` at `CHEMISTRYNETWORK=17`). Design: two bands split at 24.6 eV, absorbers HI and HeI.

| test | checks |
|---|---|
| `test_two_source_bands` | the two source bands add up to the table's photon rate and energy (1e-10), band 1 = table, mean energy inside the band (cool stars at the 24.6 eV edge), root node = sum (1e-12) |
| `test_helium_absorbers` | node species HI and HeI per cell and root node (1e-12) |
| `test_band_sweeps` | photon flux per band against a direct sweep (band 0 through HI, band 1 through HI + HeI; median 20 %, tree error at tau ~ 1); band mean energies inside the bands |
| `test_cross_sections_as_blackbody_integrals` | the walk's constants = stella_dev's band means at 4e4 K (1e-3) |
| `test_helium_ionised` | StarBench m10: x(He+) / ABHE at least 0.9 where x(H+) > 0.9 |
| `test_helium_ledger` | StarBench m10: helium recombinations at most 1.3 Q1, He++ below 1e-3 ABHE |
| `test_helium_fills_the_h_zone` | StarBench m10: R_He / R_IF in 0.85-1.05, the analytic nested zones |
| `test_total_ledger` | xfail, bar missed and kept: recombinations 0.85-1.7 Q (measured 0.73-1.35; stella_dev net 17 0.59-1.06) |
| `test_against_stella_dev` | xfail, bar missed and kept: within 4 points of R_IF and 0.10 of R_He / R_IF of stella_dev (measured -29.1 % vs -35.3 %, 0.94 vs 0.73); cause not found, not the band split |
