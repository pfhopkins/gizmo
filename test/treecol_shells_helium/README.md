# treecol_shells_helium

Helium in the TreeRay photoionisation (`GALSF_RESOLVEDISM_TREECOL_PIXELS_RADIAL_SHELLS_PHOTOION` at `CHEMISTRYNETWORK=17`). Design: two bands split at 24.6 eV, absorbers HI and HeI.

| test | checks |
|---|---|
| `test_two_source_bands` | the two source bands add up to the table's photon rate and energy (1e-10), band 1 = table, mean energy inside the band (cool stars at the 24.6 eV edge), root node = sum (1e-12) |
| `test_helium_absorbers` | node species HI and HeI per cell and root node (1e-12) |
