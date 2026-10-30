# photoion_per_direction

Per-direction photoionisation around single stars (`GALSF_FB_PHOTOIONIZATION_PER_DIRECTION`). Design note:
design_photoionisation_pixelised.md; reference: stella_dev galaxy_sf/resolvedism_photoion.cc.

Problem: a 32^3 lattice of 1 Msun cells at n_H = 10 cm^-3, no gravity, no cooling (the gas stays static), with single
stars from the stellar-evolution tables.

Checks:
- `test_ionising_stars_in_the_photoionisation_bin`: all ionising stars share one cadence. With MaxSizeTimestep 20 kyr and
  PhotoionMaxTimestep 1 kyr, a 35 Msun star 1 Myr old steps at most 1 kyr, while the gas keeps its coarse step. A 35 Msun
  star 1000 yr old steps onto its ionising onset (15 kyr) and then keeps to the same bin. Without the change both stars
  step with the gas.
