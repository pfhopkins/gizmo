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
- `test_stromgren_sphere`: a 35 Msun star 1 Myr old near the centre (Q = 7.7e48 /s, Stromgren radius 12.7 pc, 9 cells).
  Per pixel the front is where the cells' recombination rates (case B, 2.6e-13 cm^3/s, at each cell's density), summed
  outward, use up Q/12. The smallest and largest of the 12 fronts (log columns 10, 11) lie within one cell of the
  Stromgren radius. Without the change there is no photoionisation log.
  With the stamp (cells inside the fronts ionised until the next solve, `PhotoionizedUntil`): the recombination rate of
  the ionised cells is Q to 2% (measured 0.9994-0.99999; log column 7), their volume radius lies within one cell of the
  Stromgren radius, and static gas keeps the same cells at every solve. Without the stamp column 7 is 0.
- `test_ranks_and_order`: 1, 2 and 4 ranks and a permuted particle order give the same fronts, search passes and
  ionised cells (the histogram sums are integers).
