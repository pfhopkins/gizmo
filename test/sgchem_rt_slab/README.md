# sgchem_rt_slab

A slab lit by one star through the moment solver, the initial conditions of `treecol_pixels` `test_slab_shielding`: n_H ~ 1e3, N_H ~ 1e22 through it, 40 Msun star 3 pc in front of the face, 1 Msun cells, 4 ranks. Long on a loaded login node (80 minutes): run it under a job.

- `test_bins_attenuate`: field on the axis against the analytic point source with the chemistry's dust law, 20 %, tau < 3. Open (expected failure): 1.10 in the face slice, then near 0.5.
- `test_attenuation_with_depth`: flat ratio to analytic for 1 < tau < 6: the dust acts once, in the bins.
- `test_shielding`: deep cells hold over 30 x the H2 of the lit face and x_H2 > 1e-2 after 0.15 Myr.
