# sgchem_rt_slab

A slab lit by one star through the moment solver, the initial conditions of `treecol_pixels` `test_slab_shielding`: n_H ~ 1e3, N_H ~ 1e22 through it, 40 Msun star 3 pc in front of the face, 1 Msun cells, 4 ranks. Long on a loaded login node (80 minutes): run it under a job.

- `test_bins_attenuate`: field on the axis against the analytic point source with the chemistry's dust law, 20 %, tau < 3. Fails so far.
- `test_attenuation_with_depth`: flat ratio to analytic for 1 < tau < 6: the dust acts once, in the bins.
- `test_shielding`: deep cells hold over 30 x the H2 of the lit face and x_H2 > 1e-2 after 0.15 Myr.
- `test_h2_transition_matches_treecol`: the same slab through the TreeCol walk (treecol_pixels, 75 steps of 0.002 to the same time): the axis x_H2 in four depth bins agrees within a factor 2. Both paths apply the H2 self-shielding of Draine & Bertoldi (1996) once, in the chemistry, with the pixel H2 columns; the LW bin's opacity is dust only.
