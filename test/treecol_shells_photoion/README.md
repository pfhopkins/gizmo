# treecol_shells_photoion

StarBench D-type expansion of an HII region (Bisbas et al. 2015) with the TreeRay photoionisation
(`GALSF_RESOLVEDISM_TREECOL_PIXELS_RADIAL_SHELLS_PHOTOION`): the photons go through the gravity walk on per-direction HI
shells and enter SGChem as rates. Conditions as `test/photoion_starbench`, the same lattice (57 pc, n_H ~ 100, one star at
the centre), with `TEST_TREECOL_SHELLS_PHOTOION_FIXED_RATE_TWO_TEMPERATURE_GAS`: Q = 1e49 /s, T(x) = T_n + (1e4 K - T_n) x.
Gravity stays on (the walk carries the rays), so the shell self-collapses after about 2.5 Myr: the grading stops there.

| test | checks |
|---|---|
| `test_photon_ledger` | sum alpha_B x^2 n_H^2 V / Q in 0.85-1.7 in every D-type snapshot (stella_dev on the same estimator: net 5 0.9-1.3 Q, net 17 1.0-1.7 Q) |
| `test_starbench_coarse_rung` | m10 (40^3): median R_IF / R_StarBench - 1 for 0.2 < t < 2.5 Myr within 4 points of stella_dev's -25.0 % (net 5, one band, NSIDE 2; its net 17 ladder run -22.3 %) |
