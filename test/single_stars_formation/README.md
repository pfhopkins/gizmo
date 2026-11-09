# single_stars_formation

Star formation of IMF-sampled single stars (`GALSF_SINGLE_STARS_COLLECT_GAS_FROM_NEIGHBORS`). In the star-formation
step the IMF mass is drawn first (keyed draw from the mass-weighted Kroupa IMF), the acceptance probability is
(m / M_drawn)(1 - e^-p), and an accepted cell spawns a zero-mass single star (`spawn_star_from_gas`) that then collects
its drawn mass from the gas around it. With this acceptance the formed stars follow the number-weighted Kroupa IMF
(LYRA, Gutcke et al. 2021). The star-formation log counts the collected mass.

Problem: a 16^3 lattice of 1 Msun cells at n_H ~ 1e4 cm^-3, about 100 K, with a random velocity field (no gravity, no
cooling). Every cell is above the density threshold, which is the only criterion here; high efficiency.

Checks:
1. the stars of the first star-forming step (IDs, drawn and collected masses, radii, passes) are bitwise the same on
   1, 2 and 4 ranks;
2. 10 steps (several hundred births): every star within 1% of its drawn mass; the drawn masses of the formed stars pass
   a Kolmogorov-Smirnov test against the number-weighted Kroupa IMF on [0.08, 350] Msun.

Reference: the IMF (analytic) and identity between decompositions.
