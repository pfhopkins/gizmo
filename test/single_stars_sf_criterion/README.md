# single_stars_sf_criterion

The star-formation criterion of IMF-sampled single stars (`GALSF_SINGLE_STARS_SF_CRITERION`, which implies the gas
collection and the star record), as the reference implementation: the density threshold; no star formation
in a cell marked ionised; Jeans mass pi^(5/2)/6 c_s^3 / (G^(3/2) rho^(1/2)) below `FacSfThreshMJ` x `DesNumNgb` x m, with
the physical density and the physical sound speed from the gas pressure; rate m / t_sf. After a birth the parent cell
forms no star until its density is recomputed, and then for `SFCooldownFreeFallFraction` x its free-fall time.

Problem: a 16^3 lattice of 1 Msun cells at n_H ~ 1e4 cm^-3 with a random velocity field, a uniform magnetic field and
temperatures spread so that the Jeans mass lies on either side of the limit (no gravity).

Checks:
1. one step at negligible efficiency: a cell has a non-zero SF rate exactly where the criterion, recomputed from the
   snapshot, says so (cells within 0.1% of the limit may differ: the snapshot holds the end-of-step energy); about half
   the cells pass;
2. the same with the density threshold above every cell: no cell forms stars although the Jeans test would pass;
3. 16 steps at high efficiency (several hundred births): a cell's next birth comes at least
   `SFCooldownFreeFallFraction` x its free-fall time later;
4. negative control: the same run without the cool-down has repeat births sooner than that interval.

Reference: the criterion, recomputed from the snapshot. Not tested here: the ionised-cell case, which needs a
photoionisation module that marks cells (not combined with single stars yet).
