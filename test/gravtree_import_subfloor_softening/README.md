# gravtree_import_subfloor_softening

With `ADAPTIVE_GRAVSOFT_FORALL` the gravity walk takes, per target, the larger of its kernel radius and the fixed softening of
its type (`grav_target_select_soft_and_zeta`), and with adaptive softening opens every node within max(that softening, the
node's) + len sqrt(3)/2 (`NEIGHBORS_MUST_BE_COMPUTED_EXPLICITLY_IN_FORCETREE`). The rank that packs the subtrees
other ranks import must assume the same softening; otherwise the walk opens imported nodes that arrived without children and
the run stops. Stella runs have no gas kernel floor (`MinGasKernelRadiusFractional 0`), so kernels below `SofteningGas` occur
(test/sn_driven_box at 0.1 Msun on 32 ranks stopped this way).

Problem: a cold 16^3 gas lattice, kernels about 2 cells, fixed gas softening 5 cells, 4 steps, 1 and 8 ranks.

- `test_import_carries_what_the_walk_opens`: 8 ranks complete without an import shortfall; start-up gravity agrees with 1 rank
  (99th percentile below 1 %, maximum below 3 % of the median |a|).
- `test_mass_and_energy[1, 8]`: mass constant to 1e-12; E_kin + E_th + E_pot changes by less than 1e-3 of |E_pot|.
