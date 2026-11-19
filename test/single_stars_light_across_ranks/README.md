# single_stars_light_across_ranks

A star that starts to shine or goes dark between tree builds must light the gas the same on every rank. The light switch
refreshes the node luminosities without a build; other ranks' copies of a rank's nodes (imported subtrees) are refreshed
only by the exchange in a build.

Problem: 16^3 lattice of 1 Msun cells at n_H = 1 (thin), SGChem network 5, TreeCol band fluxes from the gravity walk,
radiation pressure per cell, 2 kyr, no tree or domain rebuild after the start (`DomainBuild_ActiveFraction`,
`TreeRebuild_ActiveFraction` 1e4). `switch`: a 40 Msun star ending its pre-main sequence and one reaching the end of its life,
both near 1 kyr, at (0.3, 0.3, 0.3) L and (0.7, 0.7, 0.7) L. `control`: one 40 Msun star shining throughout. 1, 2 and 4 ranks.

- `test_light_switch_seen_across_ranks[2, 4]`: G0_FUV of the final snapshot (cells above 1e-3 of the maximum) and the
  radiation-pressure momentum per step (MomWinds.txt) on N ranks against 1 rank. Bar, set before the first run:
  |X_N / X_1 - 1| <= max(2 x the control's largest deviation, 1 %), the 1 % being the node refresh tolerance.
- `test_mass_and_energy[switch, control]`: mass constant to 1e-12; gas energy change minus the ledger's chemistry and floor
  rows below 1e-3 of the initial thermal energy.

Inputs: `SUNDIALS_DIR`; tables from `GIZMO_STELLA_TABLES_DIR` (else `scripts/get_stella_tables.sh`).
- `test_faint_star_drift_seen` (1, 2, 4 ranks): a bright 40 Msun star and a group of 8 faint stars at the opposite corner, each faint
  star's 6-13.6 eV luminosity changing by at least 4 % over the run (mass and age picked from the tables, change below 1 % of
  the root's); against a run that rebuilds the tree every step, G0_FUV agrees to 2 % in the cells beyond 2.5 cells from the
  group that the group lights most (there the walk uses the group's node). The refresh tolerance used to be 1 % of the root's
  luminosity, so faint stars never triggered a refresh.
