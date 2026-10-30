# single_stars_collect_gas

A new single star (`GALSF_SINGLE_STARS_COLLECT_GAS_FROM_NEIGHBORS`) is spawned with zero mass and then collects its drawn
IMF mass from the gas around it, in passes with a growing radius (LYRA, Gutcke et al. 2021; radius rule, 50% per-pass
cap, mass floor, 1% acceptance and 10-pass limit as in the reference implementation). Donors shared by several new stars
split what they can give in proportion to the requests; every transfer is a whole multiple of one power-of-two mass
quantum, so all sums are integer sums: mass is conserved exactly and the result does not depend on rank count, thread
count or particle order. A star that does not reach its drawn mass is a failed draw and every quantum goes back.

Problem: a regular lattice of 1 Msun cells (24^3, or 4^3 for the failed-draw cases), uniform density and temperature,
metallicity varying across the box (gradient in x, modulation in y), bulk velocity plus a shear v_y(x). The lattice is
written as snapshot 0 and the run starts from it with start flag 2 (initial conditions proper do not carry metallicity).
Stars of given drawn masses are placed at given cells at the first step by the test-only flag
`TEST_SINGLE_STARS_BIRTH_AT_GIVEN_CELLS`. The main build switches merging of light cells off
(`MERGESPLIT_HARDCODE_MIN_MASS`), so the snapshot after one step shows the collection itself.

Checks:
1. shared donors (two stars one cell apart; five stars of 8-60 Msun within two cells): every star within 1% of its drawn
   mass; total mass exactly conserved (exact rational sums); donors keep their metallicity bitwise; metal mass conserved
   to the rounding of the stars' stored metallicity (bound: sum over stars of M_star x ulp(Z_star)); star metallicity
   inside the range the gas spans; momentum conserved to 1e-12; every donor keeps mass;
2. unreachable draw (100 Msun from a box of 64 Msun): the draw fails, no star exists, every gas mass and metallicity is
   bitwise its initial value; with a second, reachable star in the same box only that star forms, totals exact;
3. the five-star case gives bitwise the same masses, metallicities and velocities (by ID) on 1, 2 and 4 ranks, with
   2 ranks x 2 threads, and with the lattice stored in a permuted order;
4. donors on other ranks: three stars at the box edges collect across the periodic wrap; on 4 ranks every star takes
   mass from donors on another rank (counted in `single_star_births.txt`, column 11); the result equals the 1-rank result
   bitwise;
5. with the default merging of light cells (second build), the five-star case still forms all stars, and mass and metal
   mass hold to the merge's rounding.

Reference: conservation laws (exact rational sums of the snapshot values) and identity between decompositions. The
agreement with the reference implementation for a star whose donors are not shared is checked separately with the
reference code (not part of this test, which runs GIZMO only).
