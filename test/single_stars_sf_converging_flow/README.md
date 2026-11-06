# single_stars_sf_converging_flow

`GALSF_SINGLE_STARS_SF_CONVERGING_FLOW_ONLY` adds the convergent-flow requirement (bit 2 of the old `GALSF_SFR_CRITERION`) to
the single-star criterion: no star formation in a cell whose velocity divergence is zero or positive.

Problem: the lattice of `single_stars_sf_criterion` (random velocity field, temperatures on both sides of the Jeans limit),
one step, snapshot with the velocity divergence (`OUTPUT_VORTICITY`).

Check: every cell with a star-formation rate has div v < 0; both signs of div v occur; the forming set is the cells with a
passing Jeans test and div v < 0 (cells within 1e-3 of the Jeans limit may differ).
