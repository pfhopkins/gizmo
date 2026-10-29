# dust_hu_unit

Unit test of `galaxy_sf/resolvedism_dust_kernels.h`, the Hu+19/23/26 dust microphysics used by `GALSF_RESOLVEDISM_DUST`.

Checks against analytic solutions or fine-step RK4 references:
- MRN effective size and bin weights;
- thermal sputtering e-folding time a/(3nY);
- drag time (Hu+26 eq. 3);
- the exact linear-drag solution (Hu+19 eq. 14);
- supersonic drag with and without betatron, and the nonthermal sputtering integral;
- a shock kick;
- exact logistic growth;
- the 300 K growth cut;
- carbon growth capped at C_tot − C_CO;
- silicate growth capped by the scarcest element pool.

Run with `pytest test/dust_hu_unit`.
