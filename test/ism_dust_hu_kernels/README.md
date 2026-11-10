# ism_dust_hu_kernels

Unit test of `src/solids/ism_dust_hu_kernels.h`, the dust microphysics of Hu et al. 2019, 2023, 2026 (no GIZMO state, host and device).

Checks against analytic solutions or fine-step RK4 references:
- MRN effective size and bin weights;
- thermal sputtering e-folding time a/(3nY);
- drag time (Hu+26 eq. 3);
- the exact linear-drag solution (Hu+19 eq. 14);
- supersonic drag with and without betatron, and the nonthermal sputtering integral;
- a shock kick;
- exact logistic growth;
- the 300 K growth cut;
- carbon growth capped at C_tot - C_CO;
- silicate growth capped by the scarcest element pool.

Compiled with plain -O2 and with the production flags. No GIZMO build or simulation is needed.
