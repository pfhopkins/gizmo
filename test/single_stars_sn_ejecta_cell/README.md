# single_stars_sn_ejecta_cell

`GALSF_SINGLE_STARS_SN_EJECTA_CELL=f`: a supernova whose kernel holds less gas than f times its ejecta mass becomes one new gas cell (ejecta mass, thermal explosion energy, yields, star's velocity) at the next merge/split point instead of a kernel deposit. The split routine then refines the cell and hydro drives the blast. Failed SN, direct-collapse BH, AGB deaths and Type Ia stay in the loop.

Problem: 16^3 lattice of 1 Msun cells at n_H = 10 with SGChem network 5, tables, 2 ranks; four stars past their lifetime. Only the 150 and 320 Msun stars eject more than their kernel holds. Bars are in the test's docstring.
