# treecol_nongas

The TreeCol column densities (`RT_USE_TREECOL_FOR_NH`, six angular bins) of a gas cell must not change when a gasless star
sits among the gas: stella_dev's walk kept the gas mass of the previous leaf for star, dark-matter and sink leaves and added it
to the bins (fixed there).

Problem: 8^3 gas cells (1e-4 code masses), one star of 1e-10 of a cell's mass next to a cell in the second run. The bins of the
first ten gas cells at step 0 are read from the `GIZMO_TREECOL_DIAG` print.

Check: no bin of those cells changes by more than 1e-5 of the cell's largest bin.
