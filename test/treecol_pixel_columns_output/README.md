# treecol_pixel_columns_output

`OUTPUT_TREECOL_PIXEL_COLUMNS` (with `GALSF_RESOLVEDISM_TREECOL_PIXELS`): the per-pixel columns of the gravity walk go to
the snapshot as `TreeColumnPixels` for gas, in g cm^-2, shape (N, 3 x pixels), order [column][pixel] with the columns
dust-weighted gas, H2, CO. The column values are the stored 16-bit log10 values (or floats with
`GALSF_RESOLVEDISM_TREECOL_PIXELS_STORED_AS_FLOAT`).

Problem: the lattice of `treecol_pixels` (12^3 cells of 0.1-3 solar metallicity, six young stars,
`TreeColShieldingLength` 8 pc), one step, 1 and 2 ranks.

Check: the field is present with 36 finite non-negative values per cell; the dust-weighted column summed over the pixels
equals the direct sum over the cells (median 1e-2, 99th percentile 5e-2 of the relative difference, the tree-error bars
of `treecol_pixels`).
