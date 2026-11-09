# mechfb_healpix_weights

`GALSF_FB_MECHANICAL_HEALPIX_WEIGHTS` in the mechanical feedback loop: the sky of the source is cut in the 12 NSIDE-1 HEALPix pixels, each
occupied pixel gets the same share, and the cells of a pixel inside the source's own kernel share it equally (weight 1 / (occupied
pixels x cells in the pixel)). One extra pass counts the cells per pixel.

Problem: one star in a density gradient along z (1 Msun cells, n_H from 1 to 1000 cm^-3, scale height 0.5 pc, 100 cm^-3 at the star),
no gravity, no cooling, 1 yr steps, on 1 and 2 ranks.

Checks:
1. a 20 Msun supernova: every cell's ejecta mass is 1 / (occupied pixels x cells in the pixel) to 1e-9, the energy per occupied pixel
   equal to 2e-3, total 1e51 erg to 2e-3 (with the face-area weights a cell's share is off by a factor 21);
2. a wind deposit: mass shares |q| / sum |q| to 1e-9 with q = w (r - mean of w r), r the unit vector to the cell, kicks along q to 1e-4,
   total momentum of the gas zero to 1e-6 of the summed kicks.

Reference: the weights themselves, from the positions in the initial snapshot.
