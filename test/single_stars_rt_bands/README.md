# single_stars_rt_bands

IMF-sampled single stars source the moment solver's bins from the stellar tables (`GALSF_SINGLE_STARS_RT_TABLE_BANDS`). Two 40 Msun main-sequence stars at opposite corners of a thin periodic box and one dead star, 2 ranks, so every photon stays in the bins.

- `test_band_energy`: energy per bin over the box, as a rate between two snapshots, equals the sum of table luminosities x c_red / c (2 %). Bins: 13.6-24.6, 24.6-54.4, 54.4-70, >70, 8-11.2, 11.2-13.6, 3.444-6, 0.4133-3.444, 6-8 eV. Their sum is below L_bol. Photons in the ionising bins at the solver's mean photon energies against the table's rate: 0.7-1.3.
- `test_single_ionising_bin`: with one ionising bin the bin holds the table's photon rate at the bin's mean photon energy (2 %).

Reference: the stella-tables file through `treecol_pixels/stella_table_bands.py`. Uses his cooling module, so no chemistry library is needed. The first injection arrives two steps after t = 0.

- `test_excluded_channels`: `GALSF_FB_RADPRESSURE_SINGLE_STAR_BANDS`, `GALSF_FB_FIRE_RT_HIIHEATING` and `GALSF_FB_PHOTOIONIZATION_PER_DIRECTION` do not build with this flag: radiation pressure and photoionisation are counted once, in the bins.
