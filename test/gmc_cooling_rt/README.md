# GMC cooling test

This test initializes a $$2\times10^4 M_\odot$$ giant molecular cloud at $1M_\odot$ resolution, runs it for about a crossing time, and verifies that the temperature-density statistics are in agreement with a pre-run setup, within a reasonable tolerance. This is the version that runs with on-the-fly radiation transfer. The gas starts cold and neutral (`InitGasTemp` = 100 K in the params, which overrides the internal energy in the ICs).

This is a benchmark test. Failing this test does not necessarily imply that there is a problem, but rather indicates that something has changed that should be noted.

The reference is `gmc_cooling_rt_exact_coldstart_facemean_wakekick.hdf5`, fetched from the test-file mirror: cold start, RT/microphysics fixes, the arithmetic face-normal magnetic field under constrained gradients, and the wake-up kick for woken MFM cells (PR #68), which moves the urad_FUV statistic by ~12-15% in the lowest-density bins and ~5% elsewhere. Older checkouts use `gmc_cooling_rt_exact_coldstart_facemean.hdf5` (before the wake-up kick), `gmc_cooling_rt_exact_coldstart.hdf5` (before the face-normal change) or `gmc_cooling_rt_exact.hdf5` (before the RT/microphysics fixes).

Compile-time flags used for this setup:
```
  SINGLE_STAR_STARFORGE_DEFAULTS
  MAGNETIC
  COOLING
  SINGLE_STAR_FB_RAD
  METALS
  COOL_METAL_LINES_BY_SPECIES
  BOX_PERIODIC
  GRAVITY_NOT_PERIODIC
  ADAPTIVE_TREEFORCE_UPDATE=0.0625
  RT_SPEEDOFLIGHT_REDUCTION=1e-4
```
