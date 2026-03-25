# GMC cooling test

This test initializes a $$2\times10^4 M_\odot$$ giant molecular cloud at $1M_\odot$ resolution, runs it for about a crossing time, and verifies that the temperature-density statistics are in agreement with a pre-run setup, within a reasonable tolerance. This is the version that runs with on-the-fly radiation transfer.

This is a benchmark test. Failing this test does not necessarily imply that there is a problem, but rather indicates that something has changed that should be noted.

Compile-time flags used for this setup:
```
  SINGLE_STAR_STARFORGE_DEFAULTS
  SINGLE_STAR_FB_RAD
  COOLING
  MAGNETIC
  BOX_PERIODIC
  GRAVITY_NOT_PERIODIC
  ADAPTIVE_TREEFORCE_UPDATE=0.0625
  RT_SPEEDOFLIGHT_REDUCTION=3e-5
```
