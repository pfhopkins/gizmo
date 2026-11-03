# GMC cooling test

This test initializes a $$2\times10^4 M_\odot$$ giant molecular cloud at $1M_\odot$ resolution, runs it for about a crossing time, and verifies that the temperature-density statistics are in agreement with a pre-run setup, within a reasonable tolerance. 

This is a benchmark test. Failing this test does not necessarily imply that there is a problem, but rather indicates that something has changed that should be noted.

`test_gmc_cooling_uvb_table` runs the same build for a few steps with three UV background tables: the shipped `cooling/TREECOOL`, whose `##` header must be skipped so that every data row loads; a table of comments only; and a table with a malformed row. The last two must stop the run with a message naming the problem.

Compile-time flags used for this setup:
```
  SINGLE_STAR_STARFORGE_DEFAULTS
  COOLING
  MAGNETIC
  BOX_PERIODIC
  GRAVITY_NOT_PERIODIC
  ADAPTIVE_TREEFORCE_UPDATE=0.0625
```
