# GMC cooling test

This test initializes a $$2\times10^4 M_\odot$$ giant molecular cloud at $1M_\odot$ resolution, runs it for about a crossing time, and verifies that the temperature-density statistics are in agreement with a pre-run setup, within a reasonable tolerance. 

This is a benchmark test. Failing this test does not necessarily imply that there is a problem, but rather indicates that something has changed that should be noted.

Compile-time flags used for this setup:
```
  SINGLE_STAR_STARFORGE_DEFAULTS
  COOLING
  MAGNETIC
  BOX_PERIODIC
  GRAVITY_NOT_PERIODIC
  ADAPTIVE_TREEFORCE_UPDATE=0.0625
```

## Variants (optional — run by hand)

The default run is `Config.sh` + `gmc_cooling.params`, which is what `pytest
test/gmc_cooling` builds. These variants reuse the same `gmc_cooling_ics` and
each write to their own output directory, so they can coexist with it.

| Variant | Config | Params | What it changes |
| --- | --- | --- | --- |
| default | `Config.sh` | `gmc_cooling.params` | the benchmark above, to `TimeMax=1.0` |
| particle mesh | `Config_pmgrid.sh` | `gmc_cooling_pmgrid.params` | drops `SINGLE_STAR_STARFORGE_DEFAULTS` and the adaptive tree-force update, adds `PMGRID=64`: exercises the short-range/long-range gravity split on this cloud. Short run (`TimeMax=0.025`), writes `output_pmgrid/` |
| quick | `Config.sh` | `gmc_cooling_quick.params` | a shortened run; also used by `test/benchmark/benchmark_setup.sh` |

`Config_pmgrid.sh` and its params were formerly the separate
`gmc_cooling_pmgrid/` directory, which pointed its `InitCondFile` back at this
one. `test/gravtree` carries a related periodic+PM variant that also runs on
these ICs, but exercises the tree walk rather than the cooling physics.
