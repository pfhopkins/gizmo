# gravtree — gravity tree accuracy

The classic tree-force error test. Each variant is built once and run twice from the same IC:

1. with the parameterfile's own (production) opening criterion, and
2. with the walk tightened towards direct summation (`ErrTolTheta=0.15`, `ErrTolForceAcc=1e-6`;
   `ErrTolTheta` must lie in (0.1, 0.9)).

Both runs take a single step, and `test_gravtree.py` compares their `snapshot_000`, which holds the
start-up force evaluation, so the difference is the tree error alone. Particles are matched by
`ParticleIDs`.

For the gravitational acceleration (`OUTPUT_ACCELERATION` with `OUTPUT_HYDROACCELERATION`, so the
hydro term is excluded) and the potential, the tight walk is a converged reference: the test bounds
the median and 99th percentile of the per-particle relative error. A defect in the walk (a missed
node, a wrong softening or periodic image) shows up as a large error on some subset of particles
even when the median stays small.

The sinks and rt variants also compare quantities gathered opportunistically on the same walk,
whose openings are chosen for force accuracy alone; for these the tight run is only a
tighter-walk reference:

- `Sink_Distance` is the distance to the nearest sink the walk encounters. Outside the short-range
  region, whether a cell encounters the sink at all depends on which nodes are opened (the PM cull
  is a per-axis box test, so large accepted nodes reach further than small ones). Values are
  compared only where both walks report one, and there they must agree exactly.
- `PhotonEnergy` is the deliberately crude long-range radiation estimate of
  `GALSF_FB_FIRE_RT_LONGRANGE`, which rides on the gravity walk with no luminosity term in the
  opening criterion and has no converged solution to compare against. A heavy tail in the per-cell
  error is expected (measured p99 ~ 70%), so only the median and the energy-weighted error are gated.

## Variants

All variants build `Config.sh` plus the flags listed (the test appends them), and need
`DEVELOPER_MODE` (in `Config.sh`) so that the opening-criterion parameters are read.

| Variant | Extra flags | Params | IC | Compared |
| --- | --- | --- | --- | --- |
| default | — | `gravtree.params` | `../evrard/evrard_ics` | `Acceleration`; fixed gas softening, so every softening branch of the walk is the plain one |
| evalpotential | `EVALPOTENTIAL`, `OUTPUT_POTENTIAL` | `gravtree_evalpotential.params` | `../evrard/evrard_ics` | `Acceleration`, `Potential` (accumulated on the walk) |
| pmgrid | `BOX_PERIODIC`, `GRAVITY_NOT_PERIODIC`, `PMGRID=64` | `gravtree_pmgrid.params` | `../gmc_cooling/gmc_cooling_ics` | `Acceleration` (tree + PM) |
| sinks | galaxy flags + `SINK_CALC_DISTANCES`, `OUTPUT_SINK_DISTANCES`, `SINGLE_STAR_SINK_DYNAMICS`, ... | `gravtree_sinks.params` | `../isodisk/isodisk_ics` | `Acceleration`, `Sink_Distance` |
| rt | galaxy flags + `GALSF_FB_FIRE_RT_LONGRANGE` | `gravtree_rt.params` | `../isodisk/isodisk_ics` | `Acceleration`, `PhotonEnergy` (radiation carried on the walk) |

"Galaxy flags" are `COOLING`, `GALSF`, `GALSF_SFR_CRITERION=(1+256)`, `METALS`, `BOX_PERIODIC`,
`GRAVITY_NOT_PERIODIC`, `PMGRID=64`, `ADAPTIVE_GRAVSOFT_FORGAS`; the isolated-disk IC carries the
central black hole (a Type-5 sink) and the star particles that act as radiation sources. Those two
variants need the cooling tables (`get_cooling_tables`).

## Running

```bash
pytest test/gravtree                 # all variants
pytest test/gravtree -k pmgrid       # one variant
```

Each run writes the params' `OutputDir` (production) and the same name with `_tight` appended; the
printed line per quantity gives the statistics and the limits.

## Tolerances

Set to about 3x the values measured on a recorded run (np=2, Kokkos-OpenMP build):

| Variant | Quantity | Measured | Limits |
| --- | --- | --- | --- |
| default, evalpotential | Acceleration | median 1.3e-3, p99 2.1e-3 | 4e-3, 7e-3 |
| evalpotential | Potential | median 2.7e-4, p99 5.7e-4 | 1e-3, 2e-3 |
| pmgrid | Acceleration | median 3.3e-3, p99 4.2e-3 | 1e-2, 1.5e-2 |
| sinks, rt | Acceleration | median 1.6e-3, p99 3.4e-3 | 5e-3, 1e-2 |
| sinks | Sink_Distance (shared cells) | max 0 (3536 of 35000 cells shared) | 1e-6 |
| rt | PhotonEnergy | median 6.5e-3, energy-weighted 2.3e-2 | 2e-2, 7e-2 |

The gates are not blind: a deliberately loose criterion (`ErrTolTheta=0.85`, `ErrTolForceAcc=0.0099`)
on the default variant gives median 4.7e-3 and p99 8.8e-3, and fails both acceleration limits.
