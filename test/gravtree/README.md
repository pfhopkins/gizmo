# gravtree — gravity tree-walk validation

A family of short runs whose purpose is not to reproduce a physical solution
but to check that the gravity tree walk itself is correct: that it produces
the same forces under different softening, boundary, potential-evaluation and
sink configurations, and that the tree build survives degenerate particle
positions.

These are build-vs-build comparisons rather than comparisons against an exact
solution, so they are not part of the pytest sweep. Each variant is one
`Config_*.sh` plus one `.params`, and each writes to its own output directory
so the variants can coexist.

This directory replaces the former `gravtree_vanilla`, `gravtree_vanilla_eval`,
`gravtree_pmgrid`, `gravtree_sinks` and `gravtree_collocated` directories;
nothing was dropped in the move.

## Variants

| Variant | Config | Params | ICs | What it exercises |
| --- | --- | --- | --- | --- |
| default | `Config.sh` | `gravtree.params` | `../evrard/evrard_ics` | plain walk, fixed gas softening (no `ADAPTIVE_GRAVSOFT_FORGAS`), so every gating branch in `gpu_gravtree.cc` is satisfied |
| potential | `Config_evalpotential.sh` | `gravtree_evalpotential.params` | `../evrard/evrard_ics` | adds `EVALPOTENTIAL` — the potential accumulated alongside the force |
| periodic + PM | `Config_pmgrid.sh` | `gravtree_pmgrid.params` | `../gmc_cooling/gmc_cooling_ics` | `BOX_PERIODIC` + `GRAVITY_NOT_PERIODIC` + `PMGRID=64`: the short-range/long-range split |
| sinks | `Config_sinks.sh` | `gravtree_sinks.params` | `../isodisk_thermalfb/isodisk_thermalfb_ics` | `SINK_CALC_DISTANCES` + `SINGLE_STAR_*`: sink distance/timestep quantities gathered during the walk |
| collocated | see caveat below | `gravtree_collocated.params` | `gravtree_collocated_ics` (built by `make_collocated_ics.py`) | deliberately bit-identical particle positions, to exercise the tree-build random-octant fallback |

## Running a variant

```bash
cp test/gravtree/Config_pmgrid.sh src/Config.sh     # pick a variant
# ... build GIZMO ...
cd test/gravtree && mpirun -np 2 ../../src/GIZMO gravtree_pmgrid.params
```

To compare two builds, run the same params under each build and rename
`output*/` between runs. For the sinks variant there is a checker:

```bash
python compare_sinks.py                    # defaults: output_reference vs output_kokkos
python compare_sinks.py DIR_A DIR_B
```

It matches particles by `ParticleIDs` and reports scale-normalized
`max|diff|/max|reference|` for positions, velocities and the sink distance /
timestep fields, with per-field tolerances.

## Caveats

- **`gravtree_collocated` has no tracked `Config.sh`** and never did. Its own
  header states it mirrors the `gmc_cooling_quick.params` recipe, and it sets
  `BiniX/Y/Z`, `Grain_*` and an 11-species `InitMetallicity` — so it needs a
  gmc_cooling-style config (magnetic + cooling + metals + grains), *not* the
  stripped `Config.sh` in this directory. Build it from `../gmc_cooling/Config.sh`
  adjusted for the 10 pc box, or add a `Config_collocated.sh` here once the
  intended flag set is confirmed.

- The sinks variant reads the **isodisk thermal-feedback ICs**, which already
  carry a `PartType5` sink, so the sink-side comparison is meaningful from
  `snapshot_000`. It also reads that directory's cooling tables; fetch them the
  usual way (`get_cooling_tables`, which accepts a symlink to a sibling test).
  It previously read `thermalfb_snap003`, a snapshot from a one-off run that
  was never tracked here.
