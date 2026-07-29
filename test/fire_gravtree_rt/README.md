# fire_gravtree_rt — radiation energy carried on the gravity tree walk

Checks `RT_USE_GRAVTREE_SAVE_RAD_ENERGY`: the radiation energy density
accumulated during the gravity tree walk must not depend on which build
performed the walk. Runs the isodisk thermal-feedback ICs — which carry the
star particles that act as the long-range radiation sources — for a few steps,
and compares `PhotonEnergy` / `Rad_E_gamma` particle-by-particle.

The ICs and cooling tables are read from `../isodisk_thermalfb/`; fetch the
tables the usual way (`get_cooling_tables`, which accepts a symlink to a
sibling test). This test previously read `thermalfb_snap003`, a snapshot from
a one-off run that was never tracked in this repository.

This is a build-vs-build comparison, so it is not part of the pytest sweep —
there is one `Config.sh` and one `fire_gravtree_rt.params`, run twice.

## Running it

```bash
# 1. reference build (switch Makefile.systype to the non-Kokkos entry), then:
cd test/fire_gravtree_rt && mpirun -np 2 ../../src/GIZMO fire_gravtree_rt.params
mv output output_reference

# 2. Kokkos build (Makefile.systype -> MacBookCellar_Kokkos), then the same run:
mpirun -np 2 ../../src/GIZMO fire_gravtree_rt.params
mv output output_kokkos

# 3. compare
python compare_rt.py                       # defaults to output_reference vs output_kokkos
python compare_rt.py DIR_A DIR_B           # or name the two directories
```

`compare_rt.py` matches particles by `ParticleIDs` and reports, per field, the
scale-normalized `max|diff|/max|reference|`. It exits non-zero if any field
exceeds 1e-4, and warns if the reference run emitted no radiation at all
(which would make the comparison vacuous).

## Files

- `Config.sh` — the single build config for both runs
- `fire_gravtree_rt.params` — the single params for both runs (`OutputDir output`)
- `compare_rt.py` — the comparison driver
