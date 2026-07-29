# hernquist_sidm — B2 AGSForce GPU port activating test

Small DM-only Hernquist halo (Type=1 only, ~1000 particles default) with
DM_SIDM=2 and ADAPTIVE_GRAVSOFT_FORALL=2 so the AGSForce loop actually
dispatches on Type=1 — i.e. `ags_force_evaluate_gpu` is exercised.

**Setup notes (corrected 2026-05).** This is an *isolated* halo — an open,
non-periodic domain: the Config files carry no `BOX_PERIODIC` and the params
no `BoxSize`. `make_ic.py` builds the IC using the gravitational constant in
the params' code units (kpc / 1e10 Msun / km/s → G ≈ 4.3e4), NOT G=1, so the
halo is in virial equilibrium for the units GIZMO integrates in. The IC file
`hernquist_sidm_ics.hdf5` is generated output (gitignored) — regenerate it
with the step-1 command below. `hernquist_dmfuzzy.params` runs the same IC as
fuzzy dark matter (`DM_FUZZY=1`) instead of SIDM.

## Three-part validation protocol (see feedback_gpu_port_validation_protocol.md)

### Part (a) — verify GPU path fires
Temporary printfs are added inside `ags_density_evaluate_gpu` and
`ags_force_evaluate_gpu` during validation. Remove before commit.

### Part (b) — compile + run with activating flags
Config.sh is the GPU-path variant; Config_reference.sh is the
reference. Both can be built with the same Makefile.systype by
copy-and-build.

### Part (c) — bit-compare GPU output vs CPU tree-walk reference
Workflow:

```bash
# 1. Generate IC (one-time)
cd test/hernquist_sidm && python make_ic.py 10 1.0 1.0

# 2. Build CPU tree-walk reference (no Kokkos, no GPU)
#    → switch Makefile.systype to "MacBookCellar" (non-Kokkos)
cp test/hernquist_sidm/Config_reference.sh Config.sh
make clean && make -j
mkdir -p test/hernquist_sidm/output_reference
cd test/hernquist_sidm && mpirun -np 2 ../../GIZMO hernquist_sidm.params
mv output output_reference

# 3. Build Kokkos-OpenMP path (GPU code path, Mac)
#    → switch Makefile.systype to "MacBookCellar_Kokkos"
cp test/hernquist_sidm/Config.sh Config.sh
make clean && make -j
mkdir -p test/hernquist_sidm/output_kokkos
cd test/hernquist_sidm && mpirun -np 2 ../../GIZMO hernquist_sidm.params
mv output output_kokkos

# 4. Diff snapshots
python -c "
import h5py, numpy as np
for k in ['Coordinates', 'Velocities', 'AGS_KernelRadius', 'dtime_sidm']:
    a = h5py.File('output_reference/snapshot_001.hdf5')['PartType1'][k][:]
    b = h5py.File('output_kokkos/snapshot_001.hdf5')['PartType1'][k][:]
    print(f'{k}: max abs diff = {np.max(np.abs(a-b)):.3e}, '
          f'max rel diff = {np.max(np.abs(a-b)/(np.abs(a)+1e-30)):.3e}')
"

# 5. Same on Vista: rsync the test dir + IC over, repeat steps 2/3 there.
```

Tolerances: since the CPU SIDM RNG stream is now the counter-based RNG
(same as GPU), small FP differences are expected from kernel order but
the scatter decisions should be identical. Flag anything beyond a few
parts in 1e-12 as suspect.

## Files
- `make_ic.py` — IC generator (Hernquist DF + Von Neumann velocity sampler)
- `Config.sh` — GPU path (adaptive softening + SIDM; Kokkos always active)
- `Config_reference.sh` — CPU tree-walk reference (same flags minus the neighbor-list one)
- `hernquist_sidm.params` — SIDM params, including DM_InteractionCrossSection etc.
- `hernquist_dmfuzzy.params` — same IC run as fuzzy dark matter (`DM_FUZZY=1`)
