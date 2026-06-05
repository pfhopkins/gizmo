# CBE Vlasov-moment test problems

Four pytest-bench test problems for the GIZMO CBE integrator
(`CBE_INTEGRATOR`), ported from the 1D Python harness at
`/Users/phopkins/Documents/work/papers/numerical_methods/cbe_cosmology/python_harness/`
(see `HARNESS_RESULTS_AND_FINDINGS.md` there). They evolve a per-particle
mixture-of-Gaussians velocity distribution via Godunov face fluxes — no
gravity, no hydro, no gas, pure CBE moment advection on Type=1 particles.

| dir | harness analogue | dim | NBASIS | streams |
|---|---|---|---|---|
| `cbe_two_stream`   | `test_two_stream_boosted` / `_density_wave` (rest) | 1D | 2 | +1, -1 |
| `cbe_density_wave` | `test_two_stream_density_wave` | 1D | 2 | +1, -1, density-modulated spacing |
| `cbe_free_slot_1d` | `test_free_slot` | 1D | 4 | +1, 0, -1, -2 (+localized +2 perturbation) |
| `cbe_free_slot_3d` | `test_free_slot` in 3D | 3D slab | 4 | as 1D, velocities along x |

Each directory is a self-contained pytest-bench test: `Config.sh`,
`<name>.params`, `make_ic.py`, `test_<name>.py`. Shared IC-writer + analysis
code lives in `test/cbe_vlasov_common.py`.

## How the streams get into GIZMO — `VlasovMoments` IC dataset (C7 reader)

CBE basis moments are injected per-particle through an HDF5 dataset
`/PartType1/VlasovMoments` of shape `(N, NBASIS*NMOMENTS)`, basis-major
(`index = NMOMENTS*basis + moment`), read straight into
`P[i].CBE_basis_moments[basis][moment]`. NMOMENTS (no
`CBE_INTEGRATOR_SECONDMOMENT`) = dimension+1: 1D → `[m, p_x]`, 3D →
`[m, p_x, p_y, p_z]`.

`make_ic.py` also sets `Velocities` = the mass-weighted-mean basis velocity
and `Masses` = the basis-mass sum, because `do_cbe_initialization()`
renormalizes `Σ_basis m → P.Mass` and shifts momenta so `Σ_basis p =
P.Mass·P.Vel`. Setting `Velocities` to the mean makes that closure a no-op,
so the prescribed per-stream velocities survive.

**Dependency:** the `VlasovMoments` IC-read path is the "C7 IC reader",
committed on `dmheat_cbe_c5_on_wave5next` (`gpu_bench_new` working tree) but
**not yet on this branch** (`dmheat_cbe_c6_archive_20260530` lineage). It
must be merged in before running — until then GIZMO ignores the dataset and
synthesizes a cold single-stream default, so the multi-stream physics is not
represented. Configs and ICs are ready now; runtime is gated on that merge.

## Build & run (after the C7 reader merge)

```bash
cd <worktree root>
python3 test/<name>/make_ic.py            # writes test/<name>/<name>_ics.hdf5
# pytest (user-driven), or by hand:
cp test/<name>/Config.sh . && make -j8
cp GIZMO test/<name>/ && cd test/<name>
OMP_NUM_THREADS=1 mpirun -np 2 ./GIZMO <name>.params 0
```

All four Config.sh **compile clean on Mac (`MacBookCellar_Kokkos`)** as of
2026-06-02 (this is the only validation done so far — no runtime yet).

## Validation gates (`test_<name>.py`)

- Global conservation: total mass to ≤1e-10 relative, total momentum to ≤1e-9.
- `output/cbe_diagnostics.txt` first row: `face_res_max` (col 2) ≤ 1e-11,
  `bracket_fail` (col 4) = 0.
- Controlled `STOP_WHEN_BELOW_MINTIMESTEP` stop, no NaN/segv.
- Per-stream band-mass profiles plotted (`<name>_streams.png`); free-slot
  reports per-band masses (the harness "free-slot preserves +2 and dominant
  +1" comparison) — reported, not hard-gated.
- `cbe_diagnostics.txt` col-9 `free_slot_count` should be non-zero for the
  free-slot problems (the fallback fired).

The harness `outputs/<test>/*.npz` are the quantitative anchors to
cross-compare against once these run.
