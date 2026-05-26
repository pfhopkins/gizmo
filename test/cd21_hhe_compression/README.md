# Phase 17h — CD21 H/He EOS smoke test

Substellar / giant-planet H/He equation of state from
**Chabrier & Debras 2021, ApJ 917:4** (CD21), loaded through the existing
ANEOS dispatch (`EOS_ANEOS`) — no new C++ branch.

## Pipeline

1. Download the native CD21 ASCII table for the desired Y from
   <http://perso.ens-lyon.fr/gilles.chabrier/DirEOS/DirEOS2021.tar.gz>
   and extract `TABLEEOS_2021_TP_Y0275_v1` next to this README as
   `cd21_y0275.dat`.
2. Convert to the SESAME format consumed by `eos/aneos.cc::aneos_read_table`:
   ```
   python python_src/eos_tools/cms_to_sesame.py \
       --input  cd21_y0275.dat \
       --output cd21_y0275.sesame \
       --y 0.275 --mat-id 2721
   ```
   The converter:
   - inverts the native (logT, logP) grid to a uniform (logρ, logT) grid
     (intersection of native ρ ranges across T to avoid extrapolation),
     CGS units throughout;
   - computes adiabatic sound speed via
     `gamma1 = 1 / [ dlnρ/dlnP|_T + ∇_ad · dlnρ/dlnT|_P ]`;
   - emits ncols=6 SESAME (`rho T P u S cs`).
3. Build the IC:
   ```
   python make_ics.py
   ```
   Writes `compression_ics.hdf5`: 128 1D-periodic particles at uniform
   ρ = 10⁻² g/cm³, T = 10³ K, with a single-mode sinusoidal velocity
   perturbation (1 km/s amplitude) to drive a small adiabatic compression
   wave. CompositionType = 0.

## Self-test (no download required)

The converter has a synthetic ideal-gas round-trip:
```
python python_src/eos_tools/cms_to_sesame.py --selftest
```
Passes when P, u, cs round-trip to <1e-5 against the analytic ideal-gas
solution.

## Test Config / params

`Config.sh`: `HYDRO_MESHLESS_FINITE_MASS`, `BOX_SPATIAL_DIMENSION=1`,
`BOX_PERIODIC`, `SELFGRAVITY_OFF`, `EOS_ANEOS`.
`cd21_hhe_compression.params` points `AneosTable0` at `cd21_y0275.sesame`.
TimeMax = 3×10⁻⁴ s ≈ 10 sound-crossing times. Smoke checks only that the
run completes with finite, positive (rho, P, u).

## Status (2026-05-07)

- Converter: written, self-test PASS (Phase 17d).
- CD21 Y=0.275 .sesame table: shipped (built end-to-end from the native
  v1 table at ens-lyon.fr).
- IC builder: shipped.
- C++ side: zero changes; reuses the existing 17c dispatch in `eos/eos.cc`.
- Physics validation (Jupiter/Saturn polytrope vs published profiles):
  deferred. This is a compile/smoke entry only.
