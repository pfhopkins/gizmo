# Phase 17d — CD21 H/He EOS smoke test

Substellar / giant-planet H/He equation of state from
**Chabrier & Debras 2021, ApJ 917:4** (CD21), loaded through the existing
ANEOS dispatch (`EOS_ANEOS`) — no new C++ branch.

## Pipeline

1. Download the native CD21 ASCII table (logT, logP) for the desired Y from
   the publication's electronic supplement.
2. Convert to SESAME format consumed by `eos/aneos.cc::aneos_read_table`:
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
3. Place the `.sesame` file in this directory (or upload to
   `tapir.caltech.edu/~phopkins/sims/` for distribution).

## Self-test (no download required)

The converter has a synthetic ideal-gas round-trip:
```
python python_src/eos_tools/cms_to_sesame.py --selftest
```
Passes when P, u, cs round-trip to <1e-5 against the analytic ideal-gas
solution.

## Test Config / params

`Config.sh` mirrors the existing `aneos_shocktube` (1D, periodic, no gravity,
EOS_ANEOS, double precision). `cd21_hhe_compression.params` points
`AneosTable0` at `cd21_y0275.sesame`. No new in-tree C++ machinery is added —
this exists purely to confirm the converter output round-trips through the
ANEOS loader.

## Status

- Converter: written, self-test PASS.
- C++ side: zero changes; reuses the existing 17c dispatch in `eos/eos.cc`.
- Physics validation (Jupiter/Saturn polytrope vs published profiles):
  deferred. This is a compile/smoke entry only.
