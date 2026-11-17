# CD21 H/He EOS test

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
   python cms_to_sesame.py \
       --input  cd21_y0275.dat \
       --output cd21_y0275.sesame \
       --y 0.275 --mat-id 2721 --logP-min -6.5 --logP-max 8
   ```
   The converter:
   - keeps the native pressures 10^-6.5 .. 10^8 GPa by default (the published
     table holds placeholder values in its extreme-pressure corners;
     `--full-pressure-range` keeps them, for auditing only);
   - inverts the native (logT, logP) grid to a uniform (logρ, logT) grid
     (intersection of native ρ ranges across T to avoid extrapolation),
     CGS units throughout;
   - takes the adiabatic sound speed from the table's own P and u along the
     first-law adiabat, `cs² = P_ρ + P_T (P/ρ² − u_ρ)/u_T`;
   - sets a lowest valid temperature per density above the cold, dense states
     where the published u falls as T rises (there one energy names several
     temperatures): a validity boundary, not a repair; it refuses if any such
     state remains above it;
   - emits ncols=6 SESAME (`rho T P u S cs`) with the `lowest_valid_T_index`
     block, and prints the command and the input and output SHA-256 hashes.
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
python cms_to_sesame.py --selftest
```
Passes when P, u, cs round-trip to <1e-3 against the analytic ideal-gas
solution (measured ~3e-6), and when the same gas with an injected u that
falls as T rises (and one corrupted density point) gets the point masked, a
validity floor above the fall, a block that reads back exactly, and the falls
reported with `--no-defect-floor`.

## Test Config / params

`Config.sh`: `HYDRO_MESHLESS_FINITE_MASS`, `BOX_SPATIAL_DIMENSION=1`,
`BOX_PERIODIC`, `SELFGRAVITY_OFF`, `EOS_ANEOS`.
`cd21_hhe_compression.params` points `AneosTable0` at `cd21_y0275.sesame`.
TimeMax = 3×10⁻⁴ s ≈ 10 sound-crossing times. Smoke checks only that the
run completes with finite, positive (rho, P, u).

