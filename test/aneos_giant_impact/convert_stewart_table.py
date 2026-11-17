"""Convert Stewart group M-ANEOS tables (SWIFT format, SI) to GIZMO SESAME format (CGS).

Input format (SWIFT/Stewart):
  12 comment lines starting with #
  YYYYMMDD
  nrho nT
  rho[0] rho[1] ... rho[nrho-1]    (kg/m^3)
  T[0] T[1] ... T[nT-1]            (K)
  u[0,0] P[0,0] cs[0,0] S[0,0]     (J/kg, Pa, m/s, J/K/kg)
  ...  (T-slow, rho-fast ordering: for each T, loop over all rho)

Output format (GIZMO SESAME):
  mat_id nrho nT ncols
  rho[0]  (g/cm^3, one per line)
  ...
  T[0]    (K, one per line)
  ...
  rho T P u S cs   (CGS: g/cm^3, K, dyn/cm^2, erg/g, erg/g/K, cm/s)
  ...  (rho-slow, T-fast ordering: for each rho, loop over all T)
  lowest_valid_T_index
  k[0] ... k[nrho-1]   (per density: index of the lowest temperature row the EOS is valid at)

Resolution: the density axis is kept as tabulated (its resolution is what the
consistency of bilinearly interpolated P and u depends on); every T_stride-th
temperature row is kept (default 4, which costs little accuracy).

Invalid region: in these tables, states where ANEOS returned a negative entropy
(the cold, tensile corner below each density's lowest valid temperature) had
their energy replaced by a sentinel and their other values by filler, and the
negative entropy survives as the marker. Those rows are a contiguous block at the
bottom of every density column; the converter writes, per density, the lowest
kept row above that block, and GIZMO treats it as the lower edge of the table at
that density.
"""

import numpy as np
import sys


# SI -> CGS conversion factors
KGM3_TO_GCMC = 1.0e-3       # kg/m^3 -> g/cm^3
PA_TO_DYNCM2 = 10.0         # Pa -> dyn/cm^2
JKG_TO_ERGG = 1.0e4          # J/kg -> erg/g
MS_TO_CMS = 1.0e2            # m/s -> cm/s
JKKG_TO_ERGKG = 1.0e4        # J/K/kg -> erg/K/g


def read_stewart_table(fname):
    """Read a Stewart/SWIFT format ANEOS table."""
    with open(fname) as f:
        lines = f.readlines()

    # Skip comment lines
    data_lines = [l.strip() for l in lines if not l.startswith('#')]

    # date_str = data_lines[0]
    nrho, nT = map(int, data_lines[1].split())

    # Parse all remaining numbers
    vals = []
    for l in data_lines[2:]:
        vals.extend(l.split())
    vals = np.array(vals, dtype=np.float64)

    rho = vals[:nrho]                          # kg/m^3
    T = vals[nrho:nrho + nT]                   # K
    data = vals[nrho + nT:].reshape(nT, nrho, 4)  # [iT, irho, (u,P,cs,S)]

    return rho, T, data, nrho, nT


def kept_indices(n, stride):
    """every stride-th index of an axis of length n, always including the last"""
    idx = list(range(0, n, stride))
    if idx[-1] != n - 1: idx.append(n - 1)
    return np.array(idx)


def lowest_valid_rows(data):
    """per density column, the index of the first row above the block of negative-entropy rows"""
    bad = data[:, :, 3] < 0  # [iT, irho]
    nT, nrho = bad.shape
    low = np.zeros(nrho, dtype=int)
    for i in range(nrho):
        rows = np.where(bad[:, i])[0]
        if len(rows) == 0: continue
        if not bad[:rows[-1] + 1, i].all():
            raise ValueError(f"density column {i}: the negative-entropy rows are not one block at the bottom of the column")
        low[i] = rows[-1] + 1
    return low


def write_gizmo_sesame(fname, mat_id, rho, T, data, low):
    """Write GIZMO SESAME format. data is [nT, nrho, 4] with (u, P, cs, S) in SI."""
    nrho, nT = len(rho), len(T)

    with open(fname, 'w') as f:
        f.write(f"{mat_id} {nrho} {nT} 6\n")

        # Density grid (CGS)
        for r in rho:
            f.write(f"{r * KGM3_TO_GCMC:.15e}\n")

        # Temperature grid (K, unchanged)
        for t in T:
            f.write(f"{t:.15e}\n")

        # 2D data: rho-slow, T-fast
        # For each rho, loop over all T
        for irho in range(nrho):
            r_cgs = rho[irho] * KGM3_TO_GCMC
            for iT in range(nT):
                t = T[iT]
                u_si, P_si, cs_si, S_si = data[iT, irho, :]

                P_cgs = max(P_si * PA_TO_DYNCM2, 1e-30)
                u_cgs = max(u_si * JKG_TO_ERGG, 1e-30)
                S_cgs = S_si * JKKG_TO_ERGKG   # signed: the entropy zero point is a convention
                cs_cgs = max(cs_si * MS_TO_CMS, 1e-30)

                f.write(f"{r_cgs:.10e} {t:.10e} {P_cgs:.10e} {u_cgs:.10e} {S_cgs:.10e} {cs_cgs:.10e}\n")

        f.write("lowest_valid_T_index\n")
        f.write(" ".join(str(int(k)) for k in low) + "\n")

    print(f"Wrote {fname}: mat_id={mat_id}, {nrho}x{nT} grid")


def convert(input_file, output_file, mat_id, T_stride=4):
    print(f"Reading {input_file}...")
    rho, T, data, nrho, nT = read_stewart_table(input_file)
    print(f"  Original grid: {nrho} x {nT}")
    low_native = lowest_valid_rows(data)

    keep_T = kept_indices(nT, T_stride)
    # per density: the first kept row at or above the native lowest valid row (at most the second-to-last kept row)
    low = np.minimum(np.searchsorted(keep_T, low_native), len(keep_T) - 2)
    T, data = T[keep_T], data[keep_T, :, :]
    print(f"  Kept every {T_stride}th temperature row: {nrho} x {len(T)}; lowest valid T at the median density "
          f"{T[low[nrho // 2]]:.4g} K, highest lowest-valid T {T[low.max()]:.4g} K")

    write_gizmo_sesame(output_file, mat_id, rho, T, data, low)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: convert_stewart_table.py <input.txt> <output.sesame> <mat_id> [T_stride (default 4)]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    mat_id = int(sys.argv[3])
    T_stride = int(sys.argv[4]) if len(sys.argv) > 4 else 4

    convert(input_file, output_file, mat_id, T_stride)
