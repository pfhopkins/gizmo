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


def downsample(rho, T, data, target_nrho, target_nT):
    """Downsample by picking evenly-spaced indices in log space."""
    nrho_orig, nT_orig = len(rho), len(T)

    idx_rho = np.unique(np.linspace(0, nrho_orig - 1, target_nrho).astype(int))
    idx_T = np.unique(np.linspace(0, nT_orig - 1, target_nT).astype(int))

    rho_ds = rho[idx_rho]
    T_ds = T[idx_T]
    data_ds = data[np.ix_(idx_T, idx_rho)]  # data is [iT, irho, 4]

    return rho_ds, T_ds, data_ds


def write_gizmo_sesame(fname, mat_id, rho, T, data):
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
                S_cgs = max(S_si * JKKG_TO_ERGKG, 1e-30)
                cs_cgs = max(cs_si * MS_TO_CMS, 1e-30)

                f.write(f"{r_cgs:.10e} {t:.10e} {P_cgs:.10e} {u_cgs:.10e} {S_cgs:.10e} {cs_cgs:.10e}\n")

    print(f"Wrote {fname}: mat_id={mat_id}, {nrho}x{nT} grid")


def convert(input_file, output_file, mat_id, target_nrho=200, target_nT=200):
    print(f"Reading {input_file}...")
    rho, T, data, nrho, nT = read_stewart_table(input_file)
    print(f"  Original grid: {nrho} x {nT}")

    if target_nrho < nrho or target_nT < nT:
        rho, T, data = downsample(rho, T, data, target_nrho, target_nT)
        print(f"  Downsampled to: {len(rho)} x {len(T)}")

    write_gizmo_sesame(output_file, mat_id, rho, T, data)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: convert_stewart_table.py <input.txt> <output.sesame> <mat_id> [nrho] [nT]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    mat_id = int(sys.argv[3])
    nrho = int(sys.argv[4]) if len(sys.argv) > 4 else 200
    nT = int(sys.argv[5]) if len(sys.argv) > 5 else 200

    convert(input_file, output_file, mat_id, nrho, nT)
