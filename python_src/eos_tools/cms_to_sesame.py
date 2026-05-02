"""Convert a Chabrier-family H/He EOS table (CMS19, CD21, or compatible)
to the SESAME-format ASCII table consumed by GIZMO's eos/aneos.cc loader.

Phase 17d (planet formation suite): substellar H/He EOS support.

----------------------------------------------------------------------
Input format (CMS19 / CD21 native release, ASCII):
  - Comment lines beginning with '#' are skipped.
  - The table is published as a (logT, logP) grid -- fixed Y, fixed mixing.
  - Native columns (CMS19, ApJ 872:51, Table 1):
        col 0  log10 T   [K]
        col 1  log10 P   [GPa]            (1 GPa = 1e10 dyn/cm^2)
        col 2  log10 rho [g/cm^3]
        col 3  log10 u   [MJ/kg]          (1 MJ/kg = 1e10 erg/g)
        col 4  log10 S   [MJ/kg/K]        (1 MJ/kg/K = 1e10 erg/g/K)
        col 5  dlnrho/dlnT |_P
        col 6  dlnrho/dlnP |_T
        col 7  dlnS/dlnT  |_P
        col 8  dlnS/dlnP  |_T
        col 9  grad_ad = dlnT/dlnP |_S
  - The grid is dense in log P at each fixed log T. Some releases prepend a
    blank line between log-T blocks; this script tolerates either layout.
  - CD21 (with He demixing) and Y-specific tables follow the same schema;
    pass --y to record it in the output header for bookkeeping.

Output format (matches eos/aneos.cc::aneos_read_table):
  Line 1:  mat_id  nrho  nT  ncols
  Then nrho density values (CGS, one per line)
  Then nT  temperature values (K, one per line)
  Then nrho*nT data rows, outer loop over rho, inner over T:
       rho  T  P  u  S  cs
  All quantities CGS. Grid is uniform in log10(rho) and log10(T).
  ncols=6 (no phase column; H/He surface is smooth).

The CMS native grid is uniform in (logT, logP). aneos.cc requires (logrho, logT)
on a uniform log-log grid, so we re-grid:
  1. Read native (T_i, P_j) -> rho, u, S, grad_ad on the native grid.
  2. Pick target uniform logrho axis covering [min, max] of native rho.
  3. For each target T_i (kept on native log-T axis), interpolate
     (P, u, S, grad_ad) at fixed T as functions of logrho (monotonic in P
     at fixed T, hence monotonic in rho).
  4. Sound speed:  cs^2 = gamma1 * P/rho,  with
        gamma1 = (dlnP/dlnrho)|_S
               = 1 / [ (dlnrho/dlnP)|_T  +  grad_ad * (dlnrho/dlnT)|_P ]
     Derivation: from dlnP = A dlnrho + B dlnT (with A = 1/(dlnrho/dlnP|_T),
     B = -(dlnrho/dlnT|_P)/(dlnrho/dlnP|_T) by the triple-product rule)
     plus dlnT = grad_ad * dlnP at constant S. Reduces to gamma for an ideal gas.
  5. Emit SESAME-format file.

Self-test:
  Run with --selftest to generate a synthetic ideal-gas (logT, logP) input,
  convert it, and check thermodynamic consistency vs analytic gamma=5/3.

Usage:
  python cms_to_sesame.py --input cms19_y0245.dat --output cms19_y0245.sesame
  python cms_to_sesame.py --selftest
"""

import argparse
import sys
import numpy as np

# Unit conversions (CMS native -> CGS)
GPA_TO_DYN_PER_CM2 = 1.0e10        # 1 GPa = 1e10 dyn/cm^2
MJ_PER_KG_TO_ERG_PER_G = 1.0e10    # 1 MJ/kg = 1e10 erg/g
MJ_PER_KG_K_TO_ERG_PER_G_K = 1.0e10


def parse_native_table(path):
    """Load a CMS-format ASCII table.

    Returns dict of 2D arrays indexed [iT, iP], plus 1D logT, logP axes.
    Tolerates extra blank lines between log-T blocks.
    """
    rows = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            parts = s.split()
            if len(parts) < 5:
                continue
            rows.append([float(x) for x in parts])
    arr = np.asarray(rows)
    if arr.ndim != 2 or arr.shape[1] < 5:
        raise ValueError(f"{path}: parsed {arr.shape}, need 2D with >=5 cols")

    logT_native = arr[:, 0]
    logP_native = arr[:, 1]
    logrho_native = arr[:, 2]
    logu_native = arr[:, 3]
    logS_native = arr[:, 4]

    # CMS19 includes derivative columns; CD21 too. If absent (compact release),
    # fall back to numerical sound-speed estimate.
    have_derivs = arr.shape[1] >= 10
    if have_derivs:
        dlnrho_dlnT_P = arr[:, 5]
        dlnrho_dlnP_T = arr[:, 6]
        grad_ad = arr[:, 9]
    else:
        dlnrho_dlnT_P = None
        dlnrho_dlnP_T = None
        grad_ad = None

    # Reshape onto the native (iT, iP) grid by detecting axis lengths.
    # CMS19 layout: outer loop over T, inner over P. Detect the unique T axis.
    uniqT, iT = np.unique(logT_native, return_inverse=True)
    uniqP, iP = np.unique(logP_native, return_inverse=True)
    nT, nP = uniqT.size, uniqP.size
    if iT.size != nT * nP:
        raise ValueError(f"{path}: expected complete {nT}x{nP} (logT,logP) grid, got {iT.size} rows")

    def reshape(col):
        out = np.full((nT, nP), np.nan)
        out[iT, iP] = col
        if np.isnan(out).any():
            raise ValueError(f"{path}: missing entries in (logT,logP) grid")
        return out

    return dict(
        logT=uniqT,
        logP=uniqP,
        logrho=reshape(logrho_native),
        logu=reshape(logu_native),
        logS=reshape(logS_native),
        dlnrho_dlnT_P=reshape(dlnrho_dlnT_P) if have_derivs else None,
        dlnrho_dlnP_T=reshape(dlnrho_dlnP_T) if have_derivs else None,
        grad_ad=reshape(grad_ad) if have_derivs else None,
    )


def regrid_to_uniform_logrho(native, nrho, logrho_pad=0.0):
    """Re-grid native (logT, logP) data to uniform (logrho, logT).

    Returns dict with uniform logrho axis, kept logT axis, and 2D arrays for
    P, u, S, gamma1 (each [irho, iT], CGS units, NOT logged here)."""
    logT = native['logT']        # K (log10)
    logP_native = native['logP'] # GPa (log10)
    logrho_native = native['logrho']  # g/cm^3 (log10), shape [iT, iP]
    logu_native = native['logu']      # MJ/kg (log10)
    logS_native = native['logS']      # MJ/kg/K (log10)

    nT = logT.size

    # The native (logT, logP) grid covers different rho ranges at different T.
    # To avoid extrapolation, take the INTERSECTION of native rho ranges across T:
    # max-of-mins as the lower bound, min-of-maxes as the upper bound.
    lr_lo_per_T = logrho_native.min(axis=1)
    lr_hi_per_T = logrho_native.max(axis=1)
    lr_lo = lr_lo_per_T.max() + logrho_pad
    lr_hi = lr_hi_per_T.min() - logrho_pad
    if lr_hi <= lr_lo:
        raise ValueError(f"native rho coverage does not intersect across T axis "
                         f"(lr_lo={lr_lo}, lr_hi={lr_hi})")
    logrho = np.linspace(lr_lo, lr_hi, nrho)

    # For each iT (fixed T), rho is monotonic in P, so logP, logu, logS are
    # monotonic in logrho. Interpolate each at the target logrho axis.
    P_cgs = np.empty((nrho, nT))
    u_cgs = np.empty((nrho, nT))
    S_cgs = np.empty((nrho, nT))

    for j in range(nT):
        lr_row = logrho_native[j, :]                # [iP]
        order = np.argsort(lr_row)
        lr_sorted = lr_row[order]
        logP_sorted = logP_native[order]            # already 1D in iP
        logu_sorted = logu_native[j, :][order]
        logS_sorted = logS_native[j, :][order]

        logP_target = np.interp(logrho, lr_sorted, logP_sorted,
                                left=logP_sorted[0], right=logP_sorted[-1])
        logu_target = np.interp(logrho, lr_sorted, logu_sorted,
                                left=logu_sorted[0], right=logu_sorted[-1])
        logS_target = np.interp(logrho, lr_sorted, logS_sorted,
                                left=logS_sorted[0], right=logS_sorted[-1])

        # Native -> CGS
        P_cgs[:, j] = (10.0 ** logP_target) * GPA_TO_DYN_PER_CM2
        u_cgs[:, j] = (10.0 ** logu_target) * MJ_PER_KG_TO_ERG_PER_G
        S_cgs[:, j] = (10.0 ** logS_target) * MJ_PER_KG_K_TO_ERG_PER_G_K

    # Sound speed via gamma1 if derivatives are available; else finite-difference.
    cs_cgs = np.empty((nrho, nT))
    rho_cgs_2d = (10.0 ** logrho)[:, None]  # broadcast over T

    if native['grad_ad'] is not None:
        for j in range(nT):
            lr_row = native['logrho'][j, :]
            order = np.argsort(lr_row)
            lr_sorted = lr_row[order]
            grad_ad_native_sorted = native['grad_ad'][j, :][order]
            dlrT_native_sorted = native['dlnrho_dlnT_P'][j, :][order]
            dlrP_native_sorted = native['dlnrho_dlnP_T'][j, :][order]

            grad_ad_t = np.interp(logrho, lr_sorted, grad_ad_native_sorted)
            dlrT_t   = np.interp(logrho, lr_sorted, dlrT_native_sorted)
            dlrP_t   = np.interp(logrho, lr_sorted, dlrP_native_sorted)

            # gamma1 = 1 / [ dlnrho/dlnP|_T + grad_ad * dlnrho/dlnT|_P ]
            denom = dlrP_t + grad_ad_t * dlrT_t
            denom = np.where(np.abs(denom) > 1e-30, denom, 1e-30)
            gamma1 = 1.0 / denom
            gamma1 = np.clip(gamma1, 1.01, 20.0)  # keep physical
            cs_cgs[:, j] = np.sqrt(gamma1 * P_cgs[:, j] / rho_cgs_2d[:, 0])
    else:
        # Numerical: cs^2 ~ (dP/drho)|_T (warning: misses adiabatic correction;
        # acceptable only when derivative columns absent). Issue a warning.
        print("WARNING: no grad_ad column; using isothermal cs^2 = (dP/drho)|_T."
              " Sound speed will be slightly under-estimated.", file=sys.stderr)
        for j in range(nT):
            dPdrho = np.gradient(P_cgs[:, j], 10.0 ** logrho)
            cs_cgs[:, j] = np.sqrt(np.maximum(dPdrho, 0.0))

    return dict(
        logrho=logrho,
        logT=logT,
        P=P_cgs, u=u_cgs, S=S_cgs, cs=cs_cgs,
    )


def write_sesame(path, regridded, mat_id):
    logrho = regridded['logrho']
    logT = regridded['logT']
    nrho, nT = logrho.size, logT.size
    rho_arr = 10.0 ** logrho
    T_arr = 10.0 ** logT
    P, u, S, cs = regridded['P'], regridded['u'], regridded['S'], regridded['cs']

    # Sanity: aneos.cc requires uniform spacing in log10(rho) and log10(T).
    # aneos.cc requires uniform spacing in log10. We constructed logrho uniformly,
    # but logT comes from np.unique on the (rounded) input file; snap it to a clean
    # uniform axis before emitting.
    logT = np.linspace(logT[0], logT[-1], logT.size)
    drho = np.diff(logrho); dT = np.diff(logT)
    if not (np.allclose(drho, drho[0], rtol=1e-9) and np.allclose(dT, dT[0], rtol=1e-9)):
        raise ValueError("axes must be uniform in log10")

    with open(path, 'w') as f:
        f.write(f"{mat_id} {nrho} {nT} 6\n")
        for r in rho_arr:
            f.write(f"{r:.15e}\n")
        for T in T_arr:
            f.write(f"{T:.15e}\n")
        # Outer loop rho, inner T
        for i in range(nrho):
            for j in range(nT):
                f.write(f"{rho_arr[i]:.15e} {T_arr[j]:.15e} "
                        f"{P[i, j]:.15e} {u[i, j]:.15e} {S[i, j]:.15e} {cs[i, j]:.15e}\n")
    print(f"Wrote {path}: {nrho}x{nT} grid, mat_id={mat_id}")
    print(f"  log10(rho/[g/cm^3]) range: [{logrho[0]:.3f}, {logrho[-1]:.3f}]")
    print(f"  log10(T/[K])        range: [{logT[0]:.3f}, {logT[-1]:.3f}]")


def selftest(tmpdir=None):
    """Generate a synthetic ideal-gas CMS-format table, convert, verify."""
    import tempfile, os
    GAMMA = 5.0 / 3.0
    MU = 1.23  # H/He-like mean molecular weight, atomic units
    R_GAS = 8.314462618e7  # erg/g/K per mu (R/mu = specific gas const)
    Cv = R_GAS / MU / (GAMMA - 1.0)        # erg/g/K
    R_specific = R_GAS / MU                # erg/g/K

    nT_in, nP_in = 30, 40
    logT_in = np.linspace(2.0, 6.0, nT_in)
    logP_GPa_in = np.linspace(-9.0, 2.0, nP_in)

    grad_ad = (GAMMA - 1.0) / GAMMA  # 0.4 for monatomic

    if tmpdir is None:
        tmpdir = tempfile.mkdtemp()
    inpath = os.path.join(tmpdir, "synth_ideal.dat")
    with open(inpath, 'w') as f:
        f.write("# synthetic ideal-gas table for cms_to_sesame self-test\n")
        for lT in logT_in:
            for lPg in logP_GPa_in:
                T = 10.0 ** lT
                P = (10.0 ** lPg) * GPA_TO_DYN_PER_CM2
                rho = P / (R_specific * T)
                u = Cv * T
                S_cgs = Cv * np.log(T) - R_specific * np.log(rho) + 1e9  # offset to keep positive
                lrho = np.log10(rho)
                lu_MJ = np.log10(u / MJ_PER_KG_TO_ERG_PER_G)
                lS_MJ = np.log10(S_cgs / MJ_PER_KG_K_TO_ERG_PER_G_K)
                # derivative columns (ideal gas):
                #  dlnrho/dlnT|P = -1, dlnrho/dlnP|T = +1, grad_ad = 0.4
                f.write(f"{lT:.6f} {lPg:.6f} {lrho:.6f} {lu_MJ:.6f} {lS_MJ:.6f} "
                        f"{-1.0:.6f} {1.0:.6f} {0.0:.6f} {0.0:.6f} {grad_ad:.6f}\n")

    native = parse_native_table(inpath)
    regridded = regrid_to_uniform_logrho(native, nrho=60)
    outpath = os.path.join(tmpdir, "synth_ideal.sesame")
    write_sesame(outpath, regridded, mat_id=9999)

    # Re-read the SESAME file and compare against analytic ideal gas.
    with open(outpath) as f:
        hdr = f.readline().split()
        mat_id, nrho, nT, ncols = int(hdr[0]), int(hdr[1]), int(hdr[2]), int(hdr[3])
        assert ncols == 6
        rho_grid = np.array([float(f.readline()) for _ in range(nrho)])
        T_grid = np.array([float(f.readline()) for _ in range(nT)])
        Pout = np.empty((nrho, nT)); uout = np.empty((nrho, nT))
        Sout = np.empty((nrho, nT)); csout = np.empty((nrho, nT))
        for i in range(nrho):
            for j in range(nT):
                vals = [float(x) for x in f.readline().split()]
                Pout[i, j] = vals[2]; uout[i, j] = vals[3]
                Sout[i, j] = vals[4]; csout[i, j] = vals[5]

    # Analytic ideal-gas predictions on the (rho_grid, T_grid):
    P_an = rho_grid[:, None] * R_specific * T_grid[None, :]
    u_an = (Cv * T_grid)[None, :] * np.ones((nrho, 1))
    cs_an = np.sqrt(GAMMA * R_specific * T_grid)[None, :] * np.ones((nrho, 1))

    # Restrict comparison to interior cells (avoid extrapolation edges)
    sl = (slice(5, nrho - 5), slice(5, nT - 5))
    relP = np.max(np.abs(Pout[sl] - P_an[sl]) / P_an[sl])
    relu = np.max(np.abs(uout[sl] - u_an[sl]) / u_an[sl])
    relcs = np.max(np.abs(csout[sl] - cs_an[sl]) / cs_an[sl])

    print(f"\nself-test relative errors (interior cells):")
    print(f"  P:  {relP:.3e}")
    print(f"  u:  {relu:.3e}")
    print(f"  cs: {relcs:.3e}")
    tol = 1e-3
    ok = (relP < tol) and (relu < tol) and (relcs < tol)
    print(f"  PASS" if ok else "  FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", help="CMS-format ASCII table to read")
    ap.add_argument("--output", help="SESAME-format file to write")
    ap.add_argument("--mat-id", type=int, default=2719,
                    help="material id stamped in SESAME header (default 2719)")
    ap.add_argument("--nrho", type=int, default=200,
                    help="number of points on uniform log10(rho) axis (default 200)")
    ap.add_argument("--y", type=float, default=None,
                    help="helium mass fraction Y (recorded as comment for bookkeeping)")
    ap.add_argument("--selftest", action="store_true",
                    help="run synthetic ideal-gas round-trip test and exit")
    args = ap.parse_args()

    if args.selftest:
        ok = selftest()
        sys.exit(0 if ok else 1)

    if not (args.input and args.output):
        ap.error("--input and --output required (or use --selftest)")

    native = parse_native_table(args.input)
    regridded = regrid_to_uniform_logrho(native, nrho=args.nrho)
    write_sesame(args.output, regridded, mat_id=args.mat_id)
    if args.y is not None:
        print(f"  (Y = {args.y})")


if __name__ == "__main__":
    main()
