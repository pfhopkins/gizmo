"""Convert a Chabrier-family H/He EOS table (CMS19, CD21, or compatible)
to the SESAME-format ASCII table consumed by GIZMO's eos/aneos.cc loader.

substellar H/He EOS support.

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
  All quantities CGS. The rho axis written here is uniform in log10(rho); the T axis
  is the native one. aneos.cc accepts any strictly increasing axes.
  ncols=6 (no phase column; H/He surface is smooth).

The CMS native grid is in (logT, logP). aneos.cc tabulates in (logrho, logT), so
we re-grid:
  1. Read native (T_i, P_j) -> rho, u, S, grad_ad on the native grid.
  2. Pick target uniform logrho axis covering [min, max] of native rho.
  3. For each target T_i (kept on native log-T axis), interpolate
     (P, u, S, grad_ad) at fixed T as functions of logrho (monotonic in P
     at fixed T, hence monotonic in rho).
  4. Sound speed: the adiabatic sound speed of the table's own P(rho, T) and u(rho, T) along the first-law adiabat GIZMO
     integrates (du = P / rho^2 drho):  cs^2 = P_rho + P_T (P / rho^2 - u_rho) / u_T,  derivatives taken in log space on
     the converted grid. (The published derivative columns are not consistent with the published values at the percent level,
     so they are compared against, and written only where the first-law cs^2 is not positive: below the floor, where the table
     is never evaluated, or anywhere with --no-defect-floor; the count is printed.)
  5. Validity floor (on by default): the published tables contain cold, dense states where u falls as T rises at fixed rho
     ((d u / d T)_rho < 0): there one energy names several temperatures, which no equation-of-state solve can resolve. The defects
     are found on the NATIVE (T, P) grid; the floor built from them depends on the output grid through its padding (a defect counts
     at the output densities within two spacings of it, and the floor sits two output temperature rows above it), and the check
     after conversion guarantees no defect remains above it at the chosen resolution. Each output density's lowest valid
     temperature row is made non-decreasing in density and slope-limited, so the boundary has no density-to-density jumps. This is a
     validity boundary, not a repair: states below it are outside the table (GIZMO takes the state at the boundary and counts
     it). The conversion refuses if any fall remains above the floor, or if cs^2 <= 0 there. --no-defect-floor writes the table
     without it, for auditing only.
  6. Emit SESAME-format file (with the lowest_valid_T_index block when a floor is set), and print the command, the input and
     output SHA-256 hashes and what the floor removed.

Self-test:
  Run with --selftest to generate a synthetic ideal-gas (logT, logP) input,
  convert it, and check thermodynamic consistency vs analytic gamma=5/3.

Usage:
  python cms_to_sesame.py --input cms19_y0245.dat --output cms19_y0245.sesame
  python cms_to_sesame.py --selftest
"""

import argparse
import hashlib
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
    lr_lo_per_T = np.nanmin(logrho_native, axis=1)   # points masked as corrupt (NaN) are left out everywhere below
    lr_hi_per_T = np.nanmax(logrho_native, axis=1)
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
        order = order[np.isfinite(lr_row[order])]
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
            order = order[np.isfinite(lr_row[order])]
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


NATIVE_DEFECT_THRESHOLD = -1.0e-6   # (dln u / dln T)_rho below this on the native grid marks a defect (flat-region noise is ~1e-15)
SOLVE_TOLERANCE_LOG = np.log10(1.0 + 1.0e-8)   # GIZMO's temperature-solve tolerance on u, as log10(u) (aneos.h)
FLOOR_MAX_SLOPE = 2.0   # steepest rise of the validity floor, in dex of T per dex of density


def first_law_soundspeed(regridded):
    """cs^2 = P_rho + P_T (P / rho^2 - u_rho) / u_T on the converted grid, every derivative from the logs (exact for power laws)."""
    lr, lT = regridded['logrho'], regridded['logT']
    P, u = regridded['P'], regridded['u']
    R, T = np.meshgrid(10.0 ** lr, 10.0 ** lT, indexing='ij')
    dl = lambda q, ax, x: np.gradient(np.log10(q), x, axis=ax)       # d log10(q) / d log10(x)
    P_rho, P_T = P / R * dl(P, 0, lr), P / T * dl(P, 1, lT)
    u_rho, u_T = u / R * dl(u, 0, lr), u / T * dl(u, 1, lT)
    with np.errstate(divide='ignore', invalid='ignore'):
        return P_rho + P_T * (P / R ** 2 - u_rho) / u_T


def mask_nonmonotone_density(native):
    """Along each native isotherm the density must rise strictly with pressure. Where a pair of neighbours violates that, the point
    farther from the median of its nearby pressures' densities is corrupt and is MASKED (set missing, never replaced): the regrid and
    the defect search then leave it out. Returns (native with the mask applied, report lines naming every masked point)."""
    lr = native['logrho'].copy(); masked = []
    for j in range(lr.shape[0]):
        while True:
            idx = np.nonzero(np.isfinite(lr[j]))[0]
            bad = np.nonzero(np.diff(lr[j, idx]) <= 0)[0]
            if bad.size == 0: break
            a, b = idx[bad[0]], idx[bad[0] + 1]
            def distance(k):
                near = idx[np.abs(idx - k) <= 3]; near = near[near != k]
                return abs(lr[j, k] - np.median(lr[j, near])) if near.size else 0.0
            k = a if distance(a) > distance(b) else b
            masked.append((native['logT'][j], native['logP'][k], lr[j, k]))
            lr[j, k] = np.nan
    out = dict(native); out['logrho'] = lr
    report = [f"native points masked (density not rising with pressure along the isotherm): {len(masked)}"]
    if masked:
        m = np.array(masked); values, counts = np.unique(np.round(m[:, 2], 6), return_counts=True)
        for v, c in zip(values, counts):
            s = np.abs(m[:, 2] - v) < 1e-6
            if c > 3:    # one value repeated: a placeholder the table stores, reported as one group
                report.append(f"  {c} points all with log10 rho = {v:.4f} (a placeholder): log10 T {m[s, 0].min():.2f}..{m[s, 0].max():.2f}, "
                              f"log10 P {m[s, 1].min():.2f}..{m[s, 1].max():.2f}")
            else:
                for lT, lP, x in m[s]: report.append(f"  log10 T {lT:.3f}, log10 P {lP:.3f}: log10 rho {x:.4f}")
    return out, report


def native_defects(native):
    """(dln u / dln T)_rho on the native (T, P) grid from its own centred differences:
    (dln u/dln T)_P - (dln u/dln P)_T (dln rho/dln T)_P / (dln rho/dln P)_T. Returns the defect points' (log10 rho, log10 T)."""
    lT, lP, lr, lu = native['logT'], native['logP'], native['logrho'], native['logu']
    with np.errstate(divide='ignore', invalid='ignore'):
        cv = (np.gradient(lu, lT, axis=0)
              - np.gradient(lu, lP, axis=1) * np.gradient(lr, lT, axis=0) / np.gradient(lr, lP, axis=1))
    bad = np.isfinite(cv) & (cv < NATIVE_DEFECT_THRESHOLD)
    return lr[bad], np.broadcast_to(lT[:, None], lr.shape)[bad]


def defect_floor(native, regridded):
    """Per output density, the lowest valid temperature row: above every native defect within two output density spacings,
    made non-decreasing in density. Returns the row indices (0 = no floor)."""
    lr_out, lT_out = regridded['logrho'], regridded['logT']
    d_lr, d_lT = native_defects(native)
    reach = 2.0 * (lr_out[1] - lr_out[0])   # a native point influences the output densities within two spacings of it
    floor_lT = np.full(lr_out.size, -np.inf)
    for i, x in enumerate(lr_out):
        near = np.abs(d_lr - x) <= reach
        if near.any(): floor_lT[i] = d_lT[near].max()
    floor_lT = np.maximum.accumulate(floor_lT)                                   # non-decreasing in density
    # no steps: each rise is ramped in from lower densities at FLOOR_MAX_SLOPE (it only ever raises the floor)
    if np.isfinite(floor_lT).any():
        ramp = np.array([np.max(floor_lT[i:] - FLOOR_MAX_SLOPE * (lr_out[i:] - lr_out[i])) for i in range(lr_out.size)])
        floor_lT = np.maximum(floor_lT, ramp)
    has_floor = floor_lT >= lT_out[0]                                            # a ramp below the table's coldest row sets nothing
    rows = np.where(has_floor, np.searchsorted(lT_out, floor_lT, side='right') + 2, 0)   # two rows clear of the defect's stencil
    if np.any(rows > lT_out.size - 2):
        raise ValueError("the defect floor leaves no valid temperature cell at some density")
    return rows


def restrict_pressure(native, logP_min=None, logP_max=None):
    """Keep only the native pressure columns inside [logP_min, logP_max] (log10 GPa; None = no bound): the published tables hold
    placeholder values in their extreme-pressure corners (CD21: u up to 1e199 MJ/kg, below ~2 mbar and near 1e13 GPa), which the
    regrid would otherwise blend into real states. A validity boundary of the source, like the defect floor."""
    keep = np.ones(native['logP'].size, bool)
    if logP_min is not None: keep &= native['logP'] >= logP_min
    if logP_max is not None: keep &= native['logP'] <= logP_max
    if keep.sum() < 3: raise ValueError("the pressure range keeps fewer than 3 native pressures")
    return {k: (v[keep] if k == 'logP' else (v[:, keep] if (v is not None and np.ndim(v) == 2) else v)) for k, v in native.items()}


def convert(native, nrho, use_defect_floor=True):
    """The whole conversion: regrid, first-law sound speed, validity floor; refuses an inconsistent result. Returns (regridded, floor
    rows or None, report lines)."""
    native, mask_report = mask_nonmonotone_density(native)
    rg = regrid_to_uniform_logrho(native, nrho=nrho)
    cs_published = rg['cs']
    cs2 = first_law_soundspeed(rg)
    rows = defect_floor(native, rg) if use_defect_floor else np.zeros(rg['logrho'].size, int)
    lu = np.log10(rg['u']); falls = np.diff(lu, axis=1) < -SOLVE_TOLERANCE_LOG          # cell (j, j+1) of column i falls
    above = np.arange(rg['logT'].size)[None, :] >= rows[:, None]                         # rows at or above the floor
    falls_above = falls & above[:, :-1]
    bad_cs = ~(np.isfinite(cs2) & (cs2 > 0)) & above
    T = 10.0 ** rg['logT']; rho = 10.0 ** rg['logrho']
    report = list(mask_report)
    if use_defect_floor:
        raised = rows > 0
        report.append(f"defect floor: {raised.sum()} of {rho.size} densities; native defect points {native_defects(native)[0].size}")
        if raised.any():
            for lo, hi in [(0, 0.5), (0.5, 3), (3, 30), (30, np.inf)]:
                s = raised & (rho >= lo) & (rho < hi)
                if s.any(): report.append(f"  rho {lo:g}-{hi:g} g/cc: {s.sum()} densities, floor T {T[rows[s]].min():.4g}-{T[rows[s]].max():.4g} K")
            jumps = np.abs(np.diff(np.log10(T[rows])))[raised[1:] & raised[:-1]]
            report.append(f"  largest floor step between neighbouring densities: {jumps.max() if jumps.size else 0:.3f} dex in T")
    report.append(f"u falling with T above the valid floor: {falls_above.sum()} cells; cs^2 <= 0 above it: {bad_cs.sum()} points")
    if falls_above.any():
        i, j = np.nonzero(falls_above)
        report.append(f"  falls at rho {rho[i].min():.3g}-{rho[i].max():.3g} g/cc, T {T[j].min():.4g}-{T[j].max():.4g} K")
    if use_defect_floor and (falls_above.any() or bad_cs.any()):
        raise ValueError("\n".join(report + ["refusing: the validity floor leaves a defect above it"]))
    # where the first-law cs^2 is not positive (only below the floor, where the table is never evaluated -- or anywhere with
    # --no-defect-floor) the published-derivative value is written instead, so the file stays finite; the count is reported
    substituted = ~(np.isfinite(cs2) & (cs2 > 0))
    report.append(f"published-derivative cs written where the first-law cs^2 is not positive: {substituted.sum()} points "
                  f"({(substituted & above).sum()} of them at or above the floor)")
    rg['cs'] = np.where(~(np.isfinite(cs2) & (cs2 > 0)), cs_published, np.sqrt(np.where(np.isfinite(cs2) & (cs2 > 0), cs2, 1.0)))
    ratio = cs_published[above & np.isfinite(cs2) & (cs2 > 0)] / rg['cs'][above & np.isfinite(cs2) & (cs2 > 0)]
    if ratio.size: report.append("published-derivative cs / first-law cs above the floor: p1/p50/p99 "
                                 + "/".join(f"{np.percentile(ratio, q):.3f}" for q in (1, 50, 99)))
    return rg, (rows if use_defect_floor else None), report


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''): h.update(block)
    return h.hexdigest()


def write_sesame(path, regridded, mat_id, floor_rows=None):
    logrho = regridded['logrho']
    logT = regridded['logT']
    nrho, nT = logrho.size, logT.size
    rho_arr = 10.0 ** logrho
    T_arr = 10.0 ** logT
    P, u, S, cs = regridded['P'], regridded['u'], regridded['S'], regridded['cs']

    # Write the T axis the data were computed on; aneos.cc requires only strictly increasing axes.
    if not (np.all(np.diff(logrho) > 0) and np.all(np.diff(logT) > 0)):
        raise ValueError("axes must be strictly increasing")

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
        if floor_rows is not None:
            f.write("lowest_valid_T_index\n" + "\n".join(str(int(k)) for k in floor_rows) + "\n")
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
    regridded, rows, report = convert(native, nrho=60)
    print("\n".join(report))
    outpath = os.path.join(tmpdir, "synth_ideal.sesame")
    write_sesame(outpath, regridded, mat_id=9999, floor_rows=rows)
    if rows is not None and np.any(rows > 0):
        print("self-test: an ideal gas must get no defect floor"); return False

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

    # Known positives for the validity machinery: the same gas with u = Cv T (1 + 4 (1e3 K / T)^2), which falls as T rises below
    # 2000 K at every density, and one native density corrupted. The conversion must mask exactly that point, set a floor above
    # 2000 K with nothing falling above it, write a block that reads back exactly, and report the falls when the floor is off.
    defpath = os.path.join(tmpdir, "synth_defect.dat")
    with open(defpath, 'w') as f:
        for i_T, lT in enumerate(logT_in):
            for i_P, lPg in enumerate(logP_GPa_in):
                T = 10.0 ** lT; P = (10.0 ** lPg) * GPA_TO_DYN_PER_CM2
                lrho = np.log10(P / (R_specific * T)) - (5.0 if (i_T, i_P) == (12, 20) else 0.0)
                u = Cv * T * (1.0 + 4.0 * (1.0e3 / T) ** 2)
                f.write(f"{lT:.6f} {lPg:.6f} {lrho:.6f} {np.log10(u / MJ_PER_KG_TO_ERG_PER_G):.6f} {0.0:.6f} "
                        f"{-1.0:.6f} {1.0:.6f} {0.0:.6f} {0.0:.6f} {grad_ad:.6f}\n")
    nat_def = parse_native_table(defpath)
    rg_def, rows_def, rep_def = convert(nat_def, nrho=60)
    print("\n".join(rep_def))
    T_def = 10.0 ** rg_def['logT']
    masked_ok = any(l.startswith("native points masked") and l.rstrip().endswith(": 1") for l in rep_def)
    floor_ok = rows_def is not None and np.all(rows_def > 0) and T_def[rows_def].min() > 2.0e3
    defout = os.path.join(tmpdir, "synth_defect.sesame")
    write_sesame(defout, rg_def, mat_id=9998, floor_rows=rows_def)
    tok = open(defout).read().split()
    n_head = 4 + rg_def['logrho'].size + T_def.size + 6 * rg_def['logrho'].size * T_def.size
    block_ok = tok[n_head] == "lowest_valid_T_index" and np.array_equal(np.array(tok[n_head + 1:], int), rows_def)
    _, _, rep_audit = convert(nat_def, nrho=60, use_defect_floor=False)
    audit_ok = any(l.startswith("u falling with T above the valid floor:") and not l.startswith("u falling with T above the valid floor: 0 ")
                   for l in rep_audit)
    print(f"  injected defects: corrupt point masked {masked_ok}, floor above 2000 K {floor_ok}, block reads back {block_ok}, "
          f"audit mode reports the falls {audit_ok}")
    ok = ok and masked_ok and floor_ok and block_ok and audit_ok
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
    ap.add_argument("--logP-min", type=float, default=-6.5,
                    help="lowest native log10(P / GPa) to use (default -6.5: CD21 holds placeholder values below ~2 mbar)")
    ap.add_argument("--logP-max", type=float, default=8.0,
                    help="highest native log10(P / GPa) to use (default 8: and near 1e13 GPa)")
    ap.add_argument("--full-pressure-range", action="store_true",
                    help="use every native pressure, placeholder corners included (auditing only)")
    ap.add_argument("--no-defect-floor", action="store_true",
                    help="write the table without the validity floor (auditing only: one energy then names several temperatures in places)")
    ap.add_argument("--selftest", action="store_true",
                    help="run synthetic ideal-gas round-trip test and exit")
    args = ap.parse_args()

    if args.selftest:
        ok = selftest()
        sys.exit(0 if ok else 1)

    if not (args.input and args.output):
        ap.error("--input and --output required (or use --selftest)")

    if args.full_pressure_range:
        args.logP_min = args.logP_max = None
        print("WARNING: --full-pressure-range keeps the table's placeholder corners: for auditing only")
    native = restrict_pressure(parse_native_table(args.input), args.logP_min, args.logP_max)
    print(f"native pressures used: 10^{native['logP'][0]:.2f} .. 10^{native['logP'][-1]:.2f} GPa ({native['logP'].size} of the table)")
    regridded, rows, report = convert(native, nrho=args.nrho, use_defect_floor=not args.no_defect_floor)
    print("\n".join(report))
    if args.no_defect_floor:
        print("WARNING: written WITHOUT the validity floor (--no-defect-floor): for auditing only")
    write_sesame(args.output, regridded, mat_id=args.mat_id, floor_rows=rows)
    if args.y is not None:
        print(f"  (Y = {args.y})")
    print("command: " + " ".join(sys.argv))
    print(f"input  sha256 {sha256(args.input)}  {args.input}")
    print(f"output sha256 {sha256(args.output)}  {args.output}")


if __name__ == "__main__":
    main()
