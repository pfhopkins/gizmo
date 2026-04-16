#!/usr/bin/env python3
"""
Unit tests for stellar_tables_unified.hdf5 — checks the raw table data
and simulates the C++ interpolation logic to catch bad entries.

Motivated by PPISN crash (job 150573): M=150 at Z=0.001 had rem_mass=4.22
instead of ~50, producing Z_ej=0.51 ejecta from a 0.1-solar progenitor.
"""

import sys
import numpy as np

TABLE_PATH = "galaxy_sf/stellar_tables/stellar_tables_unified.hdf5"

# Remnant type codes (must match resolvedism_stellar_tables.h)
REM_WD    = 0
REM_ECSN  = 1
REM_CCSN  = 2
REM_FSN   = 3
REM_PPISN = 4
REM_PISN  = 5
REM_DBH   = 6

REM_NAMES = {0: "WD", 1: "ECSN", 2: "CCSN", 3: "FSN", 4: "PPISN", 5: "PISN", 6: "DBH"}

# Solar metallicity (project convention)
SOLAR_Z = 0.02

# Element indices
ELEM_H  = 0
ELEM_He = 1
ELEM_C  = 2

nfail_total = 0
npass_total = 0


def load_table():
    import h5py
    f = h5py.File(TABLE_PATH, "r")
    tbl = {
        "M": f["M_init"][:],
        "Z": f["Z"][:],
        "logM": f["log_M_init"][:],
        "logZ": f["log_Z"][:],
        "rem_type": f["remnant_type"][:],
        "rem_mass": f["remnant_mass"][:],
        "net_yields": f["net_yields"][:],
        "elems": [e.decode() for e in f["elements"][:]],
    }
    f.close()
    return tbl


def check(condition, msg, detail=""):
    global nfail_total, npass_total
    if condition:
        npass_total += 1
    else:
        nfail_total += 1
        print(f"  FAIL: {msg}")
        if detail:
            print(f"        {detail}")


def test_pisn_remnant_zero(tbl):
    """PISN must have remnant mass = 0 (complete disruption)."""
    print("\n--- PISN remnant = 0 ---")
    for iz in range(len(tbl["Z"])):
        for im in range(len(tbl["M"])):
            rt = int(tbl["rem_type"][iz, im])
            rm = tbl["rem_mass"][iz, im]
            if rt == REM_PISN:
                check(rm < 0.01,
                      f"PISN with rem_mass={rm:.2f} at M={tbl['M'][im]:.1f} Z={tbl['Z'][iz]:.4f}")
    print("  done")


def test_ppisn_remnant_massive(tbl):
    """PPISN must leave a massive BH remnant (>30 Msun)."""
    print("\n--- PPISN remnant > 30 Msun ---")
    for iz in range(len(tbl["Z"])):
        for im in range(len(tbl["M"])):
            rt = int(tbl["rem_type"][iz, im])
            rm = tbl["rem_mass"][iz, im]
            if rt == REM_PPISN:
                check(rm > 30.0,
                      f"PPISN with rem_mass={rm:.2f} at M={tbl['M'][im]:.1f} Z={tbl['Z'][iz]:.4f}",
                      f"Expected >30 Msun for pulsational pair-instability BH")
    print("  done")


def test_remnant_less_than_minit(tbl):
    """Remnant mass must be < M_init for all entries."""
    print("\n--- Remnant < M_init ---")
    for iz in range(len(tbl["Z"])):
        for im in range(len(tbl["M"])):
            rm = tbl["rem_mass"][iz, im]
            M = tbl["M"][im]
            check(rm <= M * 1.01,
                  f"rem_mass={rm:.2f} > M_init={M:.2f} at Z={tbl['Z'][iz]:.4f}")
    print("  done")


def test_remnant_positive_non_pisn(tbl):
    """Non-PISN entries with M >= 0.5 must have remnant mass > 0."""
    print("\n--- Non-PISN remnant > 0 ---")
    for iz in range(len(tbl["Z"])):
        for im in range(len(tbl["M"])):
            rt = int(tbl["rem_type"][iz, im])
            rm = tbl["rem_mass"][iz, im]
            M = tbl["M"][im]
            if rt != REM_PISN and M >= 0.5:
                check(rm > 0,
                      f"rem_mass=0 for {REM_NAMES.get(rt, rt)} at M={M:.2f} Z={tbl['Z'][iz]:.4f}")
    print("  done")


def test_remnant_continuity(tbl):
    """Remnant mass should not jump by >10x between adjacent mass grid points
    within the same remnant type (for explosive types)."""
    print("\n--- Remnant mass continuity ---")
    skip_types = {REM_WD, REM_FSN, REM_DBH}
    for iz in range(len(tbl["Z"])):
        for im in range(1, len(tbl["M"])):
            rt = int(tbl["rem_type"][iz, im])
            rt_prev = int(tbl["rem_type"][iz, im - 1])
            if rt != rt_prev or rt in skip_types:
                continue
            rm = tbl["rem_mass"][iz, im]
            rm_prev = tbl["rem_mass"][iz, im - 1]
            if rm_prev < 1.0 or rm < 1.0:
                continue
            ratio = rm / rm_prev
            check(0.1 < ratio < 10.0,
                  f"rem jump {rm_prev:.1f}->{rm:.1f} (x{ratio:.1f}) "
                  f"M={tbl['M'][im-1]:.0f}->{tbl['M'][im]:.0f} Z={tbl['Z'][iz]:.4f} type={REM_NAMES.get(rt, rt)}")
    print("  done")


def test_ejecta_metallicity(tbl):
    """Ejecta metallicity should not exceed 50% for any massive star entry.
    Uses same yield computation as the C++ code."""
    print("\n--- Ejecta metallicity < 50% ---")
    solar_X = np.array([
        0.7381, 0.2485, 2.36e-3, 6.91e-4, 5.72e-3, 3.26e-7, 1.25e-3,
        2.98e-5, 5.91e-4, 5.57e-5, 6.65e-4, 5.16e-6, 3.10e-4, 3.15e-6,
        7.37e-5, 2.93e-6, 6.44e-5, 3.48e-8, 3.59e-6, 2.30e-7, 1.37e-5,
        9.17e-6, 1.17e-3, 3.30e-6, 6.99e-5, 7.20e-7, 1.67e-6
    ])

    for iz in range(len(tbl["Z"])):
        Z_birth = tbl["Z"][iz]
        Z_frac = Z_birth / SOLAR_Z
        X_birth = solar_X.copy()
        X_birth[:2] *= (1.0 - Z_birth) / (1.0 - SOLAR_Z)
        X_birth[2:] *= Z_frac

        for im in range(len(tbl["M"])):
            M = tbl["M"][im]
            if M < 8.0:
                continue
            rt = int(tbl["rem_type"][iz, im])
            if rt in (REM_FSN, REM_DBH, REM_PISN):
                continue  # FSN/DBH have no yields; PISN are legitimately metal-rich
            rm = tbl["rem_mass"][iz, im]
            Mej = M - rm
            if Mej < 0.1:
                continue

            net_y = tbl["net_yields"][iz, im, :]
            metal_mass = 0.0
            for k in range(ELEM_C, len(net_y)):
                M_elem_ej = net_y[k] + X_birth[k] * Mej
                if M_elem_ej > 0:
                    metal_mass += M_elem_ej
            Z_ej = metal_mass / Mej

            check(Z_ej < 0.5,
                  f"Z_ej={Z_ej:.3f} ({Z_ej/SOLAR_Z:.0f}x solar) at M={M:.1f} Z={Z_birth:.4f} "
                  f"type={REM_NAMES.get(rt, rt)} rem={rm:.1f} Mej={Mej:.1f}")
    print("  done")


def test_specific_ppisn_150(tbl):
    """The specific entry that caused the crash: M=150 at Z=0.001.
    This entry has rem_type=PPISN but rem_mass=4.22 — clearly wrong."""
    print("\n--- Specific: M=150 Z=0.001 PPISN entry ---")
    im = np.argmin(np.abs(tbl["M"] - 150.0))
    iz = np.argmin(np.abs(tbl["Z"] - 0.001))

    rt = int(tbl["rem_type"][iz, im])
    rm = tbl["rem_mass"][iz, im]
    M = tbl["M"][im]
    Z = tbl["Z"][iz]

    print(f"  M={M:.1f}, Z={Z:.4f}, rem_type={REM_NAMES.get(rt, rt)}, rem_mass={rm:.3f}")
    print(f"  Mej={M - rm:.1f}")

    check(rt == REM_PPISN or rt == REM_PISN,
          f"Expected PPISN or PISN at M=150 Z=0.001, got {REM_NAMES.get(rt, rt)}")

    if rt == REM_PPISN:
        check(rm > 30.0,
              f"PPISN rem_mass={rm:.2f} — should be >30 Msun (massive BH)",
              "This is the root cause of the job 150573 crash")

    # Check net yields
    net_y = tbl["net_yields"][iz, im, :]
    print(f"  Key net yields: H={net_y[0]:.1f} He={net_y[1]:.1f} C={net_y[2]:.1f} O={net_y[4]:.1f}")
    metal_sum = sum(net_y[ELEM_C:])
    print(f"  Total net metal yield: {metal_sum:.1f} Msun")

    # Compare with neighbors
    for im2 in [im - 1, im + 1]:
        if 0 <= im2 < len(tbl["M"]):
            rt2 = int(tbl["rem_type"][iz, im2])
            rm2 = tbl["rem_mass"][iz, im2]
            print(f"  Neighbor M={tbl['M'][im2]:.1f}: type={REM_NAMES.get(rt2, rt2)} rem={rm2:.2f}")
    print("  done")


def test_interpolation_146msun(tbl):
    """Simulate linear interpolation for M=146 at Z=0.001 (between grid points).
    The C++ code uses linear interpolation in log(M)."""
    print("\n--- Interpolation: M=146 at Z=0.001 ---")
    iz = np.argmin(np.abs(tbl["Z"] - 0.001))

    logM_target = np.log10(146.0)
    logM = tbl["logM"]

    # Find bracketing indices
    im_lo = np.searchsorted(logM, logM_target) - 1
    im_hi = im_lo + 1
    if im_hi >= len(logM):
        print("  SKIP: M=146 beyond grid")
        return

    M_lo, M_hi = tbl["M"][im_lo], tbl["M"][im_hi]
    rm_lo = tbl["rem_mass"][iz, im_lo]
    rm_hi = tbl["rem_mass"][iz, im_hi]
    rt_lo = int(tbl["rem_type"][iz, im_lo])
    rt_hi = int(tbl["rem_type"][iz, im_hi])

    # Linear interpolation weight in log(M)
    w = (logM_target - logM[im_lo]) / (logM[im_hi] - logM[im_lo])
    rm_interp = rm_lo * (1.0 - w) + rm_hi * w

    print(f"  M_lo={M_lo:.0f} (type={REM_NAMES.get(rt_lo, rt_lo)} rem={rm_lo:.1f})")
    print(f"  M_hi={M_hi:.0f} (type={REM_NAMES.get(rt_hi, rt_hi)} rem={rm_hi:.1f})")
    print(f"  weight={w:.3f}")
    print(f"  Interpolated rem_mass={rm_interp:.1f} for M=146")

    Mej_interp = 146.0 - rm_interp
    print(f"  Interpolated Mej={Mej_interp:.1f}")

    # With the current broken table, this interpolation produces nonsense
    check(rm_interp > 30.0,
          f"Interpolated PPISN rem_mass={rm_interp:.1f} at M=146 — should be >30",
          f"Interpolating between rem={rm_lo:.1f} (M={M_lo:.0f}) and rem={rm_hi:.1f} (M={M_hi:.0f})")

    # Interpolate net yields and compute ejecta Z
    ny_lo = tbl["net_yields"][iz, im_lo, :]
    ny_hi = tbl["net_yields"][iz, im_hi, :]
    ny_interp = ny_lo * (1.0 - w) + ny_hi * w
    metal_yield = sum(max(0, ny_interp[k]) for k in range(ELEM_C, len(ny_interp)))
    if Mej_interp > 0:
        Z_ej = metal_yield / Mej_interp
        print(f"  Interpolated ejecta Z={Z_ej:.3f} ({Z_ej/SOLAR_Z:.0f}x solar)")
        check(Z_ej < 0.5,
              f"Interpolated Z_ej={Z_ej:.3f} at M=146 — unphysical")
    print("  done")


def test_ppisn_pisn_boundary(tbl):
    """At fixed Z, the transition PPISN->PISN should happen at most once
    with increasing mass, and not oscillate."""
    print("\n--- PPISN/PISN boundary monotonicity ---")
    for iz in range(len(tbl["Z"])):
        transitions = []
        for im in range(1, len(tbl["M"])):
            rt = int(tbl["rem_type"][iz, im])
            rt_prev = int(tbl["rem_type"][iz, im - 1])
            if rt != rt_prev and rt in (REM_PPISN, REM_PISN) and rt_prev in (REM_PPISN, REM_PISN):
                transitions.append((tbl["M"][im - 1], tbl["M"][im],
                                    REM_NAMES[rt_prev], REM_NAMES[rt]))
        # Should have at most one PPISN->PISN transition
        ppisn_to_pisn = [t for t in transitions if t[2] == "PPISN" and t[3] == "PISN"]
        pisn_to_ppisn = [t for t in transitions if t[2] == "PISN" and t[3] == "PPISN"]
        check(len(pisn_to_ppisn) == 0,
              f"PISN->PPISN reverse transition at Z={tbl['Z'][iz]:.4f}: {pisn_to_ppisn}",
              "Type boundary should be monotonic in mass")
    print("  done")


# ============================================================
# Interpolation fidelity tests
# Reimplement the C++ interp2d / nearest-neighbor logic and
# verify it reproduces table values at grid points and stays
# bounded between grid points.
# ============================================================

def _bsearch(grid, val):
    """Replicates stbl_idx_bsearch: returns (i0, f) where
    grid[i0] <= val < grid[i0+1] and f is the fractional position."""
    N = len(grid)
    if val <= grid[0]:
        return 0, 0.0
    if val >= grid[N - 1]:
        return N - 2, 1.0
    i0 = int(np.searchsorted(grid, val, side="right")) - 1
    i0 = max(0, min(i0, N - 2))
    f = (val - grid[i0]) / (grid[i0 + 1] - grid[i0])
    return i0, f


def _interp2d(arr, logM, logZ, tbl):
    """Replicates C++ interp2d for a 2D [NZ x NM] array."""
    iz, fz = _bsearch(tbl["logZ"], logZ)
    im, fm = _bsearch(tbl["logM"], logM)
    v00 = arr[iz,     im]
    v01 = arr[iz,     im + 1]
    v10 = arr[iz + 1, im]
    v11 = arr[iz + 1, im + 1]
    return (1 - fz) * (1 - fm) * v00 + (1 - fz) * fm * v01 + fz * (1 - fm) * v10 + fz * fm * v11


def _nearest2d(arr, logM, logZ, tbl):
    """Replicates C++ stellar_remnant_type (nearest-neighbor)."""
    iz, fz = _bsearch(tbl["logZ"], logZ)
    im, fm = _bsearch(tbl["logM"], logM)
    iz_near = iz + 1 if fz > 0.5 else iz
    im_near = im + 1 if fm > 0.5 else im
    iz_near = min(iz_near, len(tbl["Z"]) - 1)
    im_near = min(im_near, len(tbl["M"]) - 1)
    return int(arr[iz_near, im_near])


def test_interp_at_grid_points(tbl):
    """At exact grid points, interp2d must return the exact table value
    for remnant_mass and net_yields."""
    print("\n--- Interpolation at grid points ---")
    for iz in range(len(tbl["Z"])):
        logZ = tbl["logZ"][iz]
        for im in range(len(tbl["M"])):
            logM = tbl["logM"][im]

            # Remnant mass
            rm_table = tbl["rem_mass"][iz, im]
            rm_interp = _interp2d(tbl["rem_mass"], logM, logZ, tbl)
            if rm_table != 0:
                reldiff = abs(rm_interp - rm_table) / max(abs(rm_table), 1e-30)
            else:
                reldiff = abs(rm_interp)
            check(reldiff < 1e-10,
                  f"rem_mass interp={rm_interp:.4f} vs table={rm_table:.4f} "
                  f"(reldiff={reldiff:.2e}) at M={tbl['M'][im]:.1f} Z={tbl['Z'][iz]:.4f}")

            # Remnant type (nearest-neighbor)
            rt_table = int(tbl["rem_type"][iz, im])
            rt_interp = _nearest2d(tbl["rem_type"], logM, logZ, tbl)
            check(rt_interp == rt_table,
                  f"rem_type interp={rt_interp} vs table={rt_table} "
                  f"at M={tbl['M'][im]:.1f} Z={tbl['Z'][iz]:.4f}")

            # Net yields (check C and O)
            for k in [ELEM_C, 4]:  # C and O
                ny_table = tbl["net_yields"][iz, im, k]
                ny_interp = _interp2d(tbl["net_yields"][:, :, k], logM, logZ, tbl)
                if abs(ny_table) > 1e-6:
                    reldiff_y = abs(ny_interp - ny_table) / abs(ny_table)
                else:
                    reldiff_y = abs(ny_interp - ny_table)
                check(reldiff_y < 1e-10,
                      f"net_yield[{tbl['elems'][k]}] interp={ny_interp:.4e} vs table={ny_table:.4e} "
                      f"at M={tbl['M'][im]:.1f} Z={tbl['Z'][iz]:.4f}")
    print("  done")


def test_interp_between_grid_bounded(tbl):
    """Between grid points, interpolated remnant_mass must be bounded by
    the four surrounding corner values (within bilinear convex hull).
    Tests midpoints between all adjacent grid pairs."""
    print("\n--- Interpolation between grid points: bounded ---")
    nZ = len(tbl["Z"])
    nM = len(tbl["M"])
    for iz in range(nZ - 1):
        logZ_mid = 0.5 * (tbl["logZ"][iz] + tbl["logZ"][iz + 1])
        for im in range(nM - 1):
            logM_mid = 0.5 * (tbl["logM"][im] + tbl["logM"][im + 1])

            corners = [
                tbl["rem_mass"][iz, im],
                tbl["rem_mass"][iz, im + 1],
                tbl["rem_mass"][iz + 1, im],
                tbl["rem_mass"][iz + 1, im + 1],
            ]
            lo = min(corners)
            hi = max(corners)
            rm_interp = _interp2d(tbl["rem_mass"], logM_mid, logZ_mid, tbl)

            # Bilinear interpolation is bounded by corners
            check(rm_interp >= lo - 1e-10 and rm_interp <= hi + 1e-10,
                  f"rem_mass interp={rm_interp:.2f} outside [{lo:.2f}, {hi:.2f}] "
                  f"at M_mid={10**logM_mid:.1f} Z_mid={10**logZ_mid:.4f}")
    print("  done")


def test_interp_massive_star_sweep(tbl):
    """Sweep M from 100 to 300 Msun at Z=0.001 in fine steps.
    Check that interpolated remnant_mass doesn't produce ejecta Z > 50%
    and that remnant_mass doesn't drop below 0."""
    print("\n--- Interpolation sweep M=100-300 at Z=0.001 ---")
    iz = np.argmin(np.abs(tbl["Z"] - 0.001))
    logZ = tbl["logZ"][iz]

    solar_X = np.array([
        0.7381, 0.2485, 2.36e-3, 6.91e-4, 5.72e-3, 3.26e-7, 1.25e-3,
        2.98e-5, 5.91e-4, 5.57e-5, 6.65e-4, 5.16e-6, 3.10e-4, 3.15e-6,
        7.37e-5, 2.93e-6, 6.44e-5, 3.48e-8, 3.59e-6, 2.30e-7, 1.37e-5,
        9.17e-6, 1.17e-3, 3.30e-6, 6.99e-5, 7.20e-7, 1.67e-6
    ])
    Z_birth = tbl["Z"][iz]
    X_birth = solar_X.copy()
    X_birth[:2] *= (1.0 - Z_birth) / (1.0 - 0.014)
    X_birth[2:] *= Z_birth / 0.014

    masses = np.linspace(100, 300, 201)
    for M_test in masses:
        logM = np.log10(M_test)
        rm = _interp2d(tbl["rem_mass"], logM, logZ, tbl)
        rt = _nearest2d(tbl["rem_type"], logM, logZ, tbl)

        check(rm >= 0,
              f"Negative rem_mass={rm:.2f} at M={M_test:.1f}")

        if rt in (REM_FSN, REM_DBH):
            continue
        Mej = M_test - rm
        if Mej < 0.1:
            continue

        # Compute ejecta Z from interpolated yields
        metal_mass = 0.0
        for k in range(ELEM_C, 27):
            ny = _interp2d(tbl["net_yields"][:, :, k], logM, logZ, tbl)
            M_elem_ej = ny + X_birth[k] * Mej
            if M_elem_ej > 0:
                metal_mass += M_elem_ej
        Z_ej = metal_mass / Mej

        # PISN are legitimately metal-rich, skip them
        if rt != REM_PISN:
            check(Z_ej < 0.5,
                  f"Interpolated Z_ej={Z_ej:.3f} ({Z_ej/SOLAR_Z:.0f}x solar) "
                  f"at M={M_test:.1f} type={REM_NAMES.get(rt, rt)} rem={rm:.1f}")
    print("  done")


if __name__ == "__main__":
    print(f"Loading table: {TABLE_PATH}")
    tbl = load_table()
    print(f"Grid: {len(tbl['Z'])} Z x {len(tbl['M'])} M, {len(tbl['elems'])} elements")

    test_pisn_remnant_zero(tbl)
    test_ppisn_remnant_massive(tbl)
    test_remnant_less_than_minit(tbl)
    test_remnant_positive_non_pisn(tbl)
    test_remnant_continuity(tbl)
    test_ejecta_metallicity(tbl)
    test_specific_ppisn_150(tbl)
    test_interpolation_146msun(tbl)
    test_ppisn_pisn_boundary(tbl)
    test_interp_at_grid_points(tbl)
    test_interp_between_grid_bounded(tbl)
    test_interp_massive_star_sweep(tbl)

    print(f"\n{'='*60}")
    print(f"Results: {npass_total} passed, {nfail_total} FAILED")
    if nfail_total > 0:
        print(f"TABLE HAS {nfail_total} PROBLEM(S)")
        sys.exit(1)
    else:
        print("All checks passed")
        sys.exit(0)
