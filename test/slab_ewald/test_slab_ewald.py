"""slab_ewald: forces from GRAVITY_PERIODIC_XY (self-gravity periodic or shear-periodic in x/y, open in z).

Sources (one heavy particle + a random cloud) and massless tracers on a 1 x 1 x 4 box. The tracer accelerations
are compared with an independent 2D Ewald sum over the (shear-)periodic lattice, which is itself checked against
a brute-force lattice sum. The shear variant (BOX_SHEARING=3, q = Omega = 1, t = 0.37) has its x-images shifted
by s = 0.37 in y, and adds the shearing-sheet tidal term 2 q Omega^2 (x - L/2), which is subtracted here.
Without the x/y image correction (nearest image only) the errors are O(1).
"""

import pytest
import numpy as np
import h5py
from pathlib import Path
from scipy.special import erf, erfc

from gizmo.test import build_and_run_test, default_mpi_ranks, get_final_snapshot

TEST_NAME = "slab_ewald"
TEST_DIR = Path(__file__).parent
SHEAR_FLAGS = ("BOX_SHEARING=3", "BOX_SHEARING_Q=1")
T0 = 0.37


def generate_ics():
    ic_file = TEST_DIR / f"{TEST_NAME}_ics.hdf5"
    if not ic_file.exists():
        from make_slab_ewald_ics import make_slab_ewald_ics
        make_slab_ewald_ics(str(ic_file))


def lattice_accel(dr, s, L=1.0):
    """acceleration at the target from unit masses on the full lattice, dr = source - target (N,3), G = 1"""
    A, alpha = L * L, 2.0 / L
    r = -np.asarray(dr, dtype=float)  # target - source
    g = np.zeros_like(r)
    s = s - L * np.floor(s / L + 0.5)
    for n in range(-5, 6):
        for m in range(-6, 7):
            p = r + np.array([n * L, -n * s + m * L, 0.0])
            rho = np.linalg.norm(p, axis=1)
            dfdr = -erfc(alpha * rho) / rho**2 - 2 * alpha / np.sqrt(np.pi) * np.exp(-(alpha * rho) ** 2) / rho
            g += (dfdr / rho)[:, None] * p
    for h in range(-8, 9):
        for l in range(-8, 9):
            if h == 0 and l == 0:
                continue
            kx, ky = 2 * np.pi * h / L + 2 * np.pi * l * s / A, 2 * np.pi * l / L
            k = np.hypot(kx, ky)
            if k > 12 * alpha:
                continue
            a, z = k / (2 * alpha), r[:, 2]
            ep, em = np.exp(k * z) * erfc(a + alpha * z), np.exp(-k * z) * erfc(a - alpha * z)
            ph = kx * r[:, 0] + ky * r[:, 1]
            g[:, 0] -= np.pi / A * np.sin(ph) * kx / k * (ep + em)
            g[:, 1] -= np.pi / A * np.sin(ph) * ky / k * (ep + em)
            g[:, 2] += np.pi / A * np.cos(ph) * (ep - em)
    g[:, 2] -= 2 * np.pi / A * erf(alpha * r[:, 2])
    return g


def brute_lattice_accel(dr, s, L=1.0, R=400.0):
    """direct sum over lattice points within R plus the uniform-sheet tail beyond R (z only)"""
    n, m = np.meshgrid(np.arange(-int(R) - 2, int(R) + 3), np.arange(-int(R) - 3 - int(R), int(R) + 4 + int(R)), indexing="ij")
    R_lat = np.column_stack([(n * L).ravel(), (-n * s + m * L).ravel()])
    R_lat = R_lat[np.hypot(R_lat[:, 0], R_lat[:, 1]) <= R]
    out = []
    for d in np.atleast_2d(dr):
        p = np.column_stack([d[0] + R_lat[:, 0], d[1] + R_lat[:, 1], np.full(len(R_lat), d[2])])
        r3 = np.linalg.norm(p, axis=1) ** 3
        a = (p / r3[:, None]).sum(axis=0)
        a[2] += 2 * np.pi / L**2 * d[2] / np.hypot(R, d[2])
        out.append(a)
    return np.array(out)


def test_reference_lattice_sum():
    """the Ewald reference used below agrees with a brute-force lattice sum"""
    rng = np.random.default_rng(1)
    dr = np.column_stack([rng.uniform(-0.5, 0.5, 6), rng.uniform(-0.5, 0.5, 6), rng.uniform(-1.5, 1.5, 6)])
    for s in (0.0, 0.37):
        e, b = lattice_accel(dr, s), brute_lattice_accel(dr, s)
        # brute-force truncation leaves ~|dr| pi / R in-plane; z is exact up to the continuum tail
        assert np.abs(e - b).max() < 5e-3, f"s={s}: Ewald reference vs brute force max diff {np.abs(e - b).max():.2e}"


def nearest_image(dr, s, L=1.0):
    """GIZMO's NEAREST_XYZ: shear y-shift applied before the x-wrap"""
    dr = dr.copy()
    sx = np.where(dr[:, 0] > L / 2, 1, np.where(dr[:, 0] < -L / 2, -1, 0))
    dr[:, 1] += s * sx
    dr[:, 0] -= L * sx
    dr[:, 1] -= L * np.round(dr[:, 1] / L)
    return dr


@pytest.mark.parametrize("extra_flags", [(), SHEAR_FLAGS], ids=["periodic", "shear"])
def test_slab_ewald(extra_flags):
    generate_ics()
    build_and_run_test(TEST_NAME, min(default_mpi_ranks(), 4), 1, extra_config_flags=extra_flags)
    snap = get_final_snapshot(TEST_NAME, extra_flags)
    with h5py.File(snap, "r") as F:
        t = F["Header"].attrs["Time"]
        xs, ms = F["PartType1/Coordinates"][:], F["PartType1/Masses"][:]
        xt, at = F["PartType2/Coordinates"][:], F["PartType2/Acceleration"][:]
    shear = bool(extra_flags)
    s = (t % 1.0) if shear else 0.0
    if shear:
        at = at.copy()
        at[:, 0] -= 2.0 * (xt[:, 0] - 0.5)  # shearing-sheet tidal term, q = Omega = 1 (tracers start at rest)
    expect = np.zeros_like(at)
    for x, m in zip(xs, ms):
        dr = nearest_image(np.asarray(x)[None, :] - xt, s)
        expect += m * lattice_accel(dr, s)
    err = np.linalg.norm(at - expect, axis=1) / np.linalg.norm(expect, axis=1)
    print(f"{'shear' if shear else 'periodic'}: s = {s:.4f}; relative force error median {np.median(err):.2e}, 99% {np.percentile(err, 99):.2e}, max {err.max():.2e}")
    # tree truncation (ErrTolForceAcc 0.0025) + trilinear table interpolation, both ~1e-3
    assert np.median(err) < 3e-3 and err.max() < 3e-2
