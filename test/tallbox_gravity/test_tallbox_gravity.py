"""Tall-box gravity: x/y-periodic self-gravity by image sums, open in z (GRAVITY_PERIODIC_XY_IMAGES).

The box is 1 x 1 x 8 (BOX_PERIODIC, BOX_LONG_Z=8, BOX_OUTFLOW_Z). Gravity of a source is summed over the
source's nearest x/y image and the (2N+1)^2 - 1 images around it, N = GravityPeriodicXYImages = 8; there is
no wrap in z. Collisionless particles only. Each IC is run on 1 and 2 MPI ranks, once with the parameterfile's
opening criterion and once with the walk tightened towards direct summation (ErrTolTheta=0.15,
ErrTolForceAcc=1e-6); snapshot_000 holds the start-up force and the final snapshot the evolved state.

Checks (limits fixed before the first run, see README.md):
  slab   the tight-walk acceleration against a direct sum over the same images (numpy), for a slab plus
         layers 0.03 from each z face: median and p99 of |a - a_ref| / median|a_ref|, and the maximum;
         the production walk against the tight one; net force sum(m a) against sum(m |a|)
  sheet  probes above and below an exact lattice sheet against the analytic field of the square patch of
         images, a_z = 4 G Sigma arctan(W^2 / (h sqrt(2 W^2 + h^2))), W = (N + 1/2) L; the sheet's own
         acceleration against 2 pi G Sigma
  both   the run conserves mass (to 1e-12) and net momentum (sum(m v) against sum(m |v|)) over its steps
"""
import sys
from os import chdir, path

import h5py
import numpy as np
import pytest

from gizmo.test import build_gizmo_for_test, default_mpi_ranks, parse_params, run_test

sys.path.insert(0, path.dirname(path.abspath(__file__)))
import make_ics  # noqa: E402

TEST_NAME = "tallbox_gravity"
TIGHT = {"ErrTolTheta": 0.15, "ErrTolForceAcc": 1e-6}
N_IMG, G = 8, 1.0
LIMITS = {
    "slab_median": 1e-3, "slab_p99": 5e-3, "slab_max": 2e-2,   # tight walk vs direct sum
    "slab_prod_median": 4e-3, "slab_prod_p99": 1.5e-2,          # production walk vs tight walk
    "slab_net_force": 2e-3,                                     # |sum m a| / sum m |a|
    "sheet_probe": 2e-3,                                        # relative, vs the analytic patch field
    "sheet_self": 5e-3,                                         # |a_sheet| / (2 pi G Sigma)
    "momentum": 1e-2, "mass": 1e-12,
}


@pytest.fixture(scope="module")
def built():
    build_gizmo_for_test(TEST_NAME, 1)
    chdir(f"test/{TEST_NAME}/")
    make_ics.make_slab_ics("tallbox_slab_ics.hdf5")
    make_ics.make_sheet_ics("tallbox_sheet_ics.hdf5")
    yield
    chdir("../../")


def _run(kind, ranks, tight):
    params = f"tallbox_{kind}"
    outdir = parse_params(f"{params}.params")["OutputDir"] + f"_r{ranks}" + ("_tight" if tight else "")
    over = {"OutputDir": outdir}
    over.update(TIGHT if tight else {})
    run_test(TEST_NAME, ranks, 1, params_name=params, param_overrides=over)
    snaps = sorted(f for f in __import__("glob").glob(f"{outdir}/snapshot_*.hdf5"))
    return snaps[0], snaps[-1]


def _read(snap):
    with h5py.File(snap, "r") as F:
        g = F["PartType1"]
        order = np.argsort(g["ParticleIDs"][:])
        return {k: g[k][:][order] for k in ("Coordinates", "Velocities", "Masses", "Acceleration")}


def direct_sum_with_images(pos, mass, n=N_IMG, lxy=1.0):
    """a_i = G sum_j m_j sum_(a,b) d / |d|^3 over the nearest x/y image of j and the (2n+1)^2-1 around it."""
    off = np.array([(a, b) for a in range(-n, n + 1) for b in range(-n, n + 1)], float) * lxy
    acc = np.zeros_like(pos)
    for i in range(len(pos)):
        dr = np.delete(pos, i, 0) - pos[i]
        m = np.delete(mass, i)
        dr[:, :2] -= lxy * np.round(dr[:, :2] / lxy)
        d = np.broadcast_to(dr[:, None, :], (len(dr), len(off), 3)).copy()
        d[:, :, :2] += off[None, :, :]
        acc[i] = G * np.einsum("j,jkc->c", m, d / np.linalg.norm(d, axis=2)[:, :, None] ** 3)
    return acc


def _relerr(a, ref):
    return np.linalg.norm(a - ref, axis=1) / np.median(np.linalg.norm(ref, axis=1))


def _conservation(first, last):
    f, l = _read(first), _read(last)
    assert abs(l["Masses"].sum() / f["Masses"].sum() - 1) < LIMITS["mass"]
    p = (l["Masses"][:, None] * l["Velocities"]).sum(0)
    scale = (l["Masses"] * np.linalg.norm(l["Velocities"], axis=1)).sum()
    print(f"momentum: |sum m v| / sum m|v| = {np.linalg.norm(p) / scale:.3e}")
    assert np.linalg.norm(p) / scale < LIMITS["momentum"]


@pytest.mark.parametrize("ranks", (1, 2))
def test_slab_against_direct_sum(built, ranks):
    first_t, last_t = _run("slab", ranks, tight=True)
    first_p, _ = _run("slab", ranks, tight=False)
    ref = direct_sum_with_images(*[_read(first_t)[k] for k in ("Coordinates", "Masses")])
    tight, prod = _read(first_t), _read(first_p)
    e = _relerr(tight["Acceleration"], ref)
    ep = _relerr(prod["Acceleration"], tight["Acceleration"])
    net = np.linalg.norm((tight["Masses"][:, None] * tight["Acceleration"]).sum(0)) / np.sum(
        tight["Masses"] * np.linalg.norm(tight["Acceleration"], axis=1))
    print(f"[ranks={ranks}] tight vs direct: median={np.median(e):.2e} p99={np.percentile(e, 99):.2e} max={e.max():.2e}; "
          f"production vs tight: median={np.median(ep):.2e} p99={np.percentile(ep, 99):.2e}; net force {net:.2e}")
    assert np.median(e) < LIMITS["slab_median"] and np.percentile(e, 99) < LIMITS["slab_p99"] and e.max() < LIMITS["slab_max"]
    assert np.median(ep) < LIMITS["slab_prod_median"] and np.percentile(ep, 99) < LIMITS["slab_prod_p99"]
    assert net < LIMITS["slab_net_force"]
    _conservation(first_t, last_t)


@pytest.mark.parametrize("ranks", (1, 2))
def test_sheet_against_analytic_field(built, ranks):
    first, last = _run("sheet", ranks, tight=True)
    d = _read(first)
    n_sheet = make_ics.NG ** 2
    sigma, w = 1.0, (N_IMG + 0.5) * make_ics.LXY
    probes = d["Coordinates"][n_sheet:]
    h = probes[:, 2] - make_ics.Z0
    expect = -np.sign(h) * 4 * G * sigma * np.arctan(w * w / (np.abs(h) * np.sqrt(2 * w * w + h * h)))
    got = d["Acceleration"][n_sheet:, 2]
    err = np.abs(got / expect - 1)
    self_acc = np.linalg.norm(d["Acceleration"][:n_sheet], axis=1).max() / (2 * np.pi * G * sigma)
    print(f"[ranks={ranks}] probe a_z {np.round(got, 4)} expected {np.round(expect, 4)} max rel err {err.max():.2e}; "
          f"max |a_sheet| / 2 pi G Sigma = {self_acc:.2e}")
    assert err.max() < LIMITS["sheet_probe"] and self_acc < LIMITS["sheet_self"]
    _conservation(first, last)
