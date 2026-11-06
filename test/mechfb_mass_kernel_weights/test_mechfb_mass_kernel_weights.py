"""mechfb_mass_kernel_weights test: GALSF_FB_MECHANICAL_MASS_KERNEL_WEIGHTS gives each cell inside the star's kernel the share mass x kernel,
wind and AGB along q = w (r - mean of w r) normalised by sum |q|, and every deposit closes energy, momentum, mass and metals. See README.md."""
import os
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
import h5py
import pytest
from gizmo.test import build_gizmo_for_test, get_stellar_evolution_tables, mpi_launch_command

TEST_NAME = "mechfb_mass_kernel_weights"
TEST_DIR = Path(__file__).parent
E_UNIT = 1.989e43  # erg per code energy unit (Msun (km/s)^2)
CLOSE = 1e-12
SHARE = 1e-9  # absolute on a cell's share: dM comes from differences of masses near 1
sys.path.insert(0, str(TEST_DIR))
from make_mechfb_mass_kernel_weights_ics import make_ics  # noqa: E402

BUILT = {}


@pytest.fixture(scope="module")
def tables():
    t = get_stellar_evolution_tables()
    if not os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        build_gizmo_for_test(TEST_NAME)
        shutil.move(TEST_DIR / "GIZMO", TEST_DIR / "GIZMO_events")
        build_gizmo_for_test(TEST_NAME, extra_config_flags=("GALSF_FB_RADPRESSURE_SINGLE_STAR_BANDS",))
        shutil.move(TEST_DIR / "GIZMO", TEST_DIR / "GIZMO_rp")
    return t


def run(tables, event, ranks):
    d = TEST_DIR / f"run_{event}_{ranks}"
    if (event, ranks) in BUILT:
        return BUILT[(event, ranks)]
    if not os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        if d.exists():
            shutil.rmtree(d)
        (d / "output").mkdir(parents=True)
        L = make_ics(str(d / "output" / "snapshot_000.hdf5"), tables["StellarTablesFile"], event)
        params = (TEST_DIR / f"{TEST_NAME}.params").read_text().replace("SET_BY_TEST", tables["StellarTablesFile"]).replace("20.0", str(L), 1)
        (d / f"{TEST_NAME}.params").write_text(params)
        exe = "GIZMO_rp" if event == "rp" else "GIZMO_events"
        with open(d / "run.out", "w") as out:
            subprocess.run(mpi_launch_command(ranks, 1) + [str(TEST_DIR / exe), f"{TEST_NAME}.params", "2", "0"], cwd=d, stdout=out,
                           stderr=subprocess.STDOUT, env={**os.environ, "OMP_NUM_THREADS": "1"}, check=True)
    snaps = [read(f) for f in sorted((d / "output").glob("snapshot_*.hdf5"))]
    # the first pair of snapshots with a deposit in it: exactly one deposit
    a, b = next((x, y) for x, y in zip(snaps, snaps[1:]) if np.any(y["m"] != x["m"]) or np.any(y["p"] != x["p"]))
    BUILT[(event, ranks)] = (a, b, d)
    return BUILT[(event, ranks)]


def read(fname):
    """Per ID: split cells keep the ID of their parent."""
    with h5py.File(fname, "r") as F:
        g = F["PartType0"]; s = F["PartType4"]
        ids, inv = np.unique(g["ParticleIDs"][()], return_inverse=True)
        m = g["Masses"][()]; v = g["Velocities"][()]; u = g["InternalEnergy"][()]; z = g["Metallicity"][()].reshape(len(m), -1)[:, 0]
        n = len(ids)
        x = np.zeros((n, 3)); x[inv] = g["Coordinates"][()]
        return dict(id=ids, x=x, m=np.bincount(inv, m, n), mz=np.bincount(inv, m * z, n),
                    E=np.bincount(inv, m * (u + 0.5 * (v ** 2).sum(1)), n), p=np.stack([np.bincount(inv, m * v[:, k], n) for k in range(3)], 1),
                    sx=s["Coordinates"][()], sm=s["Masses"][()], sv=s["Velocities"][()], box=F["Header"].attrs["BoxSize"])


def kernel(q):
    return np.where(q < 0.5, 1 + 6 * q * q * (q - 1), np.where(q < 1, 2 * (1 - q) ** 3, 0.0))


def receivers(a, b):
    i = np.searchsorted(b["id"], a["id"])
    dM = b["m"][i] - a["m"]; rec = dM > 1e-9 * dM.max()
    r = (a["x"] - a["sx"][0] + 0.5 * a["box"]) % a["box"] - 0.5 * a["box"]
    return rec, r, dM, (b["E"][i] - a["E"]), (b["p"][i] - a["p"]), (b["mz"][i] - a["mz"])


def kernel_radius(share, weight_of_h, rmax):
    """The star's kernel radius is not written: the one that fits the shares of the receivers."""
    lo, hi = rmax * (1 + 1e-9), 3 * rmax
    gs = 0.5 * (np.sqrt(5) - 1)
    for _ in range(80):
        h1, h2 = hi - gs * (hi - lo), lo + gs * (hi - lo)
        (lo, hi) = (lo, h2) if np.abs(weight_of_h(h1) - share).max() < np.abs(weight_of_h(h2) - share).max() else (h1, hi)
    return 0.5 * (lo + hi)


def plain_share(rm, m):
    rn = np.linalg.norm(rm, axis=1)
    return lambda h: (lambda w: w / w.sum())(m * kernel(rn / h))


def q_vectors(rm, w):
    rhat = rm / np.linalg.norm(rm, axis=1)[:, None]
    return w[:, None] * (rhat - (w[:, None] * rhat).sum(0))


def q_share(rm, m):
    rn = np.linalg.norm(rm, axis=1)

    def f(h):
        w = m * kernel(rn / h); w = w / w.sum()
        qn = np.linalg.norm(q_vectors(rm, w), axis=1)
        return qn / qn.sum()
    return f


def events_row(d):
    return np.loadtxt(d / "output" / "single_star_events.txt", ndmin=2)[0]


@pytest.mark.parametrize("ranks", [1, 2])
def test_supernova_shares_mass_kernel(tables, ranks):
    a, b, _ = run(tables, "sn", ranks)
    rec, r, dM, dE, _, _ = receivers(a, b)
    share = dM[rec] / dM[rec].sum()
    f = plain_share(r[rec], a["m"][rec]); rmax = np.linalg.norm(r[rec], axis=1).max()
    h = kernel_radius(share, f, rmax)
    assert np.abs(f(h) - share).max() < SHARE  # every cell: mass x kernel over the star's kernel
    assert np.abs(dE[rec] / dE[rec].sum() - share).max() < SHARE  # energy with the same share


@pytest.mark.parametrize("event,ranks", [("wind", 1), ("wind", 2), ("agb", 1), ("agb", 2)])
def test_wind_agb_vector_shares(tables, event, ranks):
    a, b, _ = run(tables, event, ranks)
    rec, r, dM, _, dp, _ = receivers(a, b)
    share = dM[rec] / dM[rec].sum()
    f = q_share(r[rec], a["m"][rec]); rmax = np.linalg.norm(r[rec], axis=1).max()
    h = kernel_radius(share, f, rmax)
    assert np.abs(f(h) - share).max() < SHARE  # mass shares |q| / sum |q|, q with the weight mass x kernel
    w = a["m"][rec] * kernel(np.linalg.norm(r[rec], axis=1) / h); w = w / w.sum()
    q = q_vectors(r[rec], w); qn = np.linalg.norm(q, axis=1); pn = np.linalg.norm(dp[rec], axis=1)
    assert np.abs((dp[rec] * q).sum(1) / (pn * qn) - 1).max() < 1e-5  # along q
    assert np.abs(pn / pn.sum() - share).max() < SHARE  # momentum with the mass


@pytest.mark.parametrize("ranks", [1, 2])
def test_supernova_closure(tables, ranks):
    a, b, d = run(tables, "sn", ranks)
    rec, _, dM, dE, dp, dZ = receivers(a, b)
    row = events_row(d)  # time ID kind remnant M_drawn M_trigger M_ejecta M_remnant E M_Z_ejecta ...
    assert abs(dE.sum() * E_UNIT / row[8] - 1) < CLOSE
    assert abs(dM.sum() / row[6] - 1) < CLOSE
    assert abs(dZ.sum() / row[9] - 1) < CLOSE
    assert np.linalg.norm(dp.sum(0)) < CLOSE * np.sqrt(2 * row[8] / E_UNIT * row[6])


@pytest.mark.parametrize("ranks", [1, 2])
def test_agb_closure(tables, ranks):
    a, b, d = run(tables, "agb", ranks)
    rec, _, dM, dE, dp, dZ = receivers(a, b)
    row = events_row(d)
    pn = np.linalg.norm(dp, axis=1)
    assert abs(dM.sum() / row[6] - 1) < CLOSE
    assert abs(dZ.sum() / row[9] - 1) < CLOSE
    assert abs(pn.sum() / (row[6] * 30.0) - 1) < CLOSE  # the planetary nebula at 30 km/s
    assert abs(dE.sum() / (0.5 * row[6] * 30.0 ** 2) - 1) < CLOSE
    assert np.linalg.norm(dp.sum(0)) / pn.sum() < CLOSE


@pytest.mark.parametrize("ranks", [1, 2])
def test_wind_closure(tables, ranks):
    a, b, _ = run(tables, "wind", ranks)
    rec, _, dM, dE, dp, dZ = receivers(a, b)
    pn = np.linalg.norm(dp, axis=1); M = dM.sum(); P = pn.sum()
    assert abs(M / (a["sm"][0] - b["sm"][0]) - 1) < CLOSE
    assert abs(dE.sum() / (0.5 * P * P / M) - 1) < CLOSE  # kinetic + heat = 1/2 M v^2, v = P / M
    assert np.linalg.norm(dp.sum(0)) / P < CLOSE
    assert np.abs(dZ / dZ.sum() - dM / M).max() < SHARE  # the same composition everywhere


@pytest.mark.parametrize("ranks", [1, 2])
def test_wind_closure_moving_gas(tables, ranks):
    """Lab-frame closure with gas and star moving; the wind speed is that of the run at rest."""
    a0, b0, _ = run(tables, "wind", ranks)
    _, _, dM0, _, dp0, _ = receivers(a0, b0)
    v_w = np.linalg.norm(dp0, axis=1).sum() / dM0.sum()
    a, b, _ = run(tables, "moving", ranks)
    i = np.searchsorted(b["id"], a["id"])
    dM = (b["m"][i] - a["m"]).sum(); dM_star = a["sm"][0] - b["sm"][0]
    p_gas = (b["p"][i] - a["p"]).sum(0); p_star = b["sm"][0] * b["sv"][0] - a["sm"][0] * a["sv"][0]
    ke = lambda s: 0.5 * s["sm"][0] * (s["sv"][0] ** 2).sum()
    dE = (b["E"][i] - a["E"]).sum() + ke(b) - ke(a)
    assert abs(dM / dM_star - 1) < CLOSE
    assert np.linalg.norm(p_gas + p_star) / (dM * v_w) < CLOSE
    assert abs(dE / (0.5 * dM * v_w ** 2) - 1) < CLOSE


@pytest.mark.parametrize("ranks", [1, 2])
def test_one_ray_is_radial_and_the_star_recoils(tables, ranks):
    a, b, _ = run(tables, "ray", ranks)
    rec, r, dM, _, dp, _ = receivers(a, b)
    pn = np.linalg.norm(dp, axis=1)
    assert np.abs(dp[rec, 1:]).max() < 1e-12 * pn.max()  # radial
    assert (dp[rec, 0] > 0).all()
    p_star = b["sm"][0] * b["sv"][0] - a["sm"][0] * a["sv"][0]
    assert np.linalg.norm(dp.sum(0) + p_star) / pn.sum() < CLOSE


@pytest.mark.parametrize("ranks", [1, 2])
def test_overlapping_winds_close(tables, ranks):
    """Two winds into the same cells in the same step."""
    a0, b0, _ = run(tables, "wind", ranks)
    _, _, dM0, _, dp0, _ = receivers(a0, b0)
    v_w = np.linalg.norm(dp0, axis=1).sum() / dM0.sum()
    a, b, _ = run(tables, "overlap", ranks)
    i = np.searchsorted(b["id"], a["id"])
    dM = (b["m"][i] - a["m"]).sum()
    assert abs(dM / (a["sm"] - b["sm"]).sum() - 1) < CLOSE
    assert abs((b["E"][i] - a["E"]).sum() / (0.5 * dM * v_w ** 2) - 1) < CLOSE
    assert np.linalg.norm((b["p"][i] - a["p"]).sum(0)) / (dM * v_w) < CLOSE


@pytest.mark.parametrize("ranks", [1, 2])
def test_radiation_kick_is_balanced(tables, ranks):
    """Star-side radiation pressure: share mass x kernel along r - mean of w r, no net momentum (stella_dev pass 1)."""
    a, b, _ = run(tables, "rp", ranks)
    i = np.searchsorted(b["id"], a["id"])
    dp = b["p"][i] - a["p"]; pn = np.linalg.norm(dp, axis=1); rec = pn > 1e-9 * pn.max()
    r = (a["x"] - a["sx"][0] + 0.5 * a["box"]) % a["box"] - 0.5 * a["box"]
    rn = np.linalg.norm(r[rec], axis=1); m = a["m"][rec]
    f = lambda h: (lambda w: (lambda q: np.linalg.norm(q, axis=1) / np.linalg.norm(q, axis=1).sum())(q_vectors(r[rec], w)))((m * kernel(rn / h)) / (m * kernel(rn / h)).sum())
    h = kernel_radius(pn[rec] / pn[rec].sum(), f, rn.max())
    assert np.abs(f(h) - pn[rec] / pn[rec].sum()).max() < SHARE
    assert np.linalg.norm(dp.sum(0)) / pn.sum() < CLOSE
