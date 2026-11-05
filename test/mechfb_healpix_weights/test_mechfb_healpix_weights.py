"""mechfb_healpix_weights test: GALSF_FB_MECHANICAL_HEALPIX_WEIGHTS gives each occupied NSIDE-1 pixel of the star's sky the same share.
See README.md."""
import os
import shutil
import subprocess
from pathlib import Path
import numpy as np
import h5py
import pytest
from gizmo.test import build_gizmo_for_test, get_stellar_evolution_tables, mpi_launch_command

TEST_NAME = "mechfb_healpix_weights"
TEST_DIR = Path(__file__).parent
E_UNIT = 1.989e43  # erg per code energy unit (Msun (km/s)^2)


@pytest.fixture(scope="module")
def tables():
    t = get_stellar_evolution_tables()
    if not os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        build_gizmo_for_test(TEST_NAME)
    return t


def pixel(v):
    """HEALPix RING pixel at NSIDE 1 (the code's treecol_hpx_pixel)."""
    z = v[:, 2] / np.linalg.norm(v, axis=1)
    tt = (np.arctan2(v[:, 1], v[:, 0]) % (2 * np.pi)) / (0.5 * np.pi)
    jp = np.floor(0.5 + tt - 0.75 * z).astype(int); jm = np.floor(0.5 + tt + 0.75 * z).astype(int)
    ir = 2 + jp - jm
    eq = ((ir - 1) * 4 + ((jp + jm + 1 - ir % 2) // 2) % 4)
    tmp = np.sqrt(3 * (1 - np.abs(z))); tp = tt % 1
    ring = np.floor(tp * tmp).astype(int) + np.floor((1 - tp) * tmp).astype(int) + 1
    ip = np.floor(tt * ring).astype(int) % (4 * ring)
    return np.where(np.abs(z) <= 2 / 3, eq, np.where(z > 0, 2 * ring * (ring - 1) + ip, 12 - 2 * ring * (ring + 1) + ip))


def run(tables, event, ranks):
    import sys
    sys.path.insert(0, str(TEST_DIR))
    from make_mechfb_healpix_weights_ics import make_ics
    d = TEST_DIR / f"run_{event}_{ranks}"
    if not os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        if d.exists():
            shutil.rmtree(d)
        (d / "output").mkdir(parents=True)
        L = make_ics(str(d / "output" / "snapshot_000.hdf5"), tables["StellarTablesFile"], event)
        params = (TEST_DIR / f"{TEST_NAME}.params").read_text().replace("SET_BY_TEST", tables["StellarTablesFile"]).replace("20.0", str(L), 1)
        (d / f"{TEST_NAME}.params").write_text(params)
        with open(d / "run.out", "w") as out:
            subprocess.run(mpi_launch_command(ranks, 1) + [str(TEST_DIR / "GIZMO"), f"{TEST_NAME}.params", "2", "0"], cwd=d, stdout=out,
                           stderr=subprocess.STDOUT, env={**os.environ, "OMP_NUM_THREADS": "1"}, check=True)
    snaps = sorted((d / "output").glob("snapshot_*.hdf5"))
    return read(snaps[0]), read(snaps[-1])


def read(fname):
    """Per ID: split cells keep the ID of their parent."""
    with h5py.File(fname, "r") as F:
        g = F["PartType0"]
        ids, inv = np.unique(g["ParticleIDs"][()], return_inverse=True)
        m = g["Masses"][()]; v = g["Velocities"][()]; u = g["InternalEnergy"][()]
        n = len(ids)
        x = np.zeros((n, 3)); x[inv] = g["Coordinates"][()]
        return dict(id=ids, x=x, m=np.bincount(inv, m, n), E=np.bincount(inv, m * (u + 0.5 * (v ** 2).sum(1)), n),
                    p=np.stack([np.bincount(inv, m * v[:, k], n) for k in range(3)], 1), star=F["PartType4/Coordinates"][0], box=F["Header"].attrs["BoxSize"])


def deposit(a, b):
    i = np.searchsorted(b["id"], a["id"])
    dM = b["m"][i] - a["m"]; rec = dM > 1e-9 * dM.max()
    r = (a["x"] - a["star"] + 0.5 * a["box"]) % a["box"] - 0.5 * a["box"]
    return r[rec], dM[rec], (b["E"][i] - a["E"])[rec], (b["p"][i] - a["p"])[rec], (b["p"][i] - a["p"]).sum(0)


@pytest.mark.parametrize("ranks", [1, 2])
def test_supernova_shares_per_pixel(tables, ranks):
    r, dM, dE, _, _ = deposit(*run(tables, "sn", ranks))
    assert abs(dE.sum() * E_UNIT / 1e51 - 1) < 2e-3
    pix = pixel(r); occ = np.unique(pix); n = np.bincount(pix, minlength=12)
    want = 1.0 / (len(occ) * n[pix])
    assert np.abs(dM / dM.sum() / want - 1).max() < 1e-9  # every cell: 1 / (occupied pixels x cells in its pixel)
    shares = np.bincount(pix, dE, 12)[occ] / dE.sum()
    assert np.abs(shares * len(occ) - 1).max() < 2e-3  # energy: the same share per occupied pixel


@pytest.mark.parametrize("ranks", [1, 2])
def test_wind_shares_and_balance(tables, ranks):
    r, dM, _, dp, sum_p = deposit(*run(tables, "wind", ranks))
    rhat = r / np.linalg.norm(r, axis=1)[:, None]
    pix = pixel(r); occ = np.unique(pix); n = np.bincount(pix, minlength=12)
    w = 1.0 / (len(occ) * n[pix])
    q = w[:, None] * (rhat - (w[:, None] * rhat).sum(0))
    qn = np.linalg.norm(q, axis=1)
    assert np.abs(dM / dM.sum() / (qn / qn.sum()) - 1).max() < 1e-9  # mass shares |q| / sum |q|
    pn = np.linalg.norm(dp, axis=1)
    assert np.linalg.norm(sum_p) / pn.sum() < 1e-6  # momentum balanced, over all gas: the hot receivers push on their neighbours
    assert np.abs((dp * q).sum(1) / (pn * qn) - 1).max() < 1e-4  # along q
