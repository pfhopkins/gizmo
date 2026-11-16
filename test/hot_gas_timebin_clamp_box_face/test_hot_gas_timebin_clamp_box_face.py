"""hot_gas_timebin_clamp_box_face: RESOLVEDISM_HOTSYNC_CLAMP takes its sphere with nearest-image centre and distances, so a
remnant on a periodic box face gets a sphere at the face, not one half a box away. Adiabatic gas, 1 Msun cells on a 60 x 10 x 10
lattice in a periodic 85.8 x 14.3 x 14.3 pc box (BOX_LONG_X=6), one 10 Msun star 0.17 pc inside the +x face dying 200 yr in, 2 ranks, 7.4 kyr.

Bars, written before the runs, on the snapshots just after the coarsest syncs (as hot_gas_timebin_clamp): the sphere is rebuilt
from the snapshot with nearest-image distances (hottest cell + mean nearest-image offset of the hot cells; farthest hot cell + 3
largest hot kernel radii). Hot gas lies on both sides of the x face; fewer than 5 % of the cells beyond 1.25 sphere radii are on the
hot gas's step (at most 2 x its longest step), and at least 98 % of the gas inside is on it (at most 2 x its shortest step).
"""
import shutil
import subprocess
import sys
from pathlib import Path
import h5py
import numpy as np
import pytest
from gizmo.test import build_gizmo_for_test, get_stellar_evolution_tables, mpi_launch_command

TEST_NAME = "hot_gas_timebin_clamp_box_face"
TEST_DIR = Path(__file__).parent
sys.path.insert(0, str(TEST_DIR.parent / "treecol_pixels"))
from make_treecol_pixels_ics import make_treecol_pixels_ics, lattice  # noqa: E402
from stella_table_bands import Tables  # noqa: E402

L, N_SIDE, LONG_X, M, Z, T_DEATH = 14.3, 10, 6, 10.0, 0.0142, 200.0
BOX = np.array([LONG_X * L, L, L])
U_HOT = 1.0e6 * 1.380649e-16 / ((2.0 / 3.0) * 0.6 * 1.6726e-24) / 1e10  # as the code, (km/s)^2


def nearest(d):
    return d - BOX * np.round(d / BOX)


@pytest.fixture(scope="module")
def run():
    tables = get_stellar_evolution_tables()
    build_gizmo_for_test(TEST_NAME)
    d = TEST_DIR / "output"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir()
    tab = Tables(tables["StellarTablesFile"])
    lM, lZ = np.log10(M), np.log10(Z)
    age = tab._i2(tab.t_pms, lM, lZ) + tab._i2(tab.life, lM, lZ) - T_DEATH
    gas = np.concatenate([lattice(N_SIDE, L) + [k * L, 0, 0] for k in range(LONG_X)])
    make_treecol_pixels_ics(str(d / "snapshot_000.hdf5"), L, gas, [(M, age, np.array([BOX[0] - 0.17, 0.5 * L + 0.17, 0.5 * L + 0.17]))], hsml=2.5)
    (d / "run.params").write_text((TEST_DIR / f"{TEST_NAME}.params").read_text().replace("SET_BY_TEST", tables["StellarTablesFile"]))
    with open(d / "run.out", "w") as out:
        subprocess.run(mpi_launch_command(2, 1) + [str(TEST_DIR / "GIZMO"), str(d / "run.params"), "2", "0"], cwd=TEST_DIR, stdout=out, stderr=subprocess.STDOUT, check=True)
    return d


def test_sphere_stays_at_the_box_face(run):
    snaps = sorted(run.glob("snapshot_*.hdf5"))[1:4]  # just after the coarsest syncs
    assert len(snaps) == 3
    for s in snaps:
        with h5py.File(s) as f:
            g = f["PartType0"]
            x, u, h, dt = g["Coordinates"][()], g["InternalEnergy"][()], g["KernelMaxRadius"][()], g["TimeStep"][()]
        hot = u > U_HOT
        assert hot.sum() > 10
        assert (x[hot, 0] < 0.5 * BOX[0]).any() and (x[hot, 0] > 0.5 * BOX[0]).any(), f"{s.name}: hot gas does not straddle the x face"
        ref = x[np.argmax(u)]
        c = ref + nearest(x[hot] - ref).mean(0)
        r = np.linalg.norm(nearest(x - c), axis=1)
        R = r[hot].max() + 3 * h[hot].max()
        inside, far = r < R, r > 1.25 * R
        assert far.sum() > 100
        f_in = (dt[inside] <= 2.0001 * dt[hot].min()).mean()
        f_far = (dt[far] <= 2.0001 * dt[hot].max()).mean()
        print(f"{s.name}: {hot.sum()} hot, centre x {c[0]:.2f} pc, R = {R:.2f} pc, {inside.sum()} inside (in step {f_in:.4f}), {far.sum()} far (on the hot step {f_far:.4f})")
        assert f_far < 0.05
        assert f_in >= 0.98
