"""hot_gas_timebin_clamp: RESOLVEDISM_HOTSYNC_CLAMP keeps everything inside a sphere around gas above 1e6 K on the step of the hot
gas. Adiabatic gas, 20^3 cells of 1 Msun, one 10 Msun star dying 200 yr in, 2 ranks, 5 kyr.

Bar, written before the runs: on each snapshot at least 98 % of the gas inside the sphere (as the clamp defines it) has a step at
most 2 x the shortest step of the hot gas (2 for the power-of-two bins; particles take the clamp at their next step, hence not 100 %).
"""
import shutil
import subprocess
import sys
from pathlib import Path
import h5py
import numpy as np
import pytest
from gizmo.test import build_gizmo_for_test, get_stellar_evolution_tables, mpi_launch_command

TEST_NAME = "hot_gas_timebin_clamp"
TEST_DIR = Path(__file__).parent
sys.path.insert(0, str(TEST_DIR.parent / "treecol_pixels"))
from make_treecol_pixels_ics import make_treecol_pixels_ics, lattice  # noqa: E402
from stella_table_bands import Tables  # noqa: E402

L, N_SIDE, M, Z, T_DEATH = 28.6, 20, 10.0, 0.0142, 200.0
U_HOT = 1.0e6 * 1.380649e-16 / ((2.0 / 3.0) * 0.6 * 1.6726e-24) / 1e10  # as the code, (km/s)^2


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
    make_treecol_pixels_ics(str(d / "snapshot_000.hdf5"), L, lattice(N_SIDE, L), [(M, age, np.full(3, 0.5 * L + 0.17))], hsml=2.5)
    (d / "run.params").write_text((TEST_DIR / f"{TEST_NAME}.params").read_text().replace("SET_BY_TEST", tables["StellarTablesFile"]))
    with open(d / "run.out", "w") as out:
        subprocess.run(mpi_launch_command(2, 1) + [str(TEST_DIR / "GIZMO"), str(d / "run.params"), "2", "0"], cwd=TEST_DIR, stdout=out, stderr=subprocess.STDOUT, check=True)
    return d


def test_gas_near_hot_gas_keeps_its_step(run):
    snaps = sorted(run.glob("snapshot_*.hdf5"))[1:]
    assert len(snaps) == 2
    for s in snaps:
        with h5py.File(s) as f:
            g = f["PartType0"]
            x, u, h, dt = g["Coordinates"][()], g["InternalEnergy"][()], g["KernelMaxRadius"][()], g["TimeStep"][()]
        hot = u > U_HOT
        assert hot.sum() > 10
        c = x[hot].mean(0)
        R = np.linalg.norm(x[hot] - c, axis=1).max() + 3 * h[hot].max()
        inside = np.linalg.norm(x - c, axis=1) < R
        f_in = (dt[inside] <= 2.0001 * dt[hot].min()).mean()
        print(f"{s.name}: {hot.sum()} hot, {inside.sum()} inside R = {R:.2f} pc, shortest hot step {dt[hot].min():.3e}, longest inside {dt[inside].max():.3e}, in step {f_in:.4f}")
        assert f_in >= 0.98
