"""hot_gas_timebin_clamp: RESOLVEDISM_HOTSYNC_CLAMP keeps everything inside a sphere around gas above 1e6 K on the step of the hot
gas. Adiabatic gas, 20^3 cells of 1 Msun, one 10 Msun star dying 200 yr in, 2 ranks, 7.4 kyr.

Bar, written before the runs: on each snapshot at least 98 % of the gas inside the sphere (as the clamp defines it) has a step at
most 2 x the shortest step of the hot gas (2 for the power-of-two bins; particles take the clamp at their next step, hence not 100 %).
A particle takes a new step only when it next becomes active, and the clamp wakes nobody. So the bar can hold only just after a coarsest
sync, when every cell has just been active: on such a sync a snapshot still shows the steps before the reassignment (10 %), and 1.1e-3 past
one (3.0e-3 in the run) gas that entered the growing sphere since then is not clamped yet (75 % with the flag, 6 % without). The first
choices of snapshot time were logically wrong for that reason; the snapshots moved to 3e-5 after each coarsest sync, the bar did not.

Quiet steps: MPI collectives counted on rank 0 by calling function (LD_PRELOAD, mpi_collective_callers.c, addr2line on the
non-PIE binary). No star: at most one from the clamp (find_timesteps), its start-up look at the ICs. A young 40 Msun star, winds only: none from the single-star
feedback (single_star_*), while the mechanical feedback runs.
"""
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
import h5py
import numpy as np
import pytest
from gizmo.test import build_gizmo_for_test, get_stellar_evolution_tables, mpi_launch_command, mpirun_is_open_mpi

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
    snaps = sorted(run.glob("snapshot_*.hdf5"))[1:4]  # just after the coarsest syncs; the last one, at the end time, is on one
    assert len(snaps) == 3
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


@pytest.fixture(scope="module")
def callers(run):
    """collectives on rank 0 by calling function, for a run of the test's build with the given star (M, age before death [yr]) or none"""
    tables = get_stellar_evolution_tables()
    exe = TEST_DIR / "GIZMO"  # built by the run fixture
    so = TEST_DIR / "mpi_collective_callers.so"
    subprocess.run(["mpicc", "-shared", "-fPIC", "-O1", str(TEST_DIR / "mpi_collective_callers.c"), "-o", str(so)], check=True)
    tab = Tables(tables["StellarTablesFile"])

    def count(tag, star):
        d = TEST_DIR / f"output_{tag}"
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir()
        stars = []
        if star:
            lM, lZ = np.log10(star[0]), np.log10(Z)
            stars = [(star[0], tab._i2(tab.t_pms, lM, lZ) + tab._i2(tab.life, lM, lZ) - star[1], np.full(3, 0.5 * L + 0.17))]
        make_treecol_pixels_ics(str(d / "snapshot_000.hdf5"), L, lattice(N_SIDE, L), stars, hsml=2.5)
        (d / "run.params").write_text((TEST_DIR / f"{TEST_NAME}.params").read_text().replace("SET_BY_TEST", tables["StellarTablesFile"]).replace("OutputDir                          output", f"OutputDir                          output_{tag}"))
        fwd = ["-x", "LD_PRELOAD"] if mpirun_is_open_mpi() else []
        out = subprocess.run(mpi_launch_command(2, 1) + fwd + [str(exe), str(d / "run.params"), "2", "0"], cwd=TEST_DIR, capture_output=True, text=True,
                             env={**os.environ, "LD_PRELOAD": str(so)}, check=True).stdout
        (d / "run.out").write_text(out)
        rows = re.findall(r"^MPI_CALLER (0x[0-9a-f]+) (\d+)", out, re.M)
        assert rows, out[-500:]
        names = subprocess.run(["addr2line", "-f", "-C", "-e", str(exe)] + [a for a, _ in rows], capture_output=True, text=True, check=True).stdout.splitlines()[0::2]
        by = {}
        for nm, (_, c) in zip(names, rows):
            by[nm] = by.get(nm, 0) + int(c)
        print(tag, sorted(by.items(), key=lambda x: -x[1])[:12])
        return by
    return count


def test_quiet_steps_no_clamp_collective(callers):
    by = callers("quiet", None)
    assert sum(c for f, c in by.items() if "find_timesteps" in f or "hotsync" in f) <= 1, by  # the start-up look at the ICs


def test_wind_steps_no_single_star_collective(callers):
    by = callers("winds", (40.0, 1.0e6))
    assert sum(c for f, c in by.items() if f.startswith("mechanical_fb")) > 0, by  # the feedback ran
    assert not [f for f in by if f.startswith("single_star")], by
