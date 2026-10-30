"""photoion_per_direction test: per-direction photoionisation around single stars (GALSF_FB_PHOTOIONIZATION_PER_DIRECTION).
See README.md."""
import os
import re
import shutil
import subprocess
from pathlib import Path
import numpy as np
import pytest
from gizmo.test import build_gizmo_for_test, get_stellar_evolution_tables, mpi_launch_command

TEST_NAME = "photoion_per_direction"
TEST_DIR = Path(__file__).parent
PC_KMS_IN_YR = 3.085678e13 / 3.155e7
X_H = 0.76
ALPHA_B = 2.6e-13
PC = 3.085678e18
CENTRE = [(35.0, 1.0e6, (0.5, 0.5, 0.5))]  # a 35 Msun star 1 Myr old


@pytest.fixture(scope="module")
def built():
    tables = get_stellar_evolution_tables()
    if not os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        build_gizmo_for_test(TEST_NAME)
    return tables


def run(tag, tables, stars=CENTRE, ranks=1, threads=1, seed=None, overrides=None):
    """the run directory and the lattice (cell size, n_H)"""
    import sys
    sys.path.insert(0, str(TEST_DIR))
    from make_photoion_per_direction_ics import make_photoion_per_direction_ics
    d = TEST_DIR / f"run_{tag}"
    if d.exists():
        shutil.rmtree(d)  # this run's own directory: no stale logs
    (d / "output").mkdir(parents=True)
    L, dx, n_H = make_photoion_per_direction_ics(str(d / "output" / "snapshot_000.hdf5"), stars, seed=seed)
    params = Path(TEST_DIR / f"{TEST_NAME}.params").read_text().replace("SET_BY_TEST", tables["StellarTablesFile"])
    params = params.replace("BoxSize                            45.0", f"BoxSize                            {L}")
    for k, v in (overrides or {}).items():
        params = "\n".join(f"{k:34s} {v}" if l.split() and l.split()[0] == k else l for l in params.splitlines()) + "\n"
    (d / f"{TEST_NAME}.params").write_text(params)
    with open(d / f"test_{TEST_NAME}.out", "w") as out:
        subprocess.run(mpi_launch_command(ranks, threads) + [str(TEST_DIR / "GIZMO"), f"{TEST_NAME}.params", "2", "0"],
                       cwd=d, stdout=out, stderr=subprocess.STDOUT, env={**os.environ, "OMP_NUM_THREADS": str(threads)}, check=True)
    return d, dx, n_H


def log(d):
    """the photoionisation log rows (HIIheating.txt)"""
    return np.atleast_2d(np.loadtxt(d / "output" / "HIIheating.txt"))


def timebins(out):
    """per step: [(non-cells, cells, dt)] of the occupied bins, from the run's timebin tables"""
    steps, cur = [], None
    for l in Path(out).read_text().splitlines():
        if l.startswith("Occupied timebins"):
            cur = []; steps.append(cur)
        elif cur is not None and (m := re.match(r"\s*X?\s*bin=\s*\d+\s+(\d+)\s+(\d+)\s+([0-9.eE+-]+)", l)):
            cur.append((int(m.group(1)), int(m.group(2)), float(m.group(3))))
        else:
            cur = None
    return steps


def test_ionising_stars_in_the_photoionisation_bin(built):
    """an ionising star steps at most PhotoionMaxTimestep (1 kyr here); a 35 Msun star 1000 yr old steps onto its onset
    (15 kyr) and from then on keeps to that bin too; the gas keeps its own coarse step"""
    stars = [(35.0, 1.0e6, (0.25, 0.25, 0.25)), (35.0, 1.0e3, (0.75, 0.75, 0.75))]
    d, _, _ = run("cadence", built, stars=stars, overrides={"MaxSizeTimestep": "0.02", "TimeMax": "0.06", "TimeBetSnapshot": "0.06",
                                                          "TimeBetStatistics": "0.06", "PhotoionMaxTimestep": "1000"})
    cap = 1000.0 / PC_KMS_IN_YR
    steps = [s for s in timebins(d / f"test_{TEST_NAME}.out") if not any(dt == 0 for _, _, dt in s)]
    assert len(steps) > 10
    for s in steps:
        assert sum(n for n, _, dt in s if 0 < dt <= cap) >= 1, s  # the ionising star
        assert all(dt > 10 * cap for _, c, dt in s if c > 0), s  # the gas: its own, coarse step
    assert sum(n for n, _, dt in steps[-1] if dt <= cap) == 2, steps[-1]  # both, after the onset


def stromgren_pc(Q, n_H):
    return (3 * Q / (4 * np.pi * ALPHA_B * n_H ** 2)) ** (1 / 3) / PC


def test_stromgren_sphere(built):
    """one source: every pixel front lies within one cell of the Stromgren radius"""
    d, dx, n_H = run("single", built)
    rows = log(d)
    assert len(rows) >= 3 and np.all(rows[:, 1] == 1)
    for r in rows:
        R_S = stromgren_pc(r[2] * X_H, n_H)
        assert abs(r[9] * 1e3 - R_S) < dx and abs(r[10] * 1e3 - R_S) < dx, (r[9] * 1e3, r[10] * 1e3, R_S)


def test_ranks_and_order(built):
    """the fronts do not depend on the rank count or the particle order (integer sums)"""
    ref = log(run("single_r1", built)[0])
    for tag, kw in [("single_r2", dict(ranks=2)), ("single_r4", dict(ranks=4)), ("single_perm", dict(ranks=2, seed=7))]:
        rows = log(run(tag, built, **kw)[0])
        assert rows.shape == ref.shape, tag
        assert np.array_equal(rows[:, [1, 8]], ref[:, [1, 8]]), tag  # sources, passes
        assert np.allclose(rows[:, [9, 10]], ref[:, [9, 10]], rtol=1e-6, atol=0), tag  # fronts
