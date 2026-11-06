"""single_stars_wind_dump test: TEST_DUMP_STATE_AROUND_WIND_DEPOSIT writes, in each step with a mass return (wind), the mass, velocity
and gas energy of every particle of a rank before (phase 0) and after (phase 1) the feedback loop to wind_dep_dump.<rank>.bin
(record: int phase, long long record number, double time, long long n, then n x {unsigned long long ID, int type, 5 doubles}).
A 35 Msun star 1 Myr old in the lattice of single_stars_winds_agb_ia, 0.01 code time, 2 ranks. Bars, written before the run:
both ranks have one file with at least 5 deposits as (phase 0, phase 1) pairs of the same record number, time and size; at every
deposit the total mass of all particles is conserved and the gas gains the mass the star loses, both to 1e-14 Msun (absolute, from the change
of each particle: ~64 cells of 1 Msun are touched, each rounded at 1.1e-16; the first versions asked 1e-12 of the mass the star
loses, 6e-5 to 0.35 Msun, from the totals of 13824 cells and then from the changes, logically wrong: one rounding of a cell exceeds it), and the
momentum of all particles changes by less than 5 % of (mass returned x 3000 km/s, above the table wind speed of the star), the bar of the wind test."""
import os
import math
import shutil
import struct
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
from gizmo.test import build_gizmo_for_test, get_stellar_evolution_tables, mpi_launch_command

TEST_NAME = "single_stars_wind_dump"
TEST_DIR = Path(__file__).parent
sys.path.insert(0, str(TEST_DIR.parent / "single_stars_winds_agb_ia"))
from make_single_stars_winds_agb_ia_ics import make_single_stars_winds_agb_ia_ics  # noqa: E402


@pytest.fixture(scope="module")
def built():
    tables = get_stellar_evolution_tables()
    if not os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        build_gizmo_for_test(TEST_NAME)
    return tables


def run(tables):
    d = TEST_DIR / "run_dump"
    if not os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN") and d.exists():
        shutil.rmtree(d)
    (d / "output").mkdir(parents=True, exist_ok=True)
    if not os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        L = make_single_stars_winds_agb_ia_ics(str(d / "output" / "snapshot_000.hdf5"), tables["StellarTablesFile"], which=("wind",))
        p = (TEST_DIR / f"{TEST_NAME}.params").read_text().replace("SET_BY_TEST", tables["StellarTablesFile"])
        (d / f"{TEST_NAME}.params").write_text(p.replace("BoxSize                            45.0", f"BoxSize                            {L}"))
        with open(d / f"test_{TEST_NAME}.out", "w") as out:
            subprocess.run(mpi_launch_command(2) + [str(TEST_DIR / "GIZMO"), f"{TEST_NAME}.params", "2", "0"], cwd=d, stdout=out,
                           stderr=subprocess.STDOUT, env={**os.environ, "OMP_NUM_THREADS": "1"}, check=True)
    return d / "output"


def read_dump(fname):
    recs, rec_dtype = [], np.dtype([("id", "<u8"), ("type", "<i4"), ("v", "<f8", (5,))], align=False)
    data, k = Path(fname).read_bytes(), 0
    while k < len(data):
        phase, rec, t, n = struct.unpack_from("<iqdq", data, k); k += 4 + 8 + 8 + 8
        rows = np.frombuffer(data, dtype=rec_dtype, count=n, offset=k); k += n * rec_dtype.itemsize
        recs.append((phase, rec, t, rows))
    return recs


def test_wind_dump(built):
    out = run(built)
    ranks = [read_dump(out / f"wind_dep_dump.{r}.bin") for r in (0, 1)]
    assert all(len(r) >= 10 and len(r) % 2 == 0 for r in ranks), [len(r) for r in ranks]
    for r in ranks:
        for a, b in zip(r[0::2], r[1::2]):
            assert a[0] == 0 and b[0] == 1 and a[1] == b[1] and a[2] == b[2] and len(a[3]) == len(b[3])
    assert [a[1] for a in ranks[0][0::2]] == [a[1] for a in ranks[1][0::2]]
    for k in range(0, len(ranks[0]), 2):
        before = np.concatenate([ranks[r][k][3] for r in (0, 1)]); after = np.concatenate([ranks[r][k + 1][3] for r in (0, 1)])
        before, after = before[np.argsort(before["id"], kind="stable")], after[np.argsort(after["id"], kind="stable")]
        assert np.array_equal(before["id"], after["id"])
        dm = after["v"][:, 0] - before["v"][:, 0]  # per particle, so that the sums are of small numbers
        gas = before["type"] == 0
        dstar = -math.fsum(dm[~gas])
        assert dstar > 0
        assert abs(math.fsum(dm)) < 1e-14 and abs(math.fsum(dm[gas]) - dstar) < 1e-14, (math.fsum(dm), math.fsum(dm[gas]) - dstar, dstar)
        dp = np.array([math.fsum(after["v"][:, 0] * after["v"][:, 1 + c] - before["v"][:, 0] * before["v"][:, 1 + c]) for c in range(3)])
        assert np.all(np.abs(dp) < 0.05 * dstar * 3000.0), (dp, dstar)
