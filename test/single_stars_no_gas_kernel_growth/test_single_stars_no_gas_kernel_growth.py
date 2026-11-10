"""single_stars_no_gas_kernel_growth (narrow): a single-star event whose kernel reaches no gas widens the kernel x1.4 and tries
again in the same step, at most 3 times (bubble retry). Setup of test/single_stars_forced_events (two
20 Msun stars, supernovae forced at 0.002 and 0.005), the first star in an empty sphere of 3.5 pc, its feedback kernel set to 3 pc
before the loop (TEST_SINGLE_STARS_FEEDBACK_KERNEL_AT_MOST, a star in a bubble; the port gives coupling sources kernels up to 2 kpc,
so a hole alone never leaves one empty), so it finds no gas until widened once (4.2 pc).
Bars, written before the runs, from the rule: both supernovae couple in the step they are forced, t_forced <= t <
t_forced + 1.5e-4, with zero steps without deposit; mass conserved to 1e-12."""
import os
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
import h5py
import pytest
from gizmo.test import build_gizmo_for_test, get_stellar_evolution_tables, mpi_launch_command

NAME = "single_stars_no_gas_kernel_growth"
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "single_stars_forced_events"))
from make_single_stars_forced_events_ics import make_single_stars_forced_events_ics  # noqa: E402
R_HOLE = 3.5


def total_mass(fname):
    with h5py.File(fname, "r") as F:
        return sum(F[g]["Masses"][()].sum() for g in ("PartType0", "PartType4"))


@pytest.fixture(scope="module")
def run():
    tables = get_stellar_evolution_tables()
    build_gizmo_for_test(NAME)
    d = HERE / "run"
    shutil.rmtree(d, ignore_errors=True)
    (d / "output").mkdir(parents=True)
    ic = d / "output" / "snapshot_000.hdf5"
    L = make_single_stars_forced_events_ics(str(ic))
    with h5py.File(ic, "r+") as F:
        g = F["PartType0"]; s0 = F["PartType4/Coordinates"][0]
        x = g["Coordinates"][()]; dd = x - s0; dd -= L * np.round(dd / L)
        keep = np.sqrt((dd ** 2).sum(1)) > R_HOLE
        data = {k: g[k][()][keep] for k in g.keys()}
        for k in list(g.keys()):
            del g[k]
        for k, v in data.items():
            g[k] = v
        n = int(keep.sum()); h = F["Header"]
        npart = h.attrs["NumPart_ThisFile"]; npart[0] = n; h.attrs["NumPart_ThisFile"] = npart; h.attrs["NumPart_Total"] = npart
    p = (HERE / f"{NAME}.params").read_text().replace("SET_BY_TEST", tables["StellarTablesFile"]).replace("BoxSize                            45.0", f"BoxSize                            {L}")
    (d / f"{NAME}.params").write_text(p)
    with open(d / "run.out", "w") as o:
        subprocess.run(mpi_launch_command(2) + [str(HERE / "GIZMO"), f"{NAME}.params", "2", "0"], cwd=d, stdout=o, stderr=subprocess.STDOUT,
                       env={**os.environ, "OMP_NUM_THREADS": "1"}, check=False)
    ev = d / "output" / "single_star_events.txt"
    rows = np.atleast_2d(np.loadtxt(ev)) if ev.exists() else np.zeros((0, 12))
    snaps = sorted((d / "output").glob("snapshot_*.hdf5"))
    return ic, snaps[-1], rows, (d / "run.out").read_text()


def test_both_couple_in_their_step(run):
    ic, snap, rows, log = run
    sn = {int(r[1]): r for r in rows if int(r[2]) == 1}
    print({k: (v[0], v[11]) for k, v in sn.items()}, log.count("found no gas"))
    assert sorted(sn) == [1, 2], rows
    for k, t_forced in ((1, 0.002), (2, 0.005)):
        assert t_forced <= sn[k][0] < t_forced + 1.5e-4, (k, sn[k][0])
        assert sn[k][11] == 0, (k, sn[k][11])
    assert abs(total_mass(snap) / total_mass(ic) - 1) < 1e-12
