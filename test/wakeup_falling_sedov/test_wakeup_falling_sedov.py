"""wakeup_falling_sedov: WakeupTargetBin. A point blast in cold uniform gas wakes cells on long timesteps while the periodic box
falls in a uniform field (g = -0.5 in y). Default 2 sends a woken cell to the highest active bin, 0 to the waker's bin
minus log2(WAKEUP). With g uniform and pairwise hydro forces v_cm,y = -0.5 t exactly, so a woken cell losing part of its field
kick shows up as a centre-of-mass error (1e-3).
"""
import re
import sys
from pathlib import Path
import h5py
import numpy as np
import pytest
from gizmo.test import build_and_run_test, default_mpi_ranks, problem_output_dir

NAME = "wakeup_falling_sedov"
HERE = Path(__file__).resolve().parent
G_FIELD, TOL = 0.5, 1e-3


def updates():
    """steps and cell updates from the run log"""
    log = (HERE / f"test_{NAME}.out").read_text()
    return len(re.findall(r"^Sync-Point", log, re.M)), sum(int(x) for x in re.findall(r"^Total active:.*Sum:\s+(\d+)", log, re.M))


def vcm_error(out):
    err = []
    for f in sorted(Path(out).glob("snapshot_*.hdf5"))[1:]:
        with h5py.File(f) as F:
            t, g = F["Header"].attrs["Time"], F["PartType0"]
            v = g["Velocities"][:, 1] - 0.5 * G_FIELD * g["TimeStep"][:]  # snapshots are written before the closing half kick
            err.append(abs((g["Masses"][:] * v).sum() / g["Masses"][:].sum() + G_FIELD * t) / (G_FIELD * t))
    return max(err)


@pytest.fixture(scope="module")
def runs():
    ic = HERE / f"{NAME}_ics.hdf5"
    if not ic.exists():
        sys.path.insert(0, str(HERE))
        from make_wakeup_falling_sedov_ics import make_wakeup_falling_sedov_ics
        make_wakeup_falling_sedov_ics(str(ic))
    ranks = default_mpi_ranks(max_ranks=4)
    r = {}
    build_and_run_test(NAME, ranks, 1)
    r["default"] = problem_output_dir(NAME, None, ())
    (HERE / "used2").write_text((HERE / f"test_{NAME}.out").read_text())
    r["default"] = (updates(), vcm_error(r["default"]))
    build_and_run_test(NAME, ranks, 1, param_overrides={"WakeupTargetBin": 0})
    o = problem_output_dir(NAME, None, ())
    r["waker"] = (updates(), vcm_error(o))
    return r


def test_default_is_highest_active_bin(runs):
    used = (HERE / "used2").read_text()
    assert "WakeupTargetBin not set in parameter file: defaulting to 2" in used
    print(f"default {runs['default']}, waker bin minus offset {runs['waker']}")


def test_centre_of_mass(runs):
    assert runs["default"][1] < TOL, runs["default"]
    assert runs["waker"][1] < TOL, runs["waker"]
