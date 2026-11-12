"""MHD double rarefaction with a normal field: near-vacuum and vacuum cases, compared with converged
Athena++ solutions (mhd_double_rarefaction_exact.hdf5; see README.md).

The centre of both problems is where the Riemann solver meets vacuum or near-vacuum states (faces whose
states separate faster than the fast speed), so these runs check the solver's accuracy, positivity and divB
control next to near-empty cells.
"""

import os
import sys
from pathlib import Path
from shutil import copyfile, rmtree

import h5py
import numpy as np
import pytest

from gizmo.test import (build_gizmo_for_test, run_test, default_mpi_ranks, default_omp_threads,
                        TRUNCATED_MARKER, download_test_files)

TEST_NAME = "mhd_double_rarefaction"
TEST_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TEST_DIR))
sys.path.insert(0, str(TEST_DIR.parent))
import make_mhd_double_rarefaction_ics as ics  # noqa: E402
import mhd_1d_reference as ref1d  # noqa: E402

# L1 over the clean window, normalised by the reference field's range there (mhd_1d_reference.regional_l1),
# and the mean |Bx - Bx0| / Bx0 (Bx is exactly constant in 1D). TOLERANCES: see README.md for the measured
# values they are set from.
TOL = {
    # ~10-20% above the largest value measured at 2, 4 and 16 MPI ranks
    "near_vacuum": {"vx": 0.0106, "divB_rms": 0.0039, "vy": 0.041, "By": 0.050, "rho": 0.021, "Bx_drift": 0.0119},
    "vacuum": {"vx": 0.0075, "divB_rms": 0.0108, "vy": 0.052, "By": 0.075, "rho": 0.022, "Bx_drift": 0.0133},
}

_BUILT = [False]


def _outdir(case):
    return TEST_DIR / f"output_{case}"


def _build_and_run(case, num_mpi_ranks, num_omp_threads):
    if os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        return
    if not _BUILT[0]:
        build_gizmo_for_test(TEST_NAME, num_omp_threads)
        _BUILT[0] = True
    stem = f"{TEST_NAME}_{case}_ics"
    ics.make_mhd_double_rarefaction_ics(str(TEST_DIR / f"{stem}.hdf5"), case)
    rmtree(_outdir(case), ignore_errors=True)
    t_end = ics.CASES[case]["t_end"]
    cwd = os.getcwd()
    try:
        os.chdir(TEST_DIR)
        run_test(TEST_NAME, num_mpi_ranks, num_omp_threads,
                 param_overrides={"InitCondFile": stem, "OutputDir": f"output_{case}", "TimeMax": t_end,
                                  "TimeBetSnapshot": t_end / 2})
    finally:
        os.chdir(cwd)
        log = TEST_DIR / f"test_{TEST_NAME}.out"
        if log.is_file() and _outdir(case).is_dir():
            copyfile(log, _outdir(case) / f"test_{TEST_NAME}.out")


def load_reference(case):
    """The Athena++ solution in the dict layout mhd_1d_reference.compare_snapshot expects."""
    left, right = ics.left_right(case)
    exact = TEST_DIR / f"{TEST_NAME}_exact.hdf5"
    if not exact.is_file():  # reference lives on the test-data website, fetched on demand
        cwd = os.getcwd()
        try:
            os.chdir(TEST_DIR)
            download_test_files(TEST_NAME)
        except FileNotFoundError:  # raised for the missing remote IC; this test generates its own
            pass
        finally:
            os.chdir(cwd)
    assert exact.is_file(), f"could not download {exact.name}"
    with h5py.File(exact, "r") as F:
        g = F[case]
        ref = {k: g[k][:].astype(float) for k in ("x",) + ref1d.PRIM_NAMES}
        ref.update(Bx=float(g.attrs["Bx"]), gamma=float(g.attrs["gamma"]), L=4.0, x0=2.0, nx=len(ref["x"]),
                   left=np.array(left), right=np.array(right))
    assert abs(ref["Bx"] - ics.BX) < 1e-12
    return ref


def measure(snap_path, case):
    snap = ref1d.read_snapshot(snap_path, ics.GAMMA)
    ref = load_reference(case)
    cmp = ref1d.compare_snapshot(snap, ref)
    m = dict(cmp["L1"])
    m["Bx_drift"] = float(np.mean(np.abs(snap["Bx"] - ics.BX)) / ics.BX)
    m["divB_rms"] = cmp.get("divB_rms", np.nan)
    m["P_min"] = float(snap["P"].min())
    m["rho_min"] = float(snap["rho"].min())
    m["finite"] = all(np.all(np.isfinite(snap[f])) for f in ("rho", "P", "vx", "vy", "By", "Bx"))
    return m, cmp


@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(max_ranks=8),))  # 50k-particle strip: more ranks only add communication
@pytest.mark.parametrize("num_omp_threads", (default_omp_threads(),))
@pytest.mark.parametrize("case", list(ics.CASES))
def test_mhd_double_rarefaction(case, num_mpi_ranks, num_omp_threads):
    _build_and_run(case, num_mpi_ranks, num_omp_threads)
    marker = _outdir(case) / TRUNCATED_MARKER
    assert not marker.is_file(), f"{_outdir(case)} is from a run that did not finish: {marker.read_text().strip()}"
    snaps = sorted(_outdir(case).glob("snapshot_*.hdf5"))
    assert snaps, f"no snapshots in {_outdir(case)}"
    snap = str(snaps[-1])
    with h5py.File(snap, "r") as F:
        t = float(F["Header"].attrs["Time"])
    assert abs(t - ics.CASES[case]["t_end"]) < 1e-6, f"final snapshot at t = {t}"
    m, cmp = measure(snap, case)
    print(f"[{TEST_NAME}/{case}] window x in [{cmp['window'][0]:.3f}, {cmp['window'][1]:.3f}]: "
          + " ".join(f"{k}={v:.4g}" for k, v in m.items() if k != "finite"))
    assert m["finite"], "non-finite gas fields"
    assert m["P_min"] > 0 and m["rho_min"] > 0, f"non-positive state: P_min {m['P_min']:.3g} rho_min {m['rho_min']:.3g}"
    for k, tol in TOL[case].items():
        assert m[k] < tol, f"{case}: {k} = {m[k]:.4g} exceeds {tol} (README.md has the measured values)"
