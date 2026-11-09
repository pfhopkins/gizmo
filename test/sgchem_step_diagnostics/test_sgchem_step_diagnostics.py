"""sgchem_step_diagnostics: SGCHEM_PRINT_STEP_COOLING_DIAGNOSTICS prints, on each step with a status line, one line SGCHEM_DIAG from rank 0
with its own active cells: count, energy the cooling update removed [code] and its running sum, temperature and hydrogen
density ranges (stella_dev cooling.cc). The box of sgchem_box (8^3 cells, n_H = 100, 8000 K, SGChem network 5), 1 rank, 8 equal
steps of 0.004. Bars, written before the run: one line per call of the update, 9 with the call at the start (which changes nothing)
and the one after the last snapshot (the first version of this test expected 8 lines, a falling temperature from the first line on,
and all lines adding up to the energy lost by the last snapshot: logically wrong, the snapshot at TimeMax is written before the last
update, and the snapshot temperature of the start is a stale 100 K), N_active = 512 in all; the dE_cool of lines 2 to 9 are positive
(the box cools); lines 1 to 8 add up to the thermal energy the snapshots lost, sum m (u_0 - u_1), to 1e-5, and the last running
sum is the sum of all 9; over lines 2 to 9 the largest temperature never exceeds 8000 K x 1.01 and does not rise, the smallest is above
MinGasTemp 2.73 K; the density range of every line lies inside 90..110 cm^-3."""
import re
import subprocess
import sys
from os import environ, path

import h5py
import numpy as np
import pytest

from gizmo.test import build_and_run_test, default_omp_threads, get_cooling_tables

HERE = path.dirname(path.abspath(__file__))
ROOT = path.dirname(path.dirname(HERE))
NAME = "sgchem_step_diagnostics"


def chemistry_tables_dir():
    """SGCHEM_TABLES_DIR, else the install of scripts/get_stella_tables.sh (as sgchem_box)"""
    if environ.get("SGCHEM_TABLES_DIR"):
        return environ["SGCHEM_TABLES_DIR"]
    out = subprocess.run(["bash", path.join(ROOT, "scripts", "get_stella_tables.sh")], capture_output=True, text=True,
                         env={**environ, "PYTHON": environ.get("PYTHON", sys.executable)})
    assert out.returncode == 0, f"get_stella_tables.sh failed:\n{out.stdout}\n{out.stderr}"
    return [l for l in out.stdout.splitlines() if " holds " in l][-1].split()[1]


@pytest.fixture(scope="module")
def box():
    if not environ.get("SUNDIALS_DIR"):
        pytest.skip("set SUNDIALS_DIR (SGCHEM builds against SUNDIALS 6.x or 7.x)")
    get_cooling_tables(HERE)
    sys.path.insert(0, path.join(ROOT, "test", "sgchem_box"))
    from make_sgchem_box_ics import make_ics
    make_ics(path.join(HERE, "sgchem_box_ics.hdf5"))
    build_and_run_test(NAME, 1, default_omp_threads(), param_overrides={"SGChemTablesPath": chemistry_tables_dir()})
    return path.join(HERE, "output"), open(path.join(HERE, f"test_{NAME}.out")).read()


def test_diagnostic_lines(box):
    out, log = box
    lines = [l for l in log.splitlines() if l.startswith("SGCHEM_DIAG:")]
    assert len(lines) == 9, len(lines)
    num = r"([-+0-9.eE]+)"
    rec = [[float(x) for x in re.match(rf"SGCHEM_DIAG: N_active=(\d+)\s+dE_cool={num}\[code\]\s+E_cool_cum={num}\[code\]\s+T=\[{num},{num}\]K\s+nH=\[{num},{num}\]cm", l).groups()]
           for l in lines]
    rec = np.array(rec)
    assert (rec[:, 0] == 512).all()
    with h5py.File(path.join(out, "snapshot_000.hdf5")) as s0, h5py.File(path.join(out, "snapshot_001.hdf5")) as s1:
        m = s0["PartType0/Masses"][:]
        lost = float((m * (s0["PartType0/InternalEnergy"][:] - s1["PartType0/InternalEnergy"][:])).sum())
    assert abs(rec[:-1, 1].sum() / lost - 1) < 1e-5, (rec[:-1, 1].sum(), lost)
    assert abs(rec[-1, 2] / rec[:, 1].sum() - 1) < 1e-4
    assert rec[0, 1] == 0 and (rec[1:, 1] > 0).all() and rec[1, 4] <= 8080 and (np.diff(rec[1:, 4]) <= 0).all() and (rec[1:, 3] > 2.73).all(), rec
    assert (rec[:, 5] > 90).all() and (rec[:, 6] < 110).all() and (rec[:, 5] <= rec[:, 6]).all(), rec[:, 5:]
