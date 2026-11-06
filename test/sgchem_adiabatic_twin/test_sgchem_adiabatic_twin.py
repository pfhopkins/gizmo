"""sgchem_adiabatic_twin: TEST_SKIP_COOLING_AND_CHEMISTRY_UPDATE gives the adiabatic twin of a cooling run (needs
COOLING_OPERATOR_SPLIT, so that the hydro work is not applied by the cooling): the box of sgchem_box (8^3 cells, n_H = 100,
T0 = 8000 K, at rest, SGChem network 5, 2 ranks) with the update left out. Bars, written before the run: after 8 steps
(0.032, 31 kyr) the temperature of every cell equals its start value to 1e-6 (sgchem_box cools the same box to 32 K in
125 kyr) and the tracked species (H2, H+, CO) are the start values bitwise."""
import subprocess
import sys
from os import environ, path

import h5py
import numpy as np
import pytest

from gizmo.test import build_and_run_test, default_omp_threads, get_cooling_tables

HERE = path.dirname(path.abspath(__file__))
ROOT = path.dirname(path.dirname(HERE))
NAME = "sgchem_adiabatic_twin"


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
    tables = chemistry_tables_dir()
    build_and_run_test(NAME, 2, default_omp_threads(), param_overrides={"SGChemTablesPath": tables})
    return path.join(HERE, "output"), tables


def test_adiabatic_twin(box):
    box, _ = box
    with h5py.File(path.join(box, "snapshot_000.hdf5")) as s0, h5py.File(path.join(box, "snapshot_001.hdf5")) as s1:
        assert s1["Header"].attrs["Time"] == 0.032
        T0, T1 = s0["PartType0/Temperature"][:], s1["PartType0/Temperature"][:]
        assert np.max(np.abs(T1 / T0 - 1)) < 1e-6, (T0.min(), T1.min())
        assert np.array_equal(s0["PartType0/SGChemTracAbund"][:], s1["PartType0/SGChemTracAbund"][:])

