"""SGChem in GIZMO on a uniform periodic gas box (8^3 cells, n_H = 100 cm^-3, T0 = 8000 K, solar metals).

test_sgchem_startup: the installed chemistry tables are checked at start-up; with the right tables the run reaches
TimeMax, with a changed table it stops with a message naming the install script.
"""

import re
import shutil
import subprocess
import sys
from os import chdir, environ, path, makedirs

import pytest

from gizmo.test import build_and_run_test, run_test, get_cooling_tables, default_omp_threads

HERE = path.dirname(path.abspath(__file__))
ROOT = path.dirname(path.dirname(HERE))
NAME = "sgchem_box"


def chemistry_tables_dir():
    """SGCHEM_TABLES_DIR, else the install of scripts/get_stella_tables.sh (built from stella-tables, hash-checked)."""
    if environ.get("SGCHEM_TABLES_DIR"):
        return environ["SGCHEM_TABLES_DIR"]
    out = subprocess.run(["bash", path.join(ROOT, "scripts", "get_stella_tables.sh")], capture_output=True, text=True,
                         env={**environ, "PYTHON": environ.get("PYTHON", sys.executable)})
    assert out.returncode == 0, f"get_stella_tables.sh failed:\n{out.stdout}\n{out.stderr}"
    return [l for l in out.stdout.splitlines() if " holds " in l][-1].split()[1]


@pytest.fixture(scope="module")
def tables():
    if not environ.get("SUNDIALS_DIR"):
        pytest.skip("set SUNDIALS_DIR (SGCHEM builds against SUNDIALS 6.x or 7.x)")
    get_cooling_tables(HERE)
    sys.path.insert(0, HERE)
    from make_sgchem_box_ics import make_ics
    make_ics(path.join(HERE, "sgchem_box_ics.hdf5"))
    return chemistry_tables_dir()


def test_sgchem_startup(tables):
    build_and_run_test(NAME, 2, default_omp_threads(), param_overrides={"SGChemTablesPath": tables})
    log = open(path.join(HERE, f"test_{NAME}.out")).read()
    assert "SGCHEM: chemistry tables built" in log
    # the energy floor of MinGasTemp = 2.73 K: atomic gas with X_H = 0.7065 (ABHE = 0.1), 3/2 k T per particle
    xh, floor = re.search(r"SGCHEM: hydrogen mass fraction (\S+), MinEgySpec (\S+)", log).groups()
    mu = 4.0 / (1.0 + 3.0 * 0.7065)
    assert float(xh) == 0.7065 and float(floor) == pytest.approx(1.5 * 1.3806e-16 * 2.73 / (mu * 1.6726e-24) / 1.0e10, rel=1e-3)

    bad = path.join(HERE, "output_tables_changed")
    makedirs(bad, exist_ok=True)
    for f in ("energy-table-x.txt", "eh2-table.txt", "H2-cooling-ratios.dat", "wss_z_collis.txt"):
        shutil.copy(path.join(tables, f), bad)
    with open(path.join(bad, "wss_z_collis.txt"), "a") as f:
        f.write("\n")
    chdir(HERE)  # same executable, changed tables
    try:
        run_test(NAME, 2, default_omp_threads(), param_overrides={"SGChemTablesPath": bad}, allow_nonzero_exit=True)
    finally:
        chdir(ROOT)
    log = open(path.join(HERE, f"test_{NAME}.out")).read()
    assert "wss_z_collis.txt has hash" in log and "scripts/get_stella_tables.sh" in log
    assert "SGCHEM: chemistry tables built" not in log
