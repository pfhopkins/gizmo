"""GALSF_FB_PHOTOIONIZATION_RECOMBINATION_AT_CELL_TEMPERATURE: the recombination bill of a cell scales with the chemistry's case-B
rate at the stored temperature above 1e4 K (stella_dev resolvedism_photoion.cc). One star in uniform gas at rest, 100 K and 3e4 K.
The ionised volume at fixed Q goes as 1 / alpha_B. Bar: cells(3e4 K) / cells(100 K) within 5 % of alpha_B(1e4 K) / alpha_B(3e4 K),
case B of Hui & Gnedin 1997 (chem table k13 is that fit)."""
import os
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
from gizmo.test import build_gizmo_for_test, get_stellar_evolution_tables, get_cooling_tables, mpi_launch_command

NAME = "photoion_recomb_temperature"
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "photoion_per_direction"))
from make_photoion_per_direction_ics import make_photoion_per_direction_ics  # noqa: E402


def alpha_b(T):
    lam = 2 * 157807.0 / T
    return 2.753e-14 * lam ** 1.5 / (1 + (lam / 2.74) ** 0.407) ** 2.242


def run(tag, T, tables):
    d = HERE / f"run_{tag}"
    shutil.rmtree(d, ignore_errors=True)
    (d / "output").mkdir(parents=True)
    L, dx, n_H = make_photoion_per_direction_ics(str(d / "output" / "snapshot_000.hdf5"), [(35.0, 1.0e6, (0.5, 0.5, 0.5))], T=T, n=32, n_H=30.0)
    for f in ("TREECOOL", "spcool_tables"):
        if (HERE / f).exists():
            os.symlink(HERE / f, d / f)
    p = (HERE / f"{NAME}.params").read_text().replace("SET_BY_TEST", tables["StellarTablesFile"])
    p = p.replace("SET_TABLES", os.environ.get("SGCHEM_TABLES_DIR", os.path.dirname(tables["StellarTablesFile"])))
    p = p.replace("BoxSize                            45.0", f"BoxSize                            {L}")
    (d / "run.params").write_text(p)
    with open(d / "run.out", "w") as o:
        subprocess.run(mpi_launch_command(2, 1) + [str(HERE / "GIZMO"), "run.params", "2", "0"], cwd=d, stdout=o, stderr=subprocess.STDOUT,
                       env={**os.environ, "OMP_NUM_THREADS": "1"}, check=True)
    return np.atleast_2d(np.loadtxt(d / "output" / "HIIheating.txt"))


def test_ionised_cells_follow_alpha_b():
    if not os.environ.get("SUNDIALS_DIR"):
        pytest.skip("set SUNDIALS_DIR (SGCHEM builds against SUNDIALS 6.x or 7.x)")
    t = get_stellar_evolution_tables()
    get_cooling_tables(str(HERE))
    build_gizmo_for_test(NAME)
    cold = run("cold", 100.0, t)
    hot = run("hot", 3.0e4, t)
    ratio, want = hot[-1, 3] / cold[-1, 3], alpha_b(1.0e4) / alpha_b(3.0e4)
    print(f"cells {cold[-1, 3]} {hot[-1, 3]} ratio {ratio} bar {want}")
    assert ratio == pytest.approx(want, rel=0.05)
