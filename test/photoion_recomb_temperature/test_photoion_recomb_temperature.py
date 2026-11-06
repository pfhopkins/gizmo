"""GALSF_FB_PHOTOIONIZATION_RECOMBINATION_AT_CELL_TEMPERATURE: the recombination bill of a cell scales with the chemistry's case-B
rate at the stored temperature above 1e4 K (stella_dev resolvedism_photoion.cc). One star in uniform gas at rest, n_H = 1, 100 K and
3e4 K. At fixed Q the ionised cells satisfy N <alpha_B(max(T, 1e4 K))> = const. Bar: N <alpha_B> of the two runs agree within 5 %, T from
the snapshot in the N cells nearest the star, alpha_B of Hui & Gnedin 1997 (chem table k13 is that fit). First version of the bar,
N(hot) / N(cold) = alpha_B(1e4) / alpha_B(3e4), was wrong: gas at 3e4 K ionises itself and settles near 1.6e4 K within a step."""
import os
import shutil
import subprocess
import sys
from pathlib import Path
import h5py
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
    L, dx, n_H = make_photoion_per_direction_ics(str(d / "output" / "snapshot_000.hdf5"), [(15.0, 1.0e6, (0.5, 0.5, 0.5))], T=T, n=32, n_H=1.0)
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
    n = int(np.atleast_2d(np.loadtxt(d / "output" / "HIIheating.txt"))[-1, 3])
    with h5py.File(sorted((d / "output").glob("snapshot_*.hdf5"))[-1]) as f:
        g = f["PartType0"]; x = g["Coordinates"][:]; c = f["PartType4/Coordinates"][0]; T_gas = g["Temperature"][:]; box = f["Header"].attrs["BoxSize"]
    dx_c = x - c; dx_c -= box * np.round(dx_c / box)
    near = np.argsort((dx_c ** 2).sum(1))[:n]
    return n, np.median(T_gas[near]), n * np.mean(alpha_b(np.maximum(T_gas[near], 1.0e4))) / alpha_b(1.0e4)


def test_ionised_cells_follow_alpha_b():
    if not os.environ.get("SUNDIALS_DIR"):
        pytest.skip("set SUNDIALS_DIR (SGCHEM builds against SUNDIALS 6.x or 7.x)")
    t = get_stellar_evolution_tables()
    get_cooling_tables(str(HERE))
    build_gizmo_for_test(NAME)
    cold = run("cold", 100.0, t)
    hot = run("hot", 3.0e4, t)
    print(f"cold (cells, T, N alpha) {cold}, hot {hot}")
    assert 10 ** 4.1 < hot[1] < 10 ** 4.3, "the hot cells stay above 1e4 K"
    assert hot[2] == pytest.approx(cold[2], rel=0.05)
