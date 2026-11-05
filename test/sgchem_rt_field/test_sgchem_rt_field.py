"""sgchem_rt_field: SGChem reads G0, G0_LW and G0_dust from the moment solver's bins (reports/design/design_m1_coupling.md).
A uniform molecular box at n_H = 100 cm^-3, no stars shining. The M1 build has its bins seeded in the initial snapshot with
the band energies of a Draine field of G0 Habing (6-8, 8-11.2, 11.2-13.6, 3.444-6, 0.4133-3.444 eV); the reference is the
same box without radiation transport, with the same G0 through SGChemG0. Dust-free, so the half-cell dust column and the
absorption leave the field as seeded. H2 fraction and temperature agree within 1 %.
  test_habing_gate: G0 = 10.
  test_floor: empty bins and SGChemBackgroundG0 = 2 equal the uniform path at G0 = 0 with the same floor.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path
import h5py
import numpy as np
import pytest
from gizmo.test import build_gizmo_for_test, get_stellar_evolution_tables, get_cooling_tables, mpi_launch_command

TEST_NAME = "sgchem_rt_field"
TEST_DIR = Path(__file__).parent
sys.path.insert(0, str(TEST_DIR.parent / "treecol_pixels"))
from make_treecol_pixels_ics import make_treecol_pixels_ics, lattice  # noqa: E402

N, N_H = 8, 100.0
MSUN, PC, MP = 1.989e33, 3.085678e18, 1.6726e-24
RHO = N_H * 1.4 * MP * PC ** 3 / MSUN  # Msun / pc^3
L = N * (1.0 / RHO) ** (1.0 / 3.0)
UNIT_P = MSUN * 1e10 / PC ** 3  # erg / cm^3 per code pressure
U = dict(u68=1.877e-14, u8132=3.380e-14 - 7.799e-15, ulw=7.799e-15, unuv=4.696e-14, uopt=5.141e-13)  # per Habing (chem_interface.h)
RT = ("GALSF_SINGLE_STARS_RT_TABLE_BANDS", "RT_M1", "RT_SPEEDOFLIGHT_REDUCTION=1.0e-5", "RT_DISABLE_RAD_PRESSURE")
BUILDS = {}


@pytest.fixture(scope="module")
def tables():
    if not os.environ.get("SUNDIALS_DIR"):
        pytest.skip("set SUNDIALS_DIR (SGCHEM builds against SUNDIALS 6.x or 7.x)")
    t = get_stellar_evolution_tables()
    get_cooling_tables(str(TEST_DIR))
    return t


def build(tag, extra=()):
    if tag not in BUILDS:
        build_gizmo_for_test(TEST_NAME, extra_config_flags=tuple(extra))
        shutil.move(TEST_DIR / "GIZMO", TEST_DIR / f"GIZMO_{tag}")
        BUILDS[tag] = TEST_DIR / f"GIZMO_{tag}"
    return BUILDS[tag]


def run(exe, tag, tables, G0_bins, over):
    d = TEST_DIR / f"run_{tag}"
    if d.exists():
        shutil.rmtree(d)
    (d / "output").mkdir(parents=True)
    ic = d / "output" / "snapshot_000.hdf5"
    ab = np.zeros((N ** 3, 3)); ab[:, 0] = 0.25; ab[:, 1] = 1e-4
    make_treecol_pixels_ics(str(ic), L, lattice(N, L), [(1.0, 1e12, np.array([0.5, 0.5, 0.5]) * L)], hsml=1.0, tracabund=ab)
    if G0_bins is not None:
        V = 1.0 / RHO
        e = np.array([U["u8132"], U["ulw"], U["unuv"], U["uopt"], U["u68"]]) * G0_bins / UNIT_P * V
        with h5py.File(ic, "a") as F:
            F["PartType0/PhotonEnergy"] = np.tile(e, (N ** 3, 1))
    for f in ("TREECOOL", "spcool_tables"):
        if (TEST_DIR / f).exists():
            os.symlink(TEST_DIR / f, d / f)
    T = tables["StellarTablesFile"]
    p = (TEST_DIR / f"{TEST_NAME}.params").read_text()
    p = p.replace("StellarTablesFile           SET_BY_TEST", f"StellarTablesFile           {T}")
    p = p.replace("SGChemTablesPath            SET_BY_TEST", f"SGChemTablesPath            {os.environ.get('SGCHEM_TABLES_DIR', os.path.dirname(T))}")
    over = dict(over, BoxSize=L)
    p = "\n".join([l for l in p.splitlines() if not (l.split() and l.split()[0] in over)] + [f"{k:27s} {v}" for k, v in over.items()]) + "\n"
    (d / f"{TEST_NAME}.params").write_text(p)
    with open(d / f"test_{TEST_NAME}.out", "w") as out:
        subprocess.run(mpi_launch_command(2) + [str(exe), f"{TEST_NAME}.params", "2", "0"], cwd=d, stdout=out,
                       stderr=subprocess.STDOUT, env={**os.environ, "OMP_NUM_THREADS": "1"}, check=True)
    with h5py.File(d / "output" / "snapshot_001.hdf5", "r") as F:
        g = F["PartType0"]
        return {"H2": g["SGChemTracAbund"][:, 0].mean(), "T": g["Temperature"][()].mean()}


def compare(a, b, x0):
    print(a, b)
    assert x0 is None or a["H2"] < 0.95 * x0  # the field acted
    for k in a:
        assert abs(a[k] / b[k] - 1) < 1e-2, (k, a[k], b[k])


def test_habing_gate(tables):
    over = {"SGChemG0": 10, "SGChemG0_LW": 10, "SGChemBackgroundG0": 0}
    ref = run(build("uniform"), "uniform", tables, None, over)
    m1 = run(build("m1", RT), "m1", tables, 10.0, {"SGChemG0": 0, "SGChemG0_LW": 0, "SGChemBackgroundG0": 0})
    compare(m1, ref, 0.25)


def test_floor(tables):
    over = {"SGChemG0": 0, "SGChemG0_LW": 0, "SGChemBackgroundG0": 2000}
    ref = run(build("uniform"), "uniform_floor", tables, None, over)
    m1 = run(build("m1", RT), "m1_floor", tables, 0.0, over)
    compare(m1, ref, None)
