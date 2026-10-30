"""single_stars_metals test: the metals of single-star builds in the FIRE-3 layout (slot 0 total Z, 1 He, 2-10 C, N, O,
Ne, Mg, Si, S, Ca, Fe; H implicit), with SGChem network 5 (production chemistry). See README.md."""
import os
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
import h5py
import pytest
from gizmo.test import build_gizmo_for_test, get_stellar_evolution_tables, get_cooling_tables, mpi_launch_command

TEST_NAME = "single_stars_metals"
TEST_DIR = Path(__file__).parent
sys.path.insert(0, str(TEST_DIR))
from make_single_stars_metals_ics import make_single_stars_metals_ics  # noqa: E402

FIRE_ELEMENTS = [1, 2, 3, 4, 6, 8, 10, 12, 16, 22]  # table element (H=0 .. Zn=26) of slots 1..10
X_H_PRIMORDIAL = 0.7065  # HYDROGEN_MASSFRAC of SGChem builds
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


def solar_table(tables_file):
    """the tables' initial composition at their Z = 0.014 node (Asplund+09 present day), elements H..Zn"""
    with h5py.File(tables_file, "r") as F:
        Zt = F["Z"][()]
        iz = int(np.argmin(np.abs(Zt - 0.014)))
        assert Zt[iz] == 0.014
        return F["X_init"][()].reshape(len(Zt), -1)[iz].astype(float)


def run(exe, tag, tables, metals, stars=(), flag=2, over=None, ranks=2):
    d = TEST_DIR / f"run_{tag}"
    if d.exists():
        shutil.rmtree(d)
    (d / "output").mkdir(parents=True)
    ic = d / ("snapshot.hdf5" if flag == 0 else "output/snapshot_000.hdf5")
    L = make_single_stars_metals_ics(str(ic), metals, stars)
    for f in ("TREECOOL", "spcool_tables"):
        if (TEST_DIR / f).exists():
            os.symlink(TEST_DIR / f, d / f)
    T = tables["StellarTablesFile"]
    over = {"BoxSize": L, **(over or {})}
    p = (TEST_DIR / f"{TEST_NAME}.params").read_text()
    p = p.replace("StellarTablesFile           SET_BY_TEST", f"StellarTablesFile           {T}")
    p = p.replace("SGChemTablesPath            SET_BY_TEST", f"SGChemTablesPath            {os.environ.get('SGCHEM_TABLES_DIR', os.path.dirname(T))}")
    p = "\n".join([l for l in p.splitlines() if not (l.split() and l.split()[0] in over)] + [f"{k:27s} {v}" for k, v in over.items()]) + "\n"
    (d / f"{TEST_NAME}.params").write_text(p)
    with open(d / f"test_{TEST_NAME}.out", "w") as out:
        subprocess.run(mpi_launch_command(ranks) + [str(exe), f"{TEST_NAME}.params", str(flag), "0"], cwd=d, stdout=out,
                       stderr=subprocess.STDOUT, env={**os.environ, "OMP_NUM_THREADS": "1"}, check=True)
    return d


def gas_metals(snap):
    with h5py.File(snap, "r") as F:
        m = F["PartType0/Masses"][()]
        return m, F["PartType0/Metallicity"][()].reshape(len(m), -1)


@pytest.mark.parametrize("z_solar", [1.0, 0.3])
def test_solar_abundances(tables, z_solar):
    """a fresh start with InitMetallicity: total Z = 0.014 x InitMetallicity and the tables' solar element pattern, as
    stella_dev (Asplund+09 present day; the tables' initial composition at Z = 0.014); He primordial + scaled excess"""
    d = run(build("default"), f"solar_{z_solar}", tables, np.zeros(11), flag=0, over={"InitMetallicity": z_solar})
    _, Z = gas_metals(d / "output" / "snapshot_000.hdf5")
    X = solar_table(tables["StellarTablesFile"])
    expect = np.concatenate([[0.014], X[FIRE_ELEMENTS]]) * z_solar
    expect[1] = (1 - X_H_PRIMORDIAL) + (X[1] - (1 - X_H_PRIMORDIAL)) * z_solar
    assert np.all(np.abs(Z / expect - 1) <= 1e-10), (Z[0], expect)
