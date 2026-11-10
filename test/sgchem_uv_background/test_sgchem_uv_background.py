"""sgchem_uv_background: SGChem in a comoving run takes the UV background from TREECOOL.

The sgchem_box lattice (n_H = 100 cm^-3 physical, 8000 K, solar metals) at a = 0.25 (z = 3), 125 kyr, twice: with the
default TREECOOL and with every rate in the table set to 1e-40.
test_runs_comoving: both runs reach TimeMax (SGChem used to stop on ComovingIntegrationOn).
test_uvb_ionises: with the background the cells hold more electrons and more H+ than without, at the level of the
  Gamma_HI floor (the final x_e is above 1e-3 and above 3 x the run without).
test_uvb_rates_in_log: the start-up log names the redshift of the table lookup and the default TREECOOL (it begins with
  '##' comment lines) is read with its rows.
"""
import glob
import shutil
import sys
from os import chdir, environ, makedirs, path, symlink

import h5py
import numpy as np
import pytest

from gizmo.test import build_and_run_test, get_cooling_tables, run_test, default_omp_threads

HERE = path.dirname(path.abspath(__file__))
ROOT = path.dirname(path.dirname(HERE))
NAME = "sgchem_uv_background"
sys.path.insert(0, path.join(ROOT, "test", "sgchem_box"))
sys.path.insert(0, HERE)


def final_snapshot(d):
    return sorted(glob.glob(path.join(d, "output", "snapshot_*.hdf5")))[-1]


@pytest.fixture(scope="module")
def runs():
    if not environ.get("SUNDIALS_DIR"):
        pytest.skip("set SUNDIALS_DIR (SGCHEM builds against SUNDIALS 6.x or 7.x)")
    from test_sgchem_box import chemistry_tables_dir
    from make_ics import make_ics
    tables = chemistry_tables_dir()
    if environ.get("SPCOOL_TABLES") and not path.exists(path.join(HERE, "spcool_tables")):
        symlink(environ["SPCOOL_TABLES"], path.join(HERE, "spcool_tables"))
    get_cooling_tables(HERE)
    make_ics(path.join(HERE, f"{NAME}_ics.hdf5"))
    build_and_run_test(NAME, 2, default_omp_threads(), param_overrides={"SGChemTablesPath": tables})
    var = path.join(HERE, "output_no_uvb")
    makedirs(var, exist_ok=True)
    for f in ("GIZMO", f"{NAME}_ics.hdf5", f"{NAME}.params"):
        shutil.copy(path.join(HERE, f), var)
    if not path.exists(path.join(var, "spcool_tables")):
        symlink(path.realpath(path.join(HERE, "spcool_tables")), path.join(var, "spcool_tables"))
    with open(path.join(var, "TREECOOL"), "w") as out:  # same redshift grid, every rate 1e-40
        for l in open(path.join(HERE, "TREECOOL")):
            out.write(l if l.lstrip().startswith("#") or not l.strip() else l.split()[0] + " " + " ".join(["1e-40"] * 6) + "\n")
    chdir(var)
    try:
        run_test("no_uvb", 2, default_omp_threads(), params_name=NAME, param_overrides={"SGChemTablesPath": tables})
    finally:
        chdir(ROOT)
    return {"uvb": HERE, "no_uvb": var}


def test_runs_comoving(runs):
    for k, d in runs.items():
        with h5py.File(final_snapshot(d)) as f:
            a = f["Header"].attrs["Time"]
        assert 0.25 < a <= 0.2500143 * (1 + 1e-9), (k, a)


def electrons(d):
    with h5py.File(final_snapshot(d)) as f:
        return np.mean(f["PartType0/ElectronAbundance"][:])


def test_uvb_ionises(runs):
    xe, xe0 = electrons(runs["uvb"]), electrons(runs["no_uvb"])
    print(f"mean x_e with the background {xe:.4g}, without {xe0:.4g}")
    assert xe > 1e-3 and xe > 3 * xe0


def test_uvb_rates_in_log(runs):
    log = open(path.join(HERE, f"test_{NAME}.out")).read()
    assert "SGCHEM: UV background from TREECOOL, z = 3" in log
    assert "with 0 non-zero UVB entries" not in log  # data/cooling/TREECOOL starts with '##' comment lines
