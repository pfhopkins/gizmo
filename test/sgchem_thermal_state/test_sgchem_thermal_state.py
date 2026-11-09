"""ThermalProperties under SGChem hands back the stored chemistry state (7.17) and the mean weight of the chemistry's H/He mix (7.18).

The uniform box of test/sgchem_box with molecular ICs. Bars, from the code on stella_dev (cooling.cc ThermalProperties, eos.cc
Get_Gas_Mean_Molecular_Weight_mu): T = stored chemistry temperature; n(H0) = 1 - 2 x(H2) - x(H+) (atomic H only);
n(H+) = n_e = x(H+); mean weight 1 / (X (1 - f_mol/2) + Y/4 + n_e X_universal) with X = 1 / (1 + 4 ABHE), metals not counted.
"""

import re
import sys
from os import environ, path

import h5py
import numpy as np
import pytest

from gizmo.test import build_and_run_test, get_cooling_tables

HERE = path.dirname(path.abspath(__file__))
ROOT = path.dirname(path.dirname(HERE))
NAME = "sgchem_thermal_state"
sys.path.insert(0, path.join(ROOT, "test", "sgchem_box"))
ABHE, X_UNIVERSAL = 0.1, 0.7065


@pytest.fixture(scope="module")
def rows():
    from test_sgchem_box import chemistry_tables_dir
    from make_sgchem_box_ics import make_ics
    if not environ.get("SUNDIALS_DIR"):
        pytest.skip("set SUNDIALS_DIR (SGCHEM builds against SUNDIALS 6.x or 7.x)")
    get_cooling_tables(HERE)
    ic = path.join(HERE, "sgchem_thermal_state_ics.hdf5")
    make_ics(ic)
    with h5py.File(ic, "r+") as f:
        k = f["PartType0/ParticleIDs"][:] % 4
        f["PartType0/SGChemTracAbund"] = np.column_stack([0.05 + 0.1 * k, np.full(len(k), 0.02), np.full(len(k), 1.0e-5)])
    build_and_run_test(NAME, 2, 1, param_overrides={"SGChemTablesPath": chemistry_tables_dir()})
    with open(path.join(HERE, f"test_{NAME}.out")) as f:
        out = np.array([[float(v) for v in m.split()] for m in re.findall(r"^THERMALPROPS (.*)$", f.read(), re.M)])
    assert len(out) > 8, "the hook prints the cells of the run"
    return dict(zip("T mu ne nH0 nHp Tchem xH2 xHp fmol".split(), out.T))


def test_temperature_is_the_stored_one(rows):
    assert np.all(rows["T"] == rows["Tchem"])


def test_species_from_the_stored_abundances(rows):
    assert rows["xH2"].max() > 0.3, "the IC holds molecular gas"
    assert np.allclose(rows["nH0"], np.maximum(0, 1 - 2 * rows["xH2"] - rows["xHp"]), rtol=1e-14)
    assert np.all(rows["nHp"] == rows["xHp"]) and np.all(rows["ne"] == rows["xHp"])


def test_mean_weight_of_the_chemistry_mix(rows):
    x = 1 / (1 + 4 * ABHE)
    mu = 1 / (x * (1 - 0.5 * rows["fmol"]) + (1 - x) / 4 + rows["ne"] * X_UNIVERSAL)
    assert np.allclose(rows["mu"], mu, rtol=1e-14)
