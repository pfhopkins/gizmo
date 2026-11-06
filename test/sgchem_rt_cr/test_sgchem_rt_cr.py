"""sgchem_rt_cr: the moment solver and the cosmic-ray fluid together with SGChem (needs the cr series). A uniform box from a
fresh start at x(H+) = 0.02, ambient cosmic rays, no stars. After 10 steps: electrons at the network's value within a factor 2,
no gas speed above 1 km/s, gas + cosmic-ray + bin energy within 10 % of the start."""
import sys
from os import environ, path
import glob
import h5py
import numpy as np
import pytest
from gizmo.test import build_and_run_test, get_cooling_tables, get_stellar_evolution_tables, problem_output_dir

HERE = path.dirname(path.abspath(__file__))
ROOT = path.dirname(path.dirname(HERE))
NAME = "sgchem_rt_cr"
X_HP = 0.02
sys.path.insert(0, path.join(ROOT, "test", "sgchem_box"))


def energy(f):
    g = f["PartType0"]; m = g["Masses"][()]
    return (m * g["InternalEnergy"][()]).sum(), (0.5 * m * (g["Velocities"][()] ** 2).sum(1)).sum(), g["PhotonEnergy"][()].sum(), g["CosmicRayEnergy"][()].sum()


def test_m1_and_cr_fresh_start():
    from test_sgchem_box import chemistry_tables_dir
    from make_sgchem_box_ics import make_ics
    if not environ.get("SUNDIALS_DIR"):
        pytest.skip("set SUNDIALS_DIR (SGCHEM builds against SUNDIALS 6.x or 7.x)")
    get_cooling_tables(HERE)
    t = get_stellar_evolution_tables()
    ic = path.join(HERE, "sgchem_rt_cr_ics.hdf5")
    make_ics(ic)
    with h5py.File(ic, "r+") as f:
        n = len(f["PartType0/ParticleIDs"])
        f["PartType0/SGChemTracAbund"] = np.column_stack([np.zeros(n), np.full(n, X_HP), np.zeros(n)])
    build_and_run_test(NAME, 2, 1, param_overrides={"SGChemTablesPath": chemistry_tables_dir(), "StellarTablesFile": t["StellarTablesFile"]})
    out = problem_output_dir(NAME, None, ())
    s = sorted(glob.glob(path.join(out, "snapshot_*.hdf5")))
    with h5py.File(s[0]) as a, h5py.File(s[-1]) as b:
        ea, eb = energy(a), energy(b)
        ne0, ne1 = a["PartType0/ElectronAbundance"][()], b["PartType0/ElectronAbundance"][()]
        v = np.sqrt((b["PartType0/Velocities"][()] ** 2).sum(1)).max()
    print("energies start", ea, "end", eb, "Ne", ne0.min(), ne1.min(), "vmax", v)
    assert np.all(ne0 > 0.5 * ne1) and np.all(ne0 < 2 * ne1)
    assert v < 1.0
    assert abs(sum(eb) / sum(ea) - 1) < 0.1
