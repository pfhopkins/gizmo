"""SGCHEM_KEEP_IC_ABUNDANCES: the chemistry abundances of a fresh start come from the IC file.

The uniform box of test/sgchem_box (8^3 cells, 8 kyr) with an IC that carries SGChemTracAbund (x(H2), x(H+), x(CO) per H,
different in every cell). Built with the flag, snapshot 0 holds the IC values and the first steps leave them in place;
built without it, snapshot 0 holds the default start (0, 1e-4, 0) whatever the IC says.
"""

import sys
from os import environ, path

import h5py
import shutil
import numpy as np
import pytest

from gizmo.test import build_and_run_test, get_cooling_tables, problem_output_dir

HERE = path.dirname(path.abspath(__file__))
ROOT = path.dirname(path.dirname(HERE))
NAME = "sgchem_keep_ic"
FLAG = "SGCHEM_KEEP_IC_ABUNDANCES"
sys.path.insert(0, path.join(ROOT, "test", "sgchem_box"))


def ic_abundances(ids):
    """x(H2), x(H+), x(CO) of the cell with each ID: H2 0.05-0.35, H+ 0.02, CO 1e-5 - 1.5e-5."""
    k = np.asarray(ids) % 4
    return np.column_stack([0.05 + 0.1 * k, np.full(len(k), 0.02), 1.0e-5 * (1 + 0.5 * (k // 2))])


@pytest.fixture(scope="module")
def runs():
    from test_sgchem_box import chemistry_tables_dir
    from make_sgchem_box_ics import make_ics
    if not environ.get("SUNDIALS_DIR"):
        pytest.skip("set SUNDIALS_DIR (SGCHEM builds against SUNDIALS 6.x or 7.x)")
    get_cooling_tables(HERE)
    ic = path.join(HERE, "sgchem_keep_ic_ics.hdf5")
    make_ics(ic)
    with h5py.File(ic, "r+") as f:
        f["PartType0/SGChemTracAbund"] = ic_abundances(f["PartType0/ParticleIDs"][:])
    tables = chemistry_tables_dir()
    out = {}
    for name, flags in (("default", ()), ("keep", (FLAG,))):
        build_and_run_test(NAME, 2, 1, extra_config_flags=flags, param_overrides={"SGChemTablesPath": tables})
        out[name] = problem_output_dir(NAME, None, flags)
    return out


def read(out, snap):
    with h5py.File(path.join(out, f"snapshot_{snap:03d}.hdf5")) as f:
        o = np.argsort(f["PartType0/ParticleIDs"][:])
        return f["PartType0/ParticleIDs"][:][o], f["PartType0/SGChemTracAbund"][:][o], f["PartType0/SGChemDustTemp"][:][o]


def test_keep_ic_abundances(runs):
    ids, x, tdust = read(runs["keep"], 0)
    assert np.array_equal(x, ic_abundances(ids)), "snapshot 0 holds the IC abundances"
    assert np.all(tdust == 10.0), "the dust temperature is still set at the start"
    ids, x1, _ = read(runs["keep"], 1)
    dev = np.max(np.abs(x1 / ic_abundances(ids) - 1), axis=0)
    print(f"x(H2), x(H+), x(CO) after 4 kyr vs the IC, max relative change: {dev}")
    assert dev[0] < 0.05 and dev[2] < 0.05, "H2 and CO evolve from the IC values (H+ recombines within 4 kyr)"


def test_default_resets_abundances(runs):
    ids, x, _ = read(runs["default"], 0)
    assert np.all(x == [0.0, 1.0e-4, 0.0]), "without the flag the start is the default, whatever the IC holds"


POPIII = (FLAG, "GALSF", "GALSF_SINGLE_STARS_FROM_IMF", "GALSF_SINGLE_STARS_SF_CRITERION", "GALSF_SINGLE_STARS_POPIII")


@pytest.fixture(scope="module")
def popIII_runs():
    """the lattice squeezed to n_H = 1e7 (box 10 -> 0.215 pc, about 7 Msun per cell) at 300 K with the IC H2 fractions kept and a
    slow uniform compression; pristine (InitMetallicity 0) and enriched (1e-3 Zsun); SF at full efficiency over 8 kyr (about half a free-fall time)"""
    from test_sgchem_box import chemistry_tables_dir
    from make_sgchem_box_ics import make_ics
    if not environ.get("SUNDIALS_DIR"):
        pytest.skip("set SUNDIALS_DIR (SGCHEM builds against SUNDIALS 6.x or 7.x)")
    get_cooling_tables(HERE)
    ic = path.join(HERE, "sgchem_keep_ic_ics.hdf5")  # the test's IC name (each fixture writes its own before its runs)
    make_ics(ic)
    shrink = (100.0 / 1.0e7) ** (1.0 / 3.0)
    with h5py.File(ic, "r+") as f:
        L = f["Header"].attrs["BoxSize"] * shrink
        f["Header"].attrs["BoxSize"] = L
        x = f["PartType0/Coordinates"][...] * shrink
        f["PartType0/Coordinates"][...] = x
        f["PartType0/Velocities"][...] = -0.5 * (x - L / 2) / L  # km/s: converging, |v| <= 0.25 km/s
        u300 = 1.5 * 1.380649e-16 * 300.0 / (1.22 * 1.6726e-24) / 1e10
        f["PartType0/InternalEnergy"][...] = u300
        f["PartType0/SGChemTracAbund"] = ic_abundances(f["PartType0/ParticleIDs"][:])
    tables = chemistry_tables_dir()
    common = {"SGChemTablesPath": tables, "BoxSize": L, "CritPhysDensity": 1e3, "PartAllocFactor": 4,
              "SfEffPerFreeFall": 1.0, "FacSfThreshMJ": 1.0, "SFCooldownFreeFallFraction": 0.1, "IMFSampleMinMass": 0.08}
    out = {}
    for name, Z in (("pristine", 0.0), ("enriched", 1e-3)):
        build_and_run_test(NAME, 2, 1, extra_config_flags=POPIII, param_overrides={**common, "InitMetallicity": Z})
        d = problem_output_dir(NAME, None, POPIII)
        dest = path.join(HERE, f"output_popIII_{name}")
        if path.exists(dest):
            shutil.rmtree(dest)
        shutil.copytree(d, dest)
        out[name] = dest
    return out


def new_stars(out):
    import glob
    with h5py.File(sorted(glob.glob(path.join(out, "snapshot_*.hdf5")))[-1]) as f:
        if "PartType4" not in f:
            return {}
        s = f["PartType4"]
        return {k: s[k][()] for k in ("Masses", "SingleStarDrawnMass", "SingleStarPopIII", "SingleStarBirthMetallicity")}


def test_popIII_stars_in_pristine_gas(popIII_runs):
    """bar (set before the run): pristine gas at n_H = 1e7 with H2 forms stars through the Pop III channel: every new star is
    labelled Pop III, drawn from the Pop III IMF (>= 1 Msun), and holds its drawn mass (the whole remainder collected, 1 %);
    the same gas at 1e-3 Zsun forms no Pop III star"""
    p, e = new_stars(popIII_runs["pristine"]), new_stars(popIII_runs["enriched"])
    print(f"pristine: {len(p.get('Masses', []))} stars, drawn {np.sort(p.get('SingleStarDrawnMass', []))}")
    print(f"enriched: {len(e.get('Masses', []))} stars, labels {e.get('SingleStarPopIII', [])}")
    assert len(p.get("Masses", [])) > 0
    assert np.all(p["SingleStarPopIII"] == 1) and np.all(p["SingleStarDrawnMass"] >= 1.0)
    assert np.allclose(p["Masses"], p["SingleStarDrawnMass"], rtol=0.01)
    assert len(e.get("Masses", [])) == 0 or np.all(e["SingleStarPopIII"] == 0)
