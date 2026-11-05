"""treecol_shells_helium: helium in the TreeRay photoionisation (net 17). Sources and absorbers on the tree, from the first
gravity computation. See README.md."""
import os
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
from gizmo.test import build_gizmo_for_test, get_stellar_evolution_tables, get_cooling_tables, mpi_launch_command

TEST_NAME = "treecol_shells_helium"
TEST_DIR = Path(__file__).parent
PIX_DIR = TEST_DIR.parent / "treecol_pixels"
sys.path.insert(0, str(PIX_DIR))
from make_treecol_pixels_ics import make_treecol_pixels_ics, lattice  # noqa: E402
from stella_table_bands import Tables  # noqa: E402

EV = 1.60217733e-12
L_BOX, SHIELD = 20.0, 3.0
STARS = [(8.0, 0.5), (15.0, 0.6), (30.0, 0.2), (60.0, 0.5), (100.0, 0.2)]  # (M, fraction of the main sequence)
PIXELS = ("GALSF_RESOLVEDISM_TREECOL_PIXELS=12", "TEST_DUMP_TREECOL_PIXELS_AND_EXIT", "BOX_PERIODIC", "GRAVITY_NOT_PERIODIC")


@pytest.fixture(scope="module")
def dump():
    if not os.environ.get("SUNDIALS_DIR"):
        pytest.skip("set SUNDIALS_DIR (SGCHEM builds against SUNDIALS 6.x or 7.x)")
    tables = get_stellar_evolution_tables()
    get_cooling_tables(str(TEST_DIR))
    build_gizmo_for_test(TEST_NAME, extra_config_flags=PIXELS)
    exe = TEST_DIR / "GIZMO_dump"
    shutil.move(TEST_DIR / "GIZMO", exe)
    tab = Tables(tables["StellarTablesFile"])
    rng = np.random.default_rng(13)
    n = 12; pos = lattice(n, L_BOX) + rng.uniform(-0.2, 0.2, (n ** 3, 3)) * (L_BOX / n)
    stars = []
    for M, f in STARS:
        lM, lZ = np.log10(M), np.log10(0.0142)
        stars.append((M, tab._i2(tab.t_pms, lM, lZ) + f * tab._i2(tab.life, lM, lZ), rng.uniform(0.15, 0.85, 3) * L_BOX))
    ab = np.zeros((n ** 3, 7)); ab[:, 0] = rng.uniform(0, 0.1, n ** 3); ab[:, 1] = rng.uniform(0, 0.9, n ** 3)
    ab[:, 3] = rng.uniform(0, 0.08, n ** 3); ab[:, 4] = rng.uniform(0, 0.02, n ** 3)
    d = TEST_DIR / "run_dump"
    if d.exists():
        shutil.rmtree(d)
    (d / "output").mkdir(parents=True)
    make_treecol_pixels_ics(str(d / "output" / "snapshot_000.hdf5"), L_BOX, pos, stars, tracabund=ab)
    for f in ("TREECOOL", "spcool_tables"):
        if (TEST_DIR / f).exists():
            os.symlink(TEST_DIR / f, d / f)
    p = (PIX_DIR / "treecol_pixels.params").read_text()
    p = p.replace("StellarTablesFile           SET_BY_TEST", f"StellarTablesFile           {tables['StellarTablesFile']}")
    tabdir = os.environ.get("SGCHEM_TABLES_DIR", os.path.dirname(tables["StellarTablesFile"]))
    p = p.replace("SGChemTablesPath            SET_BY_TEST", f"SGChemTablesPath            {tabdir}")
    over = {"BoxSize": L_BOX, "TreeColShieldingLength": SHIELD}
    p = "\n".join([l for l in p.splitlines() if not (l.split() and l.split()[0] in over)] + [f"{k} {v}" for k, v in over.items()]) + "\n"
    (d / "treecol_pixels.params").write_text(p)
    with open(d / "run.out", "w") as out:
        subprocess.run(mpi_launch_command(1, 1) + [str(exe), "treecol_pixels.params", "2", "0"], cwd=d, stdout=out,
                       stderr=subprocess.STDOUT, env={**os.environ, "OMP_NUM_THREADS": "1"}, check=True)
    assert "TREECOL DUMP written" in (d / "run.out").read_text()
    return d / "output", tab, stars, ab


def load(out, what):
    r = np.vstack([np.atleast_2d(np.loadtxt(f)) for f in sorted(out.glob(f"treecol_dump_{what}_*.txt")) if f.stat().st_size > 0])
    return r[np.argsort(r[:, 0])]


def test_two_source_bands(dump):
    """two bands from the tables split at 24.6 eV: photons and energy add up to the star's, photon rates and energy in band 1 as the tables"""
    out, tab, stars, _ = dump
    bins = np.loadtxt(out / "treecol_dump_bins_0.txt").astype(int)
    assert len(bins) == 9
    rows = load(out, "stars")[:, 2:]
    L0, Q0, L1, Q1 = (rows[:, bins[k]] for k in (5, 6, 7, 8))
    Q0, Q1 = Q0 / (13.6 * EV), Q1 / (13.6 * EV)
    want = np.array([tab.ionising(M, 0.0142, age) for M, age, _ in stars])
    np.testing.assert_allclose(L0 + L1, want[:, 0], rtol=1e-10)
    np.testing.assert_allclose(Q0 + Q1, want[:, 1], rtol=1e-10)
    he = np.array([tab.ionising_above_he(M, 0.0142, age) for M, age, _ in stars])
    np.testing.assert_allclose(L1, he[:, 0], rtol=1e-10)
    np.testing.assert_allclose(Q1, he[:, 1], rtol=1e-10)
    assert np.all(L1 / Q1 / EV > 24.6 * (1 - 1e-9)) and np.all(L1 / Q1 / EV < 54.4 * (1 + 1e-9))
    root = np.loadtxt(out / "treecol_dump_root_lum_0.txt")
    np.testing.assert_allclose(root[bins[5:]], rows[:, bins[5:]].sum(0), rtol=1e-12)


def test_helium_absorbers(dump):
    """node species: HI = (1 - x_H+) per H, HeI = ABHE - x_He+ - x_He++ per H, cell by cell and in the root node"""
    out, _, _, ab = dump
    cells = load(out, "cells")
    hi = (1 - ab[:, 1]) * 0.7065
    hei = (0.1 - ab[:, 3] - ab[:, 4]) * 0.7065
    np.testing.assert_allclose(cells[:, 4], hi, rtol=1e-12)
    np.testing.assert_allclose(cells[:, 5], hei, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.loadtxt(out / "treecol_dump_root_species_0.txt")[3:5], [hi.sum(), hei.sum()], rtol=1e-12)
