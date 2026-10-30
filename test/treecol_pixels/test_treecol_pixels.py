"""treecol_pixels: per-direction (HEALPix) columns and stellar band fluxes accumulated in the gravity walk
(GALSF_RESOLVEDISM_TREECOL_PIXELS...). Each test builds its configuration, runs from the initial conditions until the
first gravity computation is done (TEST_DUMP_TREECOL_PIXELS_AND_EXIT writes the tree radiation inputs and results and
stops) and checks them against exact values.
  test_star_band_luminosities: single stars of 1-100 Msun before, on and after the main sequence and dead shine in the
    disjoint table bands (PE 8-13.6, LW 11.2-13.6, 6-8, NUV 3.444-6, OPT eV) at their current age: each band equals the
    table interpolation to 1e-10, OPT + NUV + 6-8 + PE = OPT + NUV(3.444-8) + PE(8-13.6) of the tables, LW <= PE, and
    the root node of the tree holds their sum (1 and 2 ranks).
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
from gizmo.test import build_gizmo_for_test, get_stellar_evolution_tables, get_cooling_tables, mpi_launch_command

TEST_NAME = "treecol_pixels"
TEST_DIR = Path(__file__).parent
sys.path.insert(0, str(TEST_DIR))
from make_treecol_pixels_ics import make_treecol_pixels_ics, lattice  # noqa: E402
from stella_table_bands import Tables  # noqa: E402

BUILDS = {}


def chemistry_tables_dir(stellar_tables_file):
    return os.environ.get("SGCHEM_TABLES_DIR", os.path.dirname(stellar_tables_file))


@pytest.fixture(scope="module")
def tables():
    if not os.environ.get("SUNDIALS_DIR"):
        pytest.skip("set SUNDIALS_DIR (SGCHEM builds against SUNDIALS 6.x or 7.x)")
    t = get_stellar_evolution_tables()
    get_cooling_tables(str(TEST_DIR))
    return t


def build(tag, extra=()):
    """the test's Config.sh plus extra flags, as TEST_DIR/GIZMO_<tag>"""
    if tag not in BUILDS:
        build_gizmo_for_test(TEST_NAME, extra_config_flags=tuple(extra))
        shutil.move(TEST_DIR / "GIZMO", TEST_DIR / f"GIZMO_{tag}")
        BUILDS[tag] = TEST_DIR / f"GIZMO_{tag}"
    return BUILDS[tag]


def run(exe, tag, tables, L, gas_pos, stars, ranks=1, overrides=None, **ic):
    d = TEST_DIR / f"run_{tag}"
    if d.exists():
        shutil.rmtree(d)  # this run's own directory
    (d / "output").mkdir(parents=True)
    make_treecol_pixels_ics(str(d / "output" / "snapshot_000.hdf5"), L, gas_pos, stars, **ic)
    for f in ("TREECOOL", "spcool_tables"):
        if (TEST_DIR / f).exists():
            os.symlink(TEST_DIR / f, d / f)
    params = (TEST_DIR / f"{TEST_NAME}.params").read_text()
    params = params.replace("StellarTablesFile           SET_BY_TEST", f"StellarTablesFile           {tables['StellarTablesFile']}")
    params = params.replace("SGChemTablesPath            SET_BY_TEST", f"SGChemTablesPath            {chemistry_tables_dir(tables['StellarTablesFile'])}")
    over = {"BoxSize": L, **(overrides or {})}
    lines = [l for l in params.splitlines() if not (l.split() and l.split()[0] in over)]
    params = "\n".join(lines + [f"{k:27s} {v}" for k, v in over.items()]) + "\n"
    (d / f"{TEST_NAME}.params").write_text(params)
    with open(d / f"test_{TEST_NAME}.out", "w") as out:
        subprocess.run(mpi_launch_command(ranks) + [str(exe), f"{TEST_NAME}.params", "2", "0"], cwd=d, stdout=out,
                       stderr=subprocess.STDOUT, env={**os.environ, "OMP_NUM_THREADS": "1"}, check=True)
    assert "TREECOL DUMP written" in (d / f"test_{TEST_NAME}.out").read_text()
    return d / "output"


def load(out, what):
    rows = [np.atleast_2d(np.loadtxt(f)) for f in sorted(out.glob(f"treecol_dump_{what}_*.txt")) if f.stat().st_size > 0]
    return np.vstack(rows) if rows else np.zeros((0, 0))


STAR_SPECS = [(1.0, 0.5), (3.0, 0.3), (8.0, 0.01), (8.0, 0.97), (15.0, 0.6), (25.0, 1e-4), (40.0, 0.9), (60.0, 0.5),
              (100.0, 0.2), (20.0, -0.5), (30.0, 1.05)]  # (M, fraction of the main-sequence life; < 0: before it, > 1: dead)


def star_list(tab, L):
    rng = np.random.default_rng(4)
    stars = []
    for M, f in STAR_SPECS:
        lM, lZ = np.log10(M), np.log10(0.0142)
        t_pms, life = tab._i2(tab.t_pms, lM, lZ), tab._i2(tab.life, lM, lZ)
        age = t_pms * (1 + f) if f < 0 else t_pms + f * life
        stars.append((M, age, rng.uniform(0.2, 0.8, 3) * L))
    return stars


@pytest.mark.parametrize("ranks", [1, 2])
def test_star_band_luminosities(tables, ranks):
    tab = Tables(tables["StellarTablesFile"])
    L = 20.0
    stars = star_list(tab, L)
    out = run(build("bands"), f"bands_np{ranks}", tables, L, lattice(12, L), stars, ranks=ranks)
    bins = np.loadtxt(out / "treecol_dump_bins_0.txt").astype(int)
    rows = load(out, "stars")
    assert len(rows) == len(stars)
    N_gas = 12 ** 3
    shone = 0
    for row in rows:
        M, age, _ = stars[int(row[0]) - N_gas - 1]
        age += row[1] * 3.085678e13 / 3.155e7
        got, want = row[2:][bins], tab.bands(M, 0.0142, age)
        assert np.allclose(got, want, rtol=1e-10, atol=0), (M, age, got, want)
        if want.any():
            shone += 1
            assert got[1] <= got[0]
        np.testing.assert_allclose(got[4] + got[3] + got[2] + got[0], want.sum() - want[1], rtol=1e-12)
    assert shone == len(stars) - 2  # the pre-main-sequence and the dead star are dark
    root = np.loadtxt(out / "treecol_dump_root_0.txt")
    np.testing.assert_allclose(root[bins], rows[:, 2:][:, bins].sum(0), rtol=1e-12)
