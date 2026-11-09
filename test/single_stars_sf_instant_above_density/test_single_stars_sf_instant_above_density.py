"""single_stars_sf_instant_above_density test: with GALSF_SINGLE_STARS_SF_INSTANT_ABOVE_DENSITY every cell above nHcutoffSF
forms a single star at once, whatever the Jeans test and the rate say, with the mass drawn from the number-weighted Kroupa
IMF. Lattice of single_stars_sf_criterion (n_H ~ 7e3 cm^-3, efficiency 1e-12). See README.md."""
import os
import shutil
import sys
from pathlib import Path
import numpy as np
import pytest
from gizmo.test import build_gizmo_for_test, run_test

TEST_NAME = "single_stars_sf_instant_above_density"
TEST_DIR = Path(__file__).parent
N_CELLS = 16 ** 3


@pytest.fixture(scope="module")
def built():
    build_gizmo_for_test(TEST_NAME)
    return True


def run(tag, cutoff, ranks=2):
    """births (rows of single_star_births.txt) of the one step"""
    sys.path.insert(0, str(TEST_DIR.parent / "single_stars_sf_criterion"))
    from make_single_stars_sf_criterion_ics import make_single_stars_sf_criterion_ics
    d = TEST_DIR / f"run_{tag}"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir()
    os.chdir(d)
    try:
        make_single_stars_sf_criterion_ics(f"{TEST_NAME}_ics.hdf5")
        text = (TEST_DIR / f"{TEST_NAME}.params").read_text().replace("nHcutoffSF                         1e3", f"nHcutoffSF                         {cutoff}")
        Path(f"{TEST_NAME}.params").write_text(text)
        os.symlink(TEST_DIR / "GIZMO", "GIZMO")
        run_test(TEST_NAME, num_mpi_ranks=ranks)
        births = Path("output/single_star_births.txt").resolve()
        return np.atleast_2d(np.loadtxt(births)) if births.exists() else np.zeros((0, 12))
    finally:
        os.chdir("../../../")


def test_every_cell_forms_above_cutoff(built):
    """bar (written before the run): cutoff below the lattice density: at least 95% of the cells form; the drawn masses are
    number-weighted Kroupa (76% below 0.5 Msun, bar 0.65 to 0.85; a mass-weighted draw would give 26%)"""
    rows = run("below", "1e3")
    assert len(rows) >= 0.95 * N_CELLS
    frac = np.mean(rows[:, 4] < 0.5)
    assert 0.65 < frac < 0.85


def test_none_below_cutoff(built):
    """cutoff above the lattice density: no star (the efficiency is 1e-12)"""
    assert len(run("above", "1e9")) == 0
