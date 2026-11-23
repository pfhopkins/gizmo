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


def run(tag, cutoff, ranks=2, eff=None):
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
        if eff is not None:
            text = text.replace("SfEffPerFreeFall                   1e-12", f"SfEffPerFreeFall                   {eff}")
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


def test_split_above_cutoff_when_x_covers_two_draws(built):
    """bars (written before the run, stella_dev resolvedism_sf_chain): efficiency 1e6, so X = m pfac ~ 1 Msun per cell; a cell whose
    first mass-weighted draw M0 is below X/2 takes the sub-event split, the rest one number-weighted star. At least 3% of the cells
    form two or more stars (one star per event, the old rule, gives 0); the formed masses stay number-weighted Kroupa (0.65 to 0.85
    below 0.5 Msun); at least 85% of the cells form."""
    rows = run("split", "1e3", eff="1e6")
    ids, counts = np.unique(rows[:, 1].astype(np.int64), return_counts=True)
    multi = np.mean(counts >= 2) * len(ids) / N_CELLS
    frac = np.mean(rows[:, 4] < 0.5)
    print(f"cells forming: {len(ids)} of {N_CELLS}; with >= 2 stars: {multi:.3f}; stars {len(rows)}; below 0.5 Msun {frac:.3f}")
    assert multi >= 0.03
    assert 0.65 < frac < 0.85
    assert len(ids) >= 0.85 * N_CELLS
