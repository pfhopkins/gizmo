"""single_stars_formation test: star formation of IMF-sampled single stars (GALSF_SINGLE_STARS_COLLECT_GAS_FROM_NEIGHBORS):
the IMF mass is drawn first, the acceptance probability is (m / M_drawn)(1 - e^-p), and an accepted cell spawns a
zero-mass star that collects its drawn mass. Dense lattice, the density threshold is the only criterion. See README.md."""
import os
import shutil
from pathlib import Path
import numpy as np
import pytest
from gizmo.test import build_gizmo_for_test, run_test

TEST_NAME = "single_stars_formation"
TEST_DIR = Path(__file__).parent


@pytest.fixture(scope="module")
def built():
    build_gizmo_for_test(TEST_NAME)
    return True


def run(tag, ranks=2, **overrides):
    """one run in its own directory; returns the birth log rows"""
    import sys
    sys.path.insert(0, str(TEST_DIR))
    from make_single_stars_formation_ics import make_single_stars_formation_ics
    d = TEST_DIR / f"run_{tag}"
    if d.exists():
        shutil.rmtree(d)  # this run's own directory: no stale logs or snapshots
    d.mkdir()
    os.chdir(d)
    try:
        make_single_stars_formation_ics(f"{TEST_NAME}_ics.hdf5")
        lines = []
        for line in Path(TEST_DIR / f"{TEST_NAME}.params").read_text().splitlines():
            key = line.split()[0] if line.split() else ""
            lines.append(f"{key}    {overrides[key]}" if key in overrides else line)
        Path(f"{TEST_NAME}.params").write_text("\n".join(lines) + "\n")
        os.symlink(TEST_DIR / "GIZMO", "GIZMO")
        run_test(TEST_NAME, num_mpi_ranks=ranks)
        births = Path("output/single_star_births.txt").resolve()
    finally:
        os.chdir("../../../")
    return np.atleast_2d(np.loadtxt(births)) if births.exists() else np.zeros((0, 12))


def first_step(rows):
    t0 = rows[:, 0].min()
    sel = rows[rows[:, 0] == t0]
    return sel[np.lexsort((sel[:, 2], sel[:, 1]))]


def test_first_births_identical_across_ranks(built):
    """the stars of the first star-forming step (IDs, drawn and collected masses, radii, passes) are bitwise the same on
    1, 2 and 4 ranks (later steps differ between rank counts: the hydro step itself is not bitwise rank-independent)"""
    short = dict(TimeMax=0.00025, TimeBetSnapshot=0.00025)
    cols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11]  # all but the count of grants from other ranks
    ref = first_step(run("births_r2", ranks=2, **short))[:, cols]
    assert len(ref) > 20
    for r in (1, 4):
        other = first_step(run(f"births_r{r}", ranks=r, **short))[:, cols]
        assert other.tobytes() == ref.tobytes(), r


def kroupa_number_cdf(x, m_min=0.08, m_max=350.0):
    def N(y):  # cumulative number-weighted Kroupa from m_min (2 m^-1.3 below 0.5, m^-2.3 above)
        y = np.clip(y, m_min, m_max)
        lo = 2 * (np.minimum(y, 0.5) ** -0.3 - m_min ** -0.3) / -0.3
        hi = np.where(y > 0.5, (0.5 ** -1.3 - y ** -1.3) / 1.3, 0.0)
        return lo + hi
    return N(x) / N(m_max)


def test_masses_and_imf(built):
    """10 steps: every star within 1% of its drawn mass; the drawn masses of the formed stars follow the number-weighted
    Kroupa IMF (acceptance probability m_cell / M_drawn, below 1 here)"""
    from scipy import stats
    rows = run("imf")
    assert len(rows) > 300
    assert np.all(rows[:, 8] == 1) and np.all(np.abs(rows[:, 5] / rows[:, 4] - 1) <= 0.01)
    assert stats.kstest(rows[:, 4], kroupa_number_cdf).pvalue > 1e-3
