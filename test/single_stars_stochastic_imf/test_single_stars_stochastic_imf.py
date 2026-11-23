"""single_stars_stochastic_imf test (GALSF_SINGLE_STARS_STOCHASTIC_IMF, experimental): the 16^3 lattice of single_stars_sf_criterion
(1 Msun cells, all above CritPhysDensity), standard star formation at efficiency 1e3 converts the cells; each new star particle of mass
m carries one drawn star with probability m / MassPerStarIMF = 1/4, mass from dN/dM ~ M^-2.3 in [8, 120] Msun. No star dies in the
1e-5 code-time run. Bars (written before the run):
  - at least 90 % of the cells become stars;
  - the fraction of stars with a drawn star is 1/4 within 3 sigma of the binomial;
  - every drawn mass in [8, 120]; the drawn masses against the M^-2.3 CDF: KS sqrt(N) D < 1.63 (p > 0.01);
  - stars without a drawn star carry 0; every star with one has its birth metallicity (> 0);
  - total baryon mass and metal mass constant to 1e-12."""
import os
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
import h5py
import pytest
from gizmo.test import build_gizmo_for_test, get_stellar_evolution_tables, mpi_launch_command

TEST_NAME = "single_stars_stochastic_imf"
TEST_DIR = Path(__file__).parent
sys.path.insert(0, str(TEST_DIR.parent / "single_stars_sf_criterion"))
from make_single_stars_sf_criterion_ics import make_single_stars_sf_criterion_ics  # noqa: E402

N_CELLS, P_STAR, MMIN, MMAX = 16 ** 3, 0.25, 8.0, 120.0


@pytest.fixture(scope="module")
def run():
    tables = get_stellar_evolution_tables()
    build_gizmo_for_test(TEST_NAME)
    d = TEST_DIR / "run"
    if d.exists():
        shutil.rmtree(d)
    (d / "output").mkdir(parents=True)
    make_single_stars_sf_criterion_ics(str(d / f"{TEST_NAME}_ics.hdf5"))
    p = (TEST_DIR / f"{TEST_NAME}.params").read_text().replace("SET_BY_TEST", tables["StellarTablesFile"])
    (d / f"{TEST_NAME}.params").write_text(p)
    with open(d / f"test_{TEST_NAME}.out", "w") as out:
        subprocess.run(mpi_launch_command(2) + [str(TEST_DIR / "GIZMO"), f"{TEST_NAME}.params"], cwd=d, stdout=out,
                       stderr=subprocess.STDOUT, env={**os.environ, "OMP_NUM_THREADS": "1"}, check=True)
    return sorted((d / "output").glob("snapshot_*.hdf5"))


def _totals(snap):
    with h5py.File(snap, "r") as F:
        M = Z = 0.0
        for t in ("PartType0", "PartType4"):
            if t in F:
                m = F[t]["Masses"][()]; z = F[t]["Metallicity"][()]; z = z[:, 0] if z.ndim > 1 else z
                M += m.sum(); Z += (m * z).sum()
        return M, Z


def test_stochastic_imf(run):
    first, last = run[0], run[-1]
    with h5py.File(last, "r") as F:
        s = F["PartType4"]
        mdraw = s["SingleStarDrawnMass"][()]
        zb = s["SingleStarBirthMetallicity"][()]
    n = len(mdraw)
    k = int((mdraw > 0).sum())
    sigma = np.sqrt(n * P_STAR * (1 - P_STAR))
    print(f"stars {n}, with a drawn star {k} (expected {n * P_STAR:.1f} +- {sigma:.1f})")
    assert n >= 0.9 * N_CELLS
    assert abs(k - n * P_STAR) <= 3 * sigma
    m = np.sort(mdraw[mdraw > 0])
    assert m.min() >= MMIN and m.max() <= MMAX
    cdf = (MMIN ** -1.3 - m ** -1.3) / (MMIN ** -1.3 - MMAX ** -1.3)
    emp_hi = np.arange(1, len(m) + 1) / len(m)
    emp_lo = np.arange(0, len(m)) / len(m)
    D = max(np.max(emp_hi - cdf), np.max(cdf - emp_lo))
    print(f"KS sqrt(N) D = {np.sqrt(len(m)) * D:.3f}")
    assert np.sqrt(len(m)) * D < 1.63
    assert np.all(zb[mdraw > 0] > 0) and np.all(mdraw[mdraw <= 0] == 0)
    (M0, Z0), (M1, Z1) = _totals(first), _totals(last)
    print(f"mass {M0:.15g} -> {M1:.15g}, metals {Z0:.15g} -> {Z1:.15g}")
    assert abs(M1 / M0 - 1) <= 1e-12 and abs(Z1 / Z0 - 1) <= 1e-12
