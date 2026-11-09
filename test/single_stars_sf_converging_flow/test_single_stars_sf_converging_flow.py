"""single_stars_sf_converging_flow test: with GALSF_SINGLE_STARS_SF_CONVERGING_FLOW_ONLY a cell forms stars only where the
single-star Jeans criterion passes and the velocity divergence is negative. Same lattice as single_stars_sf_criterion
(random velocities, so both signs of div v occur). See README.md."""
import os
import shutil
import sys
from pathlib import Path
import numpy as np
import h5py
import pytest
from gizmo.test import build_gizmo_for_test, run_test

TEST_NAME = "single_stars_sf_converging_flow"
TEST_DIR = Path(__file__).parent
G_CODE = 6.6743e-8 * 1.989e33 * (3.085678e18 / 1e5) ** 2 / 3.085678e18 ** 3  # pc, Msun, km/s


@pytest.fixture(scope="module")
def built():
    build_gizmo_for_test(TEST_NAME)
    return True


def run(tag, ranks=2):
    sys.path.insert(0, str(TEST_DIR.parent / "single_stars_sf_criterion"))
    from make_single_stars_sf_criterion_ics import make_single_stars_sf_criterion_ics
    d = TEST_DIR / f"run_{tag}"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir()
    os.chdir(d)
    try:
        make_single_stars_sf_criterion_ics(f"{TEST_NAME}_ics.hdf5")
        Path(f"{TEST_NAME}.params").write_text((TEST_DIR / f"{TEST_NAME}.params").read_text())
        os.symlink(TEST_DIR / "GIZMO", "GIZMO")
        run_test(TEST_NAME, num_mpi_ranks=ranks)
        return sorted(Path("output").resolve().glob("snapshot_*.hdf5"))[-1]
    finally:
        os.chdir("../../../")


def test_forming_cells_converge(built):
    """bar (written before the run): every cell with a star-formation rate has div v < 0; at least 10% of the cells form;
    the forming set equals {Jeans passes and div v < 0} except cells within 1e-3 of the Jeans limit"""
    with h5py.File(run("decide"), "r") as F:
        g = F["PartType0"]
        rho, u, m, sfr, divv = g["Density"][()], g["InternalEnergy"][()], g["Masses"][()], g["StarFormationRate"][()], g["VelocityDivergence"][()]
    forms = sfr > 0
    cs = np.sqrt(5 / 3 * 2 / 3 * u)
    ratio = 2.9155697 * cs ** 3 / (G_CODE ** 1.5 * np.sqrt(rho)) / (1.0 * 32 * m)
    assert np.mean(divv < 0) > 0.2 and np.mean(divv > 0) > 0.2  # both signs are present
    assert forms.mean() > 0.1
    assert not np.any(divv[forms] >= 0)
    mismatch = forms != ((ratio < 1) & (divv < 0))
    assert np.all(np.abs(ratio[mismatch] - 1) < 1e-3)
