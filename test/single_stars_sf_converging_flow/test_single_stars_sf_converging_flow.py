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


def run(tag, ranks=2, post_ic=None, **overrides):
    sys.path.insert(0, str(TEST_DIR.parent / "single_stars_sf_criterion"))
    from make_single_stars_sf_criterion_ics import make_single_stars_sf_criterion_ics
    d = TEST_DIR / f"run_{tag}"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir()
    os.chdir(d)
    try:
        make_single_stars_sf_criterion_ics(f"{TEST_NAME}_ics.hdf5")
        if post_ic:
            post_ic(f"{TEST_NAME}_ics.hdf5")
        lines = [l for l in (TEST_DIR / f"{TEST_NAME}.params").read_text().splitlines() if not (l.split() and l.split()[0] in overrides)]
        Path(f"{TEST_NAME}.params").write_text("\n".join(lines + [f"{k}    {v}" for k, v in overrides.items()]) + "\n")
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


def compressed_at(a, eps):
    """the lattice at scale factor a (h = 1, comoving positions), no random motions, a uniform compression
    Vel_IC = -eps (x - centre): physical peculiar div v = -3 eps a^-1/2 (GIZMO: Vel = a^3/2 Vel_IC)"""
    def post(f):
        with h5py.File(f, "r+") as F:
            L = F["Header"].attrs["BoxSize"] / a
            x = F["PartType0/Coordinates"][...] / a
            F["Header"].attrs["Time"] = a; F["Header"].attrs["BoxSize"] = L
            F["PartType0/Coordinates"][...] = x
            F["PartType0/Velocities"][...] = -eps * (x - L / 2)
    return post


@pytest.mark.parametrize("hubble_fraction", [0.3, 3.0])
def test_hubble_flow_counts(built, hubble_fraction):
    """comoving run at a = 0.5, compression of the given fraction of the Hubble rate: at 0.3 the peculiar flow converges but
    the physical flow (Hubble flow included) expands, no cell may form; at 3 it converges, the Jeans-unstable cells form.
    Cells away from the periodic seam (where the compression wraps) only"""
    a = 0.5
    G, H0, box = 6.672e-8 * 1.989e33 / (3.085678e18 * 1e5 * 1e5), (1.0 * 3.2407789e-18) * (3.085678e18 / 1e5), 2.56 / a
    omega = 16 ** 3 * 1.0 / (box * box * box) / (3 * H0 * H0 / (8 * np.pi * G))  # the box holds Omega_M (check_omega, its constants)
    H = H0 * np.sqrt(omega / a ** 3)  # H(a), no Lambda
    snap = run(f"hubble_{hubble_fraction}", post_ic=compressed_at(a, hubble_fraction * H * a ** 0.5), ComovingIntegrationOn=1,
               TimeBegin=a, TimeMax=a * (1 + 1e-6), TimeOfFirstSnapshot=a, TimeBetSnapshot=1.000001, TimeBetStatistics=1.000001,
               MaxSizeTimestep=1e-6, Omega0=omega, OmegaLambda=0, OmegaBaryon=1e-6 * omega, HubbleParam=1.0, BoxSize=box,
               **{f"Softening{t}MaxPhys": 0.01 for t in ("Gas", "Halo", "Disk", "Bulge", "Stars", "Bndry")})
    # Omega_b sets the comoving overdensity threshold (1000 x the mean baryon density): small, so the cells clear it
    with h5py.File(snap, "r") as F:
        g = F["PartType0"]
        rho, u, m, sfr, x = g["Density"][()] / a ** 3, g["InternalEnergy"][()], g["Masses"][()], g["StarFormationRate"][()], g["Coordinates"][()]
    keep = np.all(np.abs(x - box / 2) < box / 4, axis=1)
    forms = (sfr > 0)[keep]
    ratio = (2.9155697 * np.sqrt(5 / 3 * 2 / 3 * u) ** 3 / (G_CODE ** 1.5 * np.sqrt(rho)) / (1.0 * 32 * m))[keep]
    assert np.mean(ratio < 0.9) > 0.2  # the Jeans test alone would let these form
    if hubble_fraction < 1:
        assert not np.any(forms), forms.sum()
    else:
        sure = np.abs(ratio - 1) > 1e-3
        assert np.all(forms[sure] == (ratio[sure] < 1))
