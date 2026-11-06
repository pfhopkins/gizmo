"""mhd_cleaning_off_options test: TEST_MHD_NO_DEDNER_CLEANING removes the Dedner divergence cleaning and
TEST_MHD_NO_POWELL_SOURCE_TERMS the Powell terms proportional to div B (in DtB, the acceleration and the energy). Orszag-Tang
vortex on a 32^2 lattice, 0.2 time units, three builds. See README.md."""
import glob
import h5py
import numpy as np
import pytest
from gizmo.test import build_and_run_test, clean_test_outputs, variant_output_dir

TEST_NAME = "mhd_cleaning_off_options"
IC_FILE = f"test/{TEST_NAME}/{TEST_NAME}_ics.hdf5"
N = 32
BUILDS = {"default": (), "no_dedner": ("TEST_MHD_NO_DEDNER_CLEANING",), "no_powell": ("TEST_MHD_NO_POWELL_SOURCE_TERMS",)}


def write_ics():
    rng = np.random.default_rng(3)
    g = (np.arange(N) + 0.5) / N
    X, Y = np.meshgrid(g, g, indexing="ij")
    xy = np.stack([X.ravel(), Y.ravel()], 1) + rng.uniform(-0.1, 0.1, (N * N, 2)) / N
    pos = np.zeros((N * N, 3)); pos[:, :2] = xy % 1.0
    rho0, p0, B0 = 25.0 / (36 * np.pi), 5.0 / (12 * np.pi), 1.0 / np.sqrt(4 * np.pi)
    vel = np.zeros((N * N, 3)); vel[:, 0] = -np.sin(2 * np.pi * pos[:, 1]); vel[:, 1] = np.sin(2 * np.pi * pos[:, 0])
    B = np.zeros((N * N, 3)); B[:, 0] = -B0 * np.sin(2 * np.pi * pos[:, 1]); B[:, 1] = B0 * np.sin(4 * np.pi * pos[:, 0])
    with h5py.File(IC_FILE, "w") as F:
        h = F.create_group("Header")
        h.attrs["NumPart_ThisFile"] = [N * N, 0, 0, 0, 0, 0]; h.attrs["NumPart_Total"] = [N * N, 0, 0, 0, 0, 0]
        h.attrs["NumPart_Total_HighWord"] = [0] * 6; h.attrs["MassTable"] = [0] * 6
        h.attrs["BoxSize"] = 1.0; h.attrs["Time"] = 0.0; h.attrs["Redshift"] = 0.0; h.attrs["NumFilesPerSnapshot"] = 1
        p = F.create_group("PartType0")
        p["Coordinates"] = pos; p["Velocities"] = vel; p["MagneticField"] = B
        p["Masses"] = np.full(N * N, rho0 / (N * N)); p["InternalEnergy"] = np.full(N * N, p0 / ((5.0 / 3.0 - 1.0) * rho0))
        p["ParticleIDs"] = np.arange(1, N * N + 1, dtype=np.uint32)


def run(tag):
    flags = BUILDS[tag]
    clean_test_outputs(TEST_NAME, flags)
    build_and_run_test(TEST_NAME, 2, 0, flags)
    snap = sorted(glob.glob(variant_output_dir(TEST_NAME, flags) + "/snapshot_*.hdf5"))[-1]
    with h5py.File(snap, "r") as F:
        g = F["PartType0"]; o = np.argsort(g["ParticleIDs"][()])
        assert float(F["Header"].attrs["Time"]) >= 0.2 - 1e-9
        return {k: g[k][()][o] for k in ("MagneticField", "Velocities", "DivBcleaningFunctionPhi", "DivergenceOfMagneticField")}


@pytest.fixture(scope="module")
def results():
    write_ics()
    return {tag: run(tag) for tag in BUILDS}


def test_dedner_cleaning_off(results):
    """bar (written before the run): the cleaning field is nonzero in the default run (max |phi| > 1e-6) and exactly zero
    in every cell without the cleaning"""
    assert np.abs(results["default"]["DivBcleaningFunctionPhi"]).max() > 1e-6
    assert np.all(results["no_dedner"]["DivBcleaningFunctionPhi"] == 0)


def test_powell_terms_off(results):
    """bar (written before the run): without the Powell terms the evolved magnetic field and velocity differ from the
    default run (max relative difference of B above 1e-6), while the cleaning field stays nonzero"""
    d, p = results["default"], results["no_powell"]
    dB = np.abs(p["MagneticField"] - d["MagneticField"]).max() / np.abs(d["MagneticField"]).max()
    print("no_powell: max |dB| / max |B| =", dB)
    assert dB > 1e-6 and np.abs(p["DivBcleaningFunctionPhi"]).max() > 1e-6
