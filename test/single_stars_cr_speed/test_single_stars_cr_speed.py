"""single_stars_cr_speed: with the CR fluid and single stars the CR transport speed defaults to 300 km/s (3e7 cm/s) instead of the
speed of light. A uniform magnetised box with a CR slab of half-width 2 pc around x = L/2, B along x, large diffusion
coefficient: the CRs free-stream along the field. Two builds: no CR speed flag, and CRFLUID_SPEEDOFLIGHT_REDUCTION=(1.0).

An earlier version counted system steps of the supernova box; the CR timestep did not limit that box, so it could not tell the speeds apart.

Bars (set before the first run of this version): at t = 0.002 pc/(km/s), D = |x - L/2| holding 99 % of the CR energy above the
ambient, D - 2 pc <= 2 v t with v = 300 km/s in the default build (1.2 pc); in the build at the speed of light it exceeds 5 x that (6 pc).
"""
import shutil
import subprocess
from pathlib import Path
import h5py
import numpy as np
from gizmo.test import build_gizmo_for_test, get_stellar_evolution_tables, mpi_launch_command

TEST_NAME = "single_stars_cr_speed"
TEST_DIR = Path(__file__).parent
N, DX = 24, 1.0
L = N * DX
E_AMB, E_SLAB, W = 1e-3, 1.0, 2.0
T_END = 0.002


def make_ics(fname):
    g = (np.arange(N) + 0.5) * DX
    pos = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    n = len(pos)
    u = 1.5 * 1.380649e-16 * 1e4 / (1.27 * 1.6726e-24) / 1e10
    b = np.zeros((n, 3))
    b[:, 0] = 1e-6
    with h5py.File(fname, "w") as F:
        h = F.create_group("Header")
        npart = np.array([n, 0, 0, 0, 0, 0])
        h.attrs["NumPart_ThisFile"] = npart; h.attrs["NumPart_Total"] = npart
        h.attrs["NumPart_Total_HighWord"] = np.zeros(6, dtype=np.uint32); h.attrs["MassTable"] = np.zeros(6)
        h.attrs["Time"] = 0.0; h.attrs["Redshift"] = 0.0; h.attrs["BoxSize"] = L; h.attrs["NumFilesPerSnapshot"] = 1
        h.attrs["Omega0"] = h.attrs["OmegaLambda"] = h.attrs["HubbleParam"] = 0.0
        h.attrs["Flag_Sfr"] = 1; h.attrs["Flag_Cooling"] = 0; h.attrs["Flag_StellarAge"] = 1; h.attrs["Flag_Metals"] = 1
        h.attrs["Flag_Feedback"] = 1; h.attrs["Flag_DoublePrecision"] = 1
        p = F.create_group("PartType0")
        p["Coordinates"] = pos; p["Velocities"] = np.zeros((n, 3)); p["ParticleIDs"] = np.arange(1, n + 1, dtype=np.uint32)
        p["Masses"] = np.ones(n); p["InternalEnergy"] = np.full(n, u); p["Metallicity"] = np.full((n, 1), 0.014)
        p["SmoothingLength"] = np.full(n, 2.0 * DX); p["MagneticField"] = b
        p["CosmicRayEnergy"] = np.where(np.abs(pos[:, 0] - 0.5 * L) < W, E_SLAB, E_AMB)


def run(label, flags, tables):
    build_gizmo_for_test(TEST_NAME, 0, flags)
    d = TEST_DIR / f"run_{label}"
    shutil.rmtree(d, ignore_errors=True)
    (d / "output").mkdir(parents=True)
    make_ics(d / "output" / "snapshot_000.hdf5")
    params = (TEST_DIR / f"{TEST_NAME}.params").read_text().replace("SET_BY_TEST", tables["StellarTablesFile"])
    (d / f"{TEST_NAME}.params").write_text(params)
    with open(d / f"test_{TEST_NAME}.out", "w") as out:
        subprocess.run(mpi_launch_command(2, 1) + [str(TEST_DIR / "GIZMO"), f"{TEST_NAME}.params", "2", "0"], cwd=d, stdout=out,
                       stderr=subprocess.STDOUT, check=True)
    with h5py.File(sorted((d / "output").glob("snapshot_*.hdf5"))[-1], "r") as F:
        x, e = F["PartType0/Coordinates"][()][:, 0], F["PartType0/CosmicRayEnergy"][()]
    dist = np.abs(x - 0.5 * L)
    order = np.argsort(dist)
    excess = np.cumsum(np.clip(e[order] - E_AMB, 0, None))
    return dist[order][np.searchsorted(excess, 0.99 * excess[-1])]


def test_single_stars_cr_speed():
    tables = get_stellar_evolution_tables()
    d_default = run("default", (), tables)
    d_light = run("light", ("CRFLUID_SPEEDOFLIGHT_REDUCTION=(1.0)",), tables)
    print(f"D(99%) default {d_default:.2f} pc, speed of light {d_light:.2f} pc")
    assert d_default - W <= 2 * 300.0 * T_END
    assert d_light - W > 5 * 2 * 300.0 * T_END
