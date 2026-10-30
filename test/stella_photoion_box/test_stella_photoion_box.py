"""stella_photoion_box: Stella subgrid photoionisation of a uniform box by one fixed-Q star, across MPI ranks.

A Q = 1e49 /s star sits at the centre of a 40 pc box of n_H = 10 cm^-3 gas (1 Msun cells, 4 ranks). Its Stromgren sphere
(~15 pc) spans many tree nodes on every other rank, so one star sends many export entries; with BufferSize = 1 MB that
exceeded what the exchange held while the per-direction histograms rode in each entry (12 kB). Checks:
  - the run reaches TimeMax (no "buffer appears too small to hold a single particle" abort);
  - every front solve spends the photon budget: recombinations in the marked cells / Q within 3% of 1 (the code's ledger);
  - the HII region in the uniform medium is round: largest/smallest per-direction front <= 1.2.
"""

import os
import re
import shutil
import subprocess
import tempfile
from os.path import abspath, dirname, join

import h5py
import numpy as np
import pytest

from gizmo.test import build_gizmo_for_test, mpi_launch_command

TEST_NAME = "stella_photoion_box"
TEST_DIR = dirname(abspath(__file__))
REPO = abspath(join(TEST_DIR, "..", ".."))
OUT = join(TEST_DIR, "output")
NRANKS = 4  # the star's sphere must reach tree nodes on several other ranks

KPC, MSUN, M_P, K_B = 3.085678e21, 1.989e33, 1.6726e-24, 1.3807e-16
BOX_PC, N_H, T_GAS, Z_SUN, M_CELL = 40.0, 10.0, 1.0e4, 0.1, 1.0
LEDGER_TOL, ASYM_MAX = 0.03, 1.2


def make_ics(fname):
    """Uniform lattice, gas only; the cell nearest the centre has ID 1, which GALSF_RESOLVEDISM_ISOLATED_FB_TEST turns into the star."""
    x_z = Z_SUN * 0.014
    x_h = 1.0 - 0.275 - x_z
    m_gas = N_H * (BOX_PC * 3.085678e18) ** 3 * M_P / x_h / MSUN
    ng = int(round((m_gas / M_CELL) ** (1.0 / 3.0)))
    n = ng**3
    box = BOX_PC * 1e-3
    g = (np.arange(ng) + 0.5) * box / ng
    pos = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    ids = np.arange(n, dtype=np.uint32) + 2
    ids[np.argmin(np.sum((pos - 0.5 * box) ** 2, axis=1))] = 1
    ea = np.zeros((n, 27), dtype=np.float32)
    ea[:, 1] = 0.275
    for k, x in {2: 2.53e-3, 3: 7.41e-4, 4: 6.13e-3, 5: 1.34e-3, 6: 7.57e-4, 7: 7.12e-4, 8: 3.31e-4, 9: 6.87e-5, 10: 1.38e-3}.items():
        ea[:, k] = x * Z_SUN
    ea[:, 0] = 1.0 - ea.sum(axis=1)
    u = 1.5 * K_B * T_GAS / (0.6 * M_P) / 1e10  # (km/s)^2, ionised mu above 6e3 K
    with h5py.File(fname, "w") as f:
        h = f.create_group("Header")
        h.attrs["BoxSize"] = box
        h.attrs["NumPart_ThisFile"] = np.array([n, 0, 0, 0, 0, 0], dtype=np.int32)
        h.attrs["NumPart_Total"] = np.array([n, 0, 0, 0, 0, 0], dtype=np.uint32)
        h.attrs["NumPart_Total_HighWord"] = np.zeros(6, dtype=np.uint32)
        h.attrs["MassTable"] = np.zeros(6)
        h.attrs["Time"] = 0.0
        h.attrs["Redshift"] = 0.0
        h.attrs["NumFilesPerSnapshot"] = 1
        h.attrs["Flag_Metals"] = 27
        h.attrs["Flag_DoublePrecision"] = 0
        for k in ("Flag_Sfr", "Flag_Cooling", "Flag_StellarAge", "Flag_Feedback"):
            h.attrs[k] = 1
        p = f.create_group("PartType0")
        p["Coordinates"] = pos
        p["Velocities"] = np.zeros((n, 3), dtype=np.float32)
        p["Masses"] = np.full(n, m_gas / n * MSUN / 1.989e43, dtype=np.float32)
        p["ParticleIDs"] = ids
        p["InternalEnergy"] = np.full(n, u, dtype=np.float32)
        p["ElementAbundance"] = ea
        p["MagneticField"] = np.zeros((n, 3), dtype=np.float32)


def test_stella_photoion_box():
    if os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        pytest.skip("builds GIZMO; nothing to validate from external snapshots")
    cwd = os.getcwd()
    os.chdir(REPO)
    try:
        build_gizmo_for_test(TEST_NAME)
    finally:
        os.chdir(cwd)
    # outside the test tree: the gizmo/ link to the repo root would otherwise make pytest collect the whole suite
    run = tempfile.mkdtemp(prefix=f"{TEST_NAME}_")
    os.makedirs(join(run, "output"))
    os.symlink(REPO, join(run, "gizmo"))  # SGCHEM and the stellar tables open their data under gizmo/
    make_ics(join(run, f"{TEST_NAME}_ics.hdf5"))
    shutil.copy(join(TEST_DIR, f"{TEST_NAME}.params"), run)
    with open(join(run, "run.log"), "w") as log:
        r = subprocess.run(mpi_launch_command(NRANKS) + [join(TEST_DIR, "GIZMO"), f"{TEST_NAME}.params"],
                           stdout=log, stderr=subprocess.STDOUT, cwd=run)
    os.makedirs(OUT, exist_ok=True)
    shutil.copy(join(run, "run.log"), join(OUT, "run.log"))
    text = open(join(run, "run.log")).read()
    assert "too small to hold a single particle" not in text, f"exchange buffer overflow, see {OUT}/run.log"
    assert r.returncode == 0, f"GIZMO exited {r.returncode}, see {OUT}/run.log"
    snaps = sorted(s for s in os.listdir(join(run, "output")) if s.startswith("snapshot_"))
    assert snaps, f"no snapshots written, see {OUT}/run.log"
    with h5py.File(join(run, "output", snaps[-1]), "r") as f:
        assert abs(f["Header"].attrs["Time"] / 2.5e-5 - 1) < 1e-6, "run stopped before TimeMax"
    ratio = np.array([float(x) for x in re.findall(r"RESOLVEDISM PI LEDGER: .* ratio=([0-9.eE+-]+)", text)])
    asym = np.array([float(x) for x in re.findall(r"max asymmetry=([0-9.]+):1", text)])
    n_ion = [int(x) for x in re.findall(r"RESOLVEDISM PI: (\d+) ionized", text)]
    print(f"{len(ratio)} front solves: ledger ratio {ratio.min():.4f}..{ratio.max():.4f}, "
          f"max asymmetry {asym.max():.2f}, ionised cells {max(n_ion)}")
    assert len(ratio) >= 10 and max(n_ion) > 0, "no ionised region"
    assert np.all(np.abs(ratio - 1) <= LEDGER_TOL), f"photon budget not spent: ledger ratio {ratio.min():.4f}..{ratio.max():.4f}"
    assert asym.max() <= ASYM_MAX, f"HII region not round in a uniform medium: asymmetry {asym.max():.2f}"
    shutil.rmtree(run, ignore_errors=True)
