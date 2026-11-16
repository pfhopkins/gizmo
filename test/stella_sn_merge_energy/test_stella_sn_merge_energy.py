"""stella_sn_merge_energy: adiabatic Stella supernova at 1 Msun with merge/split on, gas energy against injected energy.

The Stella default build plus the isolated-feedback test hooks, without the cooling/chemistry update
(RESOLVEDISM_ADIABATIC_NOCOOL). A 12 Msun star (cell ID 1) at the centre of a uniform 30 pc periodic box (n_H = 10,
1e4 K, 1 Msun cells) explodes at TestSNTime0 = 1e-6 (~1 kyr); the run ends at 39 kyr, before the blast wraps around.
Stellar radiation is off (TestFUVZero), so the SN and the star's winds are the only sources.

Check: the gas thermal + kinetic energy at the last full synchronisation (ENERGY_BUDGET_DIAGNOSTIC), minus its
initial value, equals the energy the feedback coupled to the gas (ENERGYinfo.txt column 12) within 2%.
Merge and split counts are printed: an isolated Stella SN drains no cell below the merge threshold, so this run
has no merges; the merge energy itself is checked cell by cell in merge_split_conservation_stella_default.
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
from gizmo.conservation_trace import keep_and_check_traces

TEST_NAME = "stella_sn_merge_energy"
TEST_DIR = dirname(abspath(__file__))
REPO = abspath(join(TEST_DIR, "..", ".."))
NRANKS = 4
E_UNIT_ERG = 1.989e43 * 1e10  # 1e10 Msun (km/s)^2

KPC, MSUN, M_P, K_B = 3.085678e21, 1.989e33, 1.6726e-24, 1.3807e-16
BOX_PC, N_H, T_GAS, Z_SUN, M_CELL = 30.0, 10.0, 1.0e4, 0.1, 1.0


def make_ics(fname):
    """Uniform lattice, gas only; the cell nearest the centre has ID 1 and becomes the star."""
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
    u = 1.5 * K_B * T_GAS / (0.6 * M_P) / 1e10
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


SYNC = re.compile(r"Energy \(synced\) t=([0-9.eE+-]+) E_int=\S+ E_kin=\S+ E_tot=([0-9.eE+-]+)")
MERGE = re.compile(r"Particle split/merge check: (\d+) particles merged, (\d+) particles split")


@pytest.mark.skipif(bool(os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN")), reason="builds GIZMO")
def test_adiabatic_sn_energy_with_merge_split():
    cwd = os.getcwd()
    os.chdir(REPO)
    try:
        build_gizmo_for_test(TEST_NAME)
    finally:
        os.chdir(cwd)
    run = tempfile.mkdtemp(prefix=f"{TEST_NAME}_")
    os.makedirs(join(run, "output"))
    os.symlink(REPO, join(run, "gizmo"))
    make_ics(join(run, f"{TEST_NAME}_ics.hdf5"))
    shutil.copy(join(TEST_DIR, f"{TEST_NAME}.params"), run)
    with open(join(run, "run.log"), "w") as log:
        r = subprocess.run(mpi_launch_command(NRANKS) + [join(TEST_DIR, "GIZMO"), f"{TEST_NAME}.params"],
                           stdout=log, stderr=subprocess.STDOUT, cwd=run)
    out = join(TEST_DIR, "output")
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    for f in ("run.log", "output/ENERGYinfo.txt", "output/SNinfo.txt"):
        if os.path.exists(join(run, f)):
            shutil.copy(join(run, f), out)
    text = open(join(run, "run.log")).read()
    trace = keep_and_check_traces(join(run, "output"), out)
    shutil.rmtree(run, ignore_errors=True)
    assert r.returncode == 0, f"GIZMO exited {r.returncode}, see {out}/run.log"

    sync = np.array([[float(a), float(b)] for a, b in SYNC.findall(text)])
    assert len(sync) > 2 and abs(sync[-1, 0] / 4.0e-5 - 1) < 1e-6, "no synchronised energy line at TimeMax"
    ledger = np.loadtxt(join(out, "ENERGYinfo.txt"), ndmin=2)
    e_inj = ledger[-1, 11]
    de = (sync[-1, 1] - sync[0, 1]) * E_UNIT_ERG
    nm = sum(int(a) for a, _ in MERGE.findall(text))
    ns = sum(int(b) for _, b in MERGE.findall(text))
    err = de / e_inj - 1
    print(f"E_coupled={e_inj:.4e} erg  dE_gas={de:.4e} erg  rel={err:+.4%}  merges={nm} splits={ns}")
    assert e_inj > 0.9e51, f"SN did not couple its energy ({e_inj:.3e} erg)"
    assert abs(err) < 0.02, f"gas energy change {de:.4e} erg vs coupled {e_inj:.4e} erg ({err:+.2%})"
    assert trace.ok, trace.summary()
