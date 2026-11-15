"""mhd_kernel_volume_energy: MHD_CONSERVE_ENERGY_ON_KERNEL_VOLUME_CHANGE and MHD_RESCALE_B_TO_BOOKED_MAGNETIC_ENERGY. With MFM
MHD the cell magnetic energy follows the kernel volume, while the internal-energy equation books the magnetic work through the face
volume change; at low beta the difference is a net energy gain. Static periodic box, 20^3 cells with 20 % lattice jitter, uniform
B_z, beta about 0.14, adiabatic (the check is total-energy conservation), 2 ranks, 1.96 Myr.

Bars, written before the runs, from stella_dev on the same box (end drift of E_kin + E_th + E_B: +2.89 % without either option,
-0.20 % charging u, -0.29 % rescaling B): without either option the end drift is above +2 %; with either option |end drift| is
below 0.5 %.
"""
import shutil
import sys
import subprocess
from pathlib import Path
import numpy as np
import h5py
import pytest
from gizmo.test import build_gizmo_for_test, mpi_launch_command

TEST_NAME = "mhd_kernel_volume_energy"
TEST_DIR = Path(__file__).parent
sys.path.insert(0, str(TEST_DIR))
from make_mhd_box_ics import write_ics  # noqa: E402


UM, UL, UV = 1.989e43, 3.085678e21, 1.0e5


def energies(fn):
    """E_kin, E_th, E_B [erg] from a snapshot (code units kpc, 1e10 Msun, km/s; B in gauss)"""
    with h5py.File(fn, "r") as f:
        p = f["PartType0"]
        m = p["Masses"][:] * UM; v2 = np.sum(p["Velocities"][:] ** 2, axis=1) * UV ** 2
        u = p["InternalEnergy"][:] * UV ** 2; V = p["Masses"][:] / p["Density"][:] * UL ** 3
        B2 = np.sum(p["MagneticField"][:] ** 2, axis=1)
    return 0.5 * np.sum(m * v2), np.sum(m * u), np.sum(B2 / (8 * np.pi) * V)


def run(tag, flags=()):
    build_gizmo_for_test(TEST_NAME, extra_config_flags=tuple(flags))
    d = TEST_DIR / f"run_{tag}"
    shutil.rmtree(d, ignore_errors=True)
    (d / "output").mkdir(parents=True)
    shutil.move(TEST_DIR / "GIZMO", d / "GIZMO")
    write_ics(d / "snapshot_000.hdf5")
    shutil.copy(TEST_DIR / f"{TEST_NAME}.params", d / "run.params")
    with open(d / "run.out", "w") as out:
        subprocess.run(mpi_launch_command(2, 1) + [str(d / "GIZMO"), "run.params"], cwd=d, stdout=out, stderr=subprocess.STDOUT, check=True)
    snaps = sorted((d / "output").glob("snapshot_*.hdf5"))
    t, E = [], []
    for s in snaps:
        t.append(h5py.File(s, "r")["Header"].attrs["Time"]); E.append(energies(s))
    E = np.array(E); Et = E.sum(axis=1)
    drift = (Et[-1] - Et[0]) / Et[0]
    print(f"{tag}: {len(snaps)} snapshots to t = {t[-1]:.5g}, E_B {E[0, 2]:.4e} -> {E[-1, 2]:.4e}, E_th {E[0, 1]:.4e} -> {E[-1, 1]:.4e} erg, end drift {drift:+.5f}, largest {np.max(np.abs(Et - Et[0])) / Et[0]:.5f}")
    assert t[-1] > 0.0019
    return drift


def test_drift_without_options():
    assert run("base") > 0.02


@pytest.mark.parametrize("flag", ["MHD_CONSERVE_ENERGY_ON_KERNEL_VOLUME_CHANGE", "MHD_RESCALE_B_TO_BOOKED_MAGNETIC_ENERGY"])
def test_drift_with_option(flag):
    assert abs(run(flag.lower(), (flag,))) < 0.005
