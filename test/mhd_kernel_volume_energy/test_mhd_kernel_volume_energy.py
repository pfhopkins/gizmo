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
import pytest
from gizmo.test import build_gizmo_for_test, mpi_launch_command

TEST_NAME = "mhd_kernel_volume_energy"
TEST_DIR = Path(__file__).parent
sys.path.insert(0, str(TEST_DIR))
from make_mhd_box_ics import write_ics  # noqa: E402


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
    a = np.loadtxt(d / "output" / "energy_ledger.txt", ndmin=2)
    E = a[:, 2] + a[:, 3] + a[:, 4]
    drift = (E[-1] - E[0]) / E[0]
    print(f"{tag}: {len(a)} synchronised rows to t = {a[-1, 0]:.5g}, E_B {a[0, 4]:.5g} -> {a[-1, 4]:.5g}, E_th {a[0, 3]:.5g} -> {a[-1, 3]:.5g} (1e51 erg), end drift {drift:+.5f}, largest {np.max(np.abs(E - E[0])) / E[0]:.5f}")
    assert a[-1, 0] > 0.0019
    return drift


def test_drift_without_options():
    assert run("base") > 0.02


@pytest.mark.parametrize("flag", ["MHD_CONSERVE_ENERGY_ON_KERNEL_VOLUME_CHANGE", "MHD_RESCALE_B_TO_BOOKED_MAGNETIC_ENERGY"])
def test_drift_with_option(flag):
    assert abs(run(flag.lower(), (flag,))) < 0.005
