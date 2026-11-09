"""CRFLUID_EXCHANGE_PDV_ONLY: the CRs take only their own P dV work from the gas, no share of shock heating.

Without the flag the CR-gas exchange gives the CRs at least their pressure fraction of the whole hydro change of the gas
internal energy, shock heating included. A Sedov-type blast (total energy 1) on a periodic lattice with CR energy 2 (CR pressure
about the gas pressure of the cold gas) heats the shell the shock sweeps up; with the flag that heat stays in the gas.

The box runs the production path: MAGNETIC and COOLING_OPERATOR_SPLIT, the only path where the kick credits the exchange to the
CRs (cosmic_ray_functions.h). An earlier version of this test ran without the split: the gas pays the exchange and the CRs never
receive it, so the bar could not be met; that path is an open item of the base code, not of this flag.

Bars (set before the first run of this version):
- closure: CR + thermal + kinetic energy at the end within 1e-3 of the start, both builds;
- flag: final CR energy with the flag <= 0.99 x without;
- reference: stella_dev on the same ICs (stella_dev_reference.txt): CR energy gain within 10 % (default) or within 1e-3 of the CR
  energy (pdV only), thermal and kinetic energy within 1 %.
"""
import glob
import shutil
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

from gizmo.test import build_gizmo_for_test, mpi_launch_command

TEST_NAME = "crfluid_exchange_pdv_only"
TEST_DIR = Path(__file__).parent
sys.path.insert(0, str(TEST_DIR))
from make_crfluid_exchange_pdv_only_ics import make_crfluid_exchange_pdv_only_ics  # noqa: E402


def _reference():
    rows = [line.split() for line in (TEST_DIR / "stella_dev_reference.txt").read_text().splitlines() if line and not line.startswith("#")]
    return {r[0]: np.array([float(x) for x in r[1:]]) for r in rows}


def _energies(fname):
    with h5py.File(fname, "r") as F:
        g = F["PartType0"]
        m = g["Masses"][:]
        return np.array([g["CosmicRayEnergy"][:].sum(), (m * g["InternalEnergy"][:]).sum(), 0.5 * (m * np.sum(g["Velocities"][:] ** 2, axis=1)).sum()])


def _run(flags, label):
    build_gizmo_for_test(TEST_NAME, 0, flags)
    binary = TEST_DIR / f"GIZMO_{label}"
    shutil.move(str(TEST_DIR / "GIZMO"), binary)
    d = TEST_DIR / f"output_{label}"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir()
    make_crfluid_exchange_pdv_only_ics(str(d / "snapshot_000.hdf5"))
    params = (TEST_DIR / f"{TEST_NAME}.params").read_text().replace("OutputDir                          output", f"OutputDir                          output_{label}")
    (d / "run.params").write_text(params)
    with open(d / "run.out", "w") as out:
        subprocess.run(mpi_launch_command(2, 1) + [str(binary), str(d / "run.params"), "2", "0"], cwd=TEST_DIR, stdout=out,
                       stderr=subprocess.STDOUT, check=True)
    files = sorted(glob.glob(str(d / "snapshot_*.hdf5")))
    return _energies(files[0]), _energies(files[-1])


def test_crfluid_exchange_pdv_only():
    ref = _reference()
    start_0, end_0 = _run((), "default")
    start_1, end_1 = _run(("CRFLUID_EXCHANGE_PDV_ONLY",), "pdv_only")
    for label, s, e in (("default", start_0, end_0), ("pdv_only", start_1, end_1)):
        print(f"{label}: E_CR, E_th, E_kin {e}, stella_dev {ref[label]}, total {e.sum() / s.sum() - 1:+.2e}")
        assert abs(e.sum() / s.sum() - 1) <= 1e-3, f"{label}: CR + gas energy is not conserved"
    assert end_1[0] <= 0.99 * end_0[0], "with the flag the CRs must gain less from the shock heating"
    assert abs((end_0[0] - start_0[0]) / (ref["default"][0] - start_0[0]) - 1) <= 0.1
    assert abs(end_1[0] - ref["pdv_only"][0]) <= 1e-3 * ref["pdv_only"][0]
    for e, r in ((end_0, ref["default"]), (end_1, ref["pdv_only"])):
        assert np.all(np.abs(e[1:] / r[1:] - 1) <= 0.01)
