"""ledger_cr_rows: with OUTPUT_MASS_METAL_ENERGY_LEDGER every change of the CR energy has its row, so the state row E_CR closes against
the channels: E_CR - E_CR(0) = injected_sn + injected_wind + shock + clip + exchange + transport - streaming - collected - losses -
ionisation_from_fluid.

Bar (set before the first run): closes at every ledger line to 1e-5 of E_CR(0); the box must exchange energy (exchange row non zero).

CosmicRay_ShockAccelEfficiency: with CRFLUID_INJECTION_AT_SHOCKS=(0.1) the tag overrides the flag's value at run time. Bars (set before
the first run): tag unset gives the same shock row as tag 0.1 to 1e-6; tag 0.2 gives 1.5 to 2.5 x that. First run: the row stayed zero, this
blast never reached the 100 km/s the shock criterion asks for. Same bars; the hot cells now hold 1e5 x the energy and the run is ten times shorter.
"""
import shutil
import subprocess
import sys
from pathlib import Path
import h5py
import numpy as np
from gizmo.test import build_gizmo_for_test, mpi_launch_command

TEST_NAME = "ledger_cr_rows"
TEST_DIR = Path(__file__).parent
sys.path.insert(0, str(TEST_DIR.parent / "crfluid_exchange_pdv_only"))
from make_crfluid_exchange_pdv_only_ics import make_crfluid_exchange_pdv_only_ics  # noqa: E402

ADD = ("E_cr_injected_sn_erg", "E_cr_injected_wind_erg", "E_cr_shock_erg", "E_cr_clip_erg", "E_cr_exchange_erg", "E_cr_transport_erg")
SUB = ("E_cr_streaming_erg", "E_cr_collected_erg")


def run(label, extra="", strong=False):
    d = TEST_DIR / f"output_{label}"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir()
    make_crfluid_exchange_pdv_only_ics(str(d / "snapshot_000.hdf5"))
    params = (TEST_DIR / f"{TEST_NAME}.params").read_text().replace("OutputDir                          output", f"OutputDir                          output_{label}")
    if strong:
        with h5py.File(d / "snapshot_000.hdf5", "a") as F:
            u = F["PartType0/InternalEnergy"][()]
            F["PartType0/InternalEnergy"][...] = np.where(u > 2.0, 1e5 * u, u)
        params = params.replace("TimeMax                            0.01", "TimeMax                            0.001").replace("TimeBetSnapshot                    0.0025", "TimeBetSnapshot                    0.001")
    (d / "run.params").write_text(params + extra)
    with open(d / "run.out", "w") as out:
        subprocess.run(mpi_launch_command(2, 1) + [str(TEST_DIR / "GIZMO"), str(d / "run.params"), "2", "0"], cwd=TEST_DIR, stdout=out,
                       stderr=subprocess.STDOUT, check=True)
    names = (d / "ledger.txt").read_text().splitlines()[0].split()[1:]
    return dict(zip(names, np.atleast_2d(np.loadtxt(d / "ledger.txt")).T))


def test_ledger_cr_rows():
    build_gizmo_for_test(TEST_NAME)
    c = run("rows")
    sum_rows = sum(c[k] for k in ADD) - sum(c[k] for k in SUB)
    e_cr = c["E_CR_erg"]
    resid = (e_cr - e_cr[0]) - (sum_rows - sum_rows[0])
    print("E_CR", e_cr, "residual / E_CR(0)", resid / e_cr[0], {k: c[k][-1] for k in ADD + SUB})
    assert np.all(np.abs(resid) <= 1e-5 * e_cr[0])
    assert c["E_cr_exchange_erg"][-1] != 0


def test_shock_efficiency_tag():
    build_gizmo_for_test(TEST_NAME, 0, ("CRFLUID_INJECTION_AT_SHOCKS=(0.1)",))
    unset = run("unset", strong=True)["E_cr_shock_erg"][-1]
    tag = run("tag01", "\nCosmicRay_ShockAccelEfficiency 0.1\n", True)["E_cr_shock_erg"][-1]
    double = run("tag02", "\nCosmicRay_ShockAccelEfficiency 0.2\n", True)["E_cr_shock_erg"][-1]
    print(f"shock row: unset {unset:.5e}, tag 0.1 {tag:.5e}, tag 0.2 {double:.5e}")
    assert unset > 0
    assert abs(unset / tag - 1) < 1e-6
    assert 1.5 <= double / tag <= 2.5
