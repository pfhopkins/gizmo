"""mhd_energy_ledger: TEST_MHD_ENERGY_LEDGER_OUTPUT, the MHD energy rows in ledger.txt (OUTPUT_MASS_METAL_ENERGY_LEDGER). The box of
test/mhd_kernel_volume_energy (static periodic box, 20^3 jittered cells, uniform B_z, beta about 0.14, adiabatic, 2 ranks, 1.96 Myr),
one ledger row per snapshot.

Bars, written before the runs, from stella_dev on the same box (E_B change -1.3726657e48 erg, volume + kick terms -1.3726657e48;
drift +0.573e48, not explained by the ledger -0.011e48; E_B not booked by the u equation +0.611e48):
  test_magnetic_energy_split: the E_B change equals the kernel-volume term plus the B-kick term to 1e-6 of the initial E_B.
  test_ledger_closes: the total-energy drift minus the booked source terms (limiter and floor, Powell, Dedner face and correction,
    face-closure limiter) minus the E_B change the u equation did not book is below 5 % of the drift; that unbooked E_B change
    is the drift to 20 %.
  test_ledger_closes_psi (DIVBCLEANING_CONSERVATIVE_PSI_OVER_CH; E_psi in the total, no Dedner rows): the closure bar (rest below
    5 % of the drift), and the psi damping row is not negative.
Changed after the first run: the psi variant first also had "unbooked E_B is the drift to 20 %"; with this cleaning the Powell term
is most of the drift (+1.56e48 of +2.17e48 erg), so that bar does not apply there. The E_B state row then came from the drifted B
while the ledger is written before the second half-kick; it now uses the kicked B, as E_th and E_kin use the kicked u and v.
"""
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
from gizmo.test import build_gizmo_for_test, mpi_launch_command

TEST_NAME = "mhd_energy_ledger"
TEST_DIR = Path(__file__).parent
sys.path.insert(0, str(TEST_DIR.parent / "mhd_kernel_volume_energy"))
from make_mhd_box_ics import write_ics  # noqa: E402

SOURCES = ("E_kick_floor_erg", "E_mhd_powell_erg", "E_mhd_dedner_face_erg", "E_mhd_dedner_correction_erg", "E_mhd_closure_erg")


def run(tag, flags=()):
    build_gizmo_for_test(TEST_NAME, extra_config_flags=tuple(flags))
    d = TEST_DIR / f"run_{tag}"
    shutil.rmtree(d, ignore_errors=True)
    (d / "output").mkdir(parents=True)
    shutil.move(TEST_DIR / "GIZMO", d / "GIZMO")
    write_ics(d / "snapshot_000.hdf5")
    shutil.copy(TEST_DIR.parent / "mhd_kernel_volume_energy" / "mhd_kernel_volume_energy.params", d / "run.params")
    with open(d / "run.out", "w") as out:
        subprocess.run(mpi_launch_command(2, 1) + [str(d / "GIZMO"), "run.params"], cwd=d, stdout=out, stderr=subprocess.STDOUT, check=True)
    fn = d / "output" / "ledger.txt"
    with open(fn) as f:
        head = f.readline().split()[1:]
    names = head[:next(k for k, w in enumerate(head) if w.startswith("("))]
    a = np.loadtxt(fn, ndmin=2)
    assert a.shape[1] == len(names) and len(a) >= 5 and a[-1, 0] > 0.0019, (a.shape, len(names))
    return {n: a[:, k] for k, n in enumerate(names)}


@pytest.fixture(scope="module")
def ledger():
    return run("dedner")


@pytest.fixture(scope="module")
def ledger_psi():
    return run("psi", ("DIVBCLEANING_CONSERVATIVE_PSI_OVER_CH",))


def delta(L, n):
    return (L[n][-1] - L[n][0]) / 1e48


def closes(L, with_psi, unbooked_is_drift=True):
    dE = sum(delta(L, n) for n in ("E_th_erg", "E_kin_erg", "E_B_erg") + (("E_psi_erg",) if with_psi else ()))
    src = sum(delta(L, n) for n in SOURCES)
    unbooked = delta(L, "E_B_erg") - (delta(L, "E_mhd_B_dVB_booked_erg") - delta(L, "E_mhd_half_B2_dV_booked_erg"))
    rest = dE - src - unbooked
    print(f"drift {dE:+.4f}, sources {src:+.4f} (" + " ".join(f"{n} {delta(L, n):+.4f}" for n in SOURCES) + f"), unbooked E_B {unbooked:+.4f}, rest {rest:+.4f} (1e48 erg)")
    assert abs(rest) < 0.05 * abs(dE)
    if unbooked_is_drift:
        assert abs(unbooked - dE) < 0.2 * abs(dE)


def test_magnetic_energy_split(ledger):
    dEB, dv, dk = delta(ledger, "E_B_erg"), delta(ledger, "E_B_volume_change_erg"), delta(ledger, "E_B_kick_change_erg")
    print(f"dE_B {dEB:+.7f}, volume {dv:+.7f} + kick {dk:+.7f} = {dv + dk:+.7f} (1e48 erg)")
    assert abs(dEB - dv - dk) < 1e-6 * ledger["E_B_erg"][0] / 1e48


def test_ledger_closes(ledger):
    closes(ledger, False)


def test_ledger_closes_psi(ledger_psi):
    assert delta(ledger_psi, "E_mhd_dedner_face_erg") == 0 and delta(ledger_psi, "E_mhd_dedner_correction_erg") == 0
    print("E_psi end", ledger_psi["E_psi_erg"][-1], "damped", ledger_psi["E_psi_damping_erg"][-1], "erg")
    assert ledger_psi["E_psi_damping_erg"][-1] >= 0
    closes(ledger_psi, True, unbooked_is_drift=False)
