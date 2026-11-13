"""mhd_energy_ledger: TEST_MHD_ENERGY_LEDGER_OUTPUT, the MHD energy rows in ledger.txt (OUTPUT_MASS_METAL_ENERGY_LEDGER). The box of
test/mhd_kernel_volume_energy (static periodic box, 20^3 jittered cells, uniform B_z, beta about 0.14, adiabatic, 2 ranks, 1.96 Myr),
all cells on one fixed step (Min = MaxSizeTimestep, TimeMax/256, ~1/5 of the Courant step), and again at TimeMax/512.

Bars from the ledger's algebra, set before the run (reports/test_bars_2026_11_13.md):
  test_magnetic_energy_split: per cell E_B changes only at its B kicks (kick row) and through its volume between kicks (volume row,
    pending at output); the rows telescope, the sums are exact, so the split closes to the rounding of the per-kick differences:
    (4 N_steps + 10) x 2.2e-16 x E_B0.
  test_ledger_closes / _psi: rest = drift - unbalanced source rows - E_B the u equation did not book. Everything in the rest is
    time-integration error of the KDK (the fluxes conserve in rate form, the psi pair too); on one uniform step it is 2nd order:
    measured order log2(|rest(TimeMax/256)| / |rest(TimeMax/512)|) >= 1.5, unless below the roundoff floor (4 N + 10) x 2.2e-16 x E_tot.
  _psi also: no Dedner rows (exactly 0), psi damping row >= 0 (each kick multiplies PsiZ by a factor <= 1).
Uniform steps: on block steps the flux exchange between cells on different bins adds a 1st-order hydro residual (outside the MHD
terms) whose size is not fixed by theory; one step makes the rest a pure 2nd-order KDK error. Not tested here: the closure on
block steps. Earlier bars (rest < 5 % of the drift, unbooked E_B = drift to 20 %, split to 1e-6) were set or changed after runs.
"""
import re
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


STEPS = (256, 512)  # uniform steps per run (TimeMax / n)


def run(tag, flags=()):
    """ledger rows per step count"""
    build_gizmo_for_test(TEST_NAME, extra_config_flags=tuple(flags))
    out = {}
    for n in STEPS:
        d = TEST_DIR / f"run_{tag}_{n}"
        shutil.rmtree(d, ignore_errors=True)
        (d / "output").mkdir(parents=True)
        shutil.copy(TEST_DIR / "GIZMO", d / "GIZMO")
        write_ics(d / "snapshot_000.hdf5")
        p = (TEST_DIR / f"{TEST_NAME}.params").read_text()
        for k in ("MaxSizeTimestep", "MinSizeTimestep"):
            p = re.sub(rf"(?m)^{k}\s+\S+", f"{k:34s} {0.002 / n!r}", p)
        (d / "run.params").write_text(p)
        with open(d / "run.out", "w") as o:
            subprocess.run(mpi_launch_command(2, 1) + [str(d / "GIZMO"), "run.params"], cwd=d, stdout=o, stderr=subprocess.STDOUT, check=True)
        fn = d / "output" / "ledger.txt"
        with open(fn) as f:
            head = f.readline().split()[1:]
        names = head[:next(k for k, w in enumerate(head) if w.startswith("("))]
        a = np.loadtxt(fn, ndmin=2)
        assert a.shape[1] == len(names) and len(a) >= 5 and a[-1, 0] > 0.0019, (a.shape, len(names))
        out[n] = {k: a[:, j] for j, k in enumerate(names)}
    return out


@pytest.fixture(scope="module")
def ledger():
    return run("dedner")


@pytest.fixture(scope="module")
def ledger_psi():
    return run("psi", ("DIVBCLEANING_CONSERVATIVE_PSI_OVER_CH",))


def delta(L, n):
    return L[n][-1] - L[n][0]


def eps_bar(n, E):
    return (4 * n + 10) * 2.2e-16 * E


def rest(L, with_psi):
    dE = sum(delta(L, n) for n in ("E_th_erg", "E_kin_erg", "E_B_erg") + (("E_psi_erg",) if with_psi else ()))
    unbooked = delta(L, "E_B_erg") - (delta(L, "E_mhd_B_dVB_booked_erg") - delta(L, "E_mhd_half_B2_dV_booked_erg"))
    return dE - sum(delta(L, n) for n in SOURCES) - unbooked, dE, unbooked


def closes(runs, with_psi):
    (R1, dE1, ub1), (R2, dE2, ub2) = (rest(runs[n], with_psi) for n in STEPS)
    L = runs[STEPS[0]]
    Etot = sum(L[n][0] for n in ("E_th_erg", "E_kin_erg", "E_B_erg") + (("E_psi_erg",) if with_psi else ()))
    order = np.log2(abs(R1) / abs(R2)) if R2 != 0 else np.inf
    print(f"drift {dE1:+.6e} / {dE2:+.6e}, unbooked E_B {ub1:+.6e} / {ub2:+.6e}, rest {R1:+.6e} / {R2:+.6e} erg (steps {STEPS}), order {order:.2f}, "
          f"floor {eps_bar(STEPS[0], Etot):.2e}; " + " ".join(f"{n} {delta(L, n):+.4e}" for n in SOURCES))
    assert abs(R1) <= eps_bar(STEPS[0], Etot) or order >= 1.5


def test_magnetic_energy_split(ledger):
    for n in STEPS:
        L = ledger[n]
        dEB, dv, dk, dp = (delta(L, k) for k in ("E_B_erg", "E_B_volume_change_erg", "E_B_kick_change_erg", "E_B_volume_change_pending_erg"))
        print(f"steps {n}: dE_B {dEB:+.12e}, volume + kick + pending {dv + dk + dp:+.12e}, residual {dEB - dv - dk - dp:+.3e}, bar {eps_bar(n, L['E_B_erg'][0]):.2e} erg")
        assert abs(dEB - dv - dk - dp) <= eps_bar(n, L["E_B_erg"][0])


def test_ledger_closes(ledger):
    closes(ledger, False)


def test_ledger_closes_psi(ledger_psi):
    for n in STEPS:
        L = ledger_psi[n]
        assert delta(L, "E_mhd_dedner_face_erg") == 0 and delta(L, "E_mhd_dedner_correction_erg") == 0
        assert np.all(np.diff(L["E_psi_damping_erg"]) >= 0) and L["E_psi_damping_erg"][0] >= 0
    closes(ledger_psi, True)
