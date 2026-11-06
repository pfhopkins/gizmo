"""Iliev et al. (2006) test 1 with the legacy photoionization chemistry (RT_CHEM_PHOTOION without COOLING), scaled to
n_H = 100 cm^-3: the I-front of a 5e48 photons/s source in a static uniform hydrogen medium held at 1e4 K, against the
analytic r_I(t) = R_S (1 - exp(-t/t_rec))^(1/3). R_S and t_rec use the chemistry's own recombination coefficient,
case B 2.59e-13 cm^3/s at 1e4 K (rt_update_chemistry), so R_S = 2.50 pc and t_rec = 1223 yr.

The source injects continuously (no RT_INJECT_PHOTONS_DISCRETELY, as for any non-GALSF build) at every step, at
c_tilde = 0.1 c, well above the front speed after the first 0.1 t_rec. Asserted: r_I(t) within 10% of the analytic
curve for 0.5-4 t_rec. Both of these used to break it: the RT_ILIEV_TEST1 source emitted 13.6 eV per photon while the
chemistry counts the band's photons at their mean energy (29.6 eV for its 1e5 K blackbody), and with continuous
injection and no gas sources Rad_Je was never zeroed, so every injection added to the last."""

import sys
import numpy as np
import h5py
import pytest
from glob import glob
from pathlib import Path
from scipy.optimize import curve_fit
from gizmo.test import (build_and_run_test, get_final_snapshot, assert_final_time, variant_output_dir,
                        default_omp_threads, default_mpi_ranks)

TEST_NAME = "iliev_test1"
TEST_DIR = Path(__file__).parent
sys.path.insert(0, str(TEST_DIR))
from make_iliev_test1_ics import make_iliev_test1_ics, BOX_SIZE, N_H, PC_CM  # noqa: E402

Q_ION = 5.0e48  # photons/s
ALPHA_B = 2.59e-13  # rt_update_chemistry's case-B coefficient at the fixed 1e4 K
CODE_TIME_S, YR_S = PC_CM / 1e5, 3.15576e7
R_S = (3 * Q_ION / (4 * np.pi * ALPHA_B * N_H**2)) ** (1 / 3) / PC_CM  # pc
T_REC = 1 / (ALPHA_B * N_H) / CODE_TIME_S  # code units


def r_analytic(t):
    return R_S * (1 - np.exp(-t / T_REC)) ** (1 / 3)


def ifront_radius(snap):
    """(time, r_I): where the ionized fraction crosses 0.5, from a sigmoid fit of HII against the distance to the source"""
    with h5py.File(snap, "r") as F:
        t = float(F["Header"].attrs["Time"])
        pos, hii = F["PartType0/Coordinates"][:], F["PartType0/HII"][:]
        src = F["PartType4/Coordinates"][0]
    d = pos - src
    d -= BOX_SIZE * np.round(d / BOX_SIZE)
    r = np.linalg.norm(d, axis=1)
    sel = r < 0.45 * BOX_SIZE
    r, hii = r[sel], hii[sel]
    guess = np.median(r[np.abs(hii - 0.5) < 0.3]) if np.any(np.abs(hii - 0.5) < 0.3) else 0.5 * R_S
    try:
        popt, _ = curve_fit(lambda x, r0, w: 1 / (1 + np.exp(np.clip((x - r0) / w, -50, 50))), r, hii,
                            p0=[guess, 0.05], bounds=([0, 1e-3], [0.45 * BOX_SIZE, 1.0]), maxfev=10000)
        return t, float(popt[0])
    except (RuntimeError, ValueError):
        return t, np.nan


def plot(t, r):
    from matplotlib import pyplot as plt
    tt = np.linspace(0, t.max(), 200)
    plt.plot(tt / T_REC, r_analytic(tt) / R_S, color="k", lw=1, label="analytic (case B)")
    plt.plot(t / T_REC, r / R_S, "o", ms=4, label="GIZMO")
    plt.xlabel(r"$t / t_{\rm rec}$")
    plt.ylabel(r"$r_{\rm I} / R_{\rm S}$")
    plt.legend(fontsize="small")
    plt.savefig(TEST_DIR / "ifront.png", bbox_inches="tight")
    plt.close()


@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(),))
@pytest.mark.parametrize("num_omp_threads", (default_omp_threads(),))
def test_iliev_test1(num_mpi_ranks, num_omp_threads):
    ic = TEST_DIR / f"{TEST_NAME}_ics.hdf5"
    if not ic.exists():
        make_iliev_test1_ics(str(ic))
    build_and_run_test(TEST_NAME, num_mpi_ranks, num_omp_threads)
    assert_final_time(get_final_snapshot(TEST_NAME, ()), TEST_NAME)
    snaps = sorted(glob(f"{variant_output_dir(TEST_NAME, ())}/snapshot_*.hdf5"))
    t, r = np.array([ifront_radius(s) for s in snaps[1:]]).T
    plot(t, r)
    print(f"R_S = {R_S:.3f} pc, t_rec = {T_REC * CODE_TIME_S / YR_S:.1f} yr")
    print("  t/t_rec  r_I [pc]  r_I / analytic")
    for ti, ri in zip(t, r):
        print(f"  {ti / T_REC:7.3f}  {ri:8.4f}  {ri / r_analytic(ti):.4f}")
    window = (t >= 0.5 * T_REC * (1 - 1e-6)) & (t <= 4 * T_REC * (1 + 1e-6))
    assert window.sum() >= 10
    ratio = r[window] / r_analytic(t[window])
    assert np.all(np.isfinite(ratio)) and np.max(np.abs(ratio - 1)) <= 0.1, \
        f"r_I / analytic over 0.5-4 t_rec: {np.round(ratio, 3)}"
