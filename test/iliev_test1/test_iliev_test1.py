"""Iliev et al. (2006) test 1, scaled to n_H = 100 cm^-3: the I-front of a 5e48 photons/s source in a static uniform
hydrogen medium held at 1e4 K, against the analytic r_I(t) = R_S (1 - exp(-t/t_rec))^(1/3).

The jaco model (starforge_legacy_RT_EUV) solves the band's photons with the chemistry: the RT kick only transports them,
and each photoionization takes one photon, at c_tilde, in the same backward-Euler step. The test checks that this
conserves photons. R_S and t_rec use the model's own recombination coefficient: case A (Verner & Ferland 1996) at
1e4 K, as in the standard cooling module, so R_S = 2.13 pc and t_rec = 756 yr.

- c_tilde = 0.1 c, well above the front speed, with c_tilde sigma n_H dt ~ 40 per cell and step at this resolution
  (R_S = 11 cells): r_I(t) within 10% of the analytic curve for 0.5-4 t_rec.
- c_tilde = 1e-4 c, the production value: the front is limited by the light crossing for R_S / c_tilde ~ 7e4 yr; the
  final radius, at 3.5 times that, within 10% of R_S.

The source injects its photons discretely (RT_INJECT_PHOTONS_DISCRETELY, as STARFORGE does) at every gas step
(MaxSizeTimestep below the RT Courant step): with continuous injection and no gas sources (RT_SOURCES without bit 1),
rt_source_injection never zeroes Rad_Je, so each injection adds to the last and the injected photon rate grows with
time."""

import re
import shutil
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
from make_iliev_test1_ics import make_iliev_test1_ics, BOX_SIZE, N_H  # noqa: E402

Q_ION = 5.0e48  # photons/s
T_GAS = 1.0e4
PC_CM, YR_S, CODE_TIME_S = 3.0857e18, 3.15576e7, 3.0857e18 / 1e5


def alpha_A(T):
    """Case-A H recombination, Verner & Ferland (1996): the standard cooling module's and the jaco model's"""
    return 7.982e-11 / (np.sqrt(T / 3.148) * (1 + np.sqrt(T / 3.148)) ** 0.252 * (1 + np.sqrt(T / 7.036e5)) ** 1.748)


R_S = (3 * Q_ION / (4 * np.pi * alpha_A(T_GAS) * N_H**2)) ** (1 / 3) / PC_CM  # pc
T_REC = 1 / (alpha_A(T_GAS) * N_H) / CODE_TIME_S  # code units


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


def ifront_evolution(flags):
    snaps = sorted(glob(f"{variant_output_dir(TEST_NAME, flags)}/snapshot_*.hdf5"))
    return np.array([ifront_radius(s) for s in snaps[1:]]).T


def plot(flags, t, r):
    from matplotlib import pyplot as plt
    tt = np.linspace(0, t.max(), 200)
    plt.plot(tt / T_REC, r_analytic(tt) / R_S, color="k", lw=1, label="analytic (case A)")
    plt.plot(t / T_REC, r / R_S, "o", ms=4, label="+".join(flags))
    plt.xlabel(r"$t / t_{\rm rec}$")
    plt.ylabel(r"$r_{\rm I} / R_{\rm S}$")
    plt.legend(fontsize="small")
    name = "_".join(f.replace("=", "") for f in flags)
    plt.savefig(TEST_DIR / f"ifront_{name}.png", bbox_inches="tight")
    plt.close()


def band_totals(logfile):
    """The ionizing band over the run's cooling passes: the energy its photoionizations took and the energy it lost, and
    the cells whose photon budget did not close"""
    tot, bad = np.zeros(2), 0
    for line in open(logfile):
        m = re.match(r"jaco band energy .*photoionized (\S+) band loss (\S+) \| non-conserving cells (\d+)", line)
        if m:
            tot += [float(m.group(1)), float(m.group(2))]
            bad += int(m.group(3))
    return tot, bad


def run(num_mpi_ranks, num_omp_threads, flags, overrides=None):
    ic = TEST_DIR / f"{TEST_NAME}_ics.hdf5"
    if not ic.exists():
        make_iliev_test1_ics(str(ic))
    build_and_run_test(TEST_NAME, num_mpi_ranks, num_omp_threads, flags, param_overrides=overrides)
    time_max = float(overrides["TimeMax"]) if overrides else None
    assert_final_time(get_final_snapshot(TEST_NAME, flags), TEST_NAME, time_max=time_max)
    t, r = ifront_evolution(flags)
    plot(flags, t, r)
    log = TEST_DIR / f"test_{TEST_NAME}.out"
    shutil.copy(log, Path(variant_output_dir(TEST_NAME, flags)) / log.name)  # the next variant overwrites it
    (ionized, lost), bad = band_totals(log)
    print(f"ionizing band over the run: photoionized / band loss - 1 = {ionized / lost - 1 if lost else 0:.4g}, "
          f"non-conserving cells {bad}")
    print(f"R_S = {R_S:.3f} pc, t_rec = {T_REC * CODE_TIME_S / YR_S:.1f} yr")
    print("  t/t_rec  r_I [pc]  r_I / analytic")
    for ti, ri in zip(t, r):
        print(f"  {ti / T_REC:7.3f}  {ri:8.4f}  {ri / r_analytic(ti):.4f}")
    assert bad == 0 and abs(ionized / lost - 1) < 1e-4, "the ionizing band's photon budget does not close"
    return t, r


@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(),))
@pytest.mark.parametrize("num_omp_threads", (default_omp_threads(),))
def test_iliev_test1_rtype_front(num_mpi_ranks, num_omp_threads):
    t, r = run(num_mpi_ranks, num_omp_threads, ("RT_SPEEDOFLIGHT_REDUCTION=0.1",))
    window = (t >= 0.5 * T_REC * (1 - 1e-6)) & (t <= 4 * T_REC * (1 + 1e-6))
    assert window.sum() >= 10
    ratio = r[window] / r_analytic(t[window])
    assert np.all(np.isfinite(ratio)) and np.max(np.abs(ratio - 1)) <= 0.1, \
        f"r_I / analytic over 0.5-4 t_rec: {np.round(ratio, 3)}"


@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(),))
@pytest.mark.parametrize("num_omp_threads", (default_omp_threads(),))
def test_iliev_test1_production_c(num_mpi_ranks, num_omp_threads):
    flags = ("RT_SPEEDOFLIGHT_REDUCTION=1e-4",)
    t_light = R_S * PC_CM / (1e-4 * 2.9979e10) / CODE_TIME_S
    t_end = 3.5 * t_light
    overrides = {"TimeMax": f"{t_end:.6e}", "TimeBetSnapshot": f"{t_end / 10:.6e}", "TimeBetStatistics": f"{t_end / 10:.6e}",
                 "MaxSizeTimestep": "1.5e-03"}  # below the RT Courant step, so the source injects at every gas step
    t, r = run(num_mpi_ranks, num_omp_threads, flags, overrides)
    assert abs(r[-1] / R_S - 1) <= 0.1, f"final r_I = {r[-1]:.3f} pc vs R_S = {R_S:.3f} pc"
