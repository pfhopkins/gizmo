"""Hierarchical triple -- momentum and energy conservation across a deep timebin hierarchy.

Equal-mass 4 AU inner binary + 1 Msun tertiary at 100 AU: period ratio 88, so the tertiary
runs ~6 timebins coarser than the inner stars at every sync point (measured: gap 6 at 99.9% of
occupancy samples). This is the configuration test/binary cannot reach -- a bound pair splits
by at most ~1 bin -- and the configuration of the production momentum violation (a hardening
triple). The inner pair shares a bin by symmetry and cannot leak against itself, so any secular
COM drift is the inner<->outer channel: forces on the fine stars evaluated with the coarse
tertiary seen mid-step, up to 2^6 of their own steps from its last sync.

What this guards: the Old*-based source prediction in the Hermite gravity passes
(gravity/forcetree.cc). With it, the COM drift stays a bounded noise band (growth exponent
~0); without it, the drift is secular. THE MAGNITUDES ALONE DO NOT DISCRIMINATE AT 50 ORBITS
-- the assertions here are calibrated so that the O(dt^2) drifted-source error fails them at
this test's operating point; see the tolerance block for the calibration.

Measured from the IO_HERMITE_SYNC datasets, for the same reason as test/binary: the plain
Coordinates/Velocities are a mixed state (positions drifted, velocities at the last kick).
"""

import glob
import os
import re
from os import chdir, getcwd, path

import h5py
import numpy as np
import pytest
import matplotlib
from matplotlib import pyplot as plt

# G and the AU conversion come from gizmo.units, which mirrors GIZMO's constants.h.
# NOT astropy: the code integrates with GRAVITY_G_CGS = 6.672e-8 and SOLAR_MASS_CGS =
# 1.989e33, giving G_code = 4.300710573e-3 rather than 4.300917270e-3. Reconstructing
# energies or orbital elements with the wrong G injects a spurious term ~ dG/r that
# sweeps with the orbit -- 9.1e-4 in |dE/E| for test/binary, an order of magnitude above
# what that test measures.
from gizmo.units import G_CODE, AU_PER_PC
from gizmo.test import (
    build_and_run_test,
    clean_test_outputs,
    variant_output_dir,
    run_test,
)

TEST_NAME = "triple"
TEST_DIR = f"test/{TEST_NAME}"
IC_FILE = f"{TEST_DIR}/{TEST_NAME}_ics.hdf5"


M_IN, M3 = 0.5, 1.0
A_IN = 4.0 / AU_PER_PC
A_OUT = 100.0 / AU_PER_PC
MTOT = 2 * M_IN + M3
P_OUT = 2.0 * np.pi * np.sqrt(A_OUT ** 3 / (G_CODE * MTOT))
V_OUT = np.sqrt(G_CODE * MTOT / A_OUT)

# Operating point eta=0.00125 (see triple.params). Measured there, 50 outer orbits:
#   with the source prediction    |dE/E| 2.08e-6 (t^+1.75)   drift band max 2.83e-6 (t^+0.55)
#   with drifted sources (defect) |dE/E| 8.15e-6 (t^+1.08)   drift band max 6.27e-6 (t^+0.95)
# Magnitude ceilings are ~3x the fixed code's values and bound gross breakage only -- the
# defect passes them. What catches the defect is the SECULAR check on the drift exponent:
# 0.95 vs 0.55 across the 0.85 threshold. (test/binary carries the magnitude discrimination:
# its 1-bin split leaks 53x harder without the fix at 1000 orbits.)
MAX_DE_OVER_E = 1e-5
MAX_COM_DRIFT = 1e-5
SECULAR_EXPONENT = 0.85

# --- integration-order sweep ------------------------------------------------------------------
# The tolerances above bound the error at ONE accuracy setting. They cannot see a change that
# keeps magnitudes within tolerance while degrading the scheme's ORDER -- which is the claim the
# Hermite machinery makes, and what the source prediction restored here.
#
# FACTORS OF 2, spanning 5e-3 down to 1.25e-3 only. The first attempt used factors of 4 reaching
# 3.125e-4 and did not measure an order at all: the error fell 112x over the first leg and then
# ROSE from 5.3e-8 to 8.3e-8 over the second, i.e. by 1.25e-3 it had already reached a floor and
# the finest point was sampling round-off and hierarchy-phase noise rather than truncation error.
# A straight line through that gave eta^1.54, which passed the assertion below while measuring
# nothing. Dropping the floored point and refining between the two that remain keeps every
# sample in the regime where the error still converges.
#
# The cost of factors of 2: dt scales as sqrt(eta), so these steps are sqrt(2) in dt rather than
# the clean factor of 2 -- half a timebin, not a whole one. Timebin assignments can therefore
# shift between sweep points instead of translating rigidly, which adds some scatter of its own.
# That is the trade for staying above the floor; watch for it if the fitted order gets noisy.
SWEEP_ETAS = (5e-3, 2.5e-3, 1.25e-3)
SWEEP_ORBITS = 20
# 4th order in dt is eta^2, 2nd order is eta^1; the threshold sits between, nearer the low side
# because a 3-point fit on a finite run has real scatter.
MIN_ENERGY_ORDER = 1.5


def _ensure_ic():
    if path.exists(IC_FILE):
        return
    import subprocess
    import sys
    subprocess.run(
        [sys.executable, "make_triple_ics.py", "--m_in", str(M_IN), "--m3", str(M3),
         "--a_in_au", "4.0", "--a_out_au", "100.0", "--out", f"{TEST_NAME}_ics.hdf5"],
        cwd=TEST_DIR, check=True,
    )


def _read(snap):
    with h5py.File(snap, "r") as f:
        t = float(dict(f["Header"].attrs)["Time"])
        g = f["PartType5"]
        o = np.argsort(np.array(g["ParticleIDs"]))
        synced = "HermiteSyncCoordinates" in g and "HermiteSyncVelocities" in g
        assert synced, "IO_HERMITE_SYNC datasets missing -- the metrics are meaningless on the mixed state"
        return (t, np.array(g["Masses"])[o],
                np.array(g["HermiteSyncCoordinates"])[o],
                np.array(g["HermiteSyncVelocities"])[o])


def _trajectory(snaps):
    t, de, dr, a_in = [], [], [], []
    e0 = v0 = None
    for s in snaps:
        ti, m, x, v = _read(s)
        E = 0.5 * (m * np.sum(v ** 2, axis=1)).sum()
        for i in range(len(m)):
            for j in range(i + 1, len(m)):
                E -= G_CODE * m[i] * m[j] / np.linalg.norm(x[j] - x[i])
        vc = (m[:, None] * v).sum(0) / m.sum()
        # inner semi-major axis as an orbit-health check (vis-viva on the inner pair)
        rr = np.linalg.norm(x[1] - x[0])
        vv = np.sum((v[1] - v[0]) ** 2)
        ain = 1.0 / (2.0 / rr - vv / (G_CODE * 2 * M_IN))
        if e0 is None:
            e0, v0 = E, vc
        t.append(ti)
        de.append(abs(E / e0 - 1.0))
        dr.append(np.linalg.norm(vc - v0) / V_OUT)
        a_in.append(ain)
    return np.array(t), np.array(de), np.array(dr), np.array(a_in)


def _envelope(t, y):
    """Per-OUTER-orbit minimum: within an orbit the instantaneous values oscillate by orders
    of magnitude with the hierarchy phase; the envelope floor is the integration error."""
    orb = (t / P_OUT).astype(int)
    xs, ys = [], []
    for i in range(orb.max() + 1):
        k = orb == i
        if k.sum() < 3:
            continue
        xs.append(t[k].mean())
        ys.append(y[k].min())
    return np.array(xs), np.array(ys)


def _growth_exponent(t, y):
    xs, ys = _envelope(t, y)
    k = (xs > 0) & (ys > 0)
    if k.sum() < 8:
        return np.nan
    h = k & (xs >= 0.5 * xs[k].max())
    return np.polyfit(np.log(xs[h]), np.log(ys[h]), 1)[0]


def _plot(t, de, dr, variant_id):
    matplotlib.rcParams['text.usetex'] = False
    orbits = t / P_OUT
    fig, ax = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True)
    ax[0].plot(orbits, de, lw=.4)
    ax[1].plot(orbits, dr, lw=.4)
    for A in ax:
        A.set_xscale("log")
        A.set_yscale("log")
    ax[0].set_ylabel("|E/E$_0$ - 1|")
    ax[1].set_ylabel("|v$_{com}$| / v$_{out}$")
    ax[1].set_xlabel("outer orbits")
    ax[1].set_xlim(1, max(2.0, orbits[-1]))
    fig.tight_layout()
    fig.savefig(f"{TEST_DIR}/{TEST_NAME}_{variant_id}_conservation.png", dpi=120)
    plt.close(fig)


def _plot_convergence(etas, errs, order, variant_id):
    """log-log energy error vs ErrTolIntAccuracy, with reference slopes.

    The fitted line IS the measurement here, unlike the conservation plot -- and the reference
    slopes matter as much as the fit: a 4th-order scheme fed drifted source positions degrades
    to 2nd, so the question this figure answers is which of the two dashed lines the points lie
    along, not merely whether they are straight.
    """
    matplotlib.rcParams["text.usetex"] = False
    etas, errs = np.asarray(etas, float), np.asarray(errs, float)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.loglog(etas, errs, "o", ms=7, zorder=3)
    xs = np.array([etas.min() / 1.6, etas.max() * 1.6])
    c = np.polyfit(np.log(etas), np.log(errs), 1)
    ax.loglog(xs, np.exp(np.polyval(c, np.log(xs))), "-", lw=1.4, zorder=2,
              label=f"fit: $\\eta^{{{order:.2f}}}$  (dt$^{{{2*order:.1f}}}$)")
    for p, lab in ((1.0, "2nd order  $\\eta^1$"), (2.0, "4th order  $\\eta^2$")):
        ax.loglog(xs, errs[0] * (xs / etas[0]) ** p, "--", lw=1.0, zorder=1, label=lab)
    ax.set_xlabel("ErrTolIntAccuracy")
    ax.set_ylabel("|E/E$_0$ - 1|  (per-orbit envelope)")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{TEST_DIR}/{TEST_NAME}_{variant_id}_convergence.png", dpi=120)
    plt.close(fig)

def _order_sweep(extra_config_flags, n_ranks, n_omp, variant_id):
    """Run the same hierarchy at several ErrTolIntAccuracy values and fit the energy error's order.

    Reuses the binary built for the fiducial run -- only params change, via run_test's override
    mechanism -- so this costs runs, not rebuilds. Each point overwrites the previous one's
    snapshots, so each is analysed before the next starts.
    """
    outdir = variant_output_dir(TEST_NAME, extra_config_flags)
    etas, errs = [], []
    for eta in SWEEP_ETAS:
        # run_test resolves <name>.params relative to the TEST directory and is normally called
        # by build_and_run_test from inside it; called from the repo root it raises
        # FileNotFoundError. Enter and leave around each run, restoring on failure.
        # Remove the previous point's snapshots first. The fiducial run is far longer than a
        # sweep point, so its snapshots are NOT all overwritten -- the leftovers sit at times
        # beyond this point's TimeMax and dominate the per-orbit envelope, making every point
        # report the fiducial value and the fitted order come out ~0.
        for _stale in glob.glob(outdir + "/snapshot_*.hdf5"):
            os.remove(_stale)
        cwd = getcwd()
        try:
            chdir(TEST_DIR)
            run_test(TEST_NAME, n_ranks, n_omp, param_overrides={
                "ErrTolIntAccuracy": float(eta),
                "TimeMax": float(SWEEP_ORBITS * P_OUT),
                "TimeBetSnapshot": float(P_OUT / 4.0),
            })
        finally:
            chdir(cwd)
        snaps = sorted(glob.glob(outdir + "/snapshot_*.hdf5"),
                       key=lambda f: int(re.search(r"snapshot_(\d+)", f).group(1)))
        if len(snaps) < 16:
            pytest.fail(f"sweep point eta={eta:g} produced only {len(snaps)} snapshots")
        t, de, dr, a_in = _trajectory(snaps)
        _, de_env = _envelope(t, de)
        etas.append(eta); errs.append(de_env[-1])
        print(f"  sweep eta={eta:9.3e}   |dE/E| envelope {de_env[-1]:.3e}")
    order = np.polyfit(np.log(etas), np.log(errs), 1)[0]
    print(f"  energy error ~ eta^{order:.2f}  (dt^{2*order:.1f});  "
          f"4th order is eta^2, 2nd order eta^1, threshold {MIN_ENERGY_ORDER}")
    _plot_convergence(etas, errs, order, variant_id)
    assert order >= MIN_ENERGY_ORDER, (
        f"energy error converges as eta^{order:.2f} (dt^{2*order:.1f}), below the "
        f"eta^{MIN_ENERGY_ORDER} floor. Across a 6-bin hierarchy this is the signature of "
        f"inactive sources being seen at drifted positions -- check the Hermite source "
        f"prediction in gravity/forcetree.cc and gravity/star_direct_gravity.cc.")
    return np.asarray(etas), np.asarray(errs), order


@pytest.mark.parametrize("num_mpi_ranks", (1,))
@pytest.mark.parametrize("num_omp_threads", (1,))
@pytest.mark.parametrize("extra_config_flags", [
    pytest.param((), id="starforge_defaults"),
])
def test_triple(num_mpi_ranks, num_omp_threads, extra_config_flags, request):
    _ensure_ic()
    clean_test_outputs(TEST_NAME, extra_config_flags)
    build_and_run_test(TEST_NAME, num_mpi_ranks, num_omp_threads, extra_config_flags)

    outdir = variant_output_dir(TEST_NAME, extra_config_flags)
    snaps = sorted(glob.glob(outdir + "/snapshot_*.hdf5"),
                   key=lambda f: int(re.search(r"snapshot_(\d+)", f).group(1)))
    assert len(snaps) >= 64, f"only {len(snaps)} snapshots -- the run died early"

    t, de, dr, a_in = _trajectory(snaps)
    assert np.all(np.diff(t) >= 0), "snapshot times are not monotonic -- file ordering is wrong"

    n_orbits = t[-1] / P_OUT
    _, de_env = _envelope(t, de)
    _, dr_env = _envelope(t, dr)
    p_de = _growth_exponent(t, de)
    p_dr = _growth_exponent(t, dr)

    variant_id = request.node.callspec.id.split("-")[0]
    _plot(t, de, dr, variant_id)
    np.savez(f"{TEST_DIR}/summary_{variant_id}.npz",
             t=t, de=de, drift=dr, a_in=a_in, period_out=P_OUT)

    print(f"  {n_orbits:.1f} outer orbits, {len(snaps)} snapshots")
    print(f"  |dE/E|      envelope {de_env[-1]:.3e}   growth t^{p_de:+.2f}")
    print(f"  COM drift   band max {dr_env.max():.3e}   growth t^{p_dr:+.2f}")
    print(f"  inner orbit a/a0 - 1 = {a_in[-1]/a_in[0]-1:+.3e}")

    # the inner binary must survive untouched -- if it hardened or dissolved, the momentum
    # numbers describe a different configuration than the one this test is calibrated for
    assert abs(a_in[-1] / a_in[0] - 1) < 5e-2, "inner binary changed by >5% -- configuration lost"

    assert de_env[-1] < MAX_DE_OVER_E, (
        f"relative energy error {de_env[-1]:.3e} over {n_orbits:.0f} outer orbits "
        f"(tol {MAX_DE_OVER_E})")
    assert dr_env.max() < MAX_COM_DRIFT, (
        f"COM drift band reached {dr_env.max():.3e} (tol {MAX_COM_DRIFT}); a secular "
        f"inner<->outer momentum leak has re-opened -- check the Hermite source prediction "
        f"in gravity/forcetree.cc")
    # Order sweep, after the magnitude asserts so a magnitude regression reports from those.
    sweep_etas, sweep_errs, sweep_order = _order_sweep(
        extra_config_flags, num_mpi_ranks, num_omp_threads, variant_id)
    # re-save with the sweep included. The earlier savez above is kept deliberately: it runs
    # before the assertions, so a failing run still leaves its trajectory on disk to look at.
    np.savez(f"{TEST_DIR}/summary_{variant_id}.npz", t=t, de=de, drift=dr, a_in=a_in, period_out=P_OUT,
             sweep_etas=sweep_etas, sweep_errs=sweep_errs, sweep_order=sweep_order)

    if p_dr >= SECULAR_EXPONENT:
        pytest.fail(
            f"COM drift grows at t^{p_dr:+.2f} -- secular, not a noise band. The many-bin "
            f"momentum leak is back.")
    if p_de >= SECULAR_EXPONENT:
        pytest.xfail(
            f"energy drifts secularly (t^{p_de:+.2f}) at the truncation level -- known "
            f"behaviour, tracked but not blocking")
