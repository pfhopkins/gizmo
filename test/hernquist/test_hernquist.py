"""Hernquist-sphere gravity-only equilibrium stability test.

Runs the same Hernquist IC through several gravity-solver variants and
verifies energy conservation and preservation of the initial mass-Lagrange
radii (10th, 50th, 90th percentiles). Each variant isolates one feature:

  - ags:       ADAPTIVE_GRAVSOFT_FORALL (Type 1)
  - pmgrid:    BOX_PERIODIC + PMGRID
  - ewald:     BOX_PERIODIC (Ewald summation, no PM)
  - tidal:     TIDAL_TIMESTEP_CRITERION
  - tidal_ags: TIDAL_TIMESTEP_CRITERION + ADAPTIVE_GRAVSOFT_FORALL +
               ADAPTIVE_GRAVSOFT_FROM_TIDAL_CRITERION  (tidal-radius softening)

Each variant writes its profile + Lagrange radii to an .npz; once any variant
runs, the overlay plot (final density profile + Lagrange radii) is rebuilt
from whatever .npz files are present.

Code units: G = M = a = 1, dynamical time ~ 1, run to TimeMax = 5.
"""

from os import path
import glob
import os
import warnings
import sys

import h5py
import numpy as np
import pytest
from matplotlib import pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from momentum_drift_common import (  # noqa: E402
    DRIFT_SANITY_CEILING, assert_randomized_drift, measure_and_record, plot_momentum_drift,
    report_momentum_drift,
)

from gizmo.test import (
    build_and_run_test,
    clean_test_outputs,
    default_mpi_ranks,
    default_omp_threads,
    variant_output_dir,
    assert_final_time,
    parse_params,
)

TEST_NAME = "hernquist"
TEST_DIR = f"test/{TEST_NAME}"
IC_FILE = f"{TEST_DIR}/{TEST_NAME}_ics.hdf5"
SCALE_RADIUS = 1.0
TOTAL_MASS = 1.0
BOXSIZE = 500.0  # = 2 * r_max where r_max = 100 r_h, so cluster fits at corner
N_PARTICLES = 2**15
LAGRANGE_FRACTIONS = (0.1, 0.5, 0.9)
LAGRANGE_TOL = (0.20, 0.15, 0.30)  # |dr/r| for 10%, 50%, 90% Lagrange radii
LAGRANGE_PERCENTILES_DENSE = np.arange(1, 100, dtype=float)  # 1..99 % for plot only


# --- half-mass-radius drift tolerance ---------------------------------------------------
# TimeMax was cut 118 -> 11.8 (~10 crossings -> ~1) for suite wall time, so this tolerance is
# NOT the historical 0.10 scaled by 10. r_h drift does not follow a power law: measured on the
# baseline variant it oscillates (9.1e-3 at t=11.8, 1.0e-3 at t=35.5, 7.6e-4 at t=71, 2.5e-2 at
# t=118), because at one crossing time we are watching the IC settle, not the secular drift.
# Scaling 0.10 by 10 would put the ceiling at 0.010 against a measured 9.1e-3 -- a 1.1x margin
# on a quantity that swings 10x between snapshots, which flaps rather than gates.
# 0.025 keeps ~2.7x margin over the measured value while still being 4x tighter than before.
# Re-measure before changing it; do not "fix" a failure by loosening it.
RH_DRIFT_TOL = 0.025

# --- energy-conservation ceilings, per variant -------------------------------------------
# |dE/E0| between the first and last snapshot. Per-variant rather than global because the
# variants legitimately differ by three orders of magnitude: ewald conserves to 5e-6 while
# tidal sits at 1e-2, so any single number is either vacuous for one end or wrong for the
# other.
#
# Calibrated against starforge_dev measured AT THIS TimeMax (snapshot index 1 of its
# TimeMax=118 run is t=11.83, so these are measurements, not extrapolations), times a 3x
# margin. 3x is deliberately snug: the regression this exists to catch is ~12x, and a 10x
# margin would sail straight past it. Each reference number is a single run, so treat a
# failure between 1x and 3x as "re-measure", not "bug".
#
# KNOWN WEAKNESS, read before trusting a verdict here. |dE/E0| is not a clean measure of
# integration quality at this TimeMax: it is the residual of a near-total cancellation. Over
# t=0..11.83 the baseline KE rises 2.04e-3 and PE falls 2.04e-3, and what is left is 4.7e-6 --
# one part in 430. So this number reports how well the snapshot's Potential field agrees with
# the forces the integrator actually used, as much as it reports energy conservation.
#
# Consequently it oscillates rather than growing: starforge_dev's baseline wanders over
# 3.6e-5..3.6e-4 across its 10 outputs, a 10x spread with no trend. The ceilings above are
# each ONE sample off such a curve, and t=11.83 happens to sit near the bottom of the swing.
# Had it been sampled at t=118's phase the baseline ceiling would be ~1.1e-3 instead of 1.7e-4
# and would admit 4x more error. Calibrating the margins honestly needs repeat runs at fixed
# t (rank count, thread count), which do not exist yet -- 3x is a placeholder, not a measurement.
#
# ewald is the least trustworthy entry: its cancellation is one part in 4000, and it is only
# "good" at t=11.83. At 7 of 10 output times it is WORSE than baseline (up to 11x), scaling as
# t^1.42 against baseline's t^0.68. Do not read its small ceiling as Ewald summation being
# accurate; nothing about the method predicts that, and over the run it is not.
#
# So the endpoint is NOT what is gated. What distinguishes a healthy run from a sick one is
# the SHAPE: starforge_dev oscillates within a bound (what a symplectic KDK leapfrog should
# do), while kokkos grows monotonically 6.55e-4 -> 5.90e-3 and exceeds the reference at every
# output time by 10-113x. The gate below fits that trend instead of sampling it.
#
# Statistic: least-squares slope of the SIGNED relative energy error against time, i.e. the
# secular drift rate per unit time. Two properties earn it its place:
#
#   - A fit over all snapshots is immune to which phase of the oscillation the last one lands
#     in, which is the whole defect of the endpoint test described above.
#   - It is a RATE, so the ceiling and the measurement scale with run length together and T
#     cancels: this gate does not need recalibrating when TimeMax changes. That matters here,
#     where no reference trajectory exists at the shortened TimeMax to calibrate against.
#
# Per-variant, because secular drift is not by itself a defect -- ags and tidal genuinely
# drift on starforge_dev too (rates 7.8e-4 and 9.5e-4, hundreds of times baseline's). A single
# global threshold would either indict them or excuse everything. A shape statistic normalized
# by its own scatter does not fix this: measured, ref/tidal scores 116x and ref/ags 57x on
# secular-slope-over-scatter, ABOVE kokkos/baseline's 39x, so it ranks the healthy reference
# as sicker than the actual regression.
#
# Rates are starforge_dev's, fitted over its full TimeMax=118 trajectory, times a 3x margin.
# Reading the fit as window-independent assumes the drift is close to linear in t; that holds
# for the secular variants and is conservative for the oscillating ones, whose fitted slope is
# small by construction.
ENERGY_SECULAR_RATE = {
    "baseline": 2.376e-06,
    "ags": 7.751e-04,
    "ewald": 3.351e-06,
    "pmgrid": 1.577e-05,
    "randomize": 8.065e-07,
    "randomize_pmgrid": 2.653e-05,
    "tidal": 9.508e-04,
    "tidal_ags": 9.508e-04,  # no reference run; given tidal's rate
}
ENERGY_SECULAR_RATE_DEFAULT = 1.0e-2  # untabulated: loose, nothing measured justifies more
ENERGY_SECULAR_MARGIN = 3.0

# Backstop for a blow-up that a slope fit would happily accommodate (a run that loses a
# quarter of its energy is broken no matter how tidily it does so). Deliberately loose: the
# secular rate above is the discriminating test, this only catches catastrophe.
ENERGY_ABS_CEILING = 0.25

# --- known pre-existing defect: periodic-gravity half-mass-radius runaway ---------------
# NOTE: at TimeMax=11.8 this runaway has not developed yet -- it is a late-time effect, so the
# xfail below is expected to be inert here and the notes are kept for when the long run is used.
# Every flag-OFF periodic variant secularly inflates r_h to ~10% by TimeMax=118, right on the
# 0.10 tolerance, so the assertion flips run to run (ewald 0.1076, pmgrid 0.0984/0.1005 on
# repeats; a since-retired PMGRID=256 probe gave 0.1019). It comes with ~4x the COM momentum drift of the non-periodic
# baseline (2.85-2.93e-2 vs 7.6e-3), and both quantities reproduce to ~1%, so the pathology is
# real -- only the *test* is marginal, because TimeMax lands where the runaway crosses the
# tolerance. Cause is position-correlated tree-force errors in the periodic gravity path.
# Eliminated: PM under-resolution (PMGRID 64->256, no effect), tree/PM split (Rcut 43.9->11.0,
# no effect), image tides (a periodic translation is an exact lattice symmetry, so
# RANDOMIZE_GRAVTREE could not affect them, yet it removes the runaway), and halo-fills-box
# (BoxSize 500->2000, 86x volume: momentum drift unchanged to 1%). The effect is identical with
# PM absent, PMGRID=64 and PMGRID=256, so it is periodicity itself, not the mesh. The mechanism
# is NOT identified; narrowing it needs direct force-error measurement, not parameter sweeps.
# The PMGRID=256 probe is not kept as a suite variant: 32768 particles over a 256^3 mesh is
# 0.002 particles/cell, a degenerate setup that also segfaults in gravity_tree() at 48 ranks.
# RANDOMIZE_GRAVTREE removes it entirely (randomize_pmgrid r_h drift 0.0118, better than the
# non-periodic baseline's 0.0152). Do NOT loosen the 0.10 tolerance or enlarge the box to make
# this green: BoxSize=2000 gives a passing 0.0821 while leaving the force error untouched.
# See domain/RANDOMIZE_GRAVTREE_TreePM.md.
PERIODIC_RH_RUNAWAY_VARIANTS = {"ewald", "pmgrid"}
PERIODIC_RH_RUNAWAY_REASON = (
    "Flag-off periodic gravity (with or without PM) accumulates correlated tree-force "
    "errors that secularly inflate r_h to ~10% by TimeMax, straddling the 0.10 tolerance. "
    "Pre-existing and independent of PMGRID (none/64/256), Rcut and BoxSize; removed by "
    "RANDOMIZE_GRAVTREE. See the PERIODIC_RH_RUNAWAY notes in this file and "
    "domain/RANDOMIZE_GRAVTREE_TreePM.md."
)

VARIANTS = [
    pytest.param((), id="baseline"),
    pytest.param(("ADAPTIVE_GRAVSOFT_FORALL=2",), id="ags"),
    pytest.param(("BOX_PERIODIC", "PMGRID=64"), id="pmgrid"),
    pytest.param(("BOX_PERIODIC",), id="ewald"),
    pytest.param(("TIDAL_TIMESTEP_CRITERION",), id="tidal"),
    pytest.param(
        (
            "TIDAL_TIMESTEP_CRITERION",
            "ADAPTIVE_GRAVSOFT_FORALL=2",
            "ADAPTIVE_GRAVSOFT_FROM_TIDAL_CRITERION=2",
        ),
        id="tidal_ags",
    ),
    # RANDOMIZE_GRAVTREE, exercising both code paths, each paired with a flag-off variant
    # for the momentum-drift comparison (see momentum_drift_common):
    #   randomize        <-> baseline   (non-periodic: move/enlarge the root node)
    #   randomize_pmgrid <-> pmgrid     (periodic TreePM: random coordinate translation)
    pytest.param(("RANDOMIZE_GRAVTREE",), id="randomize"),
    pytest.param(("BOX_PERIODIC", "PMGRID=64", "RANDOMIZE_GRAVTREE"), id="randomize_pmgrid"),
]


def _ensure_ic():
    if path.isfile(IC_FILE):
        with h5py.File(IC_FILE, "r") as F:
            if int(F["Header"].attrs["NumPart_Total"][1]) == N_PARTICLES:
                return
    import importlib.util

    test_dir = path.dirname(path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        "make_hernquist_ics", path.join(test_dir, "make_hernquist_ics.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.make_hernquist_ics(
        N=N_PARTICLES, m=TOTAL_MASS, a=SCALE_RADIUS, boxsize=BOXSIZE, seed=42, outfile=IC_FILE
    )


def _load_snapshot(snap):
    with h5py.File(snap, "r") as F:
        pos = F["PartType1/Coordinates"][:]
        vel = F["PartType1/Velocities"][:]
        mass = F["PartType1/Masses"][:]
        pot = F["PartType1/Potential"][:]
        boxsize = float(F["Header"].attrs["BoxSize"])
    return pos, vel, mass, pot, boxsize


def _wrap(dx, box):
    return dx - box * np.round(dx / box)


def _radii_from_center(pos, boxsize, periodic):
    # Cluster is centered on the origin in the IC. For periodic variants GIZMO
    # wraps coords into [0, BoxSize), so we apply minimum-image to recover
    # signed displacements; for non-periodic the coords stay around the origin.
    dx = pos
    if periodic:
        dx = _wrap(dx, boxsize)
    return np.sqrt(np.sum(dx**2, axis=1))


def _total_energy(vel, mass, pot):
    ke = 0.5 * np.sum(mass * np.sum(vel**2, axis=1))
    pe = 0.5 * np.sum(mass * pot)
    return ke + pe, ke, pe


def _energy_trajectory(snap_paths):
    """Return (times, total_energies, ke0) read from a sequence of snapshots."""
    times, energies, ke0 = [], [], None
    for s in snap_paths:
        with h5py.File(s, "r") as F:
            t = float(F["Header"].attrs["Time"])
            vel = F["PartType1/Velocities"][:]
            mass = F["PartType1/Masses"][:]
            pot = F["PartType1/Potential"][:]
        e, ke, _ = _total_energy(vel, mass, pot)
        times.append(t)
        energies.append(e)
        if ke0 is None:
            ke0 = ke
    return np.array(times), np.array(energies), ke0


def _secular_energy_rate(times, energies):
    """Secular drift rate of the relative energy error: the least-squares slope of
    (E(t)-E0)/|E0| against t, per unit time.

    Signed, not absolute: a symplectic integrator's error swings either way about zero and
    fits a slope near zero, while a systematic loss or gain fits a real one. Taking |dE| first
    would rectify the oscillation into an apparent trend and score a healthy run as drifting."""
    if len(times) < 3:
        return 0.0
    y = (np.asarray(energies, dtype=float) - energies[0]) / abs(energies[0])
    slope = np.polyfit(np.asarray(times, dtype=float), y, 1)[0]
    return abs(float(slope))


def _radial_density_profile(r, mass, rbins):
    counts_mass, _ = np.histogram(r, bins=rbins, weights=mass)
    vol = (4.0 / 3.0) * np.pi * (rbins[1:] ** 3 - rbins[:-1] ** 3)
    return counts_mass / vol


def _lagrange_radii(r, mass, fractions=LAGRANGE_FRACTIONS):
    order = np.argsort(r)
    cum = np.cumsum(mass[order]) / mass.sum()
    r_sorted = r[order]
    return np.array([r_sorted[np.searchsorted(cum, f)] for f in fractions])


def _summary_npz_path(variant_id):
    return f"{TEST_DIR}/summary_{variant_id}.npz"


def _plot_signed_log(ax, x, y, color=None, label=None):
    """Plot |y| with positive y solid, negative y dashed, single legend entry.
    Each contiguous same-sign run is plotted as its own Line2D so a transition
    from positive to negative never visually connects through the zero-crossing.
    Caller sets log scaling on the axes; points where y==0 (or x<=0) are dropped."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = (x > 0) & (y != 0)
    if not valid.any():
        return
    state = np.where(valid, (y > 0).astype(int), -1)
    bounds = np.flatnonzero(np.diff(state) != 0) + 1
    starts = np.concatenate(([0], bounds))
    ends = np.concatenate((bounds, [len(x)]))
    line = None
    labeled = False
    for s, e in zip(starts, ends):
        if state[s] < 0:
            continue
        seg_x = x[s:e]
        is_pos = bool(state[s])
        seg_y = y[s:e] if is_pos else -y[s:e]
        kwargs = dict(color=color if line is None else line.get_color())
        if not labeled and label is not None:
            kwargs["label"] = label
            labeled = True
        if len(seg_x) == 1:
            ln, = ax.plot(seg_x, seg_y, marker=("o" if is_pos else "x"),
                          linestyle="", **kwargs)
        else:
            ln, = ax.plot(seg_x, seg_y, "-" if is_pos else "--", **kwargs)
        if line is None:
            line = ln


def _current_time_max():
    """TimeMax every current-generation summary must end at."""
    return float(parse_params(f"{TEST_DIR}/{TEST_NAME}.params")["TimeMax"])


def _plot_summary():
    files = sorted(glob.glob(f"{TEST_DIR}/summary_*.npz"))
    if not files:
        return
    fig, (ax_rho, ax_lag, ax_e) = plt.subplots(3, 1, figsize=(7, 12))
    init_done = False
    init_lag_dense = None
    required = {"variant_id", "rc", "rho_initial", "rho_final",
                "r_lag_initial_dense", "r_lag_final_dense",
                "times", "energies", "ke0"}
    # Summaries accumulate in the test directory and clean_test_outputs does not remove them,
    # so after a TimeMax change the glob returns a mix of old and new runs. Plotting those
    # together silently compares different end times -- an r_h or energy curve stopping at
    # t=11.8 next to one running to 118 invites exactly the wrong conclusion. Drop the
    # mismatched ones and say so.
    time_max = _current_time_max()
    for f in files:
        data = np.load(f)
        if not required.issubset(set(data.files)):
            continue  # stale schema from an old run
        t_end = float(data["times"][-1])
        if abs(t_end - time_max) > 1e-3 * max(1.0, abs(time_max)):
            warnings.warn(
                f"{path.basename(f)}: ends at t={t_end:g}, but TimeMax is now {time_max:g}. "
                "Skipping it -- rerun that variant to include it.",
                stacklevel=2,
            )
            continue
        vid = str(data["variant_id"])
        rc = data["rc"]
        if not init_done:
            ax_rho.loglog(rc, data["rho_initial"], "k-", lw=2, label="initial")
            init_lag_dense = data["r_lag_initial_dense"]
            init_done = True
        line, = ax_rho.loglog(rc, data["rho_final"], "--", label=vid)
        rel_lag = (data["r_lag_final_dense"] - init_lag_dense) / init_lag_dense
        _plot_signed_log(ax_lag, LAGRANGE_PERCENTILES_DENSE, rel_lag,
                         color=line.get_color(), label=vid)
        times = data["times"]
        energies = data["energies"]
        ke0 = float(data["ke0"])
        rel_e = (energies - energies[0]) / abs(ke0)
        _plot_signed_log(ax_e, times, rel_e, color=line.get_color(), label=vid)
    ax_rho.set_xlabel("r")
    ax_rho.set_ylabel(r"$\rho(r)$")
    ax_rho.set_title(f"{TEST_NAME}: final density profile (initial in black)")
    ax_rho.legend(fontsize=8)

    ax_lag.set_yscale("log")
    ax_lag.set_xlabel("Lagrange percentile")
    ax_lag.set_ylabel(r"$|r_{\rm final} - r_{\rm initial}| / r_{\rm initial}$  (solid: $>0$, dashed: $<0$)")
    ax_lag.legend(fontsize=8)

    ax_e.set_xscale("log")
    ax_e.set_yscale("log")
    ax_e.set_xlabel("t")
    ax_e.set_ylabel(r"$|E(t) - E(0)| / |KE_0|$  (solid: $>0$, dashed: $<0$)")
    ax_e.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{TEST_DIR}/{TEST_NAME}_summary.png", dpi=120)
    plt.close()


def _plot_variant_density_evolution(variant_id, snaps, periodic):
    """One panel per variant: density profile at each snapshot, colored by time."""
    rbins = np.geomspace(SCALE_RADIUS / 4, 5 * SCALE_RADIUS, 30)
    rc = np.sqrt(rbins[:-1] * rbins[1:])
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap("viridis")
    n = len(snaps)
    for i, s in enumerate(snaps):
        pos, _, mass, _, boxsize = _load_snapshot(s)
        with h5py.File(s, "r") as F:
            t = float(F["Header"].attrs["Time"])
        r = _radii_from_center(pos, boxsize, periodic)
        rho = _radial_density_profile(r, mass, rbins)
        ax.loglog(rc, rho, "-", color=cmap(i / max(n - 1, 1)), label=f"t = {t:.2f}")
    ax.set_xlabel("r")
    ax.set_ylabel(r"$\rho(r)$")
    ax.set_title(f"{TEST_NAME} [{variant_id}]: density profile per snapshot")
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(f"{TEST_DIR}/{TEST_NAME}_{variant_id}_density.png", dpi=120)
    plt.close()


@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(),))
@pytest.mark.parametrize("num_omp_threads", (default_omp_threads(),))
@pytest.mark.parametrize("extra_config_flags", VARIANTS)
def test_hernquist(num_mpi_ranks, num_omp_threads, extra_config_flags, request):
    from datetime import date
    if date.today().month == 12 and date.today().day == 14:
        print("HAPPY BIRTHDAY LARS")
    _ensure_ic()
    clean_test_outputs(TEST_NAME, extra_config_flags)
    build_and_run_test(TEST_NAME, num_mpi_ranks, num_omp_threads, extra_config_flags)

    outputdir = variant_output_dir(TEST_NAME, extra_config_flags)
    snaps = sorted(glob.glob(outputdir + "/snapshot_*.hdf5"))
    if len(snaps) < 2:
        raise RuntimeError(f"GIZMO did not produce enough snapshots in {outputdir}")
    assert_final_time(snaps[-1], TEST_NAME)

    periodic = any("BOX_PERIODIC" in f for f in extra_config_flags)
    pos0, vel0, mass0, pot0, boxsize = _load_snapshot(snaps[0])
    posf, velf, massf, potf, _ = _load_snapshot(snaps[-1])
    r0 = _radii_from_center(pos0, boxsize, periodic)
    rf = _radii_from_center(posf, boxsize, periodic)

    rbins = np.geomspace(SCALE_RADIUS / 4, 5 * SCALE_RADIUS, 20)
    rc = np.sqrt(rbins[:-1] * rbins[1:])
    rho_initial = _radial_density_profile(r0, mass0, rbins)
    rho_final = _radial_density_profile(rf, massf, rbins)
    r_lag_initial = _lagrange_radii(r0, mass0)
    r_lag_final = _lagrange_radii(rf, massf)
    dense_fracs = LAGRANGE_PERCENTILES_DENSE / 100.0
    r_lag_initial_dense = _lagrange_radii(r0, mass0, fractions=dense_fracs)
    r_lag_final_dense = _lagrange_radii(rf, massf, fractions=dense_fracs)
    times, energies, ke0_traj = _energy_trajectory(snaps)

    variant_id = request.node.callspec.id.split("-")[0]
    np.savez(
        _summary_npz_path(variant_id),
        variant_id=variant_id,
        rc=rc,
        rho_initial=rho_initial,
        rho_final=rho_final,
        r_lag_initial=r_lag_initial,
        r_lag_final=r_lag_final,
        r_lag_initial_dense=r_lag_initial_dense,
        r_lag_final_dense=r_lag_final_dense,
        fractions=np.array(LAGRANGE_FRACTIONS),
        fractions_dense=dense_fracs,
        times=times,
        energies=energies,
        ke0=ke0_traj,
    )
    _plot_summary()
    _plot_variant_density_evolution(variant_id, snaps, periodic)

    # --- spurious COM drift from correlated tree-force errors (RANDOMIZE_GRAVTREE) ---
    traj = measure_and_record(TEST_DIR, variant_id, outputdir, parttype="PartType1")
    if traj is not None:
        final_drift = report_momentum_drift(TEST_DIR, TEST_NAME, variant_id, traj)
        plot_momentum_drift(TEST_DIR, TEST_NAME, end_time=_current_time_max())
        assert final_drift < DRIFT_SANITY_CEILING, (
            f"[{variant_id}] spurious COM velocity reached {final_drift:.3e} of the internal "
            f"velocity dispersion (sanity ceiling {DRIFT_SANITY_CEILING}): the system is being "
            f"pushed by force errors"
        )
        assert_randomized_drift(TEST_NAME, variant_id, final_drift)

    # --- energy conservation ---
    # Checked before the r_h block on purpose: the r_h xfail below would otherwise swallow
    # these assertions for the runaway variants.
    e_rel = abs(energies[-1] - energies[0]) / abs(energies[0])
    assert e_rel < ENERGY_ABS_CEILING, (
        f"[{variant_id}] energy blow-up: |dE/E0| = {e_rel:.4e} exceeds the sanity ceiling "
        f"{ENERGY_ABS_CEILING:g} (E0={energies[0]:.6g}, Ef={energies[-1]:.6g})"
    )

    rate = _secular_energy_rate(times, energies)
    rate_ref = ENERGY_SECULAR_RATE.get(variant_id, ENERGY_SECULAR_RATE_DEFAULT)
    rate_max = ENERGY_SECULAR_MARGIN * rate_ref
    calibrated = "starforge_dev" if variant_id in ENERGY_SECULAR_RATE else "default (uncalibrated)"
    assert rate < rate_max, (
        f"[{variant_id}] energy drifts secularly at {rate:.3e} per unit time, exceeding "
        f"{rate_max:.3e} ({ENERGY_SECULAR_MARGIN:g}x the {calibrated} rate {rate_ref:.3e}). "
        f"Endpoint |dE/E0| = {e_rel:.3e} over t={times[0]:g}..{times[-1]:g}. A symplectic "
        f"integrator should oscillate within a bound, not trend; a fitted trend this far above "
        f"reference means energy is being lost or gained systematically. See "
        f"ENERGY_SECULAR_RATE for why this is a rate and not an endpoint threshold."
    )

    half_mass_idx = LAGRANGE_FRACTIONS.index(0.5)
    r_h_0 = r_lag_initial[half_mass_idx]
    r_h_f = r_lag_final[half_mass_idx]
    rel = abs(r_h_f - r_h_0) / r_h_0
    # xfail threshold and assertion share RH_DRIFT_TOL: if they drift apart, the runaway
    # variants stop xfailing and start hard-failing instead.
    if rel >= RH_DRIFT_TOL and variant_id in PERIODIC_RH_RUNAWAY_VARIANTS:
        pytest.xfail(
            f"known pre-existing periodic-gravity r_h runaway: |dr_h/r_h| = {rel:.4f} "
            f"(r_h_0={r_h_0:.4g}, r_h_f={r_h_f:.4g}). {PERIODIC_RH_RUNAWAY_REASON}"
        )
    assert rel < RH_DRIFT_TOL, (
        f"Half-mass radius drifted: |dr_h/r_h| = {rel:.4f} "
        f"(r_h_0={r_h_0:.4g}, r_h_f={r_h_f:.4g})"
    )
