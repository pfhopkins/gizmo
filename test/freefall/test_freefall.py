"""freefall: pressureless collapse of a cold uniform gas sphere (MFM + self-gravity).

Motivated by OpenGadget3's FreefallCollapse CI test (Dolag et al. 2026, arXiv:2609.17658 sec 2.5).
Code units G = M = R = 1. Until shell crossing the collapse is homologous: a shell starting at
r0 enclosing mass M(<r0) follows r = r0 cos^2(beta), t = t_ff (2/pi) (beta + sin(2 beta)/2), with
t_ff = (pi/2) sqrt(r0^3 / (2 G M(<r0))) = pi/(2 sqrt 2) for every shell of the uniform sphere.

Checks (see README.md for what each is sensitive to):
  (a) t=0 gravitational acceleration against the analytic interior field -G M r/R^3 (binned), its
      tangential residual, and particle-by-particle against a softened direct sum (pytreegrav)
      with the softening GIZMO reports having used -- the last isolates tree error from sampling;
  (b) Lagrange radii r_10, r_50, r_90 against the analytic homologous solution;
  (c) total energy (thermal + kinetic + potential);
  (d) spurious centre-of-mass drift (momentum_drift_common).
Plus a setup check that pressure is negligible, which (a)-(c) assume.

Every tolerance below is PROVISIONAL: none has been calibrated against a run.
"""

import glob
import os
import sys
from os import path

import h5py
import numpy as np
import pytest
from matplotlib import pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from momentum_drift_common import (  # noqa: E402
    DRIFT_SANITY_CEILING, assert_randomized_drift, measure_and_record, plot_momentum_drift,
    report_momentum_drift,
)

from gizmo.test import (  # noqa: E402
    assert_energy_conserved,
    assert_final_time,
    build_and_run_test,
    clean_test_outputs,
    default_mpi_ranks,
    default_omp_threads,
    gas_energy,
    parse_params,
    variant_output_dir,
)

TEST_NAME = "freefall"
TEST_DIR = f"test/{TEST_NAME}"
IC_FILE = f"{TEST_DIR}/{TEST_NAME}_ics.hdf5"
PARAMS = parse_params(path.join(path.dirname(path.abspath(__file__)), f"{TEST_NAME}.params"))

N_PARTICLES = 2**15
TOTAL_MASS = 1.0
RADIUS = 1.0
G = float(PARAMS["GravityConstantInternal"])
T_FF = 0.5 * np.pi * np.sqrt(RADIUS**3 / (2.0 * G * TOTAL_MASS))  # = pi/(2 sqrt 2) = 1.1107
# Plummer-equivalent -> kernel support radius for the cubic spline (KERNEL_FAC_FROM_FORCESOFT_TO_PLUMMER)
KERNEL_SUPPORT_PER_PLUMMER = 2.8

LAGRANGE_FRACTIONS = (0.1, 0.5, 0.9)
# r_90 sits within one softening kernel of the surface, where the softened field falls below the
# point-mass one, and the kernel grows relative to the sphere as it collapses: it lags by design.
LAGRANGE_TOL = (0.05, 0.05, 0.10)  # PROVISIONAL: untuned, not yet run
ACCEL_WINDOW = (0.2, 0.9)  # r/R: inside the edge kernel, and away from the centre where a -> 0
ACCEL_ANALYTIC_L1_TOL = 0.01  # PROVISIONAL: untuned, not yet run
ACCEL_TANGENTIAL_TOL = 0.02  # PROVISIONAL: untuned, not yet run
# a few x ErrTolForceAcc (0.0025): the relative opening criterion bounds each node's error by
# ErrTolForceAcc |a|, and the errors of the accepted nodes partly cancel
DIRECT_SUM_RMS_TOL = 0.01  # PROVISIONAL: untuned, not yet run
ENERGY_TOL = 0.02  # PROVISIONAL: untuned, not yet run
HYDRO_TO_GRAV_TOL = 0.01  # PROVISIONAL: untuned, not yet run

VARIANTS = [
    pytest.param((), id="baseline"),
    pytest.param(("ADAPTIVE_GRAVSOFT_FORGAS",), id="ags"),
    pytest.param(("TIDAL_TIMESTEP_CRITERION", "ADAPTIVE_TREEFORCE_UPDATE=0.06"), id="tidal_adaptive"),
]
FIXED_SOFTENING_VARIANTS = {"baseline", "tidal_adaptive"}
# Under ADAPTIVE_GRAVSOFT_FORGAS the tree floors a gas target's softening at 2.8 x SofteningGas
# (gravity/forcetree.cc), so the AGS variant runs with a negligible SofteningGas, as evrard does.
AGS_PARAM_OVERRIDES = {"SofteningGas": 1e-4}


def _ensure_ic():
    if path.isfile(IC_FILE):
        with h5py.File(IC_FILE, "r") as F:
            if int(F["Header"].attrs["NumPart_Total"][0]) == N_PARTICLES:
                return
    import importlib.util

    test_dir = path.dirname(path.abspath(__file__))
    spec = importlib.util.spec_from_file_location("make_freefall_ics", path.join(test_dir, "make_freefall_ics.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.make_freefall_ics(IC_FILE, N=N_PARTICLES, M=TOTAL_MASS, R=RADIUS, seed=42)


# ----------------------------------------------------------------------------- analytic solution
_BETA = np.linspace(0.0, 0.5 * np.pi, 200001)
_TAU = (2.0 / np.pi) * (_BETA + 0.5 * np.sin(2.0 * _BETA))  # t/t_ff, monotonic 0 -> 1
_XFAC = np.cos(_BETA) ** 2


def freefall_radius_factor(t, t_ff=T_FF):
    """r(t)/r0 for pressureless homologous collapse; valid for 0 <= t < t_ff."""
    tau = np.asarray(t, dtype=float) / t_ff
    if np.any(tau >= 1.0):
        raise ValueError("free-fall solution requested at or past t_ff")
    return np.interp(tau, _TAU, _XFAC)


def shell_freefall_time(r0, m_enc):
    """Free-fall time of a shell of radius r0 enclosing mass m_enc."""
    return 0.5 * np.pi * np.sqrt(r0**3 / (2.0 * G * m_enc))


def analytic_accel(pos, m_tot=TOTAL_MASS, R=RADIUS):
    """Field of a uniform sphere: -G M r/R^3 inside, -G M r/|r|^3 outside."""
    r = np.sqrt(np.sum(pos**2, axis=1))
    coef = np.where(r < R, G * m_tot / R**3, G * m_tot / np.maximum(r, 1e-300) ** 3)
    return -coef[:, None] * pos


# ----------------------------------------------------------------------------- snapshot helpers
def _load(snap, *fields):
    with h5py.File(snap, "r") as F:
        t = float(F["Header"].attrs["Time"])
        out = []
        for f in fields:
            key = f"PartType0/{f}"
            if key not in F:
                raise KeyError(f"{snap}: {key} absent; check Config.sh output flags")
            out.append(F[key][:].astype(float))
    return (t, *out)


def _com_radii(pos, mass):
    com = np.sum(mass[:, None] * pos, axis=0) / mass.sum()
    return np.sqrt(np.sum((pos - com) ** 2, axis=1))


def _lagrange_radii(r, mass, fractions=LAGRANGE_FRACTIONS):
    order = np.argsort(r)
    cum = np.cumsum(mass[order]) / mass.sum()
    r_sorted = r[order]
    return np.array([r_sorted[np.searchsorted(cum, f)] for f in fractions])


def _rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


# ----------------------------------------------------------------------------- (a) t=0 forces
def _initial_force_metrics(snap0):
    """Compare snapshot_000's gravitational acceleration with the analytic field and (if pytreegrav
    is importable) a softened direct sum. Acceleration is gravity only here because the Config sets
    OUTPUT_HYDROACCELERATION as well (file_io/io.cc, IO_ACCEL); non-cosmological, so code units."""
    t0, pos, mass, acc, soft = _load(snap0, "Coordinates", "Masses", "Acceleration", "Softening_KernelRadius")
    assert t0 == 0.0, f"first snapshot is at t={t0}, not t=0: TimeOfFirstSnapshot must be 0"
    r = np.sqrt(np.sum(pos**2, axis=1))  # IC centred on the origin, nothing has moved yet
    rhat = pos / r[:, None]
    a_r = np.sum(acc * rhat, axis=1)
    a_t = np.sqrt(np.maximum(np.sum(acc**2, axis=1) - a_r**2, 0.0))
    a_an_r = np.sum(analytic_accel(pos, mass.sum()) * rhat, axis=1)

    window = (r > ACCEL_WINDOW[0] * RADIUS) & (r < ACCEL_WINDOW[1] * RADIUS)
    edges = np.linspace(ACCEL_WINDOW[0], ACCEL_WINDOW[1], 15) * RADIUS
    idx = np.digitize(r, edges) - 1
    rc = 0.5 * (edges[1:] + edges[:-1])
    bin_dev = np.array([np.mean(a_r[idx == i] / a_an_r[idx == i]) - 1.0 for i in range(len(rc))])
    metrics = {
        "rc": rc,
        "bin_dev": bin_dev,
        "analytic_L1": float(np.mean(np.abs(bin_dev))),
        "tangential_ratio": _rms(a_t[window]) / _rms(a_r[window]),
        "soft": soft,
    }

    try:
        from pytreegrav import Accel
    except ImportError:
        metrics["direct"] = None
        return metrics
    # pytreegrav's softening is the compact-support radius of an M4 cubic spline, as is
    # Softening_KernelRadius (ForceSoftening_KernelRadius, the value GIZMO's tree used), and both
    # resolve unequal pairs with the larger of the two softenings. For the AGS variant this is the
    # gas kernel radius; the AGS zeta correction is absent from the t=0 force (zeta is zeroed in
    # init and first set by the hydro gradient pass, which runs after the t=0 gravity call), so the
    # direct sum is the like-for-like reference for every variant. The kernel radius reported is
    # the one re-converged after that gravity call, at unchanged positions.
    direct = Accel(pos, mass, softening=soft, G=G, method="bruteforce", parallel=True)
    sel = r > ACCEL_WINDOW[0] * RADIUS
    rel = np.linalg.norm(acc - direct, axis=1) / np.linalg.norm(direct, axis=1)
    rel_edges = np.linspace(ACCEL_WINDOW[0], 1.0, 17) * RADIUS
    ridx = np.digitize(r, rel_edges) - 1
    metrics["direct"] = {
        "rms": _rms(rel[sel]),
        "max": float(rel[sel].max()),
        "rc": 0.5 * (rel_edges[1:] + rel_edges[:-1]),
        "rms_binned": np.array([_rms(rel[ridx == i]) if np.any(ridx == i) else np.nan
                                for i in range(len(rel_edges) - 1)]),
        # tree-free check of the analytic field: separates sampling/softening from tree error
        "analytic_L1_direct": float(np.mean(np.abs(np.array([
            np.mean(np.sum(direct[idx == i] * rhat[idx == i], axis=1) / a_an_r[idx == i]) - 1.0
            for i in range(len(rc))])))),
    }
    return metrics


# ----------------------------------------------------------------------------- (b), (c) trajectories
def _trajectories(snaps):
    times, r_lag, energies, hydro_ratio = [], [], [], []
    for s in snaps:
        t, pos, mass, a_grav, a_hydro = _load(s, "Coordinates", "Masses", "Acceleration", "HydroAcceleration")
        times.append(t)
        r_lag.append(_lagrange_radii(_com_radii(pos, mass), mass))
        energies.append(gas_energy(s, include_potential=True))
        hydro_ratio.append(_rms(np.linalg.norm(a_hydro, axis=1)) / _rms(np.linalg.norm(a_grav, axis=1)))
    times = np.array(times)
    r_lag = np.array(r_lag)
    m_enc = np.array(LAGRANGE_FRACTIONS) * TOTAL_MASS
    t_ff_shell = shell_freefall_time(r_lag[0], m_enc)  # exact for any spherical profile pre-crossing
    r_lag_an = np.array([r_lag[0] * freefall_radius_factor(t, t_ff_shell) for t in times])
    return times, r_lag, r_lag_an, t_ff_shell, np.array(energies), np.array(hydro_ratio)


# ----------------------------------------------------------------------------- plots
def _summary_npz_path(variant_id):
    return f"{TEST_DIR}/summary_{variant_id}.npz"


def _plot_summary():
    data = {}
    for f in sorted(glob.glob(f"{TEST_DIR}/summary_*.npz")):
        d = np.load(f)
        data[str(d["variant_id"])] = d
    if not data:
        return []
    colors = {vid: f"C{i}" for i, vid in enumerate(sorted(data))}
    outs = []

    # Lagrange radii against the analytic solution, and their relative error
    fig, (ax, ax_err) = plt.subplots(2, 1, figsize=(6, 7), sharex=True, gridspec_kw={"height_ratios": (2, 1)})
    ref = next(iter(data.values()))
    t_dense = np.linspace(0, float(ref["times"][-1]), 400)
    for k in range(len(LAGRANGE_FRACTIONS)):
        ax.plot(t_dense / T_FF, ref["r_lag"][0, k] * freefall_radius_factor(t_dense, ref["t_ff_shell"][k]),
                "k-", lw=1, label="analytic" if k == 0 else None)
    markers = ("o", "s", "^")
    for vid, d in data.items():
        for k, frac in enumerate(LAGRANGE_FRACTIONS):
            ax.plot(d["times"] / T_FF, d["r_lag"][:, k], markers[k], ms=4, mfc="none", color=colors[vid],
                    label=vid if k == 0 else None)
            ax_err.plot(d["times"] / T_FF, d["r_lag"][:, k] / d["r_lag_an"][:, k] - 1, "-" + markers[k],
                        ms=3, color=colors[vid])
    ax.set_ylabel(r"$r_{10},\ r_{50},\ r_{90}$")
    ax.legend(fontsize=8, frameon=False)
    ax_err.axhline(0, color="k", lw=0.5)
    ax_err.set_xlabel(r"$t / t_{\rm ff}$")
    ax_err.set_ylabel(r"$r / r_{\rm analytic} - 1$")
    fig.tight_layout()
    outs.append(f"{TEST_DIR}/{TEST_NAME}_lagrange.png")
    fig.savefig(outs[-1], dpi=120)
    plt.close(fig)

    # energy
    fig, ax = plt.subplots(figsize=(6, 4))
    for vid, d in data.items():
        e = d["energies"]
        ax.plot(d["times"] / T_FF, (e - e[0]) / abs(e[0]), "-o", ms=3, color=colors[vid], label=vid)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel(r"$t / t_{\rm ff}$")
    ax.set_ylabel(r"$(E - E_0) / |E_0|$")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    outs.append(f"{TEST_DIR}/{TEST_NAME}_energy.png")
    fig.savefig(outs[-1], dpi=120)
    plt.close(fig)

    # t=0 acceleration: binned against the analytic field, per particle against the direct sum
    fig, (ax_an, ax_ds) = plt.subplots(1, 2, figsize=(10, 4))
    for vid, d in data.items():
        ax_an.plot(d["accel_rc"], d["accel_bin_dev"], "-o", ms=3, color=colors[vid], label=vid)
        if "direct_rc" in d.files:
            ax_ds.semilogy(d["direct_rc"], d["direct_rms_binned"], "-o", ms=3, color=colors[vid], label=vid)
    ax_an.axhline(0, color="k", lw=0.5)
    ax_an.set_xlabel("r / R")
    ax_an.set_ylabel(r"$\langle a_r / a_{\rm analytic} \rangle - 1$  (t = 0)")
    ax_an.legend(fontsize=8, frameon=False)
    ax_ds.axhline(DIRECT_SUM_RMS_TOL, color="k", lw=0.5, ls="--")
    ax_ds.set_xlabel("r / R")
    ax_ds.set_ylabel(r"rms $|a - a_{\rm direct}| / |a_{\rm direct}|$  (t = 0)")
    fig.tight_layout()
    outs.append(f"{TEST_DIR}/{TEST_NAME}_accel.png")
    fig.savefig(outs[-1], dpi=120)
    plt.close(fig)
    return outs


# ----------------------------------------------------------------------------- the test
@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(),))
@pytest.mark.parametrize("num_omp_threads", (default_omp_threads(),))
@pytest.mark.parametrize("extra_config_flags", VARIANTS)
def test_freefall(num_mpi_ranks, num_omp_threads, extra_config_flags, request):
    _ensure_ic()
    clean_test_outputs(TEST_NAME, extra_config_flags)
    overrides = AGS_PARAM_OVERRIDES if "ADAPTIVE_GRAVSOFT_FORGAS" in extra_config_flags else None
    build_and_run_test(TEST_NAME, num_mpi_ranks, num_omp_threads, extra_config_flags, param_overrides=overrides)

    outputdir = variant_output_dir(TEST_NAME, extra_config_flags)
    snaps = sorted(glob.glob(f"{outputdir}/snapshot_*.hdf5"))
    if len(snaps) < 2:
        raise RuntimeError(f"GIZMO did not produce enough snapshots in {outputdir}")
    assert_final_time(snaps[-1], TEST_NAME)
    variant_id = request.node.callspec.id.split("-")[0]

    # ---- measure everything, write the summary and plots, then assert
    force = _initial_force_metrics(snaps[0])
    times, r_lag, r_lag_an, t_ff_shell, energies, hydro_ratio = _trajectories(snaps)
    lag_err = r_lag[-1] / r_lag_an[-1] - 1.0

    summary = dict(variant_id=variant_id, times=times, r_lag=r_lag, r_lag_an=r_lag_an, t_ff_shell=t_ff_shell,
                   energies=energies, hydro_ratio=hydro_ratio, accel_rc=force["rc"], accel_bin_dev=force["bin_dev"])
    if force["direct"] is not None:
        summary.update(direct_rc=force["direct"]["rc"], direct_rms_binned=force["direct"]["rms_binned"])
    np.savez(_summary_npz_path(variant_id), **summary)
    for p in _plot_summary():
        print(f"[freefall] plot: {p}")

    print(f"\n[freefall] {variant_id}: t=0 analytic L1 = {force['analytic_L1']:.3e}, "
          f"tangential/radial = {force['tangential_ratio']:.3e}")
    if force["direct"] is not None:
        ds = force["direct"]
        print(f"[freefall] {variant_id}: vs direct sum rms = {ds['rms']:.3e}, max = {ds['max']:.3e}; "
              f"direct sum vs analytic L1 = {ds['analytic_L1_direct']:.3e} (sampling + softening floor)")
    for frac, tff_s, err in zip(LAGRANGE_FRACTIONS, t_ff_shell, lag_err):
        print(f"[freefall] {variant_id}: r_{int(100 * frac)} t_ff,shell = {tff_s:.4f}, final error = {err:+.3e}")
    print(f"[freefall] {variant_id}: (E_f - E_0)/|E_0| = {(energies[-1] - energies[0]) / abs(energies[0]):+.3e}, "
          f"max rms|a_hydro|/rms|a_grav| = {hydro_ratio.max():.3e}")

    # ---- setup validity: the analytic solution is pressureless
    assert hydro_ratio.max() < HYDRO_TO_GRAV_TOL, (
        f"[{variant_id}] rms hydro/gravity acceleration reached {hydro_ratio.max():.3e} "
        f"(tolerance {HYDRO_TO_GRAV_TOL}): the gas is not cold enough for the pressureless solution, "
        f"so the force and Lagrange-radius checks below are not testing what they claim")

    # ---- (a) t=0 forces
    if variant_id in FIXED_SOFTENING_VARIANTS:
        expected = KERNEL_SUPPORT_PER_PLUMMER * float(PARAMS["SofteningGas"])
        assert np.allclose(force["soft"], expected, rtol=1e-6), (
            f"[{variant_id}] Softening_KernelRadius spans [{force['soft'].min():.4g}, {force['soft'].max():.4g}], "
            f"expected 2.8 x SofteningGas = {expected:.4g}: the fixed-softening convention has changed")
    assert force["analytic_L1"] < ACCEL_ANALYTIC_L1_TOL, (
        f"[{variant_id}] t=0 radial acceleration deviates from -GMr/R^3 by binned L1 "
        f"{force['analytic_L1']:.3e} over r/R in {ACCEL_WINDOW} (tolerance {ACCEL_ANALYTIC_L1_TOL})")
    assert force["tangential_ratio"] < ACCEL_TANGENTIAL_TOL, (
        f"[{variant_id}] t=0 tangential acceleration is {force['tangential_ratio']:.3e} of the radial "
        f"(rms, r/R in {ACCEL_WINDOW}; tolerance {ACCEL_TANGENTIAL_TOL}): the field is not central")
    if force["direct"] is not None:
        assert force["direct"]["rms"] < DIRECT_SUM_RMS_TOL, (
            f"[{variant_id}] t=0 tree force deviates from the softened direct sum by rms "
            f"{force['direct']['rms']:.3e} (max {force['direct']['max']:.3e}) for r/R > {ACCEL_WINDOW[0]} "
            f"(tolerance {DIRECT_SUM_RMS_TOL}; ErrTolForceAcc = {PARAMS['ErrTolForceAcc']})")

    # ---- (b) Lagrange radii at the final snapshot
    for frac, err, tol in zip(LAGRANGE_FRACTIONS, lag_err, LAGRANGE_TOL):
        assert abs(err) < tol, (
            f"[{variant_id}] r_{int(100 * frac)} at t = {times[-1]:.4g} ({times[-1] / T_FF:.3f} t_ff) is "
            f"{err:+.3e} off the analytic free-fall solution (tolerance {tol})")

    # ---- (c) energy. With ADAPTIVE_GRAVSOFT_FORGAS the softened potential is not a fixed
    # functional of the positions (the softening shrinks with the collapse), so the budget closes
    # only as well as the zeta correction makes it; an ags-only failure is that, not a regression.
    assert_energy_conserved(TEST_NAME, extra_config_flags, tol=ENERGY_TOL, include_potential=True)

    # ---- (d) spurious COM drift: the sphere starts at rest, so any drift is correlated force error
    traj = measure_and_record(TEST_DIR, variant_id, outputdir, parttype="PartType0")
    if traj is not None:
        final_drift = report_momentum_drift(TEST_DIR, TEST_NAME, variant_id, traj)
        plot_momentum_drift(TEST_DIR, TEST_NAME)
        assert final_drift < DRIFT_SANITY_CEILING, (
            f"[{variant_id}] spurious COM velocity reached {final_drift:.3e} of the gravitational velocity "
            f"(sanity ceiling {DRIFT_SANITY_CEILING}): the system is being pushed by force errors")
        assert_randomized_drift(TEST_NAME, variant_id, final_drift)

    if force["direct"] is None:
        pytest.skip("all other checks passed; direct-sum force comparison skipped: pytreegrav not installed")
