"""Plummer-sphere gravity-only equilibrium stability test.

Runs the same Plummer IC through five different gravity-solver variants and
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

import h5py
import numpy as np
import pytest
from matplotlib import pyplot as plt

from gizmo.test import (
    build_and_run_test,
    clean_test_outputs,
    default_mpi_ranks,
    default_omp_threads,
    variant_output_dir,
    assert_final_time,
)

TEST_NAME = "plummer"
TEST_DIR = f"test/{TEST_NAME}"
IC_FILE = f"{TEST_DIR}/{TEST_NAME}_ics.hdf5"
SCALE_RADIUS = 1.0
TOTAL_MASS = 1.0
BOXSIZE = 300.0  # = 2 * r_max where r_max = 100 r_h, so cluster fits at corner
N_PARTICLES = 2**15
LAGRANGE_FRACTIONS = (0.1, 0.5, 0.9)
LAGRANGE_TOL = (0.20, 0.15, 0.30)  # |dr/r| for 10%, 50%, 90% Lagrange radii
LAGRANGE_PERCENTILES_DENSE = np.arange(1, 100, dtype=float)  # 1..99 % for plot only


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
]


def _ensure_ic():
    if path.isfile(IC_FILE):
        with h5py.File(IC_FILE, "r") as F:
            if int(F["Header"].attrs["NumPart_Total"][1]) == N_PARTICLES:
                return
    import importlib.util

    test_dir = path.dirname(path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        "make_plummer_ics", path.join(test_dir, "make_plummer_ics.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.make_plummer_ics(
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
    # 1 = positive, 0 = negative, -1 = invalid; segment boundaries at any change
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
    for f in files:
        data = np.load(f)
        if not required.issubset(set(data.files)):
            continue  # stale schema from an old run
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
def test_plummer(num_mpi_ranks, num_omp_threads, extra_config_flags, request):
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

    e0, ke0, pe0 = _total_energy(vel0, mass0, pot0)
    ef, _, _ = _total_energy(velf, massf, potf)
    rel_e_err = abs(ef - e0) / abs(ke0)
    assert rel_e_err < 0.04, (
        f"Energy not conserved: |dE|/KE_0 = {rel_e_err:.4f} "
        f"(E0={e0:.4g}, Ef={ef:.4g}, KE0={ke0:.4g}, PE0={pe0:.4g})"
    )

    rel = np.abs(r_lag_final - r_lag_initial) / r_lag_initial
    failures = [
        f"r_{int(f * 100)}: |dr/r|={rel[i]:.3f} (tol {LAGRANGE_TOL[i]}); "
        f"r0={r_lag_initial[i]:.4g}, rf={r_lag_final[i]:.4g}"
        for i, f in enumerate(LAGRANGE_FRACTIONS) if rel[i] > LAGRANGE_TOL[i]
    ]
    assert not failures, "Lagrange radii drifted:\n  " + "\n  ".join(failures)
