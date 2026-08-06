"""Evrard adiabatic collapse test (Hopkins 2015)

Compares the radial density, entropy, and velocity profiles at t=0.8 against
a high-resolution reference solution.
The exact solution file has columns: radius, density, entropy, velocity.
"""

import os
import sys

import pytest
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import binned_statistic
from matplotlib import pyplot as plt
import h5py

from meshoid import Meshoid
from gizmo.test import build_and_run_test, default_mpi_ranks, flush_colorbar, assert_final_time, get_final_snapshot, default_omp_threads, variant_output_dir, assert_energy_conserved

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from momentum_drift_common import (  # noqa: E402
    DRIFT_SANITY_CEILING, assert_randomized_drift, measure_and_record, plot_momentum_drift,
    report_momentum_drift,
)


_variant_profiles = {}
_VARIANT_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
_VARIANT_COLORS = ["C0", "C2", "C1", "C3", "C4", "C5", "C6", "C7"]


def _variant_label(extra_config_flags):
    return "+".join(extra_config_flags) if extra_config_flags else "baseline"


def _short(label, maxlen=30):
    return label if len(label) <= maxlen else label[: maxlen - 3] + "..."


def plot_evrard_density_slice(coords, rho, output_dir="."):
    """Plot a density slice through the Evrard collapse center."""
    M = Meshoid(coords)
    center = np.average(coords, axis=0)
    rho_slice = M.Slice(np.log10(rho), res=1024, plane="z", center=center, size=1., order=0)
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(rho_slice.T, origin="lower", cmap="inferno", extent=[-0.5, 0.5, -0.5, 0.5])
    flush_colorbar(im, ax=ax, label="log10(Density)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Evrard Collapse - Density Slice")
    fig.savefig(output_dir + "/Density_2D.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


@pytest.mark.parametrize("num_mpi_ranks,num_omp_threads", [(default_mpi_ranks(), default_omp_threads())])
@pytest.mark.parametrize(
    "extra_config_flags",
    [
        (),
        ("TIDAL_TIMESTEP_CRITERION", "ADAPTIVE_TREEFORCE_UPDATE=0.06"),
        # RANDOMIZE_GRAVTREE (non-periodic path). Evrard is initially at rest with zero net
        # momentum by symmetry, so correlated force errors show up as spurious COM drift.
        ("RANDOMIZE_GRAVTREE",),
    ],
    ids=["baseline", "tidal_adaptive", "randomize"],
)
def test_evrard(num_mpi_ranks, num_omp_threads, extra_config_flags, request):
    test_name = "evrard"
    build_and_run_test(test_name, num_mpi_ranks, num_omp_threads, extra_config_flags)

    final_snap = get_final_snapshot(test_name, extra_config_flags)
    assert_final_time(final_snap, test_name)

    # Load simulation data
    with h5py.File(final_snap, "r") as F:
        coords = F["PartType0/Coordinates"][:]
        rho_sim = F["PartType0/Density"][:]
        vel = F["PartType0/Velocities"][:]
        u_sim = F["PartType0/InternalEnergy"][:]

    # Compute radius from origin and radial velocity
    r_sim = np.sqrt(np.sum(coords**2, axis=1))
    vr_sim = np.sum(vel * coords, axis=1) / (r_sim + 1e-30)

    # Entropy: S = P / rho^gamma = (gamma-1) * u / rho^(gamma-1)
    gamma = 5.0 / 3.0
    entropy_sim = (gamma - 1) * u_sim / rho_sim ** (gamma - 1)

    # Load exact solution: radius, density, entropy, velocity
    exact = np.loadtxt(f"test/{test_name}/evrard_exact.txt")
    r_exact = exact[:, 0]
    rho_exact = exact[:, 1]
    entropy_exact = exact[:, 2]
    vr_exact = exact[:, 3]

    # Bin simulation data by radius
    r_bins = np.logspace(np.log10(r_exact.min()), np.log10(r_exact.max()), 30)
    rho_binned = binned_statistic(r_sim, rho_sim, "median", r_bins)[0]
    vr_binned = binned_statistic(r_sim, vr_sim, "median", r_bins)[0]
    entropy_binned = binned_statistic(r_sim, entropy_sim, "median", r_bins)[0]
    r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])

    # Interpolate exact solution to bin centers
    rho_exact_interp = interp1d(r_exact, rho_exact, bounds_error=False, fill_value="extrapolate")(r_centers)
    vr_exact_interp = interp1d(r_exact, vr_exact, bounds_error=False, fill_value="extrapolate")(r_centers)

    plot_evrard_density_slice(coords, rho_sim, output_dir=f"test/{test_name}")

    # Accumulate this variant's binned profiles, then re-render combined plots
    _variant_profiles[_variant_label(extra_config_flags)] = {
        "r": r_centers,
        "Density": rho_binned,
        "RadialVelocity": vr_binned,
    }
    for label, exact_vals, log in [
        ("Density", rho_exact_interp, True),
        ("RadialVelocity", vr_exact_interp, False),
    ]:
        plt.figure()
        plotter = plt.loglog if log else plt.semilogx
        plotter(r_centers, exact_vals, "-", color="black", label="Exact")
        for i, (vlabel, prof) in enumerate(_variant_profiles.items()):
            plotter(prof["r"], prof[label], _VARIANT_MARKERS[i % len(_VARIANT_MARKERS)],
                    markersize=max(10 - 2 * i, 4), markerfacecolor="none", alpha=0.85,
                    color=_VARIANT_COLORS[i % len(_VARIANT_COLORS)], label=_short(vlabel))
        plt.xlabel("r")
        plt.ylabel(label)
        plt.legend(fontsize="x-small", loc="best")
        plt.savefig(f"test/{test_name}/{label}.png", bbox_inches="tight")
        plt.close()

    # --- spurious COM drift from correlated tree-force errors (RANDOMIZE_GRAVTREE) ---
    variant_id = request.node.callspec.id.split("-")[0]
    traj = measure_and_record(f"test/{test_name}", variant_id,
                              variant_output_dir(test_name, extra_config_flags),
                              parttype="PartType0")
    if traj is not None:
        final_drift = report_momentum_drift(f"test/{test_name}", test_name, variant_id, traj)
        plot_momentum_drift(f"test/{test_name}", test_name)
        assert final_drift < DRIFT_SANITY_CEILING, (
            f"[{variant_id}] spurious COM velocity reached {final_drift:.3e} of the internal "
            f"velocity dispersion (sanity ceiling {DRIFT_SANITY_CEILING}): the system is being "
            f"pushed by force errors"
        )
        assert_randomized_drift(test_name, variant_id, final_drift)

    # Check density profile
    good = np.isfinite(rho_binned) & np.isfinite(rho_exact_interp) & (rho_exact_interp > 0)
    if np.any(good):
        L1_rho = np.nanmean(np.abs(np.log10(rho_binned[good]) - np.log10(rho_exact_interp[good])))
        assert L1_rho < 0.3, f"Log density profile L1 error {L1_rho:.4f} exceeds tolerance"

    # Energy conservation. Evrard is self-gravitating, so the budget needs the gravitational term
    # (0.5*sum(m*phi)) on top of thermal+kinetic -- hence OUTPUT_POTENTIAL in Config.sh. CAVEAT: the least
    # certain check here, since ADAPTIVE_GRAVSOFT_FORGAS evolves the softening and the potential is then
    # not a fixed functional of the configuration. A baseline failure is the setup, not a regression.
    assert_energy_conserved(test_name, extra_config_flags, tol=0.1, include_potential=True)
