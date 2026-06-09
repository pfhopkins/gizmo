"""Orszag-Tang MHD vortex test (Hopkins & Raives 2016)

Classic 2D MHD turbulence problem. Tests the development of MHD shocks from
smooth initial conditions. Since there is no exact solution, this test verifies
that the simulation runs and checks energy conservation.
"""

import glob
import pytest
import numpy as np
from matplotlib import pyplot as plt
import h5py

from meshoid import Meshoid
from gizmo.test import (
    build_and_run_test,
    clean_test_outputs,
    flush_colorbar,
    assert_final_time,
    default_mpi_ranks,
    default_omp_threads,
    variant_output_dir,
)


def _energy_series(snaps):
    """Return (time, E_thermal, E_kinetic, E_magnetic) arrays from a list of snapshots.
    Magnetic energy uses Heaviside-Lorentz code units: u_mag = B^2/2."""
    t, E_th, E_kin, E_mag = [], [], [], []
    for snap in snaps:
        with h5py.File(snap, "r") as F:
            mass = F["PartType0/Masses"][:]
            u = F["PartType0/InternalEnergy"][:]
            vel = F["PartType0/Velocities"][:]
            B = F["PartType0/MagneticField"][:]
            rho = F["PartType0/Density"][:]
            t.append(float(F["Header"].attrs["Time"]))
        E_th.append(np.sum(mass * u))
        E_kin.append(0.5 * np.sum(mass * np.sum(vel**2, axis=1)))
        E_mag.append(np.sum(0.5 * np.sum(B**2, axis=1) * mass / rho))
    return np.array(t), np.array(E_th), np.array(E_kin), np.array(E_mag)


def plot_orszag_tang_energies(test_name):
    """Two-panel plot: (left) thermal/kinetic/magnetic energy vs time per variant,
    (right) signed total-energy relative error vs time per variant."""
    test_dir = f"test/{test_name}"
    variant_dirs = sorted(glob.glob(f"{test_dir}/output*"))
    if not variant_dirs:
        return
    components = [("thermal", "-"), ("kinetic", "--"), ("magnetic", ":")]
    cmap = plt.get_cmap("tab10")
    fig, (ax, ax_err) = plt.subplots(1, 2, figsize=(13, 5))
    for i, vdir in enumerate(variant_dirs):
        snaps = sorted(glob.glob(vdir + "/snapshot_*.hdf5"))
        if len(snaps) < 2:
            continue
        t, E_th, E_kin, E_mag = _energy_series(snaps)
        label = vdir.split("/output", 1)[1].lstrip("_") or "baseline"
        color = cmap(i % 10)
        for series, (_, ls) in zip((E_th, E_kin, E_mag), components):
            ax.plot(t, series, ls=ls, color=color, label=f"{label} ({_})")
        E_tot = E_th + E_kin + E_mag
        rel_err = (E_tot - E_tot[0]) / np.abs(E_tot[0])
        ax_err.plot(t, rel_err, color=color, label=label)

    from matplotlib.lines import Line2D

    variant_handles = [
        Line2D([0], [0], color=cmap(i % 10), lw=2,
               label=(vdir.split("/output", 1)[1].lstrip("_") or "baseline"))
        for i, vdir in enumerate(variant_dirs)
    ]
    component_handles = [Line2D([0], [0], color="k", ls=ls, lw=2, label=name) for name, ls in components]
    leg1 = ax.legend(handles=variant_handles, title="variant", loc="upper left", fontsize=8)
    ax.add_artist(leg1)
    ax.legend(handles=component_handles, title="component", loc="upper right", fontsize=8)
    ax.set_xlabel("Time")
    ax.set_ylabel("Energy")
    ax.set_yscale("log")
    ax.set_title("Energy components vs time")

    ax_err.axhline(0, color="k", lw=0.5)
    ax_err.set_xlabel("Time")
    ax_err.set_ylabel(r"$(E_{\rm tot}(t) - E_{\rm tot}(0)) / |E_{\rm tot}(0)|$")
    ax_err.set_title("Total energy relative error")
    ax_err.legend(title="variant", loc="best", fontsize=8)

    fig.suptitle("Orszag-Tang")
    fig.tight_layout()
    fig.savefig(f"{test_dir}/Energies.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_orszag_tang_divB_panels(test_name):
    """3-panel slice colormap of |divB|*h/|B| across the three MHD-cleaning variants,
    on a shared color scale."""
    test_dir = f"test/{test_name}"
    variant_dirs = sorted(glob.glob(f"{test_dir}/output*"))
    panels = []
    for vdir in variant_dirs:
        snaps = sorted(glob.glob(vdir + "/snapshot_*.hdf5"))
        if not snaps:
            continue
        with h5py.File(snaps[-1], "r") as F:
            if "PartType0/DivergenceOfMagneticField" not in F:
                continue
            pos = F["PartType0/Coordinates"][:]
            divB = F["PartType0/DivergenceOfMagneticField"][:]
            B = F["PartType0/MagneticField"][:]
            mass = F["PartType0/Masses"][:]
            rho = F["PartType0/Density"][:]
        Bmag = np.sqrt(np.sum(B**2, axis=1))
        h = (mass / rho) ** (1.0 / 2.0)  # 2D: cell size from area
        field = np.log10(np.abs(divB) * h / np.maximum(Bmag, 1e-30) + 1e-10)
        M = Meshoid(pos, boxsize=1.0)
        sl = M.Slice(field, res=1024, plane="z", center=np.array([0.5, 0.5, 0.5]), size=1.0, order=0)
        label = vdir.split("/output", 1)[1].lstrip("_") or "baseline"
        panels.append((label, sl))
    if not panels:
        return
    vmin = min(np.nanmin(sl) for _, sl in panels)
    vmax = max(np.nanmax(sl) for _, sl in panels)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    im = None
    for ax, (label, sl) in zip(axes, panels):
        im = ax.imshow(sl.T, origin="lower", cmap="inferno", extent=[0, 1, 0, 1], vmin=vmin, vmax=vmax)
        ax.set_title(label)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    for ax in axes[len(panels):]:
        ax.axis("off")
    fig.suptitle(r"Orszag-Tang — $\log_{10}(|\nabla\!\cdot\!B|\,h/|B|)$")
    fig.tight_layout(rect=[0, 0, 0.92, 0.95])
    cax = fig.add_axes([0.93, 0.1, 0.015, 0.8])
    fig.colorbar(im, cax=cax)
    fig.savefig(f"{test_dir}/DivB_panels.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(),))
@pytest.mark.parametrize("num_omp_threads", (default_omp_threads(),))
@pytest.mark.parametrize(
    "extra_config_flags",
    ((), ("MHD_CONSTRAINED_GRADIENT=1",), ("MHD_MODIFIED_GRADIENT",)),
)
def test_orszag_tang(num_mpi_ranks, num_omp_threads, extra_config_flags):
    test_name = "orszag_tang"
    clean_test_outputs(test_name, extra_config_flags)
    build_and_run_test(test_name, num_mpi_ranks, num_omp_threads, extra_config_flags)

    outputdir = variant_output_dir(test_name, extra_config_flags)
    final_snap = sorted(glob.glob(outputdir + "/snapshot_*.hdf5"))[-1]
    init_snap = outputdir + "/snapshot_000.hdf5"
    assert_final_time(final_snap, test_name)

    def compute_total_energy(snapfile):
        with h5py.File(snapfile, "r") as F:
            mass = F["PartType0/Masses"][:]
            vel = F["PartType0/Velocities"][:]
            u = F["PartType0/InternalEnergy"][:]
            B = F["PartType0/MagneticField"][:]
            rho = F["PartType0/Density"][:]
        KE = 0.5 * np.sum(mass[:, None] * vel**2)
        TE = np.sum(mass * u)
        # Heaviside-Lorentz code units: u_mag = B^2/2; cell volume = mass/rho.
        ME = np.sum(0.5 * np.sum(B**2, axis=1) * mass / rho)
        return KE + TE + ME

    E_init = compute_total_energy(init_snap)
    E_final = compute_total_energy(final_snap)

    # Plot density at final time using Meshoid slice interpolation
    with h5py.File(final_snap, "r") as F:
        coords = F["PartType0/Coordinates"][:]
        rho = F["PartType0/Density"][:]
    M = Meshoid(coords, boxsize=1.0)
    rho_slice = M.Slice(np.log10(rho), res=1024, plane="z", center=np.array([0.5, 0.5, 0.5]), size=1.0, order=0)

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(
        rho_slice.T,
        origin="lower",
        cmap="viridis",
        extent=[coords[:, 0].min(), coords[:, 0].max(), coords[:, 1].min(), coords[:, 1].max()],
    )
    flush_colorbar(im, ax=ax, label="log10(Density)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    suffix = ("_" + "__".join("".join(c if c.isalnum() else "_" for c in f) for f in extra_config_flags)) if extra_config_flags else ""
    fig.savefig(f"test/{test_name}/Density_2D{suffix}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    plot_orszag_tang_divB_panels(test_name)
    plot_orszag_tang_energies(test_name)

    # Energy should be conserved to within ~10% (shock dissipation is expected)
    dE = abs(E_final - E_init) / abs(E_init)
    assert dE < 0.1, f"Total energy changed by {dE:.4f} (>10%)"
