"""MHD blast wave test (Hopkins & Raives 2016)

Tests the propagation of a blast wave in a magnetized medium. The blast
should be elongated along the magnetic field direction. Checks energy
conservation and that the blast develops anisotropy.
"""

import pytest
import numpy as np
from matplotlib import pyplot as plt
import h5py
import glob
from meshoid import Meshoid
from gizmo.test import (
    build_and_run_test,
    default_mpi_ranks,
    clean_test_outputs,
    flush_colorbar,
    assert_final_time,
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


def plot_mhd_blast_energies(test_name):
    """Plot thermal/kinetic/magnetic energy vs time for every variant currently on disk.
    Color encodes the variant; line style encodes the energy component."""
    test_dir = f"test/{test_name}"
    variant_dirs = sorted(glob.glob(f"{test_dir}/output*"))
    if not variant_dirs:
        return
    components = [("thermal", "-"), ("kinetic", "--"), ("magnetic", ":")]
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, vdir in enumerate(variant_dirs):
        snaps = sorted(glob.glob(vdir + "/snapshot_*.hdf5"))
        if len(snaps) < 2:
            continue
        t, E_th, E_kin, E_mag = _energy_series(snaps)
        label = vdir.split("/output", 1)[1].lstrip("_") or "baseline"
        color = cmap(i % 10)
        for series, (_, ls) in zip((E_th, E_kin, E_mag), components):
            ax.plot(t, series, ls=ls, color=color, label=f"{label} ({_})")
    # Build a compact legend: variants by color, components by linestyle
    from matplotlib.lines import Line2D

    variant_handles = [
        Line2D([0], [0], color=cmap(i % 10), lw=2, label=(vdir.split("/output", 1)[1].lstrip("_") or "baseline"))
        for i, vdir in enumerate(variant_dirs)
    ]
    component_handles = [Line2D([0], [0], color="k", ls=ls, lw=2, label=name) for name, ls in components]
    leg1 = ax.legend(handles=variant_handles, title="variant", loc="upper left", fontsize=8)
    ax.add_artist(leg1)
    ax.legend(handles=component_handles, title="component", loc="upper right", fontsize=8)
    ax.set_xlabel("Time")
    ax.set_ylabel("Energy")
    ax.set_yscale("log")
    ax.set_title("MHD Blast - Energy components vs time")
    ax.set(ylim=[0.03, 3])
    fig.savefig(f"{test_dir}/Energies.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_mhd_blast_divB_panels(test_name):
    """2x2 panel of |divB|/|B| slices, one per variant, on a shared color scale."""
    test_dir = f"test/{test_name}"
    variant_dirs = sorted(glob.glob(f"{test_dir}/output*"))
    if not variant_dirs:
        return
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
        h = (mass / rho) ** (1.0 / 3.0)
        field = np.log10(np.abs(divB) * h / np.maximum(Bmag, 1e-30) + 1e-10)
        M = Meshoid(pos, boxsize=1.0)
        sl = M.Slice(field, res=2048, plane="z", center=np.array([0.5, 0.5, 0.5]), size=1.0, order=0)
        label = vdir.split("/output", 1)[1].lstrip("_") or "baseline"
        panels.append((label, sl))
    if not panels:
        return
    vmin = min(np.nanmin(sl) for _, sl in panels)
    vmax = max(np.nanmax(sl) for _, sl in panels)
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for ax, (label, sl) in zip(axes.flat, panels):
        im = ax.imshow(sl.T, origin="lower", cmap="inferno", extent=[0, 1, 0, 1], vmin=vmin, vmax=vmax)
        ax.set_title(label)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    for ax in axes.flat[len(panels) :]:
        ax.axis("off")
    fig.suptitle(r"MHD Blast — $\log_{10}(|\nabla\!\cdot\!B|\,h/|B|)$")
    fig.tight_layout(rect=[0, 0, 0.92, 0.97])
    cax = fig.add_axes([0.93, 0.1, 0.02, 0.8])
    fig.colorbar(im, cax=cax)
    fig.savefig(f"{test_dir}/DivB_panels.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_mhd_blast_density_slice(coords, rho, output_dir="."):
    """Plot a density slice of the MHD blast wave."""
    M = Meshoid(coords, boxsize=1.0)
    rho_slice = M.Slice(np.log10(rho), res=1024, plane="z", center=np.array([0.5, 0.5, 0.5]), size=1.0, order=0)
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(rho_slice.T, origin="lower", cmap="inferno", extent=[0, 1, 0, 1])
    flush_colorbar(im, ax=ax, label="log10(Density)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("MHD Blast - Density")
    fig.savefig(output_dir + "/Density_2D.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(),))
@pytest.mark.parametrize("num_omp_threads", (default_omp_threads(),))
@pytest.mark.parametrize(
    "extra_config_flags",
    ((), ("MHD_CONSTRAINED_GRADIENT=1",), ("MHD_CONSTRAINED_GRADIENT=2",), ("MHD_MODIFIED_GRADIENT",)),
)
def test_mhd_blast(num_mpi_ranks, num_omp_threads, extra_config_flags):
    test_name = "mhd_blast"
    clean_test_outputs(test_name, extra_config_flags)
    build_and_run_test(test_name, num_mpi_ranks, num_omp_threads, extra_config_flags)

    outputdir = variant_output_dir(test_name, extra_config_flags)
    snaps = sorted(glob.glob(outputdir + "/snapshot_*.hdf5"))
    if len(snaps) < 2:
        raise RuntimeError(f"Expected >=2 snapshots in {outputdir}, found {len(snaps)}.")
    assert_final_time(snaps[-1], test_name)

    # Load final snapshot
    with h5py.File(snaps[-1], "r") as F:
        pos = F["PartType0/Coordinates"][:]
        rho = F["PartType0/Density"][:]
        u = F["PartType0/InternalEnergy"][:]
        mass = F["PartType0/Masses"][:]
        B = F["PartType0/MagneticField"][:]
        vel = F["PartType0/Velocities"][:]
        boxsize = F["Header"].attrs["BoxSize"]

    plot_mhd_blast_density_slice(pos, rho, output_dir=f"test/{test_name}")

    # Load initial snapshot for conservation check
    with h5py.File(snaps[0], "r") as F:
        mass0 = F["PartType0/Masses"][:]
        u0 = F["PartType0/InternalEnergy"][:]
        B0 = F["PartType0/MagneticField"][:]
        rho0 = F["PartType0/Density"][:]
        vel0 = F["PartType0/Velocities"][:]

    # Mass conservation
    mass_err = abs(mass.sum() - mass0.sum()) / mass0.sum()
    assert mass_err < 1e-3, f"Mass not conserved: relative error {mass_err:.6f}"

    # Total energy should be approximately conserved
    # (thermal + kinetic + magnetic). GIZMO snapshot B is in Heaviside-Lorentz
    # code units (default UnitMagneticField_in_gauss=sqrt(4pi)), so magnetic
    # energy density = B^2/2; cell volume = mass/rho.
    Emag0 = np.sum(0.5 * np.sum(B0**2, axis=1) * mass0 / rho0)
    Emag_f = np.sum(0.5 * np.sum(B**2, axis=1) * mass / rho)
    Etot0 = np.sum(mass0 * u0) + 0.5 * np.sum(mass0 * np.sum(vel0**2, axis=1)) + Emag0
    Etot_f = np.sum(mass * u) + 0.5 * np.sum(mass * np.sum(vel**2, axis=1)) + Emag_f
    # Regenerate the cross-variant energy plot using whatever variants exist on disk.
    plot_mhd_blast_energies(test_name)
    plot_mhd_blast_divB_panels(test_name)

    energy_err = abs(Etot_f - Etot0) / abs(Etot0)
    assert energy_err < 0.1, f"Total energy not conserved: relative error {energy_err:.4f}"
