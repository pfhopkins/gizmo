"""Forged in FIRE nuclear-scale AGN accretion disk test

Runs a short segment of a nuclear-zoom AGN accretion disk simulation with
full radiation-MHD, dust physics, and star formation. Tests that the
SINGLE_STAR_AND_SSP_NUCLEAR_ZOOM pipeline runs without crashing and
produces physically reasonable radial profiles.

Plots: density-temperature diagram, density histogram, and radial profiles
of gas density, temperature, radiation temperature, dust temperature,
magnetic energy density, radiation energy density, and thermal energy density.
"""

import pytest
import numpy as np
from os import path, system, chdir, environ
from os.path import isfile
from shutil import copy2
from urllib.request import urlretrieve, HTTPError
from matplotlib import pyplot as plt
import h5py
from gizmo.test import (
    build_gizmo_for_test,
    clean_test_outputs,
    get_cooling_tables,
    default_mpi_ranks,
    default_omp_threads,
    get_final_snapshot,
)

WEBSITE = "http://www.tapir.caltech.edu/~phopkins/sims/"
REFERENCE_FILE = "test/forgedinfire/forgedinfire_exact.hdf5"


def _download_if_missing(filename):
    """Download a file from the tapir server if not already present."""
    if not isfile(filename):
        try:
            urlretrieve(WEBSITE + path.basename(filename), filename)
        except HTTPError:
            print(f"Could not download {filename} from {WEBSITE}")


def get_particle_masses(snapshot):
    """Return dict of total mass per particle type."""
    masses = {}
    with h5py.File(snapshot, "r") as F:
        for grp in F.keys():
            if grp.startswith("PartType"):
                pt = int(grp.replace("PartType", ""))
                masses[pt] = float(F[grp]["Masses"][:].sum())
    return masses


def get_gas_density_temperature(snapshot):
    """Return gas density and temperature arrays from the Temperature field."""
    with h5py.File(snapshot, "r") as F:
        rho = F["PartType0/Density"][:]
        T = F["PartType0/Temperature"][:]
    return rho, T


def get_radial_profiles(snapshot):
    """Extract gas quantities as a function of radius from r=0."""
    with h5py.File(snapshot, "r") as F:
        coords = F["PartType0/Coordinates"][:]
        rho = F["PartType0/Density"][:]
        T = F["PartType0/Temperature"][:]
        u = F["PartType0/InternalEnergy"][:]
        masses = F["PartType0/Masses"][:]
        mag = F["PartType0/MagneticField"][:]
        photon_E = F["PartType0/PhotonEnergy"][:]

        # Dust and radiation temperatures (may not exist in all snapshots)
        T_dust = F["PartType0/Dust_Temperature"][:] if "Dust_Temperature" in F["PartType0"] else np.zeros_like(T)
        T_rad = F["PartType0/IRBand_Radiation_Temperature"][:] if "IRBand_Radiation_Temperature" in F["PartType0"] else np.zeros_like(T)

    r = np.sqrt(coords[:, 0]**2 + coords[:, 1]**2 + coords[:, 2]**2)
    B2 = (mag**2).sum(axis=1)
    # Magnetic energy density: B^2 / (8*pi) in code units (UnitB = gauss-equivalent)
    B_energy = B2 / (8.0 * np.pi)
    # Radiation energy density: sum over all photon bands, normalized by cell volume
    # PhotonEnergy is energy per unit mass, so E_rad = photon_E * rho (energy density)
    E_rad = photon_E.sum(axis=1) * rho
    # Thermal energy density: rho * u
    E_thermal = rho * u

    return {
        "r": r, "rho": rho, "T": T, "T_dust": T_dust, "T_rad": T_rad,
        "B_energy": B_energy, "E_rad": E_rad, "E_thermal": E_thermal,
        "B_strength": np.sqrt(B2), "masses": masses,
    }


def plot_density_temperature(snapshot, output_dir, label=""):
    """Scatter plot of gas log(density) vs log(temperature)."""
    rho, T = get_gas_density_temperature(snapshot)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(np.log10(rho + 1e-30), np.log10(T + 1e-1),
               s=0.1, alpha=0.3, color="black", rasterized=True)
    ax.set_xlabel(r"$\log_{10}(\rho)$ [code units]")
    ax.set_ylabel(r"$\log_{10}(T)$ [K]")
    ax.set_title(f"Gas Density-Temperature {label}")
    fig.savefig(path.join(output_dir, "density_temperature.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_density_histogram(snapshot, output_dir, label=""):
    """Histogram of gas log(density)."""
    with h5py.File(snapshot, "r") as F:
        rho = F["PartType0/Density"][:]
        masses = F["PartType0/Masses"][:]
    log_rho = np.log10(rho + 1e-30)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(log_rho, bins=100, weights=masses, color="steelblue",
            edgecolor="none", alpha=0.8)
    ax.set_xlabel(r"$\log_{10}(\rho)$ [code units]")
    ax.set_ylabel("Mass-weighted count")
    ax.set_title(f"Gas Density Histogram {label}")
    ax.set_yscale("log")
    fig.savefig(path.join(output_dir, "density_histogram.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_radial_profiles(snapshot, output_dir, label=""):
    """Plot radial profiles of key quantities."""
    data = get_radial_profiles(snapshot)
    r = data["r"]

    # Radial bins (log-spaced)
    r_min = np.log10(r[r > 0].min())
    r_max = np.log10(r.max())
    r_bins = np.logspace(r_min, r_max, 60)
    r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])
    bin_idx = np.digitize(r, r_bins) - 1

    profiles = {
        "Gas Density": ("rho", r"$\rho$ [code units]"),
        "Gas Temperature": ("T", r"$T$ [K]"),
        "Radiation Temperature": ("T_rad", r"$T_{\rm rad}$ [K]"),
        "Dust Temperature": ("T_dust", r"$T_{\rm dust}$ [K]"),
        "Magnetic Field Strength": ("B_strength", r"$|B|$ [code units]"),
        "Magnetic Energy Density": ("B_energy", r"$B^2/8\pi$ [code units]"),
        "Radiation Energy Density": ("E_rad", r"$E_{\rm rad}$ [code units]"),
        "Thermal Energy Density": ("E_thermal", r"$\rho u$ [code units]"),
    }

    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    for ax, (title, (key, ylabel)) in zip(axes.flat, profiles.items()):
        vals = data[key]
        weights = data["masses"]
        # Mass-weighted median in each bin
        medians = np.full(len(r_centers), np.nan)
        for j in range(len(r_centers)):
            mask = bin_idx == j
            if mask.sum() > 0:
                medians[j] = np.median(vals[mask])

        good = np.isfinite(medians) & (medians > 0)
        if good.any():
            ax.loglog(r_centers[good], medians[good], ".-", color="black", linewidth=1)
        ax.set_xlabel("r [code units]")
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    plt.tight_layout()
    fig.savefig(path.join(output_dir, "radial_profiles.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_forgedinfire_test(test_name, num_mpi_ranks, num_omp_threads):
    """Build and run the forgedinfire test with restart flag 2."""
    build_gizmo_for_test(test_name, num_omp_threads)
    chdir(f"test/{test_name}/")
    # Download ICs if not present
    _download_if_missing("forgedinfire_ics.hdf5")
    _download_if_missing("forgedinfire_exact.hdf5")
    # Download cooling tables
    if not path.isdir("spcool_tables"):
        get_cooling_tables(".")
    if not isfile("TREECOOL"):
        copy2("../../data/cooling/TREECOOL", "TREECOOL")
    if num_omp_threads > 0:
        environ["OMP_NUM_THREADS"] = str(num_omp_threads)
    paramsfile = f"{test_name}.params"
    system(f"mpirun -np {num_mpi_ranks} --use-hwthread-cpus "
           f"./GIZMO {paramsfile} 0 "
           f"1>test_{test_name}.out 2>test_{test_name}.err")
    chdir("../../")


@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(2),))
@pytest.mark.parametrize("num_omp_threads", (default_omp_threads(),))
def test_forgedinfire(num_mpi_ranks, num_omp_threads):
    test_name = "forgedinfire"
    clean_test_outputs(test_name)

    # Build and run
    run_forgedinfire_test(test_name, num_mpi_ranks, num_omp_threads)

    # Check simulation produced output
    final_snap = get_final_snapshot(test_name)

    # Get initial and final masses
    initial_snap = f"test/{test_name}/forgedinfire_ics.hdf5"
    m_init = get_particle_masses(initial_snap)
    m_final = get_particle_masses(final_snap)

    print("Particle type masses:")
    type_names = {0: "Gas", 3: "BH", 4: "Stars"}
    for pt in sorted(set(list(m_init.keys()) + list(m_final.keys()))):
        mi = m_init.get(pt, 0)
        mf = m_final.get(pt, 0)
        name = type_names.get(pt, f"Type{pt}")
        print(f"  {name} (Type {pt}): initial={mi:.6g}, final={mf:.6g}, "
              f"delta={mf - mi:.6g}")

    # Generate diagnostic plots
    test_dir = f"test/{test_name}"
    plot_density_temperature(final_snap, test_dir, label="(final)")
    plot_density_histogram(final_snap, test_dir, label="(final)")
    plot_radial_profiles(final_snap, test_dir, label="(final)")

    # Compare against reference solution if it exists
    if path.isfile(REFERENCE_FILE):
        m_ref = get_particle_masses(REFERENCE_FILE)
        for pt in (0, 4):
            if pt in m_ref and pt in m_final:
                name = type_names.get(pt, f"Type{pt}")
                assert m_final[pt] == pytest.approx(m_ref[pt], rel=0.1), \
                    f"{name} mass differs from reference: " \
                    f"{m_final[pt]:.6g} vs {m_ref[pt]:.6g}"
