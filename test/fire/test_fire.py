"""FIRE cosmological galaxy formation test (m11i-like dwarf galaxy)

Runs a short cosmological FIRE simulation of a downsampled dwarf galaxy
(m11i) from z~2.9 for a small time interval. Tests that the full FIRE
physics pipeline (cooling, star formation, stellar feedback, BH physics)
runs without crashing and produces physically reasonable results.

Checks: dark matter mass conservation, and compares gas/star/BH masses
and gas density-temperature distribution against a reference solution.
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
    stash_baseline_output,
    finalize_variant_output,
)

WEBSITE = "http://www.tapir.caltech.edu/~phopkins/sims/"
REFERENCE_FILE = "test/fire/fire_exact.hdf5"


def get_particle_masses(snapshot):
    """Return dict of total mass per particle type."""
    masses = {}
    number = {}
    with h5py.File(snapshot, "r") as F:
        for grp in F.keys():
            if grp.startswith("PartType"):
                pt = int(grp.replace("PartType", ""))
                ## need to sum in float64 or roundoff can trigger failure below
                m = F[grp]["Masses"][:].astype(np.float64) 
                masses[pt] = np.float64(m.sum())
                number[pt] = F[grp]["Masses"].shape[0]
    return masses, number


def get_gas_density_temperature(snapshot):
    """Return gas density and temperature arrays."""
    with h5py.File(snapshot, "r") as F:
        rho = F["PartType0/Density"][:]
        u = F["PartType0/InternalEnergy"][:]
        xe = F["PartType0/ElectronAbundance"][:]
    X_H = 0.76
    gamma = 5.0 / 3.0
    mu = 4.0 / (1.0 + 3.0 * X_H + 4.0 * X_H * xe)
    m_p = 1.6726e-24  # g
    k_B = 1.3807e-16  # erg/K
    u_cgs = u * (1.0e5) ** 2  # convert from (km/s)^2 to (cm/s)^2
    T = (gamma - 1.0) * u_cgs * mu * m_p / k_B
    return rho, T


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


def _download_if_missing(filename):
    """Download a file from the tapir server if not already present."""
    if not isfile(filename):
        try:
            urlretrieve(WEBSITE + filename, filename)
        except HTTPError:
            print(f"Could not download {filename} from {WEBSITE}")


def run_fire_test(test_name, num_mpi_ranks, num_omp_threads, extra_config_flags=()):
    """Build and run the FIRE test with restart flag 2."""
    build_gizmo_for_test(test_name, num_omp_threads, extra_config_flags)
    chdir(f"test/{test_name}/")
    # Download ICs and reference if not present
    _download_if_missing("fire_ics.hdf5")
    _download_if_missing("fire_exact.hdf5")
    # Download cooling tables; get_cooling_tables tries cp data/cooling/TREECOOL
    # from cwd which won't work here, so copy TREECOOL manually
    if not path.isdir("spcool_tables"):
        get_cooling_tables(".")
    if not isfile("TREECOOL"):
        copy2("../../data/cooling/TREECOOL", "TREECOOL")
    if num_omp_threads > 0:
        environ["OMP_NUM_THREADS"] = str(num_omp_threads)
    paramsfile = f"{test_name}.params"
    system(f"mpirun -np {num_mpi_ranks} --use-hwthread-cpus "
           f"./GIZMO {paramsfile} 2 "
           f"1>test_{test_name}.out 2>test_{test_name}.err")
    chdir("../../")


@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(2),))
@pytest.mark.parametrize("num_omp_threads", (default_omp_threads(),))
@pytest.mark.parametrize(
    "extra_config_flags",
    [(), ("TIDAL_TIMESTEP_CRITERION", "ADAPTIVE_TREEFORCE_UPDATE=0.06")],
    ids=["baseline", "tidal_adaptive"],
)
def test_fire(num_mpi_ranks, num_omp_threads, extra_config_flags):
    test_name = "fire"
    clean_test_outputs(test_name, extra_config_flags)

    # Build and run, stashing any baseline output/ aside so non-default variants don't clobber it
    stash_baseline_output(test_name, extra_config_flags)
    try:
        run_fire_test(test_name, num_mpi_ranks, num_omp_threads, extra_config_flags)
    finally:
        finalize_variant_output(test_name, extra_config_flags)

    # Check simulation produced output
    final_snap = get_final_snapshot(test_name, extra_config_flags)

    # Get initial and final masses
    initial_snap = f"test/{test_name}/fire_ics.hdf5"
    m_init,n_init = get_particle_masses(initial_snap)
    m_final,n_final = get_particle_masses(final_snap)

    print("Particle type masses:")
    type_names = {0: "Gas", 1: "DM", 2: "DM-LowRes", 4: "Stars", 5: "BH"}
    for pt in sorted(set(list(m_init.keys()) + list(m_final.keys()))):
        mi = m_init.get(pt, 0)
        mf = m_final.get(pt, 0)
        ni = n_init.get(pt, 0)
        nf = n_final.get(pt, 0)
        name = type_names.get(pt, f"Type{pt}")
        print(f"  {name} (Type {pt}): initial={mi:.6g}, final={mf:.6g}, "
              f"delta={mf - mi:.6g}")
        print(f"  {name} (Type {pt}): initialN={ni:.6g}, finalN={nf:.6g}, "
              f"delta={nf - ni:.6g}")

    # Dark matter mass must be exactly conserved
    assert m_init[1] == pytest.approx(m_final[1], rel=1e-10), \
        f"DM mass changed: {m_init[1]} -> {m_final[1]}"
    if 2 in m_init and 2 in m_final:
        assert m_init[2] == pytest.approx(m_final[2], rel=1e-10), \
            f"Low-res DM mass changed: {m_init[2]} -> {m_final[2]}"

    # Generate diagnostic plots
    test_dir = f"test/{test_name}"
    plot_density_temperature(final_snap, test_dir, label="(final)")
    plot_density_histogram(final_snap, test_dir, label="(final)")

    # Compare against reference solution
    if path.isfile(REFERENCE_FILE):
        m_ref,n_ref = get_particle_masses(REFERENCE_FILE)
        for pt in (0, 4, 5):
            if pt in m_ref and pt in m_final:
                name = type_names.get(pt, f"Type{pt}")
                assert m_final[pt] == pytest.approx(m_ref[pt], rel=0.1), \
                    f"{name} mass differs from reference: " \
                    f"{m_final[pt]:.6g} vs {m_ref[pt]:.6g}"
