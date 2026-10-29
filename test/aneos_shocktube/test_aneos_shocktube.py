"""Sod shock tube test using ANEOS tabulated EOS.

Uses an ideal-gas SESAME table (gamma=1.4) to verify that the full
ANEOS pipeline (table loading, T-inversion, EOS dispatch, hydro coupling)
reproduces the standard Sod shock tube solution. Compares against the
same reference PPM solution as the standard shocktube test.

This test validates the ANEOS implementation end-to-end in a case where
the analytic answer is known exactly.
"""

import pytest
import numpy as np
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
import h5py
from os import chdir
from os.path import isfile
from urllib.request import urlretrieve
import subprocess
import sys

from gizmo.test import (
    build_gizmo_for_test,
    run_test,
    clean_test_outputs,
    assert_final_time,
    default_mpi_ranks,
    default_omp_threads,
    get_final_snapshot,
)


WEBSITE = "http://www.tapir.caltech.edu/~phopkins/sims/"


def generate_aneos_table():
    """Generate the ideal-gas SESAME table for this test."""
    subprocess.run(
        [sys.executable, "generate_table.py"],
        check=True,
    )
    assert isfile("ideal_gas_gamma14.sesame"), "Failed to generate ANEOS table"


def add_composition_type_to_ics(ic_file):
    """Add CompositionType=0 to all particles in the IC file (required for EOS_ANEOS)."""
    with h5py.File(ic_file, "r+") as F:
        if "CompositionType" not in F["PartType0"]:
            npart = F["PartType0/Coordinates"].shape[0]
            F["PartType0"].create_dataset(
                "CompositionType", data=np.zeros(npart, dtype=np.int32)
            )


@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(max_ranks=4),))
@pytest.mark.parametrize("num_omp_threads", (default_omp_threads(),))
def test_aneos_shocktube(num_mpi_ranks, num_omp_threads):
    test_name = "aneos_shocktube"
    clean_test_outputs(test_name)
    build_gizmo_for_test(test_name, num_omp_threads)
    chdir(f"test/{test_name}/")

    # Generate the ANEOS table
    generate_aneos_table()

    # Download ICs (same as standard shocktube) and exact solution
    for f in ("shocktube_ics_emass.hdf5", "shocktube_exact.txt"):
        if not isfile(f):
            urlretrieve(WEBSITE + f, f)

    # Add CompositionType field to ICs
    add_composition_type_to_ics("shocktube_ics_emass.hdf5")

    run_test(test_name, num_mpi_ranks, num_omp_threads)
    chdir("../../")

    final_snap = get_final_snapshot(test_name)
    assert_final_time(final_snap, test_name)

    # Load simulation data
    with h5py.File(final_snap, "r") as F:
        x_sim = F["PartType0/Coordinates"][:, 0]
        rho_sim = F["PartType0/Density"][:]
        vel_sim = F["PartType0/Velocities"][:, 0]
        u_sim = F["PartType0/InternalEnergy"][:]

    # Load exact solution: x, density, pressure, entropy, x_velocity
    exact = np.loadtxt(f"test/{test_name}/shocktube_exact.txt")
    x_exact = exact[:, 0]
    rho_exact = exact[:, 1]
    vel_exact = exact[:, 4]
    P_exact = exact[:, 2]
    gamma = 1.4
    u_exact = P_exact / (rho_exact * (gamma - 1))

    # Interpolate exact solution to particle positions
    rho_interp = interp1d(x_exact, rho_exact, bounds_error=False, fill_value="extrapolate")(x_sim)
    vel_interp = interp1d(x_exact, vel_exact, bounds_error=False, fill_value="extrapolate")(x_sim)
    u_interp = interp1d(x_exact, u_exact, bounds_error=False, fill_value="extrapolate")(x_sim)

    # Compute L1 errors
    L1_rho = np.mean(np.abs(rho_sim - rho_interp)) / np.mean(np.abs(rho_interp))
    L1_vel = np.mean(np.abs(vel_sim - vel_interp)) / (np.mean(np.abs(vel_interp)) + 1e-10)
    L1_u = np.mean(np.abs(u_sim - u_interp)) / np.mean(np.abs(u_interp))

    # Plot comparison
    order = x_sim.argsort()
    for label, sim, exact_vals in [
        ("Density", rho_sim, rho_interp),
        ("Velocity", vel_sim, vel_interp),
        ("InternalEnergy", u_sim, u_interp),
    ]:
        plt.figure()
        plt.plot(x_sim[order], sim[order], ".", markersize=1, label="GIZMO (ANEOS)")
        plt.plot(x_sim[order], exact_vals[order], "-", color="red", linewidth=0.5, label="Exact")
        plt.xlabel("x")
        plt.ylabel(label)
        plt.legend()
        plt.savefig(f"test/{test_name}/{label}.png")
        plt.close()

    # Allow slightly looser tolerances than the standard shocktube since
    # ANEOS adds interpolation error from the table lookup + T-inversion
    assert L1_rho < 0.08, f"Density L1 error {L1_rho:.4f} exceeds tolerance"
    assert L1_vel < 0.15, f"Velocity L1 error {L1_vel:.4f} exceeds tolerance"
    assert L1_u < 0.15, f"Internal energy L1 error {L1_u:.4f} exceeds tolerance"


@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(max_ranks=4),))
@pytest.mark.parametrize("num_omp_threads", (default_omp_threads(),))
def test_aneos_shocktube_uniform_state_entropy(num_mpi_ranks, num_omp_threads):
    """The uniform low-density state ahead of the shock must keep its entropy, as the ideal-gas build does (to ~1e-10 at
    t=5). With the MFM face entropic switch compiled out for ANEOS, the Riemann energy flux grew its spread to ~6e-6."""
    test_name = "aneos_shocktube"
    clean_test_outputs(test_name)
    build_gizmo_for_test(test_name, num_omp_threads)
    chdir(f"test/{test_name}/")
    generate_aneos_table()
    if not isfile("shocktube_ics_emass.hdf5"):
        urlretrieve(WEBSITE + "shocktube_ics_emass.hdf5", "shocktube_ics_emass.hdf5")
    add_composition_type_to_ics("shocktube_ics_emass.hdf5")
    run_test(test_name, num_mpi_ranks, num_omp_threads)
    chdir("../../")

    final_snap = get_final_snapshot(test_name)
    assert_final_time(final_snap, test_name)
    with h5py.File(final_snap, "r") as F:
        x = F["PartType0/Coordinates"][:, 0]
        rho = F["PartType0/Density"][:]
        u = F["PartType0/InternalEnergy"][:]
    gamma = 1.4
    entropy = (gamma - 1) * u * rho ** (1 - gamma)
    untouched = (x > 12) & (x < 28)  # the initial rho=0.25 state, which the waves have not reached by t=5
    assert untouched.sum() > 10
    spread = np.ptp(entropy[untouched]) / np.median(entropy[untouched])
    assert spread < 1e-8, f"entropy spread {spread:.3e} in the uniform state ahead of the shock"
