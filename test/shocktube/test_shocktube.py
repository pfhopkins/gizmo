"""Sod shock tube test (Hopkins 2015)

Compares the final snapshot against the reference high-resolution PPM solution.
The exact solution file has columns: x, density, pressure, entropy, x_velocity.
"""

import pytest
import numpy as np
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
import h5py
from glob import glob
from gizmo.test import (build_and_run_test, assert_final_time, default_mpi_ranks, default_omp_threads, get_final_snapshot,
                        variant_output_dir, variant_suffix, gas_conservation_errors)


# Config.sh = meshless finite-mass (MFM); Config_mfv.sh = the same problem with meshless finite-volume (MFV)
@pytest.mark.parametrize("config_file", ("Config.sh", "Config_mfv.sh"))
@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(max_ranks=4),))
@pytest.mark.parametrize("num_omp_threads", (default_omp_threads(),))
def test_shocktube(num_mpi_ranks, num_omp_threads, config_file):
    test_name = "shocktube"
    build_and_run_test(test_name, num_mpi_ranks, num_omp_threads, config_file=config_file)

    final_snap = get_final_snapshot(test_name, config_file=config_file)
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
    # Internal energy from pressure: u = P / (rho * (gamma - 1))
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
        plt.plot(x_sim[order], sim[order], ".", markersize=1, label="GIZMO")
        plt.plot(x_sim[order], exact_vals[order], "-", color="red", linewidth=0.5, label="Exact")
        plt.xlabel("x")
        plt.ylabel(label)
        plt.legend()
        plt.savefig(f"test/{test_name}/{label}{variant_suffix(config_file=config_file)}.png")
        plt.close()

    assert L1_rho < 0.05, f"Density L1 error {L1_rho:.4f} exceeds tolerance"
    assert L1_vel < 0.1, f"Velocity L1 error {L1_vel:.4f} exceeds tolerance"
    assert L1_u < 0.1, f"Internal energy L1 error {L1_u:.4f} exceeds tolerance"

    # Conservation, first snapshot against last (periodic box, no gravity: mass, momentum and energy are all conserved).
    # Energy changes only at time-integration accuracy (~1e-6 here); an MFV that loses energy at its faces showed ~6e-5.
    first_snap = sorted(glob(f"{variant_output_dir(test_name, config_file=config_file)}/snapshot_*.hdf5"))[0]
    err = gas_conservation_errors(first_snap, final_snap)
    assert err["energy"] < 1e-5, f"Total energy changed by {err['energy']:.3e} (relative)"
    assert err["momentum"] < 1e-10, f"Total momentum changed by {err['momentum']:.3e} (in units of sqrt(2 M E))"
    # MFM masses are constant, so their sum is exact. Under MFV the snapshot masses are predicted between kicks
    # (the conserved mass is not written out), so their sum is not checked.
    if config_file == "Config.sh":
        assert err["mass"] < 1e-12, f"Total mass changed by {err['mass']:.3e} (relative)"
