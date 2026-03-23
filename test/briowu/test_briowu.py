"""Brio & Wu MHD shock tube test (Hopkins & Raives 2016)

Classic MHD shock tube problem testing the MHD Riemann solver. Since there is no
provided exact solution file, this test verifies that the simulation runs to
completion and produces output with physically reasonable values.
"""

import pytest
import numpy as np
from matplotlib import pyplot as plt
import h5py
from os import path
from gizmo.test import build_and_run_test, default_mpi_ranks


@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(),))
def test_briowu(num_mpi_ranks):
    test_name = "briowu"
    build_and_run_test(test_name, num_mpi_ranks)

    outputdir = f"test/{test_name}/output"
    final_snap = outputdir + "/snapshot_002.hdf5"
    if not path.isfile(final_snap):
        raise RuntimeError("GIZMO did not run successfully.")

    # Load simulation data
    with h5py.File(final_snap, "r") as F:
        x_sim = F["PartType0/Coordinates"][:, 0]
        rho_sim = F["PartType0/Density"][:]
        vel_sim = F["PartType0/Velocities"][:, 0]
        B_sim = F["PartType0/MagneticField"][:]

    # Plot profiles
    order = x_sim.argsort()
    for label, data in [
        ("Density", rho_sim),
        ("Velocity", vel_sim),
        ("By", B_sim[:, 1]),
    ]:
        plt.figure()
        plt.plot(x_sim[order], data[order], ".", markersize=1)
        plt.xlabel("x")
        plt.ylabel(label)
        plt.savefig(f"test/{test_name}/{label}.png")
        plt.close()

    # Basic sanity checks
    assert np.all(np.isfinite(rho_sim)), "Non-finite density values found"
    assert np.all(rho_sim > 0), "Negative density values found"
    assert np.all(np.isfinite(B_sim)), "Non-finite magnetic field values found"
    # Check density is in expected range for this problem
    assert rho_sim.max() < 2.0, f"Maximum density {rho_sim.max()} unreasonably high"
    assert rho_sim.min() > 0.05, f"Minimum density {rho_sim.min()} unreasonably low"
