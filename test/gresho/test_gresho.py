"""Gresho vortex test (Hopkins 2015)

The Gresho vortex has a known analytic azimuthal velocity profile:
  v_phi(r) = 5r           for r < 0.2
  v_phi(r) = 2 - 5r       for 0.2 <= r < 0.4
  v_phi(r) = 0             for r >= 0.4

Tests that the vortex is preserved over time.
"""

import pytest
import numpy as np
from matplotlib import pyplot as plt
import h5py
from os import path
from gizmo.test import build_and_run_test, default_mpi_ranks


def gresho_vphi_analytic(r):
    """Analytic azimuthal velocity profile for the Gresho vortex."""
    vphi = np.zeros_like(r)
    vphi[r < 0.2] = 5 * r[r < 0.2]
    mask = (r >= 0.2) & (r < 0.4)
    vphi[mask] = 2 - 5 * r[mask]
    return vphi


@pytest.mark.parametrize("num_mpi_ranks", (1,))
def test_gresho(num_mpi_ranks):
    test_name = "gresho"
    build_and_run_test(test_name, num_mpi_ranks)

    outputdir = f"test/{test_name}/output"
    final_snap = outputdir + "/snapshot_006.hdf5"
    if not path.isfile(final_snap):
        raise RuntimeError("GIZMO did not run successfully.")

    # Load simulation data
    with h5py.File(final_snap, "r") as F:
        coords = F["PartType0/Coordinates"][:]
        vel = F["PartType0/Velocities"][:]

    # Compute radius from center and azimuthal velocity
    x = coords[:, 0] - 0.5
    y = coords[:, 1] - 0.5
    r = np.sqrt(x**2 + y**2)
    # v_phi = (-sin(theta) * vx + cos(theta) * vy) = (-y*vx + x*vy)/r
    vphi_sim = (-y * vel[:, 0] + x * vel[:, 1]) / (r + 1e-30)

    # Analytic solution
    vphi_analytic = gresho_vphi_analytic(r)

    # Plot comparison
    order = r.argsort()
    plt.figure()
    plt.plot(r[order], vphi_sim[order], ".", markersize=0.5, alpha=0.3, label="GIZMO")
    r_plot = np.linspace(0, 0.5, 200)
    plt.plot(r_plot, gresho_vphi_analytic(r_plot), "-", color="red", linewidth=2, label="Analytic")
    plt.xlabel("r")
    plt.ylabel(r"$v_\phi$")
    plt.legend()
    plt.savefig(f"test/{test_name}/vphi.png")
    plt.close()

    # Compute L1 error on v_phi for r < 0.4 (where vortex is active)
    inside = r < 0.4
    L1 = np.mean(np.abs(vphi_sim[inside] - vphi_analytic[inside])) / np.mean(np.abs(vphi_analytic[inside]))
    assert L1 < 0.15, f"Gresho v_phi L1 error {L1:.4f} exceeds tolerance"
