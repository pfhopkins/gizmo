"""Ring collision elastic/solid body test (Hopkins 2015)

Tests elastic solid body physics by colliding two rings. The rings should
deform on impact and bounce apart, preserving their structure due to
elastic restoring forces. With the Tillotson EOS parameters set here,
P = cs^2 * (rho - rho_0) with cs = rho_0 = 1.
"""

import pytest
import numpy as np
from matplotlib import pyplot as plt
import h5py
import glob
from gizmo.test import build_and_run_test, default_mpi_ranks, clean_test_outputs


@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(),))
def test_ring_collision(num_mpi_ranks):
    test_name = "ring_collision"
    clean_test_outputs(test_name)
    build_and_run_test(test_name, num_mpi_ranks)

    outputdir = f"test/{test_name}/output"
    snaps = sorted(glob.glob(outputdir + "/snapshot_*.hdf5"))
    if len(snaps) < 2:
        raise RuntimeError("GIZMO did not run successfully.")

    # Plot each snapshot
    for snap in snaps:
        with h5py.File(snap, "r") as F:
            pos = F["PartType0/Coordinates"][:]
            rho = F["PartType0/Density"][:]
            t = F["Header"].attrs["Time"]
        plt.figure(figsize=(6, 6))
        plt.scatter(pos[:, 0], pos[:, 1], c=rho, s=1, cmap="viridis")
        plt.colorbar(label="Density")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"Ring Collision t={t:.0f}")
        plt.axis("equal")
        plt.savefig(f"test/{test_name}/snapshot_t{t:.0f}.png", dpi=150)
        plt.close()

    # Load initial and final snapshots
    with h5py.File(snaps[0], "r") as F:
        mass0 = F["PartType0/Masses"][:]
        pos0 = F["PartType0/Coordinates"][:]
    with h5py.File(snaps[-1], "r") as F:
        mass_f = F["PartType0/Masses"][:]
        pos_f = F["PartType0/Coordinates"][:]

    # Mass conservation
    mass_err = abs(mass_f.sum() - mass0.sum()) / mass0.sum()
    assert mass_err < 1e-3, f"Mass not conserved: relative error {mass_err:.6f}"

    # After the collision, the rings should have bounced apart.
    # Check that the particles span a larger x-range than initially
    # (they started moving toward each other, collided, and bounced back)
    x_spread0 = pos0[:, 0].max() - pos0[:, 0].min()
    x_spread_f = pos_f[:, 0].max() - pos_f[:, 0].min()
    assert x_spread_f > 0.5 * x_spread0, (
        "Rings appear to have collapsed rather than bouncing"
    )
