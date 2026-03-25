"""MHD rotor test (Hopkins & Raives 2016)

Tests the spinning-down of a dense rotating disk embedded in a magnetized
medium. Checks that the rotor spins down (transfers angular momentum to
the field) and that the code runs stably.
"""

import pytest
import numpy as np
from matplotlib import pyplot as plt
import h5py
import glob
from gizmo.test import build_and_run_test, default_mpi_ranks, clean_test_outputs


@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(),))
def test_rotor(num_mpi_ranks):
    test_name = "rotor"
    clean_test_outputs(test_name)
    build_and_run_test(test_name, num_mpi_ranks)

    outputdir = f"test/{test_name}/output"
    snaps = sorted(glob.glob(outputdir + "/snapshot_*.hdf5"))
    if len(snaps) < 2:
        raise RuntimeError("GIZMO did not run successfully.")

    # Load initial and final snapshots
    with h5py.File(snaps[0], "r") as F:
        pos0 = F["PartType0/Coordinates"][:]
        vel0 = F["PartType0/Velocities"][:]
        mass0 = F["PartType0/Masses"][:]
        boxsize = F["Header"].attrs["BoxSize"]
    with h5py.File(snaps[-1], "r") as F:
        pos_f = F["PartType0/Coordinates"][:]
        vel_f = F["PartType0/Velocities"][:]
        mass_f = F["PartType0/Masses"][:]
        rho_f = F["PartType0/Density"][:]
        B_f = F["PartType0/MagneticField"][:]

    # Plot final state
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(pos_f[:, 0], pos_f[:, 1], c=np.log10(rho_f), s=0.1, cmap="inferno")
    axes[0].set_title("log10(Density)")
    Bmag = np.sqrt(np.sum(B_f**2, axis=1))
    axes[1].scatter(pos_f[:, 0], pos_f[:, 1], c=Bmag, s=0.1, cmap="viridis")
    axes[1].set_title("|B|")
    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(f"test/{test_name}/Rotor_2D.png", dpi=150)
    plt.close()

    # The rotor should spin down: kinetic energy in the central region should decrease
    center = boxsize / 2.0
    r0 = np.sqrt((pos0[:, 0] - center) ** 2 + (pos0[:, 1] - center) ** 2)
    rf = np.sqrt((pos_f[:, 0] - center) ** 2 + (pos_f[:, 1] - center) ** 2)
    KE0_center = 0.5 * np.sum(mass0[r0 < 0.15] * np.sum(vel0[r0 < 0.15] ** 2, axis=1))
    KEf_center = 0.5 * np.sum(mass_f[rf < 0.15] * np.sum(vel_f[rf < 0.15] ** 2, axis=1))

    assert KEf_center < KE0_center, "Rotor should spin down (lose kinetic energy to field)"

    # Mass conservation
    mass_err = abs(mass_f.sum() - mass0.sum()) / mass0.sum()
    assert mass_err < 1e-3, f"Mass not conserved: relative error {mass_err:.6f}"
