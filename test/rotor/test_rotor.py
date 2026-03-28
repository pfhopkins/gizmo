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
from meshoid import Meshoid
from gizmo.test import build_and_run_test, default_mpi_ranks, clean_test_outputs, assert_final_time, default_omp_threads


@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(),))
@pytest.mark.parametrize("num_omp_threads", (default_omp_threads(),))
def test_rotor(num_mpi_ranks, num_omp_threads):
    test_name = "rotor"
    clean_test_outputs(test_name)
    build_and_run_test(test_name, num_mpi_ranks, num_omp_threads)

    outputdir = f"test/{test_name}/output"
    snaps = sorted(glob.glob(outputdir + "/snapshot_*.hdf5"))
    if len(snaps) < 2:
        raise RuntimeError("GIZMO did not run successfully.")
    assert_final_time(snaps[-1], test_name)

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

    # Plot final state using Meshoid slice interpolation
    M = Meshoid(pos_f, boxsize=boxsize)
    center = np.array([boxsize/2, boxsize/2, boxsize/2])
    Bmag = np.sqrt(np.sum(B_f**2, axis=1))
    rho_slice = M.Slice(np.log10(rho_f), res=1024, plane="z", center=center, size=boxsize, order=1)
    B_slice = M.Slice(Bmag, res=1024, plane="z", center=center, size=boxsize, order=1)
    extent = [0, boxsize, 0, boxsize]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(rho_slice.T, origin="lower", cmap="inferno", extent=extent)
    axes[0].set_title("log10(Density)")
    axes[1].imshow(B_slice.T, origin="lower", cmap="viridis", extent=extent)
    axes[1].set_title("|B|")
    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
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
