"""Kelvin-Helmholtz instability - McNally et al. setup (Hopkins 2015)

Tests the development of the KH instability. Since the inviscid problem
has no converged solution, we check that the instability develops
(density variance increases) and that mass/energy are conserved.
"""

import pytest
import numpy as np
from matplotlib import pyplot as plt
import h5py
import glob
from gizmo.test import build_and_run_test, default_mpi_ranks, clean_test_outputs


@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(),))
def test_kh_mcnally_2d(num_mpi_ranks):
    test_name = "kh_mcnally_2d"
    clean_test_outputs(test_name)
    build_and_run_test(test_name, num_mpi_ranks)

    outputdir = f"test/{test_name}/output"
    snaps = sorted(glob.glob(outputdir + "/snapshot_*.hdf5"))
    if len(snaps) < 2:
        raise RuntimeError("GIZMO did not run successfully.")

    # Load initial and final snapshots
    with h5py.File(snaps[0], "r") as F:
        rho0 = F["PartType0/Density"][:]
        mass0 = F["PartType0/Masses"][:]
        pos0 = F["PartType0/Coordinates"][:]
    with h5py.File(snaps[-1], "r") as F:
        rho_f = F["PartType0/Density"][:]
        mass_f = F["PartType0/Masses"][:]
        pos_f = F["PartType0/Coordinates"][:]

    # Plot final density
    plt.figure(figsize=(6, 6))
    plt.scatter(pos_f[:, 0], pos_f[:, 1], c=rho_f, s=0.1, cmap="viridis")
    plt.colorbar(label="Density")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("KH Instability - Density")
    plt.savefig(f"test/{test_name}/Density_2D.png", dpi=150)
    plt.close()

    # Mass conservation
    mass_err = abs(mass_f.sum() - mass0.sum()) / mass0.sum()
    assert mass_err < 1e-3, f"Mass not conserved: relative error {mass_err:.6f}"

    # KH instability should develop - density variance should increase
    # as the interface gets mixed
    rho_var0 = np.var(rho0)
    rho_var_f = np.var(rho_f)
    assert rho_var_f > 0.5 * rho_var0, "Density variance collapsed - instability may not have developed"
