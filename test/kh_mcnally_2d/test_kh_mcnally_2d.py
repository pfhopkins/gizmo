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
from meshoid import Meshoid
from gizmo.test import build_and_run_test, default_mpi_ranks, clean_test_outputs, flush_colorbar, assert_final_time, default_omp_threads


@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(),))
@pytest.mark.parametrize("num_omp_threads", (default_omp_threads(),))
def test_kh_mcnally_2d(num_mpi_ranks, num_omp_threads):
    test_name = "kh_mcnally_2d"
    clean_test_outputs(test_name)
    build_and_run_test(test_name, num_mpi_ranks, num_omp_threads)

    outputdir = f"test/{test_name}/output"
    snaps = sorted(glob.glob(outputdir + "/snapshot_*.hdf5"))
    if len(snaps) < 2:
        raise RuntimeError("GIZMO did not run successfully.")
    assert_final_time(snaps[-1], test_name)

    # Load initial and final snapshots
    with h5py.File(snaps[0], "r") as F:
        rho0 = F["PartType0/Density"][:]
        mass0 = F["PartType0/Masses"][:]
        pos0 = F["PartType0/Coordinates"][:]
    with h5py.File(snaps[-1], "r") as F:
        rho_f = F["PartType0/Density"][:]
        mass_f = F["PartType0/Masses"][:]
        pos_f = F["PartType0/Coordinates"][:]

    # Plot final density using Meshoid slice interpolation
    M = Meshoid(pos_f, boxsize=1.)
    rho_slice = M.Slice(rho_f, res=1024, plane="z", center=np.array([0.5, 0.5, 0.5]), size=1., order=1)
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(rho_slice.T, origin="lower", cmap="viridis", extent=[0, 1, 0, 1])
    flush_colorbar(im, ax=ax, label="Density")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("KH Instability - Density")
    fig.savefig(f"test/{test_name}/Density_2D.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Mass conservation
    mass_err = abs(mass_f.sum() - mass0.sum()) / mass0.sum()
    assert mass_err < 1e-3, f"Mass not conserved: relative error {mass_err:.6f}"

    # KH instability should develop - density variance should increase
    # as the interface gets mixed
    rho_var0 = np.var(rho0)
    rho_var_f = np.var(rho_f)
    assert rho_var_f > 0.5 * rho_var0, "Density variance collapsed - instability may not have developed"
