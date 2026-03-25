"""Rayleigh-Taylor instability test (Hopkins 2015)

Tests the development of the RT instability with analytic gravity.
The heavy fluid on top should develop characteristic mushroom-shaped
plumes as it falls through the light fluid below.
"""

import pytest
import numpy as np
from matplotlib import pyplot as plt
import h5py
import glob
from os import path, chdir
from urllib.request import urlretrieve
from gizmo.test import build_gizmo_for_test, download_test_files, run_test, default_mpi_ranks, clean_test_outputs


WEBSITE = "http://www.tapir.caltech.edu/~phopkins/sims/"


@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(),))
def test_rt(num_mpi_ranks):
    test_name = "rt"
    clean_test_outputs(test_name)
    build_gizmo_for_test(test_name)
    chdir(f"test/{test_name}/")

    # Download ICs (non-standard name on site: rt_ics.hdf5, but params references rt_ics)
    if not path.isfile("rt_ics.hdf5"):
        urlretrieve(WEBSITE + "rt_ics.hdf5", "rt_ics.hdf5")

    run_test(test_name, num_mpi_ranks)
    chdir("../../")

    outputdir = f"test/{test_name}/output"
    snaps = sorted(glob.glob(outputdir + "/snapshot_*.hdf5"))
    if len(snaps) < 2:
        raise RuntimeError("GIZMO did not run successfully.")

    # Load initial and final snapshots
    with h5py.File(snaps[0], "r") as F:
        rho0 = F["PartType0/Density"][:]
        mass0 = F["PartType0/Masses"][:]
    with h5py.File(snaps[-1], "r") as F:
        rho_f = F["PartType0/Density"][:]
        mass_f = F["PartType0/Masses"][:]
        pos_f = F["PartType0/Coordinates"][:]

    # Plot final density
    plt.figure(figsize=(4, 8))
    plt.scatter(pos_f[:, 0], pos_f[:, 1], c=rho_f, s=0.1, cmap="viridis")
    plt.colorbar(label="Density")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Rayleigh-Taylor Instability")
    plt.savefig(f"test/{test_name}/Density_2D.png", dpi=150)
    plt.close()

    # Mass conservation
    mass_err = abs(mass_f.sum() - mass0.sum()) / mass0.sum()
    assert mass_err < 1e-3, f"Mass not conserved: relative error {mass_err:.6f}"

    # RT instability should develop - the interface should mix, so the
    # density should have intermediate values that weren't present initially
    rho_median0 = np.median(rho0)
    # Count particles near the median density (within 20% of median)
    near_median0 = np.sum(np.abs(rho0 - rho_median0) < 0.2 * rho_median0)
    near_median_f = np.sum(np.abs(rho_f - rho_median0) < 0.2 * rho_median0)
    # There should be MORE particles near the median density after mixing
    assert near_median_f > near_median0, "No evidence of RT mixing"
