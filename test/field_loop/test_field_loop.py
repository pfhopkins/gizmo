"""Field loop advection test (Hopkins & Raives 2016)

Tests advection of a magnetic field loop. The loop should be preserved
after one full advection period. Checks magnetic flux conservation.
"""

import pytest
import numpy as np
import h5py
import glob
from gizmo.test import build_and_run_test, default_mpi_ranks, clean_test_outputs, assert_final_time, default_omp_threads


@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(1),))
@pytest.mark.parametrize("num_omp_threads", (0,))
def test_field_loop(num_mpi_ranks, num_omp_threads):
    test_name = "field_loop"
    clean_test_outputs(test_name)
    build_and_run_test(test_name, num_mpi_ranks, num_omp_threads)

    outputdir = f"test/{test_name}/output"
    snaps = sorted(glob.glob(outputdir + "/snapshot_*.hdf5"))
    if len(snaps) < 2:
        raise RuntimeError("GIZMO did not run successfully.")
    assert_final_time(snaps[-1], test_name)

    # Load initial and final snapshots
    with h5py.File(snaps[0], "r") as F:
        B0 = F["PartType0/MagneticField"][:]
        pos0 = F["PartType0/Coordinates"][:]
        mass0 = F["PartType0/Masses"][:]
        dens0 = F["PartType0/Density"][:]
    with h5py.File(snaps[-1], "r") as F:
        Bf = F["PartType0/MagneticField"][:]
        mass_f = F["PartType0/Masses"][:]
        dens_f = F["PartType0/Density"][:]
        divB = F["PartType0/DivergenceOfMagneticField"][:]
        dx = np.sqrt(mass_f/dens_f)
        divB_normed = divB * dx / np.sqrt(np.sum(Bf**2, axis=1))

    # Total magnetic energy should be approximately conserved
    Emag0 = np.sum(np.sum(B0**2, axis=1) * mass0/dens0)
    Emagf = np.sum(np.sum(Bf**2, axis=1) * mass_f/dens_f)

    rel_err = abs(Emagf - Emag0) / Emag0
    print("RELATIVE_ERR == ",rel_err)
    assert rel_err < 0.05, f"Magnetic energy not conserved: relative error {rel_err:.4f}"

    # Divergence of magnetic field should be approximately zero
    divB_med = np.median(np.abs(divB))
    print("DIVB == ",divB_med)
    assert divB_med < 1.e-5, f"Maximum divergence of magnetic field is too large: {divB_med:.4f}"
    
    
def flout():
    test_name = "field_loop"
    outputdir = f"./output"
    snaps = sorted(glob.glob(outputdir + "/snapshot_*.hdf5"))

    # Load initial and final snapshots
    with h5py.File(snaps[0], "r") as F:
        B0 = F["PartType0/MagneticField"][:]
        pos0 = F["PartType0/Coordinates"][:]
        mass0 = F["PartType0/Masses"][:]
        dens0 = F["PartType0/Density"][:]
    with h5py.File(snaps[-1], "r") as F:
        Bf = F["PartType0/MagneticField"][:]
        mass_f = F["PartType0/Masses"][:]
        dens_f = F["PartType0/Density"][:]
        divB = F["PartType0/DivergenceOfMagneticField"][:]
        dx = np.sqrt(mass_f/dens_f)
        divB_normed = divB * dx / np.sqrt(np.sum(Bf**2, axis=1))

    # Total magnetic energy should be approximately conserved
    Emag0 = np.sum(np.sum(B0**2, axis=1) * mass0/dens0)
    Emagf = np.sum(np.sum(Bf**2, axis=1) * mass_f/dens_f)

    rel_err = abs(Emagf - Emag0) / Emag0
    print("RELATIVE_ERR == ",rel_err)

    # Divergence of magnetic field should be approximately zero
    divB_med = np.median(np.abs(divB))
    print("DIVB == ",divB_med)
