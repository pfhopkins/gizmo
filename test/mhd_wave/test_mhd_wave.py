"""MHD linear wave propagation test (Hopkins & Raives 2015)"""

import pytest
from gizmo.test import build_and_run_test, assert_snapshots_are_close
from os import path


@pytest.mark.parametrize("num_mpi_ranks", (12,))
def test_mhd_wave(num_mpi_ranks):
    test_name = "mhd_wave"
    build_and_run_test(test_name, num_mpi_ranks)
    outputdir = f"test/{test_name}/output"
    final_snap = outputdir + "/snapshot_010.hdf5"
    if not path.isfile(final_snap):
        raise (RuntimeError("GIZMO did not run successfully."))

    assert_snapshots_are_close(
        outputdir + "/snapshot_000.hdf5",
        final_snap,
        fields_to_compare=("Density", "Velocities", "InternalEnergy", "MagneticField"),
        rtol=1e-4,
        atol=1e-7,
        plot_1D=True,
    )
