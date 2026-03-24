"""GMC cooling and chemistry test"""

import pytest
from gizmo.test import build_and_run_test, assert_snapshots_are_close, plot_1D_snapshot_comparison, default_mpi_ranks
from os import path


@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(),))
def test_soundwave(num_mpi_ranks):
    test_name = "soundwave"
    build_and_run_test(test_name, num_mpi_ranks)
    outputdir = f"test/{test_name}/output"
    final_snap = outputdir + "/snapshot_015.hdf5"
    if not path.isfile(final_snap):
        raise (RuntimeError("GIZMO did not run successfully."))

    initial_snap = outputdir + "/snapshot_000.hdf5"
    assert_snapshots_are_close(initial_snap, final_snap, rtol=1e-5, atol=1e-8)
    plot_1D_snapshot_comparison(initial_snap, final_snap, output_dir=f"test/{test_name}")
