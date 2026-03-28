"""GMC cooling and chemistry test"""

import pytest
from gizmo.test import build_and_run_test, assert_snapshots_are_close
from os import path


<<<<<<< Updated upstream
@pytest.mark.parametrize("num_mpi_ranks", (12,))
def test_soundwave(num_mpi_ranks):
=======

@pytest.mark.parametrize("num_mpi_ranks,num_omp_threads", [(8, 0), (1, 8), (2, 4)])
def test_soundwave(num_mpi_ranks, num_omp_threads):
>>>>>>> Stashed changes
    test_name = "soundwave"
    build_and_run_test(test_name, num_mpi_ranks)
    outputdir = f"test/{test_name}/output"
    final_snap = outputdir + "/snapshot_015.hdf5"
    if not path.isfile(final_snap):
        raise (RuntimeError("GIZMO did not run successfully."))

    assert_snapshots_are_close(outputdir + "/snapshot_000.hdf5", final_snap, rtol=1e-5, atol=1e-8, plot_1D=True)
