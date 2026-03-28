"""GMC cooling and chemistry test"""

import pytest
from gizmo.test import (
    build_and_run_test,
    assert_snapshots_are_close,
    plot_1D_snapshot_comparison,
    default_mpi_ranks,
    assert_final_time,
    get_final_snapshot,
)



@pytest.mark.parametrize("num_mpi_ranks,num_omp_threads", [(16, 0), (1, 16), (4, 4)])
def test_soundwave(num_mpi_ranks, num_omp_threads):
    test_name = "soundwave"
    build_and_run_test(test_name, num_mpi_ranks, num_omp_threads)
    outputdir = f"test/{test_name}/output"
    final_snap = get_final_snapshot(test_name)
    assert_final_time(final_snap, test_name)

    initial_snap = outputdir + "/snapshot_000.hdf5"
    assert_snapshots_are_close(initial_snap, final_snap, rtol=1e-5, atol=1e-8)
    plot_1D_snapshot_comparison(initial_snap, final_snap, output_dir=f"test/{test_name}")
