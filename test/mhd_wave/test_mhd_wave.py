"""MHD linear wave propagation test (Hopkins & Raives 2015)"""

import pytest
from gizmo.test import build_and_run_test, assert_snapshots_are_close, plot_1D_snapshot_comparison, assert_final_time, default_mpi_ranks, default_omp_threads, get_final_snapshot



@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(4),))
@pytest.mark.parametrize("num_omp_threads", (default_omp_threads(),))
def test_mhd_wave(num_mpi_ranks, num_omp_threads):
    test_name = "mhd_wave"
    build_and_run_test(test_name, num_mpi_ranks, num_omp_threads)
    outputdir = f"test/{test_name}/output"
    final_snap = get_final_snapshot(test_name)
    assert_final_time(final_snap, test_name)

    initial_snap = outputdir + "/snapshot_000.hdf5"
    fields = ("Density", "Velocities", "InternalEnergy", "MagneticField")
    assert_snapshots_are_close(initial_snap, final_snap, fields_to_compare=fields, rtol=1e-7, atol=1e-7)
    plot_1D_snapshot_comparison(initial_snap, final_snap, fields_to_plot=fields, output_dir=f"test/{test_name}")
