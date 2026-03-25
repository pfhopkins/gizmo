"""Shu (1977) singular isothermal sphere collapse test: checks that exactly one sink particle forms."""

import pytest
import h5py
from os import path
from gizmo.test import build_and_run_test


@pytest.mark.parametrize("num_mpi_ranks", (16,))
def test_shu1977(num_mpi_ranks):
    test_name = "shu1977"
    build_and_run_test(test_name, num_mpi_ranks)

    final_snap = f"test/{test_name}/output/snapshot_001.hdf5"
    if not path.isfile(final_snap):
        raise RuntimeError("GIZMO did not run successfully.")

    with h5py.File(final_snap, "r") as f:
        num_sinks = f["Header"].attrs["NumPart_ThisFile"][5]

    assert num_sinks == 1, f"Expected exactly 1 PartType5 particle, got {num_sinks}"
