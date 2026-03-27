"""Shu (1977) singular isothermal sphere collapse test: checks that exactly one sink particle forms."""

import pytest
import numpy as np
from matplotlib import pyplot as plt
import h5py
import glob
from os import path
from meshoid import Meshoid
from gizmo.test import build_and_run_test, flush_colorbar, assert_final_time, get_final_snapshot, default_omp_threads


def plot_shu1977_density_slice(coords, rho, boxsize, output_dir="."):
    """Plot a density slice through the Shu 1977 collapse center."""
    center = np.average(coords, axis=0)
    size = boxsize * 0.1
    M = Meshoid(coords)
    rho_slice = M.Slice(np.log10(rho), res=1024, plane="z", center=center, size=size, order=1)
    fig, ax = plt.subplots(figsize=(6, 6))
    half = size / 2
    im = ax.imshow(rho_slice.T, origin="lower", cmap="inferno", extent=[-half, half, -half, half])
    flush_colorbar(im, ax=ax, label="log10(Density)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Shu 1977 Collapse - Density Slice")
    fig.savefig(output_dir + "/Density_2D.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


@pytest.mark.parametrize("num_mpi_ranks", (8,))
@pytest.mark.parametrize("num_omp_threads", (default_omp_threads(),))
def test_shu1977(num_mpi_ranks, num_omp_threads):
    test_name = "shu1977"
    build_and_run_test(test_name, num_mpi_ranks, num_omp_threads)

    final_snap = get_final_snapshot(test_name)
    assert_final_time(final_snap, test_name)

    with h5py.File(final_snap, "r") as f:
        num_sinks = f["Header"].attrs["NumPart_ThisFile"][5]

    assert num_sinks == 1, f"Expected exactly 1 PartType5 particle, got {num_sinks}"

    with h5py.File(final_snap, "r") as F:
        coords = F["PartType0/Coordinates"][:]
        rho = F["PartType0/Density"][:]
        boxsize = F["Header"].attrs["BoxSize"]
    plot_shu1977_density_slice(coords, rho, boxsize, output_dir=f"test/{test_name}")
