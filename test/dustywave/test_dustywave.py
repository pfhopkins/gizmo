"""Dusty wave test (Hopkins & Lee 2016)

Tests the coupled dust-gas wave propagation. Compares dust and gas velocities
against the reference solution at t=1.2.
The exact solution file has columns: x-position, x-velocity of dust, x-velocity of gas
(velocities in units of 1e-4).
"""

import pytest
import numpy as np
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
import h5py
from os import path, chdir
from urllib.request import urlretrieve
from gizmo.test import build_gizmo_for_test, download_test_files, run_test, clean_test_outputs, assert_final_time, get_final_snapshot, default_mpi_ranks, default_omp_threads


WEBSITE = "http://www.tapir.caltech.edu/~phopkins/sims/"


@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(max_ranks=2),))
@pytest.mark.parametrize("num_omp_threads", (default_omp_threads(),))
def test_dustywave(num_mpi_ranks, num_omp_threads):
    test_name = "dustywave"
    clean_test_outputs(test_name)
    build_gizmo_for_test(test_name, num_omp_threads)
    chdir(f"test/{test_name}/")

    # Download ICs (standard name) and exact solution (non-standard name)
    download_test_files(test_name)
    if not path.isfile("dustwave_exact.txt"):
        urlretrieve(WEBSITE + "dustwave_exact.txt", "dustwave_exact.txt")

    run_test(test_name, num_mpi_ranks, num_omp_threads)
    chdir("../../")

    outputdir = f"test/{test_name}/output"
    # t=1.2 corresponds to snapshot 12 (TimeBetSnapshot=0.1)
    snap_file = outputdir + "/snapshot_012.hdf5"
    if not path.isfile(snap_file):
        raise RuntimeError("GIZMO did not run successfully.")
    assert_final_time(get_final_snapshot(test_name), test_name)

    # Load simulation data - gas is PartType0, dust is PartType3
    with h5py.File(snap_file, "r") as F:
        x_gas = F["PartType0/Coordinates"][:, 0]
        vx_gas = F["PartType0/Velocities"][:, 0]
        x_dust = F["PartType3/Coordinates"][:, 0]
        vx_dust = F["PartType3/Velocities"][:, 0]

    # Load exact solution: x, v_dust, v_gas (velocities in units of 1e-4)
    exact = np.loadtxt(f"test/{test_name}/dustwave_exact.txt")
    x_exact = exact[:, 0]
    vdust_exact = exact[:, 1] * 1e-4  # convert to code units
    vgas_exact = exact[:, 2] * 1e-4

    # Interpolate exact solution to particle positions
    vgas_interp = interp1d(x_exact, vgas_exact, bounds_error=False, fill_value="extrapolate")(x_gas)
    vdust_interp = interp1d(x_exact, vdust_exact, bounds_error=False, fill_value="extrapolate")(x_dust)

    # Plot
    plt.figure()
    gas_order = x_gas.argsort()
    dust_order = x_dust.argsort()
    plt.plot(x_gas[gas_order], vx_gas[gas_order], "b.", markersize=3, label="Gas (GIZMO)")
    plt.plot(x_dust[dust_order], vx_dust[dust_order], "r.", markersize=3, label="Dust (GIZMO)")
    plt.plot(x_exact, vgas_exact, "b-", linewidth=0.5, label="Gas (exact)")
    plt.plot(x_exact, vdust_exact, "r-", linewidth=0.5, label="Dust (exact)")
    plt.xlabel("x")
    plt.ylabel("v_x")
    plt.legend()
    plt.savefig(f"test/{test_name}/velocities.png")
    plt.close()

    # Compute L1 errors
    amp = np.max(np.abs(vgas_exact))
    L1_gas = np.mean(np.abs(vx_gas - vgas_interp)) / amp
    L1_dust = np.mean(np.abs(vx_dust - vdust_interp)) / amp

    assert L1_gas < 0.15, f"Gas velocity L1 error {L1_gas:.4f} exceeds tolerance"
    assert L1_dust < 0.15, f"Dust velocity L1 error {L1_dust:.4f} exceeds tolerance"
