"""Interacting blast waves test (Woodward & Colella 1984; Hopkins 2015)

Compares the final snapshot against a high-resolution reference solution.
The exact solution file has columns: i-zone, x, density, V1, V2, V3, pressure.
"""

import pytest
import numpy as np
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
import h5py

import glob
from os import chdir, getcwd
from gizmo.test import (build_and_run_test, build_gizmo_for_test, download_test_files, run_test, assert_final_time,
                        get_final_snapshot, assert_energy_conserved)


@pytest.mark.parametrize("num_mpi_ranks", (1,)) #default_mpi_ranks(),))
@pytest.mark.parametrize("num_omp_threads", (1,)) #default_omp_threads(),))
def test_interactblast(num_mpi_ranks, num_omp_threads):
    test_name = "interactblast"
    build_and_run_test(test_name, num_mpi_ranks, num_omp_threads)

    final_snap = get_final_snapshot(test_name)
    assert_final_time(final_snap, test_name)

    # Load simulation data
    with h5py.File(final_snap, "r") as F:
        x_sim = F["PartType0/Coordinates"][:, 0]
        rho_sim = F["PartType0/Density"][:]
        vel_sim = F["PartType0/Velocities"][:, 0]

    # Load exact solution: i-zone, x, density, V1, V2, V3, pressure
    exact = np.loadtxt(f"test/{test_name}/interactblast_exact.txt")
    x_exact = exact[:, 1]
    rho_exact = exact[:, 2]
    vel_exact = exact[:, 3]

    # Interpolate exact solution to particle positions
    rho_interp = interp1d(x_exact, rho_exact, bounds_error=False, fill_value="extrapolate")(x_sim)
    vel_interp = interp1d(x_exact, vel_exact, bounds_error=False, fill_value="extrapolate")(x_sim)

    # Compute L1 errors
    L1_rho = np.mean(np.abs(rho_sim - rho_interp)) / np.mean(np.abs(rho_interp))
    L1_vel = np.mean(np.abs(vel_sim - vel_interp)) / (np.mean(np.abs(vel_interp)) + 1e-10)

    # Plot comparison
    order = x_sim.argsort()
    for label, sim, exact_vals in [
        ("Density", rho_sim, rho_interp),
        ("Velocity", vel_sim, vel_interp),
    ]:
        plt.figure()
        plt.plot(x_sim[order], sim[order], ".", markersize=1, label="GIZMO")
        plt.plot(x_sim[order], exact_vals[order], "-", color="red", linewidth=0.5, label="Exact")
        plt.xlabel("x")
        plt.ylabel(label)
        plt.legend()
        plt.savefig(f"test/{test_name}/{label}.png")
        plt.close()

    assert L1_rho < 0.1, f"Density L1 error {L1_rho:.4f} exceeds tolerance"
    assert L1_vel < 0.15, f"Velocity L1 error {L1_vel:.4f} exceeds tolerance"

    # Interacting blast waves: adiabatic, no gravity, no MHD, no source after t=0.
    assert_energy_conserved(test_name, tol=0.1)


ID_OFFSET = 2**27  # above 5e7, where a reflection placed at ID x 2e-8 of the box from the face lands outside the box


def test_interactblast_large_ids():
    """The same run with every particle ID shifted up by ID_OFFSET. A particle that crosses a reflecting face is mirrored
    back through it; where it lands must not depend on its ID, so every snapshot stays inside [0, BoxSize] and the final
    state still matches the reference."""
    test_name = "interactblast"
    build_gizmo_for_test(test_name, 1)
    cwd = getcwd()
    try:
        chdir(f"test/{test_name}")
        download_test_files(test_name)
        with h5py.File(f"{test_name}_ics.hdf5", "r") as F, h5py.File(f"{test_name}_largeid_ics.hdf5", "w") as G:
            for k in F:
                F.copy(k, G)
            for k, v in F["Header"].attrs.items():
                G["Header"].attrs[k] = v
            ids = G["PartType0/ParticleIDs"][:]
            del G["PartType0/ParticleIDs"]
            G["PartType0"].create_dataset("ParticleIDs", data=np.where(ids > 0, ids + ID_OFFSET, 0).astype(ids.dtype))
        run_test(test_name, 1, 1, param_overrides={"InitCondFile": f"{test_name}_largeid_ics", "OutputDir": "output_largeid"})
    finally:
        chdir(cwd)
    snaps = sorted(glob.glob(f"test/{test_name}/output_largeid/snapshot_*.hdf5"))
    lo, hi = np.inf, -np.inf
    for fn in snaps:
        with h5py.File(fn, "r") as F:
            x = F["PartType0/Coordinates"][:, 0]
        lo, hi = min(lo, x.min()), max(hi, x.max())
    with h5py.File(snaps[-1], "r") as F:
        x_sim, rho_sim = F["PartType0/Coordinates"][:, 0], F["PartType0/Density"][:]
    exact = np.loadtxt(f"test/{test_name}/interactblast_exact.txt")
    rho_interp = interp1d(exact[:, 1], exact[:, 2], bounds_error=False, fill_value="extrapolate")(x_sim)
    L1_rho = np.mean(np.abs(rho_sim - rho_interp)) / np.mean(np.abs(rho_interp))
    print(f"\nlarge IDs: x over all {len(snaps)} snapshots in [{lo:.6g}, {hi:.6g}]; final density L1 {L1_rho:.4f}")
    assert 0 <= lo and hi <= 1, f"particles outside the reflecting box: x in [{lo:.6g}, {hi:.6g}]"
    assert L1_rho < 0.1, f"Density L1 error {L1_rho:.4f} exceeds tolerance"
