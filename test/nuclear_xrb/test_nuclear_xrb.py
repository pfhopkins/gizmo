"""1D nuclear burning stability test.

Uniform He4 box with Helmholtz EOS and nuclear network. Verifies:
1. The simulation runs stably to completion
2. Nuclear burning occurs (He4 consumed, heavier species produced)
3. Energy is released (temperature/internal energy increases)
4. Mass fractions sum to 1 (conservation)
"""

import pytest
import sys
import numpy as np
from matplotlib import pyplot as plt
import h5py
import os
from glob import glob

from gizmo.test import (build_gizmo_for_test, run_test, get_final_snapshot,
                         assert_final_time, default_mpi_ranks, default_omp_threads,
                         clean_test_outputs)


def _generate_ics_and_table(test_dir):
    """Generate IC file and ensure helm_table.dat exists."""
    sys.path.insert(0, test_dir)
    from make_xrb_ics import make_xrb_ics
    ic_path = os.path.join(test_dir, "nuclear_xrb_ics.hdf5")
    make_xrb_ics(ic_path)  # always regenerate to ensure consistency with current code


@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(),))
@pytest.mark.parametrize("num_omp_threads", (default_omp_threads(),))
def test_nuclear_xrb(num_mpi_ranks, num_omp_threads):
    test_name = "nuclear_xrb"
    test_dir = os.path.join("test", test_name)

    clean_test_outputs(test_name)
    build_gizmo_for_test(test_name, num_omp_threads)
    _generate_ics_and_table(os.path.abspath(test_dir))

    cwd = os.getcwd()
    os.chdir(test_dir)
    run_test(test_name, num_mpi_ranks, num_omp_threads)
    os.chdir(cwd)

    final_snap = get_final_snapshot(test_name)
    assert_final_time(final_snap, test_name)

    # read initial and final snapshots
    snap_dir = os.path.join("test", test_name, "output")
    snaps = sorted(glob(os.path.join(snap_dir, "snapshot_*.hdf5")))

    with h5py.File(snaps[0], "r") as F:
        T_init = F["PartType0/Temperature"][:]
        u_init = F["PartType0/InternalEnergy"][:]
        nuc_init = F["PartType0/NuclearComposition"][:]

    with h5py.File(snaps[-1], "r") as F:
        rho = F["PartType0/Density"][:]
        T_final = F["PartType0/Temperature"][:]
        u_final = F["PartType0/InternalEnergy"][:]
        nuc_final = F["PartType0/NuclearComposition"][:]

    # species: 0=He4, 1=C12, 2=O16, ... (aprox13 ordering)
    X_He4_init = nuc_init[:, 0].mean()
    X_He4_final = nuc_final[:, 0].mean()
    X_sum_final = nuc_final.sum(axis=1).mean()

    print(f"Initial: <T>={T_init.mean():.3e} K  <u>={u_init.mean():.4e}  <X(He4)>={X_He4_init:.4f}")
    print(f"Final:   <T>={T_final.mean():.3e} K  <u>={u_final.mean():.4e}  <X(He4)>={X_He4_final:.4f}")
    print(f"Final composition (particle 0):")
    species = ['He4','C12','O16','Ne20','Mg24','Si28','S32','Ar36','Ca40','Ti44','Cr48','Fe52','Ni56']
    for k, name in enumerate(species):
        if nuc_final[0, k] > 1e-6:
            print(f"  {name}: {nuc_final[0, k]:.6f}")
    print(f"  sum = {X_sum_final:.6f}")

    # plot light curve
    times, mean_T, mean_u = [], [], []
    for snap_file in snaps:
        with h5py.File(snap_file, "r") as F:
            times.append(F["Header"].attrs["Time"])
            mean_T.append(F["PartType0/Temperature"][:].mean())
            mean_u.append(F["PartType0/InternalEnergy"][:].mean())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(times, mean_T, "o-"); ax1.set_xlabel("Time [s]"); ax1.set_ylabel("Mean T [K]")
    ax2.plot(times, mean_u, "o-"); ax2.set_xlabel("Time [s]"); ax2.set_ylabel("Mean u [erg/g]")
    plt.suptitle("Nuclear Burning Test")
    plt.tight_layout()
    plt.savefig(f"test/{test_name}/xrb_lightcurve.png", dpi=150)
    plt.close()

    # validation
    assert np.all(np.isfinite(rho)), "Non-finite density"
    assert np.all(rho > 0), "Negative density"
    assert X_He4_final < X_He4_init, \
        f"He4 should be consumed: init={X_He4_init:.4f} final={X_He4_final:.4f}"
    assert abs(X_sum_final - 1.0) < 0.01, \
        f"Mass fractions should sum to 1: got {X_sum_final:.6f}"
    assert u_final.mean() > u_init.mean(), \
        f"Energy should increase from burning: u_init={u_init.mean():.4e} u_final={u_final.mean():.4e}"
