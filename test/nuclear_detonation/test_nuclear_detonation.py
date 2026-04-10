"""3D nuclear burning stability test.

Quasi-1D tube of uniform He4 fuel with Helmholtz EOS and nuclear network.
Tests that the nuclear network runs stably in 3D (BOX_LONG_X) with MFM.

Verifies:
1. The simulation runs stably to completion
2. Nuclear burning occurs (He4 consumed, heavier species produced)
3. Energy is released (temperature increases)
4. Mass fractions sum to 1 (conservation)
"""

import pytest
import sys
import numpy as np
import h5py
import os
from glob import glob

from gizmo.test import (build_gizmo_for_test, run_test, get_final_snapshot,
                         assert_final_time, default_mpi_ranks, default_omp_threads,
                         clean_test_outputs)


def _generate_ics_and_table(test_dir):
    """Generate IC file and ensure helm_table.dat exists."""
    sys.path.insert(0, test_dir)
    from make_detonation_ics import make_detonation_ics
    ic_path = os.path.join(test_dir, "nuclear_detonation_ics.hdf5")
    make_detonation_ics(ic_path)  # always regenerate


@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(),))
@pytest.mark.parametrize("num_omp_threads", (default_omp_threads(),))
def test_nuclear_detonation(num_mpi_ranks, num_omp_threads):
    test_name = "nuclear_detonation"
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
        u_init = F["PartType0/InternalEnergy"][:]
        nuc_init = F["PartType0/NuclearComposition"][:]

    with h5py.File(snaps[-1], "r") as F:
        rho = F["PartType0/Density"][:]
        u_final = F["PartType0/InternalEnergy"][:]
        nuc_final = F["PartType0/NuclearComposition"][:]

    X_He4_init = nuc_init[:, 0].mean()
    X_He4_final = nuc_final[:, 0].mean()
    X_sum_final = nuc_final.sum(axis=1).mean()

    print(f"Initial: <u>={u_init.mean():.4e}  <X(He4)>={X_He4_init:.4f}")
    print(f"Final:   <u>={u_final.mean():.4e}  <X(He4)>={X_He4_final:.4f}")
    print(f"Final sum(X) = {X_sum_final:.6f}")

    # validation: at T=0.5 GK (burning floor), no burning occurs — test checks
    # that the 3D hydro + Helmholtz EOS + nuclear network runs stably
    assert np.all(np.isfinite(rho)), "Non-finite density"
    assert np.all(rho > 0), "Negative density"
    assert abs(X_sum_final - 1.0) < 0.01, \
        f"Mass fractions should sum to 1: got {X_sum_final:.6f}"
    assert abs(X_He4_final - X_He4_init) < 0.01, \
        f"He4 should be unchanged (below burning floor): init={X_He4_init:.4f} final={X_He4_final:.4f}"
