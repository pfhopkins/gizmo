"""Current sheet reconnection test (Hopkins & Raives 2016)

Tests magnetic reconnection in a current sheet. Checks that the magnetic
energy dissipates (reconnection occurs) and that the code runs stably.
"""

import pytest
import numpy as np
import h5py
import glob
from os import path, chdir
from urllib.request import urlretrieve
from gizmo.test import build_gizmo_for_test, download_test_files, run_test, default_mpi_ranks, clean_test_outputs, assert_final_time, default_omp_threads


WEBSITE = "http://www.tapir.caltech.edu/~phopkins/sims/"


@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(max_ranks=8),))
@pytest.mark.parametrize("num_omp_threads", (default_omp_threads(),))
def test_currentsheet(num_mpi_ranks, num_omp_threads):
    test_name = "currentsheet"
    clean_test_outputs(test_name)
    build_gizmo_for_test(test_name, num_omp_threads)
    chdir(f"test/{test_name}/")

    # Download ICs (non-standard name)
    ic_file = "currentsheet_A0pt1_b0pt1_ics.hdf5"
    if not path.isfile(ic_file):
        urlretrieve(WEBSITE + ic_file, ic_file)

    run_test(test_name, num_mpi_ranks, num_omp_threads)
    chdir("../../")

    outputdir = f"test/{test_name}/output"
    snaps = sorted(glob.glob(outputdir + "/snapshot_*.hdf5"))
    if len(snaps) < 2:
        raise RuntimeError("GIZMO did not run successfully.")
    assert_final_time(snaps[-1], test_name)

    # Load initial and final snapshots
    with h5py.File(snaps[0], "r") as F:
        B0 = F["PartType0/MagneticField"][:]
    with h5py.File(snaps[-1], "r") as F:
        Bf = F["PartType0/MagneticField"][:]

    # Magnetic energy should decrease due to reconnection
    Emag0 = np.mean(np.sum(B0**2, axis=1))
    Emagf = np.mean(np.sum(Bf**2, axis=1))

    assert Emagf < Emag0, "Magnetic energy should decrease due to reconnection"
    # But shouldn't go to zero - reconnection is a gradual process
    assert Emagf > 0.01 * Emag0, "Magnetic energy dropped too much"
