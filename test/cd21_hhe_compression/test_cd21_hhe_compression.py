"""CD21 H/He compression smoke test.

Validates that GIZMO can:
  1. Load the converted CD21 (Y=0.275) SESAME table
  2. Run a tiny 1D periodic adiabatic compression without crashing
  3. Produce finite, positive (rho, P, u)

This is a crash/sanity test, not a quantitative accuracy test. Phase 17h
deliverable; physics validation (Jupiter/Saturn polytrope vs published
profiles) is deferred.
"""

import pytest
import numpy as np
import h5py
import os
import subprocess
import sys
import tarfile
from os import chdir
from os.path import isfile, dirname, abspath, join
from urllib.request import urlretrieve

from gizmo.test import (
    build_gizmo_for_test,
    run_test,
    clean_test_outputs,
    default_mpi_ranks,
    default_omp_threads,
    get_final_snapshot,
)

# Native CD21 (Chabrier & Debras 2021) H/He EOS tables. The DirEOS2021 tarball
# bundles all compositions; we extract the T-P table at Y=0.275.
DIREOS_URL = "http://perso.ens-lyon.fr/gilles.chabrier/DirEOS/DirEOS2021.tar.gz"
DIREOS_TP_Y0275 = "DirEOS2021/TABLEEOS_2021_TP_Y0275_v1"


def _download_native_table(native, test_dir):
    """Download the CD21 DirEOS2021 tarball and extract the Y=0.275 T-P table to
    `native`. Mirrors the auto-download other test harnesses do for their data."""
    tarball = join(test_dir, "DirEOS2021.tar.gz")
    if not isfile(tarball):
        urlretrieve(DIREOS_URL, tarball)
    with tarfile.open(tarball) as tf:
        with tf.extractfile(DIREOS_TP_Y0275) as src, open(native, "wb") as dst:
            dst.write(src.read())
    os.remove(tarball)


def setup_table_and_ics(test_dir):
    """Build the SESAME table + IC if absent."""
    sesame = join(test_dir, "cd21_y0275.sesame")
    native = join(test_dir, "cd21_y0275.dat")
    if not isfile(sesame):
        if not isfile(native):
            _download_native_table(native, test_dir)
        subprocess.run(
            [sys.executable,
             join(test_dir, "..", "..", "initial_conditions", "eos_tools", "cms_to_sesame.py"),
             "--input", native, "--output", sesame,
             "--y", "0.275", "--mat-id", "2721", "--nrho", "200"],
            check=True,
        )
    if not isfile(join(test_dir, "compression_ics.hdf5")):
        subprocess.run(
            [sys.executable, join(test_dir, "make_ics.py")],
            check=True, cwd=test_dir,
        )


@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(max_ranks=2),))
@pytest.mark.parametrize("num_omp_threads", (default_omp_threads(),))
def test_cd21_hhe_compression(num_mpi_ranks, num_omp_threads):
    test_name = "cd21_hhe_compression"
    test_dir = abspath(join(dirname(__file__)))

    clean_test_outputs(test_name)
    build_gizmo_for_test(test_name, num_omp_threads)
    chdir(f"test/{test_name}/")

    setup_table_and_ics(".")

    run_test(test_name, num_mpi_ranks, num_omp_threads)
    chdir("../../")

    final_snap = get_final_snapshot(test_name)
    with h5py.File(final_snap, "r") as F:
        n = F["Header"].attrs["NumPart_ThisFile"][0]
        assert n > 0
        rho = F["PartType0/Density"][:]
        u = F["PartType0/InternalEnergy"][:]
        assert np.all(np.isfinite(rho)), "non-finite density"
        assert np.all(np.isfinite(u)), "non-finite u"
        assert np.all(rho > 0), "non-positive density"
        assert np.all(u > 0), "non-positive u"

    print(f"CD21 H/He compression smoke: {n} particles, all finite + positive")
