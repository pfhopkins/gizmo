"""A NUCLEAR_NETWORK_SOLVER=1 (SkyNet) build links and starts the nuclear_detonation problem.

Needs SkyNet: set SKYNET_DIR to its install prefix and make its headers and libraries (and the HDF5 C++, Boost, GSL and LAPACK ones it
uses) visible to the compiler and linker (CPATH, LIBRARY_PATH, LD_LIBRARY_PATH, link flags). Skipped without SKYNET_DIR.
"""
import os
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from gizmo.test import build_gizmo_for_test, run_test, get_final_snapshot, assert_final_time, default_mpi_ranks, default_omp_threads, clean_test_outputs

TEST_NAME = "nuclear_skynet_link"


@pytest.mark.skipif("SKYNET_DIR" not in os.environ, reason="SKYNET_DIR (SkyNet install prefix) not set")
def test_nuclear_skynet_link():
    build_gizmo_for_test(TEST_NAME, default_omp_threads())  # a failed link raises here
    assert (Path(__file__).resolve().parent / "GIZMO").is_file()  # the harness moves the binary into the test directory


@pytest.mark.skipif("SKYNET_DIR" not in os.environ, reason="SKYNET_DIR (SkyNet install prefix) not set")
def test_nuclear_skynet_start():
    root = Path(__file__).resolve().parents[2]
    data = Path(os.environ["SKYNET_DIR"]) / "data"
    helm = root / "src" / "eos" / "helmholtz" / "helm_table.dat"
    if not helm.exists():
        shutil.copy(data / "helm_table.dat", helm)
    sys.path.insert(0, str(root / "test" / "nuclear_detonation"))
    from make_detonation_ics import make_detonation_ics
    test_dir = root / "test" / TEST_NAME
    make_detonation_ics(str(test_dir / f"{TEST_NAME}_ics.hdf5"))
    clean_test_outputs(TEST_NAME)
    build_gizmo_for_test(TEST_NAME, default_omp_threads())
    cwd = os.getcwd()
    os.chdir(test_dir)
    try:
        run_test(TEST_NAME, default_mpi_ranks(), default_omp_threads(), param_overrides={"NuclearNetworkDataFile": str(data)})
    finally:
        os.chdir(cwd)
    assert_final_time(get_final_snapshot(TEST_NAME), TEST_NAME)
    snaps = sorted((test_dir / "output").glob("snapshot_*.hdf5"))
    with h5py.File(snaps[0], "r") as f0, h5py.File(snaps[-1], "r") as f1:
        x0 = f0["PartType0/NuclearComposition"][:]
        x1 = f1["PartType0/NuclearComposition"][:]
        u1 = f1["PartType0/InternalEnergy"][:]
    assert np.all(np.isfinite(x1)) and np.all(np.isfinite(u1))
    assert x1.min() >= 0 and abs(x1.sum(axis=1) - 1).max() < 1e-6, "mass fractions must stay a partition of unity"
    assert x1[:, 0].sum() < x0[:, 0].sum(), "the network must have burned some He4"
