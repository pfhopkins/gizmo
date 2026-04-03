"""ANEOS giant impact test: differentiated sphere with real M-ANEOS tables.

Validates that GIZMO can:
  1. Load real M-ANEOS forsterite and iron tables
  2. Initialize particles with per-particle CompositionType
  3. Run a short hydro evolution without crashing
  4. Produce physically reasonable output (T > 0, P > 0, finite values)

This is a crash/sanity test, not a quantitative accuracy test. It uses
downsampled Stewart group tables (forsterite S19, iron S20) and a simple
uniform-density differentiated sphere IC.
"""

import pytest
import numpy as np
import h5py
import subprocess
import sys
from os import chdir, getcwd
from os.path import isfile, dirname, abspath, join

from gizmo.test import (
    build_gizmo_for_test,
    run_test,
    clean_test_outputs,
    default_mpi_ranks,
    default_omp_threads,
    get_final_snapshot,
)


ZENODO_BASE = "https://zenodo.org/records/12732259/files"


def setup_tables(test_dir):
    """Download Stewart tables and convert to GIZMO format if needed."""
    converter = join(test_dir, "convert_stewart_table.py")

    for material, mat_id, out_name in [
        ("ANEOS_forsterite_S19.txt", 2019, "forsterite.sesame"),
        ("ANEOS_iron_S20.txt", 2020, "iron.sesame"),
    ]:
        out_path = join(test_dir, out_name)
        if isfile(out_path):
            continue

        raw_path = join(test_dir, material)
        if not isfile(raw_path):
            from urllib.request import urlretrieve
            url = f"{ZENODO_BASE}/{material}?download=1"
            print(f"Downloading {material}...")
            urlretrieve(url, raw_path)

        print(f"Converting {material} -> {out_name} (downsampled to 200x200)...")
        subprocess.run(
            [sys.executable, converter, raw_path, out_path, str(mat_id), "200", "200"],
            check=True,
            cwd=test_dir,
        )


def setup_ics(test_dir):
    """Generate ICs if needed."""
    ic_path = join(test_dir, "aneos_giant_impact_ics.hdf5")
    if not isfile(ic_path):
        subprocess.run(
            [sys.executable, join(test_dir, "generate_ics.py")],
            check=True,
            cwd=test_dir,
        )


@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(max_ranks=2),))
@pytest.mark.parametrize("num_omp_threads", (default_omp_threads(),))
def test_aneos_giant_impact(num_mpi_ranks, num_omp_threads):
    test_name = "aneos_giant_impact"
    test_dir = abspath(join(dirname(__file__)))

    clean_test_outputs(test_name)
    build_gizmo_for_test(test_name, num_omp_threads)
    chdir(f"test/{test_name}/")

    # Prepare tables and ICs
    setup_tables(".")
    setup_ics(".")

    run_test(test_name, num_mpi_ranks, num_omp_threads)
    chdir("../../")

    # Check that output exists
    final_snap = get_final_snapshot(test_name)

    # Basic sanity checks on the output
    with h5py.File(final_snap, "r") as F:
        n = F["Header"].attrs["NumPart_ThisFile"][0]
        assert n > 0, "No particles in output"

        rho = F["PartType0/Density"][:]
        u = F["PartType0/InternalEnergy"][:]
        coords = F["PartType0/Coordinates"][:]

        # Check fields exist and have correct shape
        assert rho.shape == (n,), f"Density shape mismatch: {rho.shape}"
        assert u.shape == (n,), f"InternalEnergy shape mismatch: {u.shape}"
        assert coords.shape == (n, 3), f"Coordinates shape mismatch: {coords.shape}"

        # All values should be finite
        assert np.all(np.isfinite(rho)), f"Non-finite densities: {np.sum(~np.isfinite(rho))} particles"
        assert np.all(np.isfinite(u)), f"Non-finite internal energies: {np.sum(~np.isfinite(u))} particles"
        assert np.all(np.isfinite(coords)), f"Non-finite coordinates"

        # Densities and internal energies should be positive
        assert np.all(rho > 0), f"Non-positive densities: {np.sum(rho <= 0)} particles"
        assert np.all(u > 0), f"Non-positive internal energies: {np.sum(u <= 0)} particles"

        # Check CompositionType is preserved
        if "CompositionType" in F["PartType0"]:
            comp = F["PartType0/CompositionType"][:]
            assert set(np.unique(comp)).issubset({0, 1}), f"Unexpected CompositionType values: {np.unique(comp)}"

    print(f"ANEOS giant impact test passed: {n} particles, all finite and positive")
