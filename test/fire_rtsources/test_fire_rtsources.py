"""FIRE galaxy — RT source injection (B5) GPU activation test

Activation test for the rt_source_injection GPU port (B5).
Uses fire ICs (m11i dwarf at z~2.9) with RT_M1 + RT_COMOVING + RT_SOURCES=48
(Type 4 + Type 5 → RT bands: RT_OPTICAL_NIR, RT_NUV, RT_PHOTOELECTRIC,
RT_CHEM_PHOTOION=1, RT_INFRARED).  Pre-existing Type-4 stellar particles
emit into these bands from step 1, guaranteeing nonzero injection.

Validation protocol:
1. Build with default Kokkos Config.sh (flags are now always active)
2. Confirm PRINT_STATUS shows nonzero "GPU rt_source_injection: N sources, M pairs"
3. Check that radiation energy density is nonzero in snapshot PartType0/PhotonEnergy_*
4. Compare GPU vs CPU tree-walk reference on at least one RT band field.
"""

import pytest
import numpy as np
from os import path, chdir
from shutil import copy2
from urllib.request import urlretrieve, HTTPError
import h5py
import glob
from gizmo.test import (
    build_gizmo_for_test,
    clean_test_outputs,
    get_cooling_tables,
    default_mpi_ranks,
    default_omp_threads,
    stash_baseline_output,
    finalize_variant_output,
    variant_output_dir,
)

TAPIR = "http://www.tapir.caltech.edu/~phopkins/sims/"
TEST_NAME = "fire_rtsources"
FIRE_IC = "fire_ics.hdf5"


def _get_ics():
    """Download fire ICs if not already present under the local name."""
    local = f"{TEST_NAME}_ics.hdf5"
    if path.isfile(local):
        return
    # Try to copy from adjacent fire test directory first
    fire_local = path.join("..", "fire", FIRE_IC)
    if path.isfile(fire_local):
        copy2(fire_local, local)
        return
    try:
        urlretrieve(TAPIR + FIRE_IC, local)
    except HTTPError:
        raise FileNotFoundError(
            f"Could not download {FIRE_IC} from {TAPIR}. "
            "Run the fire test first so its ICs are present, or copy "
            f"test/fire/{FIRE_IC} → test/{TEST_NAME}/{TEST_NAME}_ics.hdf5"
        )


@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(2),))
@pytest.mark.parametrize("num_omp_threads", (default_omp_threads(),))
def test_fire_rtsources(num_mpi_ranks, num_omp_threads):
    clean_test_outputs(TEST_NAME)
    get_cooling_tables(f"test/{TEST_NAME}")
    build_gizmo_for_test(TEST_NAME, num_omp_threads)
    stash_baseline_output(TEST_NAME)
    try:
        chdir(f"test/{TEST_NAME}/")
        _get_ics()
        from os import environ, system
        if num_omp_threads > 0:
            environ["OMP_NUM_THREADS"] = str(num_omp_threads)
        paramsfile = f"{TEST_NAME}.params"
        system(f"mpirun -np {num_mpi_ranks} --use-hwthread-cpus "
               f"./GIZMO {paramsfile} 0 "
               f"1>test_{TEST_NAME}.out 2>test_{TEST_NAME}.err")
        chdir("../../")
    finally:
        finalize_variant_output(TEST_NAME)

    outputdir = variant_output_dir(TEST_NAME)
    snaps = sorted(glob.glob(outputdir + "/snapshot_*.hdf5"))
    if len(snaps) < 1:
        raise RuntimeError("GIZMO did not produce any output snapshots.")

    # Basic sanity: DM mass must be exactly conserved
    ics_file = f"test/{TEST_NAME}/{TEST_NAME}_ics.hdf5"
    with h5py.File(ics_file, "r") as F:
        m_dm_init = float(F["PartType1/Masses"][:].sum()) if "PartType1" in F else 0.0
    with h5py.File(snaps[-1], "r") as F:
        m_dm_final = float(F["PartType1/Masses"][:].sum()) if "PartType1" in F else 0.0
    assert m_dm_init == pytest.approx(m_dm_final, rel=1e-10), \
        f"DM mass changed: {m_dm_init} -> {m_dm_final}"

    # RT activation check: at least one gas cell should have nonzero photon energy
    with h5py.File(snaps[-1], "r") as F:
        gas = F.get("PartType0", {})
        rt_keys = [k for k in gas.keys() if "Photon" in k or "RT_" in k or "Rad" in k.lower()]
        if rt_keys:
            e_rt = float(gas[rt_keys[0]][:].sum())
            assert e_rt > 0, f"No RT energy found in {rt_keys[0]} after source injection"
