"""Isolated disk galaxy with thermal supernova feedback.

The isodisk IC with GALSF_FB_THERMAL (pure thermal SN energy injection) on a
COOLING + GALSF + METALS base (no FIRE_PHYSICS_DEFAULTS). Pre-existing star
particles are older than 5 Myr and produce SNe from the first timestep.

Checks: the run reaches its final time, and total gas + star + black-hole mass is
conserved to 0.1%.
"""

import pytest
import numpy as np
from os import path, chdir
from urllib.request import urlretrieve, HTTPError
import h5py
import glob
from gizmo.test import (
    build_gizmo_for_test,
    clean_test_outputs,
    get_cooling_tables,
    default_mpi_ranks,
    default_omp_threads,
    run_test,
    stash_baseline_output,
    finalize_variant_output,
    variant_output_dir,
    assert_final_time,
)

TAPIR = "http://www.tapir.caltech.edu/~phopkins/sims/"
TEST_NAME = "isodisk_thermalfb"
ISODISK_IC = "isodisk_ics.hdf5"


def _get_ics():
    """Download isodisk ICs if not already present under the local name."""
    local = f"{TEST_NAME}_ics.hdf5"
    if path.isfile(local):
        return
    try:
        urlretrieve(TAPIR + ISODISK_IC, local)
    except HTTPError:
        raise FileNotFoundError(
            f"Could not download {ISODISK_IC} from {TAPIR}. "
            "Run the isodisk test first so its ICs are present, then symlink or copy "
            f"test/isodisk/{ISODISK_IC} → test/{TEST_NAME}/{TEST_NAME}_ics.hdf5"
        )


@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(2),))
@pytest.mark.parametrize("num_omp_threads", (default_omp_threads(),))
def test_isodisk_thermalfb(num_mpi_ranks, num_omp_threads):
    clean_test_outputs(TEST_NAME)
    get_cooling_tables(f"test/{TEST_NAME}")
    build_gizmo_for_test(TEST_NAME, num_omp_threads)
    stash_baseline_output(TEST_NAME)
    try:
        chdir(f"test/{TEST_NAME}/")
        _get_ics()
        run_test(TEST_NAME, num_mpi_ranks, num_omp_threads)
        chdir("../../")
    finally:
        finalize_variant_output(TEST_NAME)

    outputdir = variant_output_dir(TEST_NAME)
    snaps = sorted(glob.glob(outputdir + "/snapshot_*.hdf5"))
    if len(snaps) < 1:
        raise RuntimeError("GIZMO did not produce any output snapshots.")
    assert_final_time(snaps[-1], TEST_NAME)

    # Basic sanity: gas mass conservation (thermal_fb deposits ejecta mass)
    with h5py.File(f"test/{TEST_NAME}/{TEST_NAME}_ics.hdf5", "r") as F:
        m0_dm = float(F["PartType1/Masses"][:].sum()) if "PartType1" in F else 0.0
        m0_gas  = float(F["PartType0/Masses"][:].sum()) if "PartType0" in F else 0.0
        m0_star = float(F["PartType4/Masses"][:].sum()) if "PartType4" in F else 0.0
        m0_star2 = float(F["PartType2/Masses"][:].sum()) if "PartType2" in F else 0.0
        m0_star3 = float(F["PartType3/Masses"][:].sum()) if "PartType3" in F else 0.0
        m0_bh = float(F["PartType5/Masses"][:].sum()) if "PartType5" in F else 0.0
        total_m0 = m0_gas + m0_star + m0_bh + m0_star2 + m0_star3
    with h5py.File(snaps[-1], "r") as F:
        mf_dm = float(F["PartType1/Masses"][:].sum()) if "PartType1" in F else 0.0
        mf_gas  = float(F["PartType0/Masses"][:].sum()) if "PartType0" in F else 0.0
        mf_star = float(F["PartType4/Masses"][:].sum()) if "PartType4" in F else 0.0
        mf_star2 = float(F["PartType2/Masses"][:].sum()) if "PartType2" in F else 0.0
        mf_star3 = float(F["PartType3/Masses"][:].sum()) if "PartType3" in F else 0.0
        mf_bh = float(F["PartType5/Masses"][:].sum()) if "PartType5" in F else 0.0
        total_mf = mf_gas + mf_star + mf_bh + mf_star2 + mf_star3

    print(m0_gas, m0_star, m0_star2, m0_star3, m0_bh, m0_dm)
    print(mf_gas, mf_star,mf_star2, mf_star3,  mf_bh, mf_dm)

    mass_err = abs(total_mf - total_m0) / (total_m0 + 1e-30)
    assert mass_err < 1e-3, f"Gas+star+BH total mass not conserved: {mass_err:.6f}"
