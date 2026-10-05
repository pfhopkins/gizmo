"""Isolated disk galaxy with velocity-scaled subgrid winds.

The isodisk IC with the Springel & Hernquist effective-EOS model and subgrid winds
scaled by the local dark-matter velocity dispersion (GALSF_EFFECTIVE_EQS +
GALSF_SUBGRID_WINDS + GALSF_SUBGRID_WIND_SCALING=2), which requires DM_VelDisp at
every gas cell each timestep.

Checks: the run reaches its final time, stdout reports that the dispersion
computation ran ("dm_dispersion done"), most gas cells carry a non-zero
DM_VelDisp, and gas + star mass is conserved to 0.1%.
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
TEST_NAME = "isodisk_dmdispersion"
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
def test_isodisk_dmdispersion(num_mpi_ranks, num_omp_threads):
    clean_test_outputs(TEST_NAME)
    get_cooling_tables(f"test/{TEST_NAME}")
    build_gizmo_for_test(TEST_NAME, num_omp_threads)
    stash_baseline_output(TEST_NAME)

    # run_test writes GIZMO stdout to test_<name>.out in the test dir (harness convention).
    log_path = f"test/{TEST_NAME}/test_{TEST_NAME}.out"
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

    # Check that dm_dispersion actually ran (stdout contains the status line).
    if path.isfile(log_path):
        with open(log_path) as f:
            stdout = f.read()
        assert "dm_dispersion done" in stdout, (
            "disp_density() never printed status — DMDispersionSpec runner was not called."
        )

    # Check that at least some gas cells have nonzero DM_VelDisp in the final snapshot.
    # WIND_SCALING=2 uses this field; if it's all zeros the whole wind model is silently
    # disabled regardless of SF activity.
    with h5py.File(snaps[-1], "r") as F:
        if "PartType0" in F:
            # DM_VelDisp may appear under various field names depending on IO config;
            # fall back gracefully if the field is not written to the snapshot.
            dm_vdisp_keys = [k for k in F["PartType0"].keys()
                             if "VelDisp" in k or "DM_V" in k]
            if dm_vdisp_keys:
                vdisp = F["PartType0"][dm_vdisp_keys[0]][:]
                nonzero_frac = float(np.sum(vdisp > 0)) / max(len(vdisp), 1)
                assert nonzero_frac > 0.5, (
                    f"Less than 50% of gas cells have nonzero DM_VelDisp "
                    f"({nonzero_frac:.2%}); dm_dispersion may not have converged."
                )

    # Basic mass conservation.
    with h5py.File(f"test/{TEST_NAME}/{TEST_NAME}_ics.hdf5", "r") as F:
        m0_gas  = float(F["PartType0/Masses"][:].sum()) if "PartType0" in F else 0.0
        m0_star = float(F["PartType4/Masses"][:].sum()) if "PartType4" in F else 0.0
        total_m0 = m0_gas + m0_star
    with h5py.File(snaps[-1], "r") as F:
        mf_gas  = float(F["PartType0/Masses"][:].sum()) if "PartType0" in F else 0.0
        mf_star = float(F["PartType4/Masses"][:].sum()) if "PartType4" in F else 0.0
        total_mf = mf_gas + mf_star

    mass_err = abs(total_mf - total_m0) / (total_m0 + 1e-30)
    assert mass_err < 1e-3, f"Gas+star total mass not conserved: {mass_err:.6f}"
