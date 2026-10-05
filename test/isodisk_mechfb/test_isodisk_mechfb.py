"""Isolated disk galaxy with mechanical feedback.

The isodisk IC with GALSF_FB_MECHANICAL (SNe and stellar mass loss) on a
COOLING + GALSF + METALS base. Pre-existing star particles are older than 5 Myr
and produce SNe from the first timestep.

Checks: the run reaches its final time, and total baryonic mass (gas plus all
stellar types) is conserved to 1%.
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
TEST_NAME = "isodisk_mechfb"
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
def test_isodisk_mechfb(num_mpi_ranks, num_omp_threads):
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

    # Basic sanity: total baryonic mass conservation.
    # mech_fb transfers mass from old stellar populations (Type2/3) into gas (Type0),
    # and SF converts gas (Type0) into new stars (Type4), so Types 0+2+3+4 must all
    # be included in the budget.  DM (Type1) and sinks (Type5) are excluded.
    bary_types = [0, 2, 3, 4]
    with h5py.File(f"test/{TEST_NAME}/{TEST_NAME}_ics.hdf5", "r") as F:
        total_m0 = sum(float(F[f"PartType{t}/Masses"][:].sum())
                       for t in bary_types if f"PartType{t}" in F)
    with h5py.File(snaps[-1], "r") as F:
        total_mf = sum(float(F[f"PartType{t}/Masses"][:].sum())
                       for t in bary_types if f"PartType{t}" in F)

    mass_err = (total_mf - total_m0) / (total_m0 + 1e-30)
    assert abs(mass_err) < 1e-2, f"Baryonic mass not conserved: {mass_err:.6f}"
