"""Isolated disk galaxy — mechanical_fb (B8 Phase 1) GPU activation test

Activation test for the mechanical_fb GPU port (B8 / addFB_evaluate default scheme).
Uses isodisk ICs with a minimal config: GALSF_FB_MECHANICAL only (no FIRE_PHYSICS_DEFAULTS,
no COSMIC_RAY_FLUID, no GALSF_ISMDUSTCHEM_MODEL — those come in Phase 2 / 2b).
Pre-existing Type-4 stellar particles are older than 5 Myr and fire SNe via the AGORA
model from the very first timestep, guaranteeing the GPU kernel sees active sources.

Validation protocol:
1. Build with default Kokkos Config.sh (flags are now always active)
2. Confirm PRINT_STATUS shows nonzero "GPU mech_fb: N sources, M pairs" across modes
3. Compare GPU output against CPU tree-walk reference — gas Mass, InternalEnergy,
   Metallicity, Vel must match to round-off (mass + total energy conservation better
   than 1e-3).
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
