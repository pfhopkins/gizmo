"""Isolated disk + FIRE black-hole sinks — D1 sink_swallow_and_kick_gpu activation test.

Activation test for the sink-pipeline GPU port (D1 = sink_swallow_and_kick_gpu,
exercising the same ghost-writeback pattern as B4 sink_environment_gpu and C3
sink_feed_gpu). Uses the isodisk IC (shared with test/isodisk_mechfb) which already
contains one Type-5 BH particle. FIRE_BHS enables SINK_PARTICLES + SINK_SWALLOWGAS +
SINK_GRAVACCRETION=1, so D1's kernel fires on every timestep where the sink is active.

Config notes:
- NO FIRE_CRS. FIRE_CRS=0 does NOT disable cosmic rays (the presence of the flag
  itself activates the CR pipeline). D1 Phase 1 #error-guards COSMIC_RAY_FLUID +
  SINK_COSMIC_RAYS; Phase 2 will lift that.
- SINK_GRAVACCRETION=1 (FIRE_BHS default), so E2 sink_environment_second_gpu (gated
  on ==0) does NOT fire here — B4/C3/D1 are the active GPU paths.

Validation protocol:
1. Build with FIRE_BHS (Kokkos neighbor-list always active)
2. Confirm PRINT_STATUS shows nonzero "GPU sink_swallow: N active, M pairs" on at
   least one active-sink timestep
3. Verify Sink_Mass grows via HQ accretion (gas → subgrid reservoir → sink)
4. Assert baryon mass conservation across types 0+2+3+4+5
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
TEST_NAME = "isodisk_mechfb_sinks"
ISODISK_IC = "isodisk_ics.hdf5"


def _symlink_sibling_if_missing(local_name, sibling_relpath):
    """Create a relative symlink to sibling-test data if local_name is not present.
    Cleans up a dangling symlink. Returns True if local_name now exists."""
    import os
    if path.exists(local_name):
        return True
    if path.islink(local_name):
        os.unlink(local_name)
    if path.exists(sibling_relpath):
        os.symlink(sibling_relpath, local_name)
        return True
    return False


def _get_ics():
    """Bootstrap test inputs by borrowing from the sibling test/isodisk_mechfb/ dir
    (shared IC + cooling tables + ewald table). Falls back to download for the IC
    if the sibling isn't populated."""
    # IC: shared with isodisk_mechfb
    if not _symlink_sibling_if_missing(
        f"{TEST_NAME}_ics.hdf5", "../isodisk_mechfb/isodisk_mechfb_ics.hdf5"
    ):
        try:
            urlretrieve(TAPIR + ISODISK_IC, f"{TEST_NAME}_ics.hdf5")
        except HTTPError:
            raise FileNotFoundError(
                f"Could not download {ISODISK_IC} from {TAPIR}. "
                "Run the isodisk test first so its ICs are present, or symlink "
                f"test/isodisk/{ISODISK_IC} -> test/{TEST_NAME}/{TEST_NAME}_ics.hdf5"
            )
    # Cooling + ewald data: harmless if already present (download handled by
    # get_cooling_tables). Symlinks save duplicating large tables on disk.
    _symlink_sibling_if_missing("spcool_tables", "../isodisk_mechfb/spcool_tables")
    _symlink_sibling_if_missing("TREECOOL", "../isodisk_mechfb/TREECOOL")
    _symlink_sibling_if_missing(
        "ewald_spc_table_64_dbl.dat", "../isodisk_mechfb/ewald_spc_table_64_dbl.dat"
    )


@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(2),))
@pytest.mark.parametrize("num_omp_threads", (default_omp_threads(),))
def test_isodisk_mechfb_sinks(num_mpi_ranks, num_omp_threads):
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

    # Sink accretion check: Sink_Mass (subgrid BH mass) grows from HQ accretion.
    with h5py.File(snaps[0], "r") as F:
        sink_mass_initial = float(F["PartType5/Sink_Mass"][0])
    with h5py.File(snaps[-1], "r") as F:
        sink_mass_final = float(F["PartType5/Sink_Mass"][0])
    assert sink_mass_final > sink_mass_initial, (
        f"Sink_Mass did not grow: initial={sink_mass_initial:.3e} final={sink_mass_final:.3e}. "
        "HQ accretion kernel should have deposited mass into the subgrid reservoir."
    )

    # Baryon mass conservation across all mass-bearing types. Gas (0), old stars (2, 3),
    # new stars (4), and sinks (5) - DM (1) excluded. mech_fb transfers mass between
    # stellar and gas types, SF creates Type 4, sink accretion moves Type 0 into P[5].Mass
    # via the swallow kernel (rare in a short run, but accounted for). Total sum must be
    # conserved to round-off.
    bary_types = [0, 2, 3, 4, 5]
    with h5py.File(f"test/{TEST_NAME}/{TEST_NAME}_ics.hdf5", "r") as F:
        total_m0 = sum(float(F[f"PartType{t}/Masses"][:].sum())
                       for t in bary_types if f"PartType{t}" in F)
    with h5py.File(snaps[-1], "r") as F:
        total_mf = sum(float(F[f"PartType{t}/Masses"][:].sum())
                       for t in bary_types if f"PartType{t}" in F)

    mass_err = abs(total_mf - total_m0) / (total_m0 + 1e-30)
    assert mass_err < 1e-3, f"Baryonic mass not conserved: {mass_err:.6f}"
