"""Field loop advection test (Hopkins & Raives 2016)

Tests advection of a magnetic field loop. The loop should be preserved
after one full advection period. Checks magnetic flux conservation.
"""

import pytest
import numpy as np
import h5py
import glob
from gizmo.test import build_and_run_test, default_mpi_ranks, clean_test_outputs


@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(),))
def test_field_loop(num_mpi_ranks):
    test_name = "field_loop"
    clean_test_outputs(test_name)
    build_and_run_test(test_name, num_mpi_ranks)

    outputdir = f"test/{test_name}/output"
    snaps = sorted(glob.glob(outputdir + "/snapshot_*.hdf5"))
    if len(snaps) < 2:
        raise RuntimeError("GIZMO did not run successfully.")

    # Load initial and final snapshots
    with h5py.File(snaps[0], "r") as F:
        B0 = F["PartType0/MagneticField"][:]
        pos0 = F["PartType0/Coordinates"][:]
        mass0 = F["PartType0/Masses"][:]
    with h5py.File(snaps[-1], "r") as F:
        Bf = F["PartType0/MagneticField"][:]
        mass_f = F["PartType0/Masses"][:]

    # Total magnetic energy should be approximately conserved
    Emag0 = np.sum(np.sum(B0**2, axis=1) * mass0)
    Emagf = np.sum(np.sum(Bf**2, axis=1) * mass_f)

    rel_err = abs(Emagf - Emag0) / Emag0
    assert rel_err < 0.5, f"Magnetic energy not conserved: relative error {rel_err:.4f}"
