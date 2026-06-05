"""cbe_free_slot_3d — 3D slab free-slot injection CBE test.

Same physics as cbe_free_slot_1d on a [1.0 x 0.0625 x 0.0625] slab (long
in x, thin in y,z), velocities along x only. Confirms the free-slot
routing + conservation survive in 3D (NMOMENTS=4) and under domain
decomposition (multi-rank).

NOTE (2026-06-02): meaningful only after the C7 IC-reader merge.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from gizmo.test import build_and_run_test, get_final_snapshot
from cbe_vlasov_common import (read_snapshot, conservation, read_cbe_diagnostics,
                               total_band_mass, plot_cbe_streams)

TEST_NAME = "cbe_free_slot_3d"
DIM = 3
V_BANDS = [(+2.0, "+2"), (+1.0, "+1"), (0.0, "0"), (-1.0, "-1"), (-2.0, "-2")]


@pytest.mark.parametrize("num_mpi_ranks,num_omp_threads", [(2, 0), (4, 0)])
def test_cbe_free_slot_3d(num_mpi_ranks, num_omp_threads):
    build_and_run_test(TEST_NAME, num_mpi_ranks, num_omp_threads)
    outputdir = f"test/{TEST_NAME}/output"
    final = get_final_snapshot(TEST_NAME)

    snap0 = read_snapshot(f"{outputdir}/snapshot_000.hdf5", DIM)
    snapf = read_snapshot(final, DIM)

    c0, cf = conservation(snap0), conservation(snapf)
    assert abs(cf["M"] - c0["M"]) <= 1e-10 * c0["M"], "total mass not conserved"
    assert np.all(np.abs(cf["P"] - c0["P"]) <= 1e-9), "total momentum not conserved"

    diag = read_cbe_diagnostics(outputdir)
    assert diag, "no cbe_diagnostics.txt rows"
    assert diag[0]["face_res_max"] <= 1e-11, "face mass-flux residual too large"
    assert diag[0]["bracket_fail"] == 0, "root-find bracket failures"

    for v, lab in V_BANDS:
        print(f"  band v={lab:>3s}: M0={total_band_mass(snap0, v):.4e}  "
              f"Mf={total_band_mass(snapf, v):.4e}")
    plot_cbe_streams(TEST_NAME, snap0, snapf, DIM, V_BANDS,
                     output_dir=f"test/{TEST_NAME}")
