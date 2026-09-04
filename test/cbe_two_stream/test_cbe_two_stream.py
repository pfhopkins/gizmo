"""cbe_two_stream — 1D counter-streaming CBE test.

Each Type=1 particle carries a (+v/-v) 2-basis internal distribution. With
zero bulk velocity the streams advect at +/-v while total density stays
uniform. No reference solver here (cf. the AGS/SIDM no-reference tests):
the gates are global conservation, the cbe_diagnostics face-residual /
bracket-fail receipt, and a clean controlled stop.

NOTE (2026-06-02): the IC's VlasovMoments dataset is only honoured once the
C7 IC-reader is merged into this branch. Until then GIZMO synthesizes a
cold single-stream default and these per-stream gates will not be meaningful.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from gizmo.test import build_and_run_test, get_final_snapshot, assert_final_time, skip_if_not_implemented
from cbe_vlasov_common import read_snapshot, conservation, read_cbe_diagnostics, plot_cbe_streams

TEST_NAME = "cbe_two_stream"

# GIZMO runs these ICs and writes VlasovMoments, but never emits the diagnostics receipt
# the gates read; the CBE integrator here contains no file output at all.
skip_if_not_implemented("cbe_diagnostics", "The CBE Vlasov-moment diagnostics output")
DIM = 1


@pytest.mark.parametrize("num_mpi_ranks,num_omp_threads", [(2, 0), (1, 2)])
def test_cbe_two_stream(num_mpi_ranks, num_omp_threads):
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

    plot_cbe_streams(TEST_NAME, snap0, snapf, DIM,
                     [(+1.0, "+1"), (-1.0, "-1")],
                     output_dir=f"test/{TEST_NAME}")
