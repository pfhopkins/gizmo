"""cbe_density_wave — 1D density-modulated counter-streaming CBE test.

Density-modulated particle spacing (rho=1+eps cos(2 pi x/L)) with the same
(+v/-v) 2-basis internal distribution. Gates: global conservation, the
cbe_diagnostics face-residual / bracket-fail receipt, clean controlled stop.

NOTE (2026-06-02): per-stream gates meaningful only after the C7 IC-reader
merge (see cbe_two_stream).
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from gizmo.test import build_and_run_test, get_final_snapshot, skip_if_not_implemented
from cbe_vlasov_common import read_snapshot, conservation, read_cbe_diagnostics, plot_cbe_streams

TEST_NAME = "cbe_density_wave"

# GIZMO runs these ICs and writes VlasovMoments, but never emits the diagnostics receipt
# the gates read; the CBE integrator here contains no file output at all.
skip_if_not_implemented("cbe_diagnostics", "The CBE Vlasov-moment diagnostics output")
DIM = 1


@pytest.mark.parametrize("num_mpi_ranks,num_omp_threads", [(2, 0), (1, 2)])
def test_cbe_density_wave(num_mpi_ranks, num_omp_threads):
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
