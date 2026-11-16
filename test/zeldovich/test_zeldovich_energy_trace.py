"""Zeldovich pancake with the energy step trace: the comoving (Layzer-Irvine) energy budget closes.

E_kin + E_th + E_pot changes only by the expansion term int (2 E_kin + E_pot + 3(gamma-1) U) dln a in a comoving run;
the trace integrates it between full syncs (LI) and must close the time-integration residual to the 1% bar of
gizmo.conservation_trace, through and after the caustic (shock heating moves kinetic into thermal energy).
"""

import pytest

from gizmo.test import build_and_run_test, clean_test_outputs, default_mpi_ranks, default_omp_threads, variant_output_dir
from gizmo.conservation_trace import check_conservation_traces

FLAGS = ("OUTPUT_ENERGY_STEP_TRACE",)


@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(),))
@pytest.mark.parametrize("num_omp_threads", (default_omp_threads(),))
def test_zeldovich_layzer_irvine_closes(num_mpi_ranks, num_omp_threads):
    test_name = "zeldovich"
    clean_test_outputs(test_name, FLAGS)
    build_and_run_test(test_name, num_mpi_ranks, num_omp_threads, FLAGS)
    out = variant_output_dir(test_name, FLAGS)
    rep = check_conservation_traces(out)
    print(rep.summary())
    rows = [dict(t.split("=", 1) for t in l.split()[1:]) for l in open(f"{out}/energy_step_trace.txt") if l.startswith("SYNC")]
    assert len(rows) > 10 and float(rows[-1]["t"]) > 0.99, "no full syncs to the end of the run"
    li = float(rows[-1]["LI"])
    assert li != 0, "Layzer-Irvine term not integrated"
    assert rep.ok, rep.summary()
