"""Magnetized core collapse test (Hopkins 2015)

Tests the collapse of a magnetized molecular cloud core with a barotropic
EOS. The core should contract and form a dense central region. Checks that
the central density increases significantly and that the magnetic field
is amplified by compression.
"""

import pytest
import numpy as np
from matplotlib import pyplot as plt
import h5py
import glob
from gizmo.test import build_and_run_test, default_mpi_ranks, clean_test_outputs, assert_final_time, default_omp_threads


@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(),))
@pytest.mark.parametrize("num_omp_threads", (default_omp_threads(),))
def test_core(num_mpi_ranks, num_omp_threads):
    test_name = "core"
    clean_test_outputs(test_name)
    build_and_run_test(test_name, num_mpi_ranks, num_omp_threads)

    outputdir = f"test/{test_name}/output"
    snaps = sorted(glob.glob(outputdir + "/snapshot_*.hdf5"))
    if len(snaps) < 2:
        raise RuntimeError("GIZMO did not run successfully.")
    assert_final_time(snaps[-1], test_name)

    # Load initial and final snapshots
    with h5py.File(snaps[0], "r") as F:
        rho0 = F["PartType0/Density"][:]
        B0 = F["PartType0/MagneticField"][:]
        mass0 = F["PartType0/Masses"][:]
    with h5py.File(snaps[-1], "r") as F:
        rho_f = F["PartType0/Density"][:]
        Bf = F["PartType0/MagneticField"][:]
        mass_f = F["PartType0/Masses"][:]
        pos_f = F["PartType0/Coordinates"][:]

    # The core should have collapsed: max density should increase significantly
    rho_max_ratio = rho_f.max() / rho0.max()
    assert rho_max_ratio > 10, (
        f"Core did not collapse enough: max density ratio = {rho_max_ratio:.1f}"
    )

    # Magnetic field should be amplified by compression
    Bmag0_max = np.max(np.sqrt(np.sum(B0**2, axis=1)))
    Bmagf_max = np.max(np.sqrt(np.sum(Bf**2, axis=1)))
    assert Bmagf_max > Bmag0_max, "Magnetic field should be amplified during collapse"

    # Mass conservation
    mass_err = abs(mass_f.sum() - mass0.sum()) / mass0.sum()
    assert mass_err < 1e-3, f"Mass not conserved: relative error {mass_err:.6f}"

    # ------------------------------------------------------------------
    # Diagnostic figures + summary table (best-effort; never fails the test).
    # Documents the collapse against first-principles physics: central-density
    # runaway and first-core bounce vs the free-fall time, exact adherence to
    # the forced barotropic EOS, virial ratio, radial profiles (rho, v_r,
    # v_phi, |B|, dt), div.B cleaning error, momentum/angular-momentum/core-
    # position conservation, per-particle timestep correctness, the timestep
    # floor, and face-on/edge-on density slices with the B field overlaid.
    # Requires the OUTPUT_* diagnostic fields in Config.sh and meshoid; if a
    # field or dependency is missing the routine degrades gracefully.
    # ------------------------------------------------------------------
    try:
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from core_analysis import make_all_core_diagnostics
        make_all_core_diagnostics(outputdir)  # -> test/core/diag_plots/*.pdf
    except Exception as e:
        print(f"[core] diagnostic analysis skipped: {e}")
