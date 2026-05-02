"""Phase 17e smoke test: Jutzi 2008 P-alpha + Grady-Kipp + Drucker-Prager
(EOS_DAMAGE_POROSITY=7) on a small basalt-on-basalt cratering impact.

Validates:
  1. GIZMO runs to completion with EOS_DAMAGE_POROSITY=7.
  2. Final snapshot contains Damage, Distention, ActiveCracks datasets.
  3. Damage field has at least one particle with D > 0 (shock triggered GK).
  4. Distention field has at least one particle below the initial alpha_0=1.275
     (porosity was compacted by shock pressure > P_s).
  5. Mass is conserved to <0.1%.

Full physics validation (Jutzi+ 2008 Fig. 4-7 crater profiles) is deferred.
"""

import glob
import os
import sys

import h5py
import numpy as np
import pytest

from gizmo.test import (
    build_and_run_test,
    clean_test_outputs,
    default_mpi_ranks,
    default_omp_threads,
    assert_final_time,
)

TEST_NAME   = "jutzi_crater"
BASALT_A0   = 1.275   # initial distention from material table (basalt, slot 15)


def _make_ics():
    """Generate the IC file if not present."""
    ic_path = f"test/{TEST_NAME}/jutzi_crater_ics.hdf5"
    if not os.path.exists(ic_path):
        import subprocess, sys
        script = f"test/{TEST_NAME}/make_ics.py"
        subprocess.run([sys.executable, script], check=True,
                       cwd=os.getcwd())
    return ic_path


@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(),))
@pytest.mark.parametrize("num_omp_threads", (default_omp_threads(),))
def test_jutzi_crater_smoke(num_mpi_ranks, num_omp_threads):
    """Smoke: build + run, check damage / distention fields are populated."""
    _make_ics()
    clean_test_outputs(TEST_NAME)
    build_and_run_test(TEST_NAME, num_mpi_ranks, num_omp_threads)

    outputdir = f"test/{TEST_NAME}/output"
    snaps = sorted(glob.glob(outputdir + "/snapshot_*.hdf5"))
    if not snaps:
        raise RuntimeError("No snapshots found; GIZMO did not run.")
    assert_final_time(snaps[-1], TEST_NAME)

    # Use the last snapshot for physics checks
    with h5py.File(snaps[-1], "r") as f:
        g = f["PartType0"]
        keys = list(g.keys())

        # Fields must be present
        assert "Damage"       in keys, "Damage field missing in snapshot"
        assert "Distention"   in keys, "Distention field missing in snapshot"
        assert "ActiveCracks" in keys, "ActiveCracks field missing in snapshot"

        masses = g["Masses"][:]
        D      = g["Damage"][:]
        alpha  = g["Distention"][:]

    with h5py.File(snaps[0], "r") as f:
        mass0 = f["PartType0/Masses"][:]

    # Mass conservation
    mass_err = abs(masses.sum() - mass0.sum()) / mass0.sum()
    assert mass_err < 1e-3, f"Mass error {mass_err:.4e} exceeds 0.1%"

    # At 3 km/s into basalt, shock pressure >> P_s=7e10 dyn/cm²:
    # expect fully-damaged particles (D > 0) in the impact zone.
    assert D.max() > 0.0, "No damaged particles found — Grady-Kipp did not fire"

    # P-alpha: P >> P_s means alpha should be driven to 1 in shocked zone.
    # At minimum, some particles should have alpha < initial alpha_0.
    assert alpha.min() < BASALT_A0, (
        f"No compaction detected: min(alpha)={alpha.min():.4f} >= alpha_0={BASALT_A0}"
    )
