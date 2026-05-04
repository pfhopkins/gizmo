"""Phase 17g smoke test: GRAIN_FLUID_PROMOTION — Type 3 grain → Type 0 solid body.

Validates:
  1. GIZMO runs to completion with GRAIN_FLUID + GRAIN_FLUID_PROMOTION.
  2. The over-threshold grain particle disappears from PartType3.
  3. A new PartType0 particle appears (N_gas+1 total gas particles in final snap).
  4. The promoted solid has CompositionType = MATERIAL_TILLOTSON_OLIVINE (5).
  5. Total (gas + grain) mass is conserved to <0.1%.
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

TEST_NAME             = "grain_promotion"
MATERIAL_OLIVINE      = 5     # MATERIAL_TILLOTSON_OLIVINE in composition_registry.h
GRAIN_MASS_CGS        = 0.02  # must match make_ics.py
THRESHOLD_CGS         = 0.01  # must match params GrainPromotion_MassThresh_cgs
N_GAS_IC              = 64    # 8×8 gas grid from make_ics.py


def _make_ics():
    ic_path = f"test/{TEST_NAME}/grain_promotion_ics.hdf5"
    if not os.path.exists(ic_path):
        import subprocess, sys
        script = f"test/{TEST_NAME}/make_ics.py"
        subprocess.run([sys.executable, script], check=True,
                       cwd=os.getcwd())
    return ic_path


@pytest.mark.parametrize("num_mpi_ranks", (2,))
@pytest.mark.parametrize("num_omp_threads", (default_omp_threads(),))
def test_grain_promotion_smoke(num_mpi_ranks, num_omp_threads):
    """Build + run; verify grain is promoted to Type-0 olivine solid."""
    _make_ics()
    clean_test_outputs(TEST_NAME)
    build_and_run_test(TEST_NAME, num_mpi_ranks, num_omp_threads)

    outputdir = f"test/{TEST_NAME}/output"
    snaps = sorted(glob.glob(outputdir + "/snapshot_*.hdf5"))
    if not snaps:
        raise RuntimeError("No snapshots found; GIZMO did not run.")
    assert_final_time(snaps[-1], TEST_NAME)

    # Read pre-promotion counts from the IC file — promotion may fire before snapshot_000
    ic_path = f"test/{TEST_NAME}/grain_promotion_ics.hdf5"
    with h5py.File(ic_path, "r") as fic:
        n_gas_ic    = int(fic["Header"].attrs["NumPart_Total"][0])
        n_grain_ic  = int(fic["Header"].attrs["NumPart_Total"][3])
        mass_ic     = fic["PartType0/Masses"][:].sum() + fic["PartType3/Masses"][:].sum()

    # Final snapshot
    with h5py.File(snaps[-1], "r") as ff:
        n_gas_f   = ff["PartType0/Masses"][:].shape[0]
        n_grain_f = ff["PartType3/Masses"][:].shape[0] if "PartType3" in ff else 0
        mass_f    = ff["PartType0/Masses"][:].sum()
        if "PartType3" in ff:
            mass_f += ff["PartType3/Masses"][:].sum()
        comp = ff["PartType0/CompositionType"][:]

    # Grain count must drop (promoted grain is no longer Type 3)
    assert n_grain_f < n_grain_ic, (
        f"Grain count did not decrease: was {n_grain_ic} in IC, now {n_grain_f}"
    )

    # Gas count must increase by exactly the number of promoted grains
    assert n_gas_f == n_gas_ic + (n_grain_ic - n_grain_f), (
        f"Gas count mismatch: IC had {n_gas_ic} gas + {n_grain_ic} grains, final has {n_gas_f} gas + {n_grain_f} grains"
    )

    # Promoted solid must have CompositionType = MATERIAL_TILLOTSON_OLIVINE
    assert MATERIAL_OLIVINE in comp, (
        f"No olivine (CompositionType={MATERIAL_OLIVINE}) particle found in final snapshot"
    )

    # Mass conservation (gas + grains together)
    mass_err = abs(mass_f - mass_ic) / mass_ic
    assert mass_err < 1e-3, f"Total mass error {mass_err:.4e} exceeds 0.1%"
