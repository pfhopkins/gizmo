"""Driven isothermal turbulence held at Mach 10 by the TURB_DRIVING_TARGET_SIGMA control loop.

32^3 periodic box, 10 pc, c_s = 0.2 km/s, solenoidal driving with a parabolic spectrum on
1 <= kL/2pi <= 3, hydro and MHD (uniform B_z, initial plasma beta 1), each with the driving force
recomputed every step or only on driving updates (TURB_DRIVING_UPDATE_FORCE_ON_TURBUPDATE, "lazy").
Starting from rest, the loop sets the gain on the driving force from e = ln(v_target / v_rms). The test
measures the Mach number from the snapshots -- independently of the loop's own measurement -- and
checks that it settles on the target and stays there. See turb_driving_common.py.
"""

import subprocess
import sys
from os import path

import numpy as np
import pytest

import turb_driving_common as common
from gizmo.test import build_and_run_test, default_mpi_ranks, variant_output_dir

TEST_NAME = "turbulent_box"
TEST_DIR = f"test/{TEST_NAME}"


def _ensure_ic():
    if path.exists(f"{TEST_DIR}/{TEST_NAME}_ics.hdf5"):
        return
    subprocess.run([sys.executable, f"make_{TEST_NAME}_ics.py"], cwd=TEST_DIR, check=True)


@pytest.mark.parametrize("lazy", (False, True), ids=("full", "lazy"))
@pytest.mark.parametrize("variant", common.VARIANTS)
def test_turbulent_box(variant, lazy):
    _ensure_ic()
    common.run_and_check(TEST_NAME, variant, lazy=lazy)


LEDGER_FLAGS = ("OUTPUT_ADDITIONAL_RUNINFO",)  # writes turb.txt
MAX_LEDGER_RESIDUAL = 0.05  # of the energy injected over the settled window


def test_turbulent_box_energy_ledger():
    """The turbulence log's energies close: over the settled window, the change of the gas kinetic + thermal energy
    equals the energy the driving injected minus the energy the enforced EOS removed (TurbDissipatedEnergy, which must
    be an energy like TurbInjectedEnergy, not a sum of specific energies)."""
    _ensure_ic()
    build_and_run_test(TEST_NAME, num_mpi_ranks=default_mpi_ranks(max_ranks=8), extra_config_flags=LEDGER_FLAGS)
    d = np.loadtxt(f"{variant_output_dir(TEST_NAME, LEDGER_FLAGS)}/turb.txt", comments="#")
    t, e_tot, e_inj, e_diss = d[:, 0], d[:, 2], d[:, 5], d[:, 6]  # per unit mass
    i0, i1 = np.searchsorted(t, common.T_SETTLED), len(t) - 1
    d_e, d_inj, d_diss = e_tot[i1] - e_tot[i0], e_inj[i1] - e_inj[i0], e_diss[i1] - e_diss[i0]
    residual = (d_e - (d_inj - d_diss)) / d_inj
    print(f"t = {t[i0]:.3f} -> {t[i1]:.3f}: d(E_kin + E_th)/M = {d_e:.5g}, injected {d_inj:.5g}, dissipated {d_diss:.5g} "
          f"(dissipated / injected {d_diss / d_inj:.4f}); ledger residual {residual:+.4f} of the injection")
    assert d_inj > 0, "no energy injected over the settled window"
    assert abs(residual) < MAX_LEDGER_RESIDUAL, f"turbulence energy ledger off by {residual:+.4f} of the injection"
