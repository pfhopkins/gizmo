"""Type-4 stars integrated by the RUMI++ library (STARS_INTEGRATED_BY_RUMI): the eccentric binary and hierarchical triple of
test/stars_hermite_direct_gravity against the same Newtonian reference orbits. Runs only with RUMI_PATH set (the library install)."""

import glob
import importlib.util
import os
import subprocess
import sys

import h5py
import numpy as np
import pytest

from gizmo.test import build_and_run_test, variant_output_dir

TEST_NAME = "stars_integrated_by_rumi"
TEST_DIR = f"test/{TEST_NAME}"
REF_DIR = "test/stars_hermite_direct_gravity"
_spec = importlib.util.spec_from_file_location("fewbody_reference", f"{REF_DIR}/fewbody_reference.py")
ref = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(ref)

pytestmark = pytest.mark.skipif(not os.environ.get("RUMI_PATH"), reason="RUMI_PATH (RUMI++ library install) not set")

T_BINARY = 2.29e-4  # ten orbits of the a = 1000 AU binary
T_TRIPLE = 3.74e-5  # two outer orbits of the triple (52 inner orbits)


def _run(ics, tmax, flags=(), params=None):
    subprocess.run([sys.executable, os.path.abspath(f"{REF_DIR}/fewbody_reference.py")], cwd=TEST_DIR, check=True)
    p = {"InitCondFile": ics, "TimeMax": tmax, "TimeBetSnapshot": tmax, "TimeBetStatistics": tmax}
    p.update(params or {})
    build_and_run_test(TEST_NAME, 1, 1, extra_config_flags=flags, param_overrides=p)
    snaps = sorted(glob.glob(variant_output_dir(TEST_NAME, flags) + "/snapshot_*.hdf5"))
    with h5py.File(f"{TEST_DIR}/{ics}.hdf5", "r") as f:  # the ICs: the first snapshot already carries the opening half kick
        g = f["PartType4"]; ic = (g["Coordinates"][:], g["Velocities"][:], g["Masses"][:])
    return ic, ref.read(snaps[-1]), open(f"{TEST_DIR}/test_{TEST_NAME}.out").read()


def _errors(ics, tmax, flags=(), params=None):
    (x0, v0, m), (x1, v1, _, t1), log = _run(ics, tmax, flags, params)
    assert abs(t1 / tmax - 1) < 1e-6
    xr, vr = ref.reference(x0, v0, m, t1)
    de = abs(ref.energy(x1, v1, m) / ref.energy(x0, v0, m) - 1)
    a0 = ref.elements(x0[:2], v0[:2], m[:2])[0]
    dx = np.max(np.linalg.norm((x1 - x1[0]) - (xr - xr[0]), axis=1)) / a0
    return de, dx, log


def test_startup_and_direct_sum():
    """the library starts (its build id in the log, no RUMI++ warning); the stars feel each other through the exact direct sum only"""
    de, dx, log = _errors("binary_ics", T_BINARY / 10)
    print(f"binary, one orbit on the direct sum: dE/E {de:.3e} position {dx:.3e}")
    assert "RUMI++: RUMI_LIB_ID api=3" in log and "RUMI++ WARNING" not in log
    assert de < 3.5e-3 and dx < 1.2e-2  # 3x measured (1.1e-3, 3.8e-3 a): leapfrog on the direct sum
