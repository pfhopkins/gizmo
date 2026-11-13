"""Type-4 stars on the 4th-order Hermite integrator with exact direct star-star gravity (STARS_HERMITE_DIRECT_GRAVITY):
an eccentric binary and a hierarchical triple against Newtonian reference orbits."""

import glob
import importlib.util
import subprocess
import sys

import numpy as np
import pytest

from gizmo.test import build_and_run_test, variant_output_dir

TEST_NAME = "stars_hermite_direct_gravity"
TEST_DIR = f"test/{TEST_NAME}"
_spec = importlib.util.spec_from_file_location("fewbody_reference", f"{TEST_DIR}/fewbody_reference.py")
ref = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(ref)

T_BINARY = 2.29e-4  # ten orbits of the a = 1000 AU binary
T_TRIPLE = 3.74e-5  # two outer orbits of the triple (52 inner orbits)


def _run(ics, tmax, flags=()):
    subprocess.run([sys.executable, "fewbody_reference.py"], cwd=TEST_DIR, check=True)
    build_and_run_test(TEST_NAME, 1, 1, extra_config_flags=flags,
                       param_overrides={"InitCondFile": ics, "TimeMax": tmax, "TimeBetSnapshot": tmax, "TimeBetStatistics": tmax})
    snaps = sorted(glob.glob(variant_output_dir(TEST_NAME, flags) + "/snapshot_*.hdf5"))
    return ref.read(snaps[0]), ref.read(snaps[-1])


def _errors(ics, tmax, flags=()):
    (x0, v0, m, _), (x1, v1, _, t1) = _run(ics, tmax, flags)
    assert abs(t1 / tmax - 1) < 1e-6
    xr, vr = ref.reference(x0, v0, m, t1)
    de = abs(ref.energy(x1, v1, m) / ref.energy(x0, v0, m) - 1)
    a0 = ref.elements(x0[:2], v0[:2], m[:2])[0]
    dx = np.max(np.linalg.norm((x1 - x1[0]) - (xr - xr[0]), axis=1)) / a0  # relative positions, so a drift of the centre does not count
    return de, dx, ref.elements(x1[:2], v1[:2], m[:2]), ref.elements(x0[:2], v0[:2], m[:2])


# bars 3x the measured errors (base Hermite: binary dE 3.6e-5, offset 3.7e-4 a; triple dE 4.4e-6, offset 2.0e-3 a_in)
def test_binary():
    de, dx, _, _ = _errors("binary_ics", T_BINARY)
    print(f"binary: dE/E {de:.3e} position {dx:.3e}")
    assert de < 1e-4 and dx < 1.2e-3


def test_triple():
    de, dx, _, _ = _errors("triple_ics", T_TRIPLE)
    print(f"triple: dE/E {de:.3e} position {dx:.3e}")
    assert de < 1.5e-5 and dx < 6e-3
