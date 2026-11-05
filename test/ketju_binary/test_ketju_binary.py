"""An isolated KETJU binary (a = 1000 AU, e = 0.9, ten orbits) with a far star, step by step of the coupling."""

import glob
import re
import subprocess
import sys

import h5py
import numpy as np
import pytest

from gizmo.test import build_and_run_test, variant_output_dir

TEST_NAME = "ketju_binary"
TEST_DIR = f"test/{TEST_NAME}"
G, AU = 43007.1, 4.848137e-9


def _run(ranks=1, flags=(), params=None):
    subprocess.run([sys.executable, f"make_{TEST_NAME}_ics.py"], cwd=TEST_DIR, check=True)
    build_and_run_test(TEST_NAME, ranks, 1, extra_config_flags=flags, param_overrides=params)
    return open(f"{TEST_DIR}/test_{TEST_NAME}.out").read()


def _elements(flags=()):
    sn = sorted(glob.glob(variant_output_dir(TEST_NAME, flags) + "/snapshot_*.hdf5"))[-1]
    with h5py.File(sn, "r") as f:
        g = f["PartType4"]; o = np.argsort(g["ParticleIDs"][:])
        m, x, v = g["Masses"][:][o], g["Coordinates"][:][o], g["Velocities"][:][o]
    dr, dv = x[0] - x[1], v[0] - v[1]; r = np.linalg.norm(dr); mu = G * (m[0] + m[1])
    a = 1 / (2 / r - dv @ dv / mu); e = np.linalg.norm(np.cross(dv, np.cross(dr, dv)) / mu - dr / r)
    return a / (1000 * AU), e


def test_parameters():
    log = _run()
    assert not re.search(r"Tag 'Ketju(RegionRadius|MinStarMass|UseStarStarSoftening|MaxStepCount)' was specified", log)


def test_centres():
    log = _run(2)
    assert re.search(r"from 3 center\(s\)", log)


@pytest.mark.parametrize("ranks", [1, 2])
def test_regions(ranks):
    log = _run(ranks)
    assert re.search(r"Found 1 region\(s\) from 3 center\(s\)", log)


def test_mstar_wrapper():
    log = _run(1, ("TEST_KETJU_MSTAR_PAIR",))
    d = re.search(r"KETJU_MSTAR_PAIR da/a (\S+) de (\S+)", log)
    assert d and abs(float(d[1])) < 1e-9 and abs(float(d[2])) < 1e-9




















