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


def test_region_frame():
    log = _run(1, ("TEST_KETJU_REGION_LOG",))
    d = re.search(r"KETJU_REGION n 2 com_error (\S+) p_error (\S+)", log)
    assert d and float(d[1]) < 1e-12 and float(d[2]) < 1e-12


def test_soft_binary():
    """softening far above the orbit: the tree's pair force is negligible, MSTAR alone holds the orbit"""
    _run(1, params={"Softening_Type4": 1e-3, "Softening_Type4_MaxPhysLimit": 1e-3, "TimeBetSnapshot": 1})
    a, e = _elements()
    assert abs(a - 1) < 1e-3 and abs(e - 0.9) < 1e-3


@pytest.mark.parametrize("ranks", [1])
def test_hard_binary(ranks):
    """softening far below the orbit: the tree's pair force must be taken out exactly"""
    _run(ranks, params={"TimeBetSnapshot": 1})
    a, e = _elements()
    assert abs(a - 1) < 1e-6 and abs(e - 0.9) < 1e-6








def test_hubble_param_units():
    """h = 0.7 with length and mass units x 0.7: the same physical binary (the code's units carry 1/h), the same orbit"""
    _run(1, params={"TimeBetSnapshot": 1, "HubbleParam": 0.7, "UnitLength_in_cm": 3.085678e21 * 0.7, "UnitMass_in_g": 1.989e43 * 0.7})
    a, e = _elements()
    assert abs(a - 1) < 1e-6 and abs(e - 0.9) < 1e-6


def test_adaptive_treeforce():
    _run(1, ("ADAPTIVE_TREEFORCE_UPDATE=0.1",), params={"TimeBetSnapshot": 1})
    a, e = _elements(("ADAPTIVE_TREEFORCE_UPDATE=0.1",))
    assert abs(a - 1) < 1e-6 and abs(e - 0.9) < 1e-6


def test_kepler_pair():
    log = _run(1, params={"KetjuKeplerPairs": 1})
    a, e = _elements()
    assert abs(a - 1) < 1e-8 and abs(e - 0.9) < 1e-8
    assert re.findall(r"region-integrations=(\d+) MSTAR accepted=(\d+)", log)[-1][0] == re.findall(r"MSTAR accepted=(\d+)", log)[-1]  # one exact drift per region, no GBS steps


def test_energy_lines():
    log = _run(1, ("OUTPUT_STAR_ENERGY_DIRECT_SUM",))
    d = re.findall(r"ETOT_INCODE: t=\S+ E=\S+ dE/\|E0\|=(\S+)", log)
    assert d and max(abs(float(x)) for x in d) < 1e-6


def test_remnant_pn():
    """stars marked as black hole remnants take PN terms: the log says so"""
    log = _run(1, ("KETJU_PN_REMNANT_TAG", "TEST_KETJU_REMNANT_AT_START", "TEST_KETJU_REGION_LOG"), params={"KetjuPNTerms": "2457"})
    assert re.search(r"KETJU_REGION_PN 2", log)
