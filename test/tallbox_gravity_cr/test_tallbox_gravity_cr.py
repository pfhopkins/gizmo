"""Tallbox on its production configuration (Config_stella_tallbox.sh, stella_tallbox.params): audit items C3, C4 and row 17.
One build, one driver run (startup + one gravity pass): a uniform gas slab of Sigma = 10 Msun/pc^2 at the midplane of the
1 x 1 x 20 kpc box, type-3 probes (1e-20 code mass) at z = 1, 5, 9.5 kpc above it.

(a) Sheet gravity. A slab periodic in x/y and open in z pulls with exactly g_z = -2 pi G Sigma at every |z| above it (discreteness
    of the 1/16 kpc lattice ~ exp(-2 pi 16 z/L), nil). The truncated image sum (n rings) is a finite square sheet of half-width
    W = (n + 1/2) L: g_z/(2 pi G Sigma) = (2/pi) arctan(W^2/(z sqrt(2W^2 + z^2))) = 0.894, 0.533, 0.29 at z = 1, 5, 9.5 L for n = 8.
    Bar (Uli): |g_self/(-2 pi G Sigma) - 1| < 1e-3 at all three, g_self = probe g_z minus the Hu+21 field (formula below).
(b) Hu+21 field (stellar disk a_z = -2 pi G Sigma_* tanh(z/H_*), vertical NFW a_z = -G M(<r) z/r^3, r = sqrt(z^2 + R0^2)): the tidal
    term must be T_zz = d a_z/dz and the potential Phi with a_z = -dPhi/dz, Phi(0) = 0, so E_pot can carry sum m Phi_ext. Central
    differences with dz = 1e-4 kpc (error (dz/H_*)^2 ~ 2e-7): bars 1e-5 of max|T_zz| and of max|a_z|. global.cc's E_pot of the
    probes must equal sum(1/2 m Phi_self + m Phi_ext) to 1e-10 (external potential at full weight, not halved as pair sums).
(c) GHOSDT cosmic rays: zeta = FactorG0 x CosmicRayIonRate is 0 at start (FactorG0 = 0 until stars form); the other SGCHEM paths
    floor zeta at 1e-21 s^-1. Bar: zeta >= 1e-21."""

import os
import shutil
import subprocess
import sys
import tempfile
from os.path import abspath, dirname, join

import numpy as np
import pytest

sys.path.insert(0, dirname(abspath(__file__)))
from gizmo.test import build_gizmo_for_test, link_test_driver
from make_tallbox_gravity_cr_ics import make_ics, SIGMA

TEST_NAME = "tallbox_gravity_cr"
TEST_DIR = dirname(abspath(__file__))
REPO = abspath(join(TEST_DIR, "..", ".."))
DRIVER = join(TEST_DIR, "output", "zeta_gravity_driver")


@pytest.fixture(scope="module")
def out():
    if os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        pytest.skip("builds GIZMO; nothing to validate from external snapshots")
    cwd = os.getcwd(); os.chdir(REPO)
    try:
        build_gizmo_for_test(TEST_NAME, post_build=lambda: link_test_driver(join(TEST_DIR, "zeta_gravity_driver.cc"), DRIVER))
    finally:
        os.chdir(cwd)
    run = tempfile.mkdtemp(prefix=f"{TEST_NAME}_"); os.makedirs(join(run, "output")); os.symlink(REPO, join(run, "gizmo"))
    make_ics(join(run, f"{TEST_NAME}_ics.hdf5"))
    r = subprocess.run([DRIVER, join(TEST_DIR, f"{TEST_NAME}.params")], capture_output=True, text=True, cwd=run)
    with open(join(TEST_DIR, "output", "driver.txt"), "w") as f:
        f.write(r.stdout + r.stderr)
    shutil.rmtree(run, ignore_errors=True)
    lines = [l.split() for l in r.stdout.splitlines()]
    assert any(l and l[0] == "ZETA" for l in lines), "driver did not finish:\n" + (r.stdout + r.stderr)[-3000:]
    return lines


def hu_az(z, G):
    two_pi_G_sig = 2 * np.pi * G * 4.0e-3; fourpiG = 4 * np.pi * G * 9.5e-4 * 17.0 ** 3
    r = np.sqrt(z * z + 64.0); x = r / 17.0
    return -two_pi_G_sig * np.tanh(z / 0.25) - fourpiG * (np.log(1 + x) - x / (1 + x)) * z / r ** 3


def test_sheet_gravity(out):
    G = float([l for l in out if l and l[0] == "G"][0][1])
    g0 = 2 * np.pi * G * SIGMA
    probes = [l for l in out if l and l[0] == "PROBE"]
    assert len(probes) == 3
    for l in probes:
        z, az = float(l[2]), float(l[3])
        ratio = (az - hu_az(z, G)) / -g0
        print(f"z = {z:.2f} kpc: g_self / (-2 pi G Sigma) = {ratio:.6f}")
        assert abs(ratio - 1) < 1e-3


def test_hu_tidal_and_potential(out):
    hu = np.array([[float(v) for v in l[1:]] for l in out if l and l[0] == "HU"])
    assert len(hu) > 0, "no hu_tallbox_field"
    z, az, tzz, phi, daz, mdphi = hu.T
    e_t = np.abs(tzz - daz).max() / np.abs(tzz).max(); e_p = np.abs(az - mdphi).max() / np.abs(az).max()
    print(f"T_zz vs d a_z/dz: {e_t:.2e}; a_z vs -dPhi/dz: {e_p:.2e}; Phi(0) = {phi[z == 0][0]:.2e}")
    assert e_t < 1e-5 and e_p < 1e-5 and phi[z == 0][0] == 0
    ep = [l for l in out if l and l[0] == "EPOT3"][0]
    assert abs(float(ep[1]) / float(ep[2]) - 1) < 1e-10, ep


def test_ghosdt_cr_floor(out):
    zeta = float([l for l in out if l and l[0] == "ZETA"][0][1])
    print(f"GHOSDT zeta at start: {zeta:.3e}")
    assert zeta >= 1e-21
