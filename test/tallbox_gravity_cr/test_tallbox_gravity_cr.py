"""Tallbox on its production configuration (Config_stella_tallbox.sh, stella_tallbox.params): audit items C3, C4 and row 17.
One build, one driver run (startup + one gravity pass): a uniform gas slab of Sigma = 10 Msun/pc^2 at the midplane of the
1 x 1 x 20 kpc box, type-3 probes (1e-20 code mass) at z = 1, 5, 9.5 kpc above it.

(a) Sheet gravity. A slab periodic in x/y and open in z pulls with exactly g_z = -2 pi G Sigma at every |z| above it (discreteness
    of the 1/16 kpc lattice ~ exp(-2 pi 16 z/L), nil). The truncated image sum (n rings) is a finite square sheet of half-width
    W = (n + 1/2) L: g_z/(2 pi G Sigma) = (2/pi) arctan(W^2/(z sqrt(2W^2 + z^2))) = 0.894, 0.533, 0.29 at z = 1, 5, 9.5 L for n = 8.
    Bar (Uli): |g_self/(-2 pi G Sigma) - 1| < 1e-3 at all three, g_self = probe g_z minus the Hu+21 field (formula below).
(b) Hu+21 field (stellar disk a_z = -2 pi G Sigma_* tanh(z/H_*), vertical NFW a_z = -G M(<r) z/r^3, r = sqrt(z^2 + R0^2)): the tidal
    term must be T_zz = d a_z/dz and the potential Phi with a_z = -dPhi/dz, Phi(0) = 0, so E_pot can carry sum m Phi_ext. Central
    differences with dz = 1e-4 kpc (error (dz/H_*)^2 ~ 2e-7): bars 1e-5 of max|T_zz| and of max|a_z|. One call of the production
    GravAccel_HuTallbox adds exactly T_zz(z) to each probe's tidal_tensorps[2][2] (1e-12). global.cc's E_pot of the probes equals
    sum(1/2 m Phi_self + m Phi_ext) with Phi_ext from this test's own formula (1e-10; full weight, not halved as pair sums):
    Phi_ext = 2 pi G Sigma_* H_* ln cosh(z/H_*) - 4 pi G rho_s r_s^3 [ln(1 + r/r_s)/r - ln(1 + R0/r_s)/R0], r = sqrt(z^2 + R0^2).
(c) GHOSDT field as Hu, Sternberg & van Dishoeck 2021 (ApJ 920, 44; arXiv:2103.03889) Sec. 2.2: u_UV = I_UV u_UV,0 with
    I_UV = max(Sigma_SFR/Sigma_SFR,0, I_UV,min), u_UV,0 = 8.94e-14 erg cm^-3 (Draine field), Sigma_SFR,0 = 2.4e-3 Msun yr^-1 kpc^-2
    (Fuchs+09), SFR = M(stars younger than 30 Myr)/30 Myr over the box area. Floor: G0 = 0.324e-2 Habing, the one floor of all paths
    (Uli 2026-11-19; Hu+21's I_UV,min = 0.002 would be 3.40e-3).
    In Habing units (u_Hab = 5.257e-14 erg cm^-3, 6-13.6 eV) G0 = max(Sigma_SFR/Sigma_SFR,0 u_UV,0/u_Hab, 0.324e-2).
    Cosmic rays stay ours (CR_SCALE_WITH_G0: zeta scales with the field): zeta = (G0 u_Hab/u_UV,0) CosmicRayIonRate, so
    0.324e-2 x 5.257/8.94 x rate before the first star (it was exactly 0). Bars (1e-12 relative): G0 = 0.324e-2, that zeta, and
    Sigma_SFR/Sigma_SFR,0 = 1 for the young-star mass that gives Sigma_SFR,0 over the 1 kpc^2 box."""

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
    trace = join(run, "output", "energy_step_trace.txt")
    sync = [l for l in open(trace)] if os.path.exists(trace) else []
    shutil.rmtree(run, ignore_errors=True)
    lines = [l.split() for l in r.stdout.splitlines()] + [["TRACE"] + l.split() for l in sync if l.startswith("SYNC")]
    assert any(l and l[0] == "ZETA" for l in lines), "driver did not finish:\n" + (r.stdout + r.stderr)[-3000:]
    return lines


def hu_az(z, G):
    two_pi_G_sig = 2 * np.pi * G * 4.0e-3; fourpiG = 4 * np.pi * G * 9.5e-4 * 17.0 ** 3
    r = np.sqrt(z * z + 64.0); x = r / 17.0
    return -two_pi_G_sig * np.tanh(z / 0.25) - fourpiG * (np.log(1 + x) - x / (1 + x)) * z / r ** 3


def hu_phi(z, G):
    two_pi_G_sig = 2 * np.pi * G * 4.0e-3; fourpiG = 4 * np.pi * G * 9.5e-4 * 17.0 ** 3
    r = np.sqrt(z * z + 64.0)
    return two_pi_G_sig * 0.25 * np.log(np.cosh(z / 0.25)) - fourpiG * (np.log(1 + r / 17.0) / r - np.log(1 + 8.0 / 17.0) / 8.0)


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
    G = float([l for l in out if l and l[0] == "G"][0][1])
    tz = np.array([[float(v) for v in l[1:]] for l in out if l and l[0] == "TZZ"])
    assert len(tz) == 3, "no tidal tensor lines"
    for zz, dt, ref in tz:
        print(f"z = {zz:.2f}: tidal_tensorps[2][2] gained {dt:.12e}, T_zz {ref:.12e}")
        assert abs(dt / ref - 1) < 1e-12
    e3 = float([l for l in out if l and l[0] == "EPOT3"][0][1])
    pp = np.array([[float(v) for v in l[2:]] for l in out if l and l[0] == "PPOT"])
    e_ref = sum(0.5 * m * pot + m * hu_phi(zz, G) for zz, m, pot in pp)
    print(f"E_pot of the probes {e3:.12e}, expected {e_ref:.12e}")
    assert abs(e3 / e_ref - 1) < 1e-10


def test_external_potential_in_energy_reports(out):
    """The production energy reports carry the external potential at full weight: the step trace's E_pot (energy_step_trace.cc,
    1/2 sum m Phi + sum m Phi_ext) and the BUDGET gravity report (run.cc energy_budget_gravity_report) must equal
    sum over all particles of (1/2 m Phi_self + m Phi_ext(z)) with Phi_ext from this test's formula; bars 1e-10 (trace, 16 printed
    digits) and 1e-8 (BUDGET, 10 printed digits)."""
    G = float([l for l in out if l and l[0] == "G"][0][1])
    pa = np.array([[float(v) for v in l[1:]] for l in out if l and l[0] == "PALL"])
    assert len(pa) > 0, "no particle potentials"
    e_ref = float(np.sum(0.5 * pa[:, 1] * pa[:, 2] + pa[:, 1] * hu_phi(pa[:, 0], G)))
    budget = [l for l in out if l[:2] == ["Energy", "(synced,grav)"]]
    tr = [l for l in out if l and l[0] == "TRACE"]
    assert budget and tr, "no BUDGET line or no step-trace SYNC line"
    e_budget = float([tok for tok in budget[-1] if tok.startswith("E_pot=")][0][6:])
    e_trace = float([tok for tok in tr[-1] if tok.startswith("E_pot=")][0][6:])
    print(f"E_pot: expected {e_ref:.12e}, step trace {e_trace:.12e}, BUDGET {e_budget:.10e}")
    assert abs(e_trace / e_ref - 1) < 1e-10
    assert abs(e_budget / e_ref - 1) < 1e-8


def test_ghosdt_field_hu21(out):
    z = [l for l in out if l and l[0] == "ZETA"][0]
    zeta, g0, rate = float(z[1]), float(z[3]), float(z[4])
    g0_ref = 0.324e-2  # the one G0 floor of all paths (Uli 2026-11-19), Habing
    zeta_ref = rate * g0_ref * 5.257e-14 / 8.94e-14  # zeta = I_UV x rate with I_UV = G0 u_Hab / u_UV,0
    iuv = [float(l[1]) for l in out if l and l[0] == "IUV1"]
    print(f"GHOSDT before the first star: G0 = {g0:.6e} (floor {g0_ref:.6e}), zeta = {zeta:.6e} (expected {zeta_ref:.6e}); I_UV at Sigma_SFR,0: {iuv}")
    assert abs(g0 / g0_ref - 1) < 1e-12
    assert rate == 3e-17  # tallbox: solar neighbourhood
    assert abs(zeta / zeta_ref - 1) < 1e-12
    assert len(iuv) == 1 and abs(iuv[0] - 1) < 1e-12
