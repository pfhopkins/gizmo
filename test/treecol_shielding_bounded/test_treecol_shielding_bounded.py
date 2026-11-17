"""TreeCol shielding column on the production configuration (Config_stella_default.sh: HEALPix NSIDE=1, RT_TREECOL_SHIELDINGLENGTH).

Derivation. A cell j at distance r in pixel p (solid angle Omega = 4 pi/12) adds Sigma_j = m_j / (Omega r^2) to pixel p, the mass
spread over the pixel's patch of the sphere of radius r. For a uniform sphere of density rho and radius R seen from its centre,
sum_j over the pixel -> integral_0^R rho Omega r^2 dr / (Omega r^2) = rho R in every pixel (code mass / code length^2, comoving
= physical here, a = 1). Inside the softening length the point-mass 1/r^2 has no meaning (the cell's mass is spread over its
kernel); upstream GIZMO's tree column (forcetree.cc, the non-shielding-length branch and its periodic-wrap variant) replaces 1/r^2 by
the gravity-softened r * kernel_gravity(r/h) with h = max(h_target, h_source), which equals 1/r^2 for r >= h and stays bounded as
r -> 0. The softening removes at most rho h of the column (rho * integral_0^h dr), h ~ 2 spacings here, so 1/8 of rho R at
R = 16 spacings; the mean over pixels, which averages the lattice discreteness, must lie within 15 % of rho R (bar set before
running). A second cell placed 1e-3 spacings from the centre cell: unsoftened it adds m/(Omega (1e-3 d)^2) = 3e5 rho d to one
pixel (2e4 rho R); softened it adds at most m/(Omega h^2) ~ 0.1 rho d. Bar: every pixel of the centre cell within 5 % of its
value without the pair, and the pair cell's pixels within 15 % of rho R in the mean, below 2 rho R each."""

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
from make_treecol_shielding_bounded_ics import make_ics, D, NR, M

TEST_NAME = "treecol_shielding_bounded"
TEST_DIR = dirname(abspath(__file__))
REPO = abspath(join(TEST_DIR, "..", ".."))
DRIVER = join(TEST_DIR, "output", f"{TEST_NAME}_driver")


def columns(pair):
    run = tempfile.mkdtemp(prefix=f"{TEST_NAME}_")  # outside the test tree, as sgchem_cooling_path
    os.makedirs(join(run, "output"))
    os.symlink(REPO, join(run, "gizmo"))  # SGCHEM and the stellar tables open their data under gizmo/
    n = make_ics(join(run, f"{TEST_NAME}_ics.hdf5"), pair)
    r = subprocess.run([DRIVER, join(TEST_DIR, f"{TEST_NAME}.params")], capture_output=True, text=True, cwd=run)
    with open(join(TEST_DIR, "output", f"driver_pair{int(pair)}.txt"), "w") as f:
        f.write(r.stdout + r.stderr)
    cols = {}
    for l in r.stdout.splitlines():
        if l.startswith("COL "):
            _, i, k, v = l.split()
            cols.setdefault(int(i), np.zeros(12))[int(k)] = float(v)
    assert cols, "driver printed no columns (exit %d):\n%s" % (r.returncode, (r.stdout + r.stderr)[-3000:])
    shutil.rmtree(run, ignore_errors=True)
    return cols, n


def test_treecol_shielding_bounded():
    if os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        pytest.skip("builds GIZMO; nothing to validate from external snapshots")
    cwd = os.getcwd()
    os.chdir(REPO)
    try:
        build_gizmo_for_test(TEST_NAME, post_build=lambda: link_test_driver(join(TEST_DIR, f"{TEST_NAME}_driver.cc"), DRIVER))
    finally:
        os.chdir(cwd)
    rhoR = M / D ** 3 * NR * D
    single, _ = columns(False)
    pair, n = columns(True)
    c0, cp, cq = single[1], pair[1], pair[n]
    print(f"rho R {rhoR:.6e}; centre mean/rhoR {c0.mean() / rhoR:.4f}, pixels {np.round(c0 / rhoR, 3)}")
    print(f"with pair: centre max change {np.abs(cp / c0 - 1).max():.3e}; pair cell mean/rhoR {cq.mean() / rhoR:.4f} max {cq.max() / rhoR:.3e}")
    assert abs(c0.mean() / rhoR - 1) < 0.15
    assert np.abs(cp / c0 - 1).max() < 0.05
    assert abs(cq.mean() / rhoR - 1) < 0.15 and cq.max() < 2 * rhoR
