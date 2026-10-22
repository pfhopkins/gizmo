"""Cosmic-ray heating, ionisation and losses in CHEMCOOL network 5 with the CR fluid (Stella cloud Config + COSMIC_RAY_FLUID).
The driver evolves a real cell to equilibrium at fixed nH and fixed CR energy density for (nH, zeta) points and the test checks:
zeta in the cell = 3e-17 e_CR / (1.8 eV cm^-3) (SGChem); chemistry CR heating = 20 eV per ionisation x (n_HI + n_H2) (Goldsmith & Langer 1978,
as SGChem) plus hadronic and Coulomb heating; CR fluid loss per step = the hadronic + Coulomb rate; equilibrium T in the dark-cloud range.
"""

import os
import shlex
import subprocess
import tempfile
from math import exp
from os.path import abspath, dirname, join

import pytest

from gizmo.test import build_gizmo_for_test

DRIVER_DIR = dirname(abspath(__file__))
REPO = abspath(join(DRIVER_DIR, "..", ".."))
DRIVER = "cr_chemistry_onezone"
ABHE = 0.079


def _link_driver(test_dir, repo=REPO):
    """Link the driver against the objects of the build that just finished, in place of core/main.o."""
    out = join(test_dir, "output")
    os.makedirs(out, exist_ok=True)
    dry = subprocess.run(["make", "-n", "-B", "GIZMO"], capture_output=True, text=True, cwd=repo).stdout.splitlines()
    compile_main = [l for l in dry if " -c core/main.cc" in l]
    link = [l for l in dry if "-o GIZMO" in l]
    assert compile_main and link, "could not read the compile/link commands from make -n"
    obj, exe = join(out, f"{DRIVER}_driver.o"), join(out, f"{DRIVER}_driver")
    cc = shlex.split(compile_main[-1])
    cc[cc.index("core/main.cc")] = join(DRIVER_DIR, f"{DRIVER}_driver.cc")
    cc[cc.index("core/main.o")] = obj
    cc.insert(1, "-I" + repo)
    ln = shlex.split(link[-1])
    ln[ln.index("core/main.o")] = obj
    ln[ln.index("GIZMO")] = exe
    for cmd in (cc, ln):
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=repo)
        assert r.returncode == 0, "driver build failed:\n%s\n%s" % (" ".join(cmd), r.stderr[-4000:])
    return exe


def run_driver(exe, repo=REPO):
    run = tempfile.mkdtemp(prefix=f"{DRIVER}_")
    os.makedirs(join(run, "output"))
    os.symlink(repo, join(run, "gizmo"))  # CHEMCOOL and the stellar tables open their data under gizmo/
    subprocess.run(["python3", join(DRIVER_DIR, f"make_{DRIVER}_ics.py"), join(run, f"{DRIVER}_ics.hdf5")], check=True)
    r = subprocess.run(["mpirun", "-np", "1", exe, join(DRIVER_DIR, f"{DRIVER}.params")], capture_output=True, text=True, cwd=run)
    assert "DONE" in r.stdout, "driver did not finish (exit %d):\n%s\n%s" % (r.returncode, r.stdout[-3000:], r.stderr[-3000:])
    return [dict(kv.split("=") for kv in l.split()[1:]) for l in r.stdout.splitlines() if l.startswith("ROW")]


def check_rows(rows):
    fails = []
    for d in rows:
        d = {k: float(v) for k, v in d.items()}
        n, z, ecr, ne, xhi, xh2 = d["nH"], d["zeta_in"], d["ecr"], d["ne"], d["xHI"], d["xH2"]
        tag = "nH=%g zeta=%g" % (n, z)
        if abs(d["zeta_cell"] / z - 1) > 1e-6 or abs(z * 1.8 / 3e-17 * 1.602e-12 / ecr - 1) > 1e-6:
            fails.append("%s: zeta in the cell %g" % (tag, d["zeta_cell"]))
        gam = 3.2e-11 * z * n * (xhi + xh2) + (0.87 * 6.37e-16 / 6 + 0.53 * 3.09e-16 * (ne + 0.57 * xhi) / (1 + 4 * ABHE)) * ecr * n
        if abs(d["gamCR"] / gam - 1) > 2e-2:
            fails.append("%s: CR heating %g, expected %g" % (tag, d["gamCR"], gam))
        # CR_cooling_and_losses: (0.87 a + 0.53 b) n_nucleon, b = 3.09e-16 (x_e + 0.57 (1 - f_ion)) X_H; neutral gas, f_ion << 1
        rate = (0.87 * 6.37e-16 + 0.53 * 3.09e-16 * 0.76 * (ne + 0.57)) * n * (1 + 4 * ABHE)
        loss = 1 - exp(-rate * d["dt_s"])
        if abs(d["lossfrac"] / loss - 1) > 5e-2:
            fails.append("%s: CR loss per step %g, expected %g" % (tag, d["lossfrac"], loss))
        if not 5 < d["T"] < 60:
            fails.append("%s: equilibrium T %g K outside the dark-cloud range" % (tag, d["T"]))
    return fails


def test_cr_chemistry_onezone():
    if os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        pytest.skip("builds GIZMO; nothing to validate from external snapshots")
    test_dir = join(REPO, "test", DRIVER)
    cwd = os.getcwd()
    os.chdir(REPO)
    try:
        build_gizmo_for_test(DRIVER, post_build=lambda: _link_driver(test_dir))
    finally:
        os.chdir(cwd)
    rows = run_driver(join(test_dir, "output", f"{DRIVER}_driver"))
    assert len(rows) == 6
    fails = check_rows(rows)
    assert not fails, "\n".join(fails)
