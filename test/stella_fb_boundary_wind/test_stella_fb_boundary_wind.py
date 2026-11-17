"""Boundary feedback (FB_G17_BOUNDARY_INJECTION, the zoom production build): winds and AGB ejecta deposited after the closing
kick must redo the hydro pass as SNe do, else the next opening kick uses pre-deposit forces in exactly those cells.

Config_stella_zoom.sh plus the isolated-feedback test hooks, adiabatic (RESOLVEDISM_ADIABATIC_NOCOOL) so the gas energy has no
sinks; non-comoving. Uniform n_H = 10, 1 Msun cells, 16 pc periodic box. A 60 Msun star blows its wind from birth and a 3 Msun
star dies as an AGB star at 2 kyr (forced). From ENERGYinfo.txt: gas thermal + kinetic energy minus its initial value minus the
booked feedback energy, relative to the booked energy, at every statistics output. The bar is the SN result of the same build
family (+0.40 % maximum, adiabatic SN at 1 Msun, reports/sn_boundary_and_approach_dt.md). The mass, element and energy step
traces must close as well.
"""

import os
import shutil
import subprocess
from os.path import abspath, dirname, join

import numpy as np
import pytest

from gizmo.test import build_gizmo_for_test, mpi_launch_command
from gizmo.conservation_trace import keep_and_check_traces

TEST_NAME = "stella_fb_boundary_wind"
TEST_DIR = dirname(abspath(__file__))
REPO = abspath(join(TEST_DIR, "..", ".."))
EXTRA_FLAGS = ("GALSF_RESOLVEDISM_ISOLATED_FB_TEST", "RESOLVEDISM_ADIABATIC_NOCOOL", "OUTPUT_ENERGY_STEP_TRACE")
BAR_PERCENT = 0.40


def test_boundary_wind_energy():
    if os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        pytest.skip("builds GIZMO")
    import importlib.util
    spec = importlib.util.spec_from_file_location("fbc", join(REPO, "test", "stella_fb_conservation", "test_stella_fb_conservation.py"))
    fbc = importlib.util.module_from_spec(spec); spec.loader.exec_module(fbc)
    cfg = open(join(REPO, "Config_stella_zoom.sh")).read()
    open(join(TEST_DIR, "Config.sh"), "w").write(cfg + "\n# test hooks\n" + "\n".join(EXTRA_FLAGS) + "\n")
    cwd = os.getcwd(); os.chdir(REPO)
    try:
        build_gizmo_for_test(TEST_NAME)
    finally:
        os.chdir(cwd)
    d = join(TEST_DIR, "run"); out = join(d, "output")
    shutil.rmtree(d, ignore_errors=True); os.makedirs(out)
    os.symlink(REPO, join(d, "gizmo"))
    fbc.make_box(join(d, "box_ics.hdf5"), fbc.ZSUN)
    base = open(join(REPO, "test", "stella_fb_conservation", "stella_fb_conservation.params")).read().splitlines()
    over = {"TestNumStars": 2, "TestStarMass": 60.0, "TestStarMass2": 3.0, "TestSNTime0": 1.0e-3, "TestSNSpacing": -0.998e-3,
            "TimeMax": 2.0e-5, "TimeBetStatistics": 2.0e-7, "TestFUVZero": 1}
    lines = [l for l in base if not (l.split() and l.split()[0] in over)] + [f"{k} {v}" for k, v in over.items()]
    open(join(d, "run.params"), "w").write("\n".join(lines) + "\n")
    with open(join(d, "run.log"), "w") as log:
        r = subprocess.run(mpi_launch_command(2) + [join(TEST_DIR, "GIZMO"), "run.params"], stdout=log, stderr=subprocess.STDOUT, cwd=d)
    text = open(join(d, "run.log")).read()
    assert r.returncode == 0, text[-1500:]
    assert "RESOLVEDISM AGB:" in text and "RESOLVEDISM WIND:" in text
    assert "G17RECOMPUTE" in text
    a = np.loadtxt(join(out, "ENERGYinfo.txt"), comments="#", ndmin=2)
    E, Efb = a[:, 4] + a[:, 5], a[:, 9]
    m = Efb > 0
    ex = (E[m] - E[0] - Efb[m]) / Efb[m] * 100
    print(f"wind + AGB at the boundary: max excess {ex.max():+.3f} %, end {ex[-1]:+.3f} % of booked {Efb[-1]:.4e}")
    rep = keep_and_check_traces(out, out)
    assert ex.max() <= BAR_PERCENT and ex.min() >= -BAR_PERCENT
    assert rep.ok, rep.summary()
