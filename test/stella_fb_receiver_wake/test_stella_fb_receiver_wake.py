"""Feedback receivers and the kick's u limiter (STL-183): RESOLVED_ISM_FB_RAISES_RECEIVER_SIGNAL_SPEED with
WAKEUP_MERGE_SPLIT_FROM_OWN_TIMESTEP.

A wind deposit wakes its receivers. Without the two flags an inactive receiver is put on the highest active bin with its
pre-injection signal speed; at its next closing kick the forces of the heated state act over that long step, the u limiter
catches the cell and the energy it adds is created (booked as kick_floor). With the flags the deposit raises the receiver's
signal speed and the wake puts it on its own timestep.

Config_stella_default.sh (feedback before the hydro forces) plus the isolated-feedback test hooks, adiabatic, the
total-energy-exact kick (it books what the limiter adds) and TEST_KICK_FLOOR_DUMP. Uniform n_H = 10, 1 Msun cells, 16 pc
periodic box at 100 K; one 60 Msun star blows its wind from birth; stellar radiation off; steps up to 10 kyr, 100 kyr.

Metric and bars, fixed before the first run (2026-11-25):
  r = max over the full syncs with wind of (kick_floor booked so far) / (wind energy booked so far), from energy_step_trace.txt.
  - with the flags: r <= 1e-2 (Uli's bar for the r26 cloud, STL-183);
  - without them: r > 1e-2 (the problem must be present in this box, otherwise the test shows nothing);
  - both runs print TEST_KICK_FLOOR_DUMP receiver lines, the run without the flags also kick lines.
"""

import os
import shutil
import subprocess
from os.path import abspath, dirname, join

import pytest

from gizmo.test import build_gizmo_for_test, mpi_launch_command

TEST_NAME = "stella_fb_receiver_wake"
TEST_DIR = dirname(abspath(__file__))
REPO = abspath(join(TEST_DIR, "..", ".."))
EXTRA_FLAGS = ("BOX_PERIODIC", "GRAVITY_NOT_PERIODIC", "GALSF_RESOLVEDISM_ISOLATED_FB_TEST", "OUTPUT_ENERGY_STEP_TRACE",
               "RESOLVEDISM_ADIABATIC_NOCOOL", "HYDRO_KICK_TOTAL_ENERGY_EXACT", "TEST_KICK_FLOOR_DUMP=1e-14")
FLAGS = ("RESOLVED_ISM_FB_RAISES_RECEIVER_SIGNAL_SPEED", "WAKEUP_MERGE_SPLIT_FROM_OWN_TIMESTEP")
BAR = 1e-2
RUN_TIMEOUT = 5400  # s


def _ratio(out):
    r = 0.0; n = 0
    for l in open(join(out, "energy_step_trace.txt")):
        if l.startswith("SYNC"):
            v = dict(t.split("=", 1) for t in l.split()[1:] if "=" in t)
            wind = float(v["b_fb_wind"])
            if wind > 0:
                r = max(r, abs(float(v["b_kick_floor"])) / wind); n += 1
    return r, n


def _run(binary, fbc, name):
    d = join(TEST_DIR, "run_" + name); out = join(d, "output")
    shutil.rmtree(d, ignore_errors=True); os.makedirs(out)
    os.symlink(REPO, join(d, "gizmo"))
    fbc.make_box(join(d, "box_ics.hdf5"), fbc.ZSUN, T=100.0)
    base = open(join(REPO, "test", "stella_fb_conservation", "stella_fb_conservation.params")).read().splitlines()
    over = {"TestNumStars": 1, "TestStarMass": 60.0, "TimeMax": 1.0e-4, "MaxSizeTimestep": 1.0e-5, "TimeBetStatistics": 1.0e-5,
            "TimeBetSnapshot": 1.0e-4, "TestFUVZero": 1}
    lines = [l for l in base if not (l.split() and l.split()[0] in over)] + [f"{k} {v}" for k, v in over.items()]
    open(join(d, "run.params"), "w").write("\n".join(lines) + "\n")
    with open(join(d, "run.log"), "w") as log:
        r = fbc.run_killable(mpi_launch_command(2) + [binary, "run.params"], RUN_TIMEOUT, stdout=log, stderr=subprocess.STDOUT, cwd=d)
    text = open(join(d, "run.log")).read()
    assert r.returncode == 0 and "Simulation ends." in text, f"{name}: " + text[-1500:]
    assert "RESOLVEDISM WIND:" in text and "RESOLVEDISM SN:" not in text
    return text, out


@pytest.fixture(scope="module")
def runs():
    if os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        pytest.skip("builds GIZMO")
    import importlib.util
    spec = importlib.util.spec_from_file_location("fbc", join(REPO, "test", "stella_fb_conservation", "test_stella_fb_conservation.py"))
    fbc = importlib.util.module_from_spec(spec); spec.loader.exec_module(fbc)
    cfg = open(join(REPO, "Config_stella_default.sh")).read()
    open(join(TEST_DIR, "Config.sh"), "w").write(cfg + "\n# test hooks\n" + "\n".join(EXTRA_FLAGS) + "\n")
    binaries = {}
    cwd = os.getcwd(); os.chdir(REPO)
    try:
        for arm, extra in (("on", FLAGS), ("off", ())):
            build_gizmo_for_test(TEST_NAME, extra_config_flags=extra)
            binaries[arm] = join(TEST_DIR, f"GIZMO_{arm}")
            shutil.move(join(TEST_DIR, "GIZMO"), binaries[arm])
    finally:
        os.chdir(cwd)
    res = {}
    for arm in ("on", "off"):
        text, out = _run(binaries[arm], fbc, arm)
        r, n = _ratio(out)
        res[arm] = (r, n, text.count("TEST_KICK_FLOOR_DUMP recv"), text.count("TEST_KICK_FLOOR_DUMP kick"))
        print(f"flags {arm}: max kick_floor / wind {r:.3e} over {n} syncs with wind; dump lines: {res[arm][2]} receivers, {res[arm][3]} kicks")
    return res


def test_dump_lines(runs):
    for arm in ("on", "off"):
        assert runs[arm][1] > 0, f"flags {arm}: no full sync with wind"
        assert runs[arm][2] > 0, f"flags {arm}: no receiver line"
    assert runs["off"][3] > 0, "without the flags no kick fired the u limiter above the dump threshold"


def test_problem_present_without_the_flags(runs):
    assert runs["off"][0] > BAR, f"without the flags the limiter adds only {runs['off'][0]:.3e} of the wind energy in this box"


def test_receivers_on_their_own_timestep(runs):
    assert runs["on"][0] <= BAR, f"with the flags the limiter adds {runs['on'][0]:.3e} of the wind energy"
