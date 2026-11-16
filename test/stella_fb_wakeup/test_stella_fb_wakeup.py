"""stella_fb_wakeup: gas cells that Stella feedback heats or kicks between their own sync points are woken the same step.

A 12 Msun star (cell ID 1, GALSF_RESOLVEDISM_ISOLATED_FB_TEST) at the centre of a uniform 20 pc box of 1 Msun cells has
winds from formation and is forced to go supernova at TestSNTime0 = 2.1e-6. The gas is held on long steps
(MaxSizeTimestep, bin 6.25e-7) and the star on short ones (MaxStellarTimestep, bin 7.8e-8), so the SN and the winds reach
gas that is not active. Every gas cell the feedback flags for wake-up must be in the pending wake-up list that
process_wake_ups visits; the WAKEUP_GATE_AUDIT build reports each one missing ("WAKEUP-LIST-AUDIT") or left behind a
shut gate ("GATE-AUDIT").
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile
from os.path import abspath, dirname, join

import numpy as np
import pytest

from gizmo.test import build_gizmo_for_test, mpi_launch_command
from gizmo.conservation_trace import keep_and_check_traces

TEST_NAME = "stella_fb_wakeup"
TEST_DIR = dirname(abspath(__file__))
REPO = abspath(join(TEST_DIR, "..", ".."))
NRANKS = 4
TIME_MAX = 1.0e-5

sys.path.insert(0, TEST_DIR)
from make_stella_fb_wakeup_ics import make_stella_fb_wakeup_ics  # noqa: E402


@pytest.mark.skipif(bool(os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN")), reason="builds GIZMO")
def test_feedback_wakeups_reach_the_pending_list():
    cwd = os.getcwd()
    os.chdir(REPO)
    try:
        build_gizmo_for_test(TEST_NAME)
    finally:
        os.chdir(cwd)
    run = tempfile.mkdtemp(prefix=f"{TEST_NAME}_")
    os.makedirs(join(run, "output"))
    os.symlink(REPO, join(run, "gizmo"))
    make_stella_fb_wakeup_ics(join(run, f"{TEST_NAME}_ics.hdf5"))
    shutil.copy(join(TEST_DIR, f"{TEST_NAME}.params"), run)
    with open(join(run, "run.log"), "w") as log:
        r = subprocess.run(mpi_launch_command(NRANKS) + [join(TEST_DIR, "GIZMO"), f"{TEST_NAME}.params"],
                           stdout=log, stderr=subprocess.STDOUT, cwd=run)
    out = join(TEST_DIR, "output")
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    for f in ("run.log", "output/SNinfo.txt"):
        if os.path.exists(join(run, f)):
            shutil.copy(join(run, f), out)
    text = open(join(run, "run.log")).read()
    sninfo = np.loadtxt(join(run, "output", "SNinfo.txt"), ndmin=2) if os.path.exists(join(run, "output", "SNinfo.txt")) else np.zeros((0, 20))
    trace = keep_and_check_traces(join(run, "output"), out)
    shutil.rmtree(run, ignore_errors=True)

    assert r.returncode == 0, f"GIZMO exited {r.returncode}, see {out}/run.log"
    assert f"Final time={TIME_MAX:g} reached" in text, f"run stopped before TimeMax ({out}/run.log)"
    assert np.any((sninfo[:, 6] == 1) & (sninfo[:, 19] == 0)), "the forced SN of star 1 was not delivered"
    missed = [int(n) for n in re.findall(r"WAKEUP-LIST-AUDIT step=\d+ task=\d+: (\d+) pending requests missing", text)]
    gate = re.findall(r"GATE-AUDIT step=\d+ task=\d+: gate SHUT with (\d+) pending", text)
    print(f"wake-up requests missing from the list: {sum(missed)} in {len(missed)} reports; shut-gate reports: {len(gate)}")
    assert not missed, f"{sum(missed)} feedback wake-up requests were not in the pending list ({out}/run.log)"
    assert not gate, f"{len(gate)} steps shut the wake-up gate with requests pending ({out}/run.log)"
    assert trace.ok, trace.summary()
