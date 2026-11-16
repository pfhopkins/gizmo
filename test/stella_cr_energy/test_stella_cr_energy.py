"""stella_cr_energy: energy step trace with the CR fluid on (COSMIC_RAY_FLUID).

The stella_fb_wakeup setup (12 Msun SN at ~2 kyr in a 20 pc periodic box, cooling and MHD on) with the CR fluid: 10% of
the SN energy into CRs, an ambient 1 eV/cm^3 CR field so the loss operator acts from the first step, CR diffusion and
the CR-gas exchange in the kicks.

Checks (gizmo.conservation_trace): every exact stage closes with the CR terms booked (the SN pass with its CR share, the
cooling pass with the CR loss operator), the mass trace closes, and the CR bookings are live (CRs injected by the SN,
losses booked). The time-integration closure is the second test.
"""

import os
import shutil
import subprocess
import sys
import tempfile
from os.path import abspath, dirname, join

import pytest

from gizmo.test import build_gizmo_for_test, mpi_launch_command
from gizmo.conservation_trace import keep_and_check_traces

TEST_NAME = "stella_cr_energy"
TEST_DIR = dirname(abspath(__file__))
REPO = abspath(join(TEST_DIR, "..", ".."))
NRANKS = 2
TIME_MAX = 1.0e-5

sys.path.insert(0, join(TEST_DIR, "..", "stella_fb_wakeup"))
from make_stella_fb_wakeup_ics import make_stella_fb_wakeup_ics  # noqa: E402


@pytest.fixture(scope="module")
def run():
    if os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        pytest.skip("builds GIZMO")
    cwd = os.getcwd()
    os.chdir(REPO)
    try:
        build_gizmo_for_test(TEST_NAME)
    finally:
        os.chdir(cwd)
    rundir = tempfile.mkdtemp(prefix=f"{TEST_NAME}_")
    os.makedirs(join(rundir, "output"))
    os.symlink(REPO, join(rundir, "gizmo"))
    make_stella_fb_wakeup_ics(join(rundir, f"{TEST_NAME}_ics.hdf5"))
    shutil.copy(join(TEST_DIR, f"{TEST_NAME}.params"), rundir)
    with open(join(rundir, "run.log"), "w") as log:
        r = subprocess.run(mpi_launch_command(NRANKS) + [join(TEST_DIR, "GIZMO"), f"{TEST_NAME}.params"],
                           stdout=log, stderr=subprocess.STDOUT, cwd=rundir)
    out = join(TEST_DIR, "output")
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    shutil.copy(join(rundir, "run.log"), out)
    rep = keep_and_check_traces(join(rundir, "output"), out)
    text = open(join(out, "run.log")).read()
    shutil.rmtree(rundir, ignore_errors=True)
    assert r.returncode == 0, f"GIZMO exited {r.returncode}, see {out}/run.log"
    assert f"Final time={TIME_MAX:g} reached" in text, f"run stopped before TimeMax ({out}/run.log)"
    return rep, out


def last_sync(out):
    rows = [l for l in open(join(out, "energy_step_trace.txt")) if l.startswith("SYNC")]
    return {k: float(v) for k, v in (t.split("=", 1) for t in rows[-1].split()[1:])}


def test_cr_terms_close_per_stage(run):
    rep, out = run
    bad = [f for f in rep.findings if f["kind"] != "energy_integration"]
    assert not bad, rep.summary()
    s = last_sync(out)
    print({k: v for k, v in s.items() if "cr" in k})
    assert s["E_cr"] > 0, "no CR energy"
    assert s["s_fb_sn_ia"] > 0 and s["b_fb_sn"] > 0, "the SN did not fire"
    assert s["b_cool_cr_loss"] < 0, "CR losses not booked"


def test_time_integration_closes(run):
    rep, _ = run
    assert rep.ok, rep.summary()
