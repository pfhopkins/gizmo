"""stella_dense_sf: a dense cold clump forms Stella stars inside diffuse warm gas (production model).

500 cells of 1 Msun at n_H = 1e6 cm^-3 and 10 K sit at the centre of a 16 pc periodic box of diffuse 1e4 K gas
(n_H = 1 cm^-3). The diffuse gas sets a long top-level step (MaxSizeTimestep = TimeMax/4) while the clump forms stars on
short ones. One run serves the checks below.
"""

import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile
from os.path import abspath, dirname, join

import h5py
import numpy as np
import pytest

from gizmo.test import build_gizmo_for_test, mpi_launch_command

TEST_NAME = "stella_dense_sf"
TEST_DIR = dirname(abspath(__file__))
REPO = abspath(join(TEST_DIR, "..", ".."))
NRANKS = 4
TIME_MAX = 4.0e-5
UNIT_MASS_MSUN = 1.0e10
UNIT_TIME_YR = 3.085678e21 / 1.0e5 / 3.15576e7
UNIT_DENSITY_CGS = 1.989e43 / 3.085678e21**3

sys.path.insert(0, TEST_DIR)
from make_stella_dense_sf_ics import make_stella_dense_sf_ics  # noqa: E402


@pytest.fixture(scope="module")
def run():
    """Build and run once; return the run log text and the output directory (kept for the module)."""
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
    make_stella_dense_sf_ics(join(rundir, f"{TEST_NAME}_ics.hdf5"))
    shutil.copy(join(TEST_DIR, f"{TEST_NAME}.params"), rundir)
    with open(join(rundir, "run.log"), "w") as log:
        r = subprocess.run(mpi_launch_command(NRANKS) + [join(TEST_DIR, "GIZMO"), f"{TEST_NAME}.params"],
                           stdout=log, stderr=subprocess.STDOUT, cwd=rundir)
    out = join(TEST_DIR, "output")
    shutil.rmtree(out, ignore_errors=True)
    shutil.copytree(join(rundir, "output"), out, ignore=shutil.ignore_patterns("restartfiles"))
    shutil.copy(join(rundir, "run.log"), out)
    shutil.rmtree(rundir, ignore_errors=True)
    text = open(join(out, "run.log")).read()
    assert r.returncode == 0, f"GIZMO exited {r.returncode}, see {out}/run.log"
    assert f"Final time={TIME_MAX:g} reached" in text, f"run stopped before TimeMax ({out}/run.log)"
    return text, out


def test_table_selftests_check_runtime_ejecta(run):
    """The start-up table self-tests on ejecta check the tables the feedback actually injects (SN and AGB element
    ejecta, cumulative winds), not the legacy net-yield arrays the loader zeroes."""
    text, out = run
    m = re.search(r"\[PASS\] Ejecta metallicity < 50%.*max Z_ej=([0-9.]+)", text)
    assert m, "ejecta-metallicity self-test does not report the ejecta it checked"
    assert 0.05 < float(m.group(1)) < 0.5, f"max ejecta Z {m.group(1)}: not metal-enriched ejecta"
    m = re.search(r"\[PASS\] Ejecta mass budget.* to ([0-9.eE+-]+)", text)
    assert m, "ejecta mass-budget self-test missing or not passing"
    assert float(m.group(1)) < 1e-3
    m = re.search(r"\[PASS\] SN and AGB element ejecta finite and >= 0 for all (\d+) entries", text)
    assert m and int(m.group(1)) > 0, "ejecta sign self-test missing or not passing"
    assert re.search(r"\[PASS\] AGB C enrichment", text), "AGB carbon self-test not passing"


def test_dense_gas_kernel_not_floored(run):
    """Gas smoothing lengths are uncapped: dense clump cells reach kernel radii well below 2.8 x SofteningGas, the
    floor MinGasKernelRadiusFractional = 1 (the default with adaptive gas softening) would impose."""
    text, out = run
    soft_gas = float(re.search(r"^SofteningGas\s+(\S+)", open(join(TEST_DIR, f"{TEST_NAME}.params")).read(), re.M).group(1))
    floor_default = 2.8 * soft_gas
    checked = 0
    for snap in sorted(glob.glob(join(out, "snapshot_*.hdf5"))):
        with h5py.File(snap, "r") as f:
            nh = f["PartType0/Density"][:] * UNIT_DENSITY_CGS * 0.7 / 1.6726e-24
            h = f["PartType0/KernelMaxRadius"][:]
        dense = nh > 1e5
        if dense.sum() < 50:
            continue
        checked += 1
        med = np.median(h[dense])
        print(f"{os.path.basename(snap)}: {dense.sum()} cells n_H > 1e5, median h {1e3 * med:.3f} pc, default floor {1e3 * floor_default:.3f} pc")
        assert med < 0.5 * floor_default, f"{os.path.basename(snap)}: dense-gas kernel radius held at the softening floor"
    assert checked, "no snapshot with dense gas"


def test_sfr_log_counts_stella_stars(run):
    """sfr.txt (rows on full-sync steps) books the Stella stars formed since the previous row: summed over the rows up
    to the last one (at TimeMax), column 5 equals the drawn masses of all stars in SFinfo.txt, and column 4 times the
    interval since the previous row gives the same mass."""
    text, out = run
    sfr = np.loadtxt(join(out, "sfr.txt"), comments="#", ndmin=2)
    sfi = np.loadtxt(join(out, "SFinfo.txt"), comments="#", ndmin=2)
    assert abs(sfr[-1, 0] / TIME_MAX - 1) < 1e-9, "no sfr.txt row at TimeMax"
    m_drawn = sfi[:, 5].sum()
    m_logged = sfr[:, 4].sum() * UNIT_MASS_MSUN
    dt_yr = np.diff(np.concatenate([[0.0], sfr[:, 0]])) * UNIT_TIME_YR
    m_rate = np.sum(sfr[:, 3] * dt_yr)
    print(f"{len(sfr)} sfr.txt rows; stars formed (SFinfo) {len(sfi)}, {m_drawn:.3f} Msun; column 5 {m_logged:.3f}, "
          f"column 4 x dt {m_rate:.3f}")
    assert m_drawn > 1.0, "the clump formed no stars"
    assert abs(m_logged - m_drawn) < 1e-3 * m_drawn, "sfr.txt column 5 misses Stella stars"
    assert abs(m_rate - m_drawn) < 1e-3 * m_drawn, "sfr.txt column 4 is not the formed mass per time"
