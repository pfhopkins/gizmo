"""sgchem_cr_zero_before_first_star: SGCHEM_COSMIC_RAYS_ZERO_BEFORE_FIRST_STAR switches the cosmic-ray ionisation rate.

The rate handed to the chemistry is exactly 0 until a star or sink exists (in the initial conditions, in the snapshot the
run starts from, or formed in the run), then that of the build (here CR_SCALE_WITH_G0, as the zoom:
clamp(G0/1.7 x CosmicRayIonRate, 1e-21, 2e-16) s^-1).

Setup: the production model with the zoom's cosmic-ray flags, the clump of test/stella_dense_sf (n_H = 1e6 cm^-3,
free-fall time 4.4e-5), 2 ranks, one snapshot file per rank. The star-formation threshold (CritPhysDensity 1.5e6) is
above the initial clump density, so the first star forms during the collapse.

What a snapshot shows: CR_IonizationRate of a cell is the rate of its last chemistry call. A snapshot at time T is
written before the calls at T, and every cell is called at least once per MaxSizeTimestep (= snapshot interval, DT).
The star-formation routine runs after the chemistry of its step. So
  * a snapshot without a star was written before the first star formed: every cell must hold 0;
  * from the snapshot after the first one with a star, every cell's last call was made with stars present: every cell > 0.

One build, five runs:
  no_star       threshold 1e30, no star ever: 0 in every cell of every snapshot.
  ic_star       threshold 1e30, one old star in the IC: > 0 in every cell from T = 2 DT (after the first calls).
  birth         stars form during the run: 0 before, > 0 after, in the file of each rank.
  restart_files birth continued from its restart files (flag 1) with SfEffPerFreeFall 1e-30 (no further star).
  restart_snap  birth continued from its last snapshot (flag 2), SfEffPerFreeFall 1e-30.
                Both: no star forms after the restart and every cell is > 0 from 2 DT after it (a restart that lost the
                switch would give 0 there).
"""
import glob
import os
import shutil
import subprocess
import sys
import tempfile
from os.path import abspath, basename, dirname, join

import h5py
import numpy as np
import pytest

from gizmo.test import build_gizmo_for_test, mpi_launch_command, write_params_with_overrides

TEST_NAME = "sgchem_cr_zero_before_first_star"
TEST_DIR = dirname(abspath(__file__))
REPO = abspath(join(TEST_DIR, "..", ".."))
NRANKS = 2
DT = 1.25e-6           # MaxSizeTimestep = TimeBetSnapshot
T_BIRTH_RUN = 3.0e-5   # TimeMax of the parameter file
T_RESTARTED = 3.5e-5   # TimeMax of the two continuations
ZETA_MIN, ZETA_MAX = 1e-21, 2e-16  # limits of CR_SCALE_WITH_G0
NO_SF = {"CritPhysDensity": "1.0e30"}  # fresh starts: no cell reaches the threshold
# restarts: star-formation time 1e30 free-fall times. Not the threshold: a restart from restart files takes a changed
# CritPhysDensity into the star-formation time only, the threshold stays that of the restart files.
NO_SF_RESTART = {"SfEffPerFreeFall": "1.0e-30"}

sys.path.insert(0, TEST_DIR)
from make_sgchem_cr_zero_before_first_star_ics import make_sgchem_cr_zero_before_first_star_ics  # noqa: E402


def _gizmo(rundir, params, *args):
    log = join(rundir, f"{params[:-len('.params')]}{'_' + '_'.join(args) if args else ''}.log")
    with open(log, "w") as out:
        r = subprocess.run(mpi_launch_command(NRANKS) + [join(TEST_DIR, "GIZMO"), params, *args], stdout=out, stderr=subprocess.STDOUT, cwd=rundir)
    text = open(log).read()
    assert r.returncode == 0, f"GIZMO exited {r.returncode}, see {log}"
    assert "reached. Simulation ends." in text, f"run stopped before TimeMax ({log})"


def _params(rundir, name, overrides):
    """Parameter file <name>.params in rundir: the test's parameters with overrides."""
    shutil.copy(join(TEST_DIR, f"{TEST_NAME}.params"), join(rundir, f"{name}.params"))
    made = write_params_with_overrides(join(rundir, f"{name}.params"), dict(overrides, OutputDir=f"output_{name}"))
    os.replace(made, join(rundir, f"{name}.params"))
    return f"{name}.params"


def _snapshots(outdir):
    """[(time, [CR_IonizationRate of the gas cells in the file of each rank], number of stars)] in time order."""
    out = []
    for d in glob.glob(join(outdir, "snapdir_*")):
        rates, n_star, time = [], 0, None
        for fn in sorted(glob.glob(join(d, "snapshot_*.hdf5"))):
            with h5py.File(fn, "r") as f:
                time = float(f["Header"].attrs["Time"])
                n_star += int(f["Header"].attrs["NumPart_ThisFile"][4]) + int(f["Header"].attrs["NumPart_ThisFile"][5])
                rates.append(f["PartType0/CR_IonizationRate"][:].astype(np.float64) if "PartType0" in f else np.zeros(0))
        out.append((time, rates, n_star))
    return sorted(out, key=lambda s: s[0])


def _describe(name, snaps):
    for time, rates, n_star in snaps:
        r = np.concatenate(rates)
        print(f"{name}: T = {time:.4e}  stars {n_star:4d}  cells per rank {[len(x) for x in rates]}  rate 0 in {int((r == 0).sum())} of {len(r)}"
              f"  min {r.min():.3e}  max {r.max():.3e}")


def _per_rank(rates):
    return len(rates) == NRANKS and all(len(r) > 0 for r in rates)


def _all_zero(rates):
    return _per_rank(rates) and all(bool((r == 0).all()) for r in rates)


def _all_on(rates):
    return _per_rank(rates) and all(bool(((r >= ZETA_MIN * (1 - 1e-6)) & (r <= ZETA_MAX * (1 + 1e-6))).all()) for r in rates)


@pytest.fixture(scope="module")
def runs():
    """Build once, run the five cases; returns {case: snapshots} (outputs kept in the test directory)."""
    if os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        pytest.skip("builds GIZMO")
    cwd = os.getcwd()
    os.chdir(REPO)
    try:
        build_gizmo_for_test(TEST_NAME)
    finally:
        os.chdir(cwd)
    rundir = tempfile.mkdtemp(prefix=f"{TEST_NAME}_")
    os.symlink(REPO, join(rundir, "gizmo"))
    ic = join(rundir, f"{TEST_NAME}_ics.hdf5")
    try:
        make_sgchem_cr_zero_before_first_star_ics(ic)
        _gizmo(rundir, _params(rundir, "no_star", NO_SF))
        _gizmo(rundir, _params(rundir, "birth", {}))
        last = max(int(basename(d)[-3:]) for d in glob.glob(join(rundir, "output_birth", "snapdir_*")))
        # continued from the restart files written at the end of birth
        shutil.copytree(join(rundir, "output_birth"), join(rundir, "output_restart_files"))
        _gizmo(rundir, _params(rundir, "restart_files", dict(NO_SF_RESTART, TimeMax=T_RESTARTED)), "1")
        # continued from the last snapshot of birth
        os.makedirs(join(rundir, "output_restart_snap"))
        shutil.copytree(join(rundir, "output_birth", f"snapdir_{last:03d}"), join(rundir, "output_restart_snap", f"snapdir_{last:03d}"))
        _gizmo(rundir, _params(rundir, "restart_snap", dict(NO_SF_RESTART, TimeMax=T_RESTARTED)), "2", str(last))
        make_sgchem_cr_zero_before_first_star_ics(ic, with_star=True)
        _gizmo(rundir, _params(rundir, "ic_star", dict(NO_SF, TimeMax=6 * DT)))
    finally:
        for d in glob.glob(join(rundir, "output_*")):
            keep = join(TEST_DIR, basename(d))
            shutil.rmtree(keep, ignore_errors=True)
            shutil.copytree(d, keep, ignore=shutil.ignore_patterns("restartfiles"))
        for log in glob.glob(join(rundir, "*.log")):
            shutil.copy(log, TEST_DIR)
        shutil.rmtree(rundir, ignore_errors=True)
    out = {case: _snapshots(join(TEST_DIR, f"output_{case}")) for case in ("no_star", "ic_star", "birth", "restart_files", "restart_snap")}
    for case, snaps in out.items():
        _describe(case, snaps)
    return out


def test_no_star_rate_is_zero(runs):
    """No star in the IC and none formed: exactly 0 in every cell of every snapshot, to the end of the run."""
    snaps = runs["no_star"]
    assert abs(snaps[-1][0] / T_BIRTH_RUN - 1) < 1e-9 and len(snaps) >= 12
    assert all(n_star == 0 for _, _, n_star in snaps), "a star formed in the run that should have none"
    bad = [time for time, rates, _ in snaps if not _all_zero(rates)]
    assert not bad, f"cosmic-ray rate not 0 without a star at T = {bad}"


def test_star_in_ic_rate_is_on(runs):
    """One star in the IC, none formed: the rate is on in every cell once every cell has been through the chemistry."""
    snaps = runs["ic_star"]
    assert all(n_star == 1 for _, _, n_star in snaps), "the run should hold the IC star and form none"
    late = [(time, rates) for time, rates, _ in snaps if time >= 2 * DT * (1 - 1e-9)]
    assert len(late) >= 4
    bad = [time for time, rates in late if not _all_on(rates)]
    assert not bad, f"cosmic-ray rate not on with a star in the IC at T = {bad}"


def test_rate_switches_on_at_first_star(runs):
    """A star forms during the run: 0 in every cell of every rank before, on in every cell of every rank after."""
    snaps = runs["birth"]
    first = next((k for k, (_, _, n_star) in enumerate(snaps) if n_star > 0), None)
    assert first is not None, "no star formed"
    before, after = snaps[:first], snaps[first + 1:]
    print(f"first snapshot with a star: T = {snaps[first][0]:.4e} ({snaps[first][2]} stars); {len(before)} snapshots before, {len(after)} after")
    # both phases must be seen: star-free snapshots late enough that a build without the switch shows the rate (T >= 2 DT, as ic_star)
    assert sum(time >= 2 * DT * (1 - 1e-9) for time, _, _ in before) >= 2, "the first star formed too early to see the rate at 0 before it"
    assert len(after) >= 2, "the first star formed too late to see the rate after it"
    bad = [time for time, rates, _ in before if not _all_zero(rates)]
    assert not bad, f"cosmic-ray rate not 0 before the first star at T = {bad}"
    bad = [time for time, rates, _ in after if not _all_on(rates)]
    assert not bad, f"cosmic-ray rate not on in every cell of every rank after the first star at T = {bad}"


@pytest.mark.parametrize("case", ["restart_files", "restart_snap"])
def test_rate_stays_on_after_restart(runs, case):
    """Restarted after the first star, with star formation switched off: no new star, and the rate stays on."""
    birth, snaps = runs["birth"], runs[case]
    t_restart, n_star = birth[-1][0], birth[-1][2]
    assert n_star > 0
    # a restart from restart files with a larger TimeMax ends at the first step boundary beyond it
    assert snaps[-1][0] >= T_RESTARTED * (1 - 1e-9), "the continuation did not reach its TimeMax"
    late = [(time, rates, n) for time, rates, n in snaps if time >= (t_restart + 2 * DT) * (1 - 1e-9)]
    assert len(late) >= 2
    assert all(n == n_star for _, _, n in late), f"stars after the restart {[n for _, _, n in late]}, at the restart {n_star}: star formation was not off"
    bad = [time for time, rates, _ in late if not _all_on(rates)]
    assert not bad, f"cosmic-ray rate not on after the restart at T = {bad}"
