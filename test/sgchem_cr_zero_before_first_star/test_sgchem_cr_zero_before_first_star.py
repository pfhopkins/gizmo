"""sgchem_cr_zero_before_first_star: SGCHEM_COSMIC_RAYS_ZERO_BEFORE_FIRST_STAR switches the cosmic-ray ionisation rate.

The rate handed to the chemistry is exactly 0 until a star or sink exists (in the initial conditions, in the snapshot the
run starts from, or formed in the run), then that of the build (here CR_SCALE_WITH_G0, as the zoom:
clamp(G0/1.7 x CosmicRayIonRate, 1e-21, 2e-16) s^-1).

Setup (port of stella_dev's test of this name): the Stella production flags with the zoom's cosmic-ray flags, a dense clump
(n_H = 1e6 cm^-3, free-fall time 4.4e-5), 2 ranks, one snapshot file per rank. The star-formation threshold (CritPhysDensity 1.5e6) is
above the initial clump density, so the first star forms during the collapse.

What a snapshot shows: CR_IonizationRate of a cell is the rate of its last chemistry call. A snapshot at time T is
written before the calls at T, and every cell is called at least once per MaxSizeTimestep (= snapshot interval, DT).
The star-formation routine runs after the chemistry of its step. So
  * a snapshot without a star was written before the first star formed: every cell must hold 0;
  * from the snapshot after the first one with a star, every cell's last call was made with stars present: every cell > 0.

The state (one integer, on every task alike) is set by stars that exist after their IMF draw, kept in the restart files
and written to the snapshot header (StarFormationHasSetIn), which a snapshot start reads; a snapshot without the header
entry falls back to a scan for stars and sinks.

One build, eight runs:
  no_star       threshold 1e30, no star ever: 0 in every cell of every snapshot.
  ic_star       threshold 1e30, one old star in the IC: > 0 in every cell from T = 2 DT (after the first calls).
  birth         stars form during the run: 0 before, > 0 after, in the file of each rank.
  restart_files birth continued from its restart files (flag 1) with SfEffPerFreeFall 1e-30 (no further star).
  restart_snap  birth continued from its last snapshot (flag 2), SfEffPerFreeFall 1e-30.
                Both: no star forms after the restart and every cell is > 0 from 2 DT after it (a restart that lost the
                switch would give 0 there).
  snap_stars_gone      as restart_snap, every star removed from the snapshot (as after a fully disrupted first star),
                       header entry kept: no star in the run, yet > 0 in every cell.
  snap_old             as restart_snap, header entry removed (a snapshot of an older build), stars kept: > 0 (scan).
  snap_old_stars_gone  header entry and stars removed: nothing says star formation has set in, 0 in every cell.
A second build with TEST_IMF_FAIL_EVERY_DRAW_BEFORE_TIME=1.5e-5, one run:
  failed_draws  as birth, but every draw of the gas collection before 1.5e-5 fails: the spawned star is removed again and
                its grants go back to the donors. The rate stays 0 until a star survives its draw.
"""
import glob
import os
import re
import shutil
import subprocess
import sys
from os.path import abspath, basename, dirname, join

import h5py
import numpy as np
import pytest

from gizmo.test import build_gizmo_for_test, get_cooling_tables, get_stellar_evolution_tables, mpi_launch_command, write_params_with_overrides

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
STATE = "StarFormationHasSetIn"  # header entry of the state
T_DRAWS_FAIL = 1.5e-5            # the second build fails every IMF draw before this time
TABLES = None                    # stellar tables file, set by the fixture
SNAP_CASES = {"snap_stars_gone": (True, False), "snap_old": (False, True), "snap_old_stars_gone": (True, True)}  # (stars removed, header entry removed)

sys.path.insert(0, TEST_DIR)
from make_sgchem_cr_zero_before_first_star_ics import make_sgchem_cr_zero_before_first_star_ics  # noqa: E402


def _gizmo(rundir, params, *args):
    log = join(rundir, f"{params[:-len('.params')]}{'_' + '_'.join(args) if args else ''}.log")
    with open(log, "w") as out:
        r = subprocess.run(mpi_launch_command(NRANKS) + [join(TEST_DIR, "GIZMO"), params, *args], stdout=out, stderr=subprocess.STDOUT, cwd=rundir,
                           env={**os.environ, "OMP_NUM_THREADS": "1"})
    text = open(log).read()
    assert r.returncode == 0, f"GIZMO exited {r.returncode}, see {log}"
    assert "reached. Simulation ends." in text, f"run stopped before TimeMax ({log})"


def _params(rundir, name, overrides):
    """Parameter file <name>.params in rundir: the test's parameters with overrides."""
    text = open(join(TEST_DIR, f"{TEST_NAME}.params")).read()
    text = text.replace("StellarTablesFile           SET_BY_TEST", f"StellarTablesFile           {TABLES}")
    text = text.replace("SGChemTablesPath            SET_BY_TEST", f"SGChemTablesPath            {os.environ.get('SGCHEM_TABLES_DIR', dirname(TABLES))}")
    assert "SET_BY_TEST" not in text
    open(join(rundir, f"{name}.params"), "w").write(text)
    made = write_params_with_overrides(join(rundir, f"{name}.params"), dict(overrides, OutputDir=f"output_{name}"))
    os.replace(made, join(rundir, f"{name}.params"))
    return f"{name}.params"


def _edit_snapshot(snapdir, drop_stars, drop_state):
    """Remove the stars and sinks and/or the header entry of the state from every file of a snapshot."""
    for fn in glob.glob(join(snapdir, "snapshot_*.hdf5")):
        with h5py.File(fn, "r+") as f:
            h = f["Header"].attrs
            if drop_stars:
                for t in (4, 5):
                    if f"PartType{t}" in f:
                        del f[f"PartType{t}"]
                for k in ("NumPart_ThisFile", "NumPart_Total"):
                    n = h[k].copy()
                    n[4] = n[5] = 0
                    h.modify(k, n)
            if drop_state and STATE in h:
                del h[STATE]


def _header_state(snapdir):
    """The header entry of the state in every file of a snapshot (None where a file has none)."""
    out = []
    for fn in sorted(glob.glob(join(snapdir, "snapshot_*.hdf5"))):
        with h5py.File(fn, "r") as f:
            out.append(int(f["Header"].attrs[STATE]) if STATE in f["Header"].attrs else None)
    return out


def _failed_draw_times(log):
    """Time of every step with a failed draw of the gas collection in a run log (the collection's line: 'Single stars: N new,
    F failed draws ...')."""
    time, out = None, []
    for line in open(log):
        m = re.match(r"Sync-Point \d+, Time: ([0-9.eE+-]+)", line)
        if m:
            time = float(m.group(1))
            continue
        m = re.match(r"Single stars: \d+ new, (\d+) failed draws", line)
        if m and int(m.group(1)) > 0:
            out.append(time)
    return out


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
    """Build and run the cases; returns {case: snapshots} (outputs kept in the test directory)."""
    if os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        pytest.skip("builds GIZMO")
    if not os.environ.get("SUNDIALS_DIR"):
        pytest.skip("set SUNDIALS_DIR (SGCHEM builds against SUNDIALS 6.x or 7.x)")
    global TABLES
    TABLES = get_stellar_evolution_tables()["StellarTablesFile"]
    get_cooling_tables(TEST_DIR)
    cwd = os.getcwd()
    os.chdir(REPO)
    try:
        build_gizmo_for_test(TEST_NAME)
    finally:
        os.chdir(cwd)
    rundir = join(TEST_DIR, "run")  # in the test directory, not in the system's temporary directory
    shutil.rmtree(rundir, ignore_errors=True)
    os.makedirs(rundir)
    for f in ("TREECOOL", "spcool_tables"):
        if os.path.exists(join(TEST_DIR, f)):
            os.symlink(join(TEST_DIR, f), join(rundir, f))
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
        for case, (drop_stars, drop_state) in SNAP_CASES.items():
            os.makedirs(join(rundir, f"output_{case}"))
            shutil.copytree(join(rundir, "output_birth", f"snapdir_{last:03d}"), join(rundir, f"output_{case}", f"snapdir_{last:03d}"))
            _edit_snapshot(join(rundir, f"output_{case}", f"snapdir_{last:03d}"), drop_stars, drop_state)
            _gizmo(rundir, _params(rundir, case, dict(NO_SF_RESTART, TimeMax=T_RESTARTED)), "2", str(last))
        make_sgchem_cr_zero_before_first_star_ics(ic, with_star=True)
        _gizmo(rundir, _params(rundir, "ic_star", dict(NO_SF, TimeMax=6 * DT)))
        os.chdir(REPO)
        try:
            build_gizmo_for_test(TEST_NAME, extra_config_flags=(f"TEST_IMF_FAIL_EVERY_DRAW_BEFORE_TIME={T_DRAWS_FAIL:g}",))
        finally:
            os.chdir(cwd)
        make_sgchem_cr_zero_before_first_star_ics(ic)
        _gizmo(rundir, _params(rundir, "failed_draws", {}))
    finally:
        for d in glob.glob(join(rundir, "output_*")):
            keep = join(TEST_DIR, basename(d))
            shutil.rmtree(keep, ignore_errors=True)
            shutil.copytree(d, keep, ignore=shutil.ignore_patterns("restartfiles"))
        for log in glob.glob(join(rundir, "*.log")):
            shutil.copy(log, TEST_DIR)
        shutil.rmtree(rundir, ignore_errors=True)
    out = {case: _snapshots(join(TEST_DIR, f"output_{case}")) for case in ("no_star", "ic_star", "birth", "restart_files", "restart_snap", "failed_draws", *SNAP_CASES)}
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
    if case == "restart_files":
        # the restart files hold the state at the end of the run: star formation of the last step comes after the last snapshot,
        # so the stars at the restart are all the run formed (the collection's lines "Single stars: N new")
        n_star = sum(int(n) for n in re.findall(r"^Single stars: (\d+) new", open(join(TEST_DIR, "birth.log")).read(), re.M))
        assert n_star >= birth[-1][2]
    # a restart from restart files with a larger TimeMax ends at the first step boundary beyond it
    assert snaps[-1][0] >= T_RESTARTED * (1 - 1e-9), "the continuation did not reach its TimeMax"
    late = [(time, rates, n) for time, rates, n in snaps if time >= (t_restart + 2 * DT) * (1 - 1e-9)]
    assert len(late) >= 2
    assert all(n == n_star for _, _, n in late), f"stars after the restart {[n for _, _, n in late]}, at the restart {n_star}: star formation was not off"
    bad = [time for time, rates, _ in late if not _all_on(rates)]
    assert not bad, f"cosmic-ray rate not on after the restart at T = {bad}"


def test_failed_draw_does_not_switch_rate_on(runs):
    """Every IMF draw before T_DRAWS_FAIL fails and its star is removed again: the rate stays 0 until a star survives."""
    snaps = runs["failed_draws"]
    failed = _failed_draw_times(join(TEST_DIR, "failed_draws.log"))
    assert failed, "no IMF draw failed"
    assert max(failed) < T_DRAWS_FAIL, "a draw failed that the test did not fail"
    first = next((k for k, (_, _, n_star) in enumerate(snaps) if n_star > 0), None)
    assert first is not None, "no star survived"
    before, after = snaps[:first], snaps[first + 1:]
    # star-free snapshots more than DT after the first failed draw: had that draw switched the rate on, every cell would show it
    probe = [time for time, _, _ in before if time > failed[0] + DT]
    print(f"{len(failed)} failed draws from T = {failed[0]:.4e} to {max(failed):.4e}; first snapshot with a star: T = {snaps[first][0]:.4e}; "
          f"{len(probe)} star-free snapshots later than DT after the first failed draw, {len(after)} snapshots after")
    assert len(probe) >= 2, "no star-free snapshot late enough after the first failed draw"
    assert len(after) >= 2, "the first star survived too late to see the rate after it"
    bad = [time for time, rates, _ in before if not _all_zero(rates)]
    assert not bad, f"cosmic-ray rate on after a failed draw, with no star, at T = {bad}"
    bad = [time for time, rates, _ in after if not _all_on(rates)]
    assert not bad, f"cosmic-ray rate not on after the first surviving star at T = {bad}"


def test_snapshot_header_holds_the_state(runs):
    """The snapshots of birth carry the state in the header of every file: 0 before the first star, 1 at the end."""
    dirs = sorted(glob.glob(join(TEST_DIR, "output_birth", "snapdir_*")))
    assert _header_state(dirs[0]) == [0] * NRANKS, f"header entry {STATE} of the first snapshot: {_header_state(dirs[0])}"
    assert _header_state(dirs[-1]) == [1] * NRANKS, f"header entry {STATE} of the last snapshot: {_header_state(dirs[-1])}"


@pytest.mark.parametrize("case", list(SNAP_CASES))
def test_snapshot_start_takes_state_from_header(runs, case):
    """Snapshot starts from the last snapshot of birth with its stars and/or its header entry removed, star formation off."""
    stars_gone, old = SNAP_CASES[case]
    birth, snaps = runs["birth"], runs[case]
    t_restart = birth[-1][0]
    assert abs(snaps[-1][0] / T_RESTARTED - 1) < 1e-9, "the continuation did not reach its TimeMax"
    late = [(time, rates, n) for time, rates, n in snaps if time >= (t_restart + 2 * DT) * (1 - 1e-9)]
    assert len(late) >= 2
    n_expected = 0 if stars_gone else birth[-1][2]
    assert all(n == n_expected for _, _, n in late), f"stars {[n for _, _, n in late]}, expected {n_expected}"
    if stars_gone and old:  # no header entry and no star: the scan finds nothing
        bad = [time for time, rates, _ in late if not _all_zero(rates)]
        assert not bad, f"cosmic-ray rate not 0 with neither the header entry nor a star at T = {bad}"
    else:
        bad = [time for time, rates, _ in late if not _all_on(rates)]
        assert not bad, f"cosmic-ray rate not on after the snapshot start at T = {bad}"
