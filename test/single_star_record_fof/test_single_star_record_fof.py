"""single_star_record_fof test: the single-star records survive FOF + SUBFIND (GALSF_SINGLE_STARS_FROM_IMF with FOF, SUBFIND,
FOF_UNIQUE_KEY_CHECK). SUBFIND ships particles between ranks with raw struct copies (subfind_distribute_particles, subfind_exchange),
so a star's record has to travel with it. FOF and SUBFIND run at every snapshot (io.cc), 5 snapshots, 4 ranks, 64 stars of
0.1-50 Msun in a 5.28 pc box with 8 gas clumps (8 FOF groups of at least 8 members, all handled by the serial SUBFIND code on
other ranks; DesLinkNgb 8 = FOF_GROUP_MIN_SIZE, the largest value the smallest group can supply).
  1. every snapshot holds all 64 stars with mass, drawn mass and birth metallicity bitwise as in the initial conditions; at
     every FOF call SUBFIND handed groups to the serial code and no key was duplicated; in at least one of these calls the
     exchange sent stars with a record to another rank (the redistribution really moved records);
  2. a run stopped halfway and continued from its restart files ends with the same records;
  3. RestartFlag 3 (FOF + SUBFIND on a snapshot, STL-143) runs, finds groups and no duplicate keys.
Before the fix the build stops at the #error in galaxy_sf/single_star_record.cc.
"""
import os
import re
import shutil
import subprocess
from pathlib import Path
import numpy as np
import h5py
import pytest
from gizmo.test import build_gizmo_for_test, run_test, mpi_launch_command

TEST_NAME = "single_star_record_fof"
TEST_DIR = Path(__file__).parent


@pytest.fixture(scope="module")
def built():
    build_gizmo_for_test(TEST_NAME)
    return True


def records(fname):
    with h5py.File(fname, "r") as F:
        s = F["PartType4"]
        o = np.argsort(s["ParticleIDs"][()])
        return (s["ParticleIDs"][()][o].astype(np.uint64), s["Masses"][()][o].astype(np.float64),
                s["SingleStarDrawnMass"][()][o].astype(np.float64), s["SingleStarBirthMetallicity"][()][o].astype(np.float64))


def run(tag, ranks, time_max=None, restart_from=None):
    import sys
    sys.path.insert(0, str(TEST_DIR))
    from make_single_star_record_fof_ics import make_single_star_record_fof_ics
    d = TEST_DIR / f"run_{tag}"
    if restart_from is None and d.exists():
        shutil.rmtree(d)
    d.mkdir(exist_ok=True)
    os.chdir(d)
    try:
        make_single_star_record_fof_ics(f"{TEST_NAME}_ics.hdf5")
        params = Path(TEST_DIR / f"{TEST_NAME}.params").read_text()
        if time_max:
            params = params.replace("TimeMax                            0.4", f"TimeMax                            {time_max}")
        Path(f"{TEST_NAME}.params").write_text(params)
        if not Path("GIZMO").exists():
            os.symlink(TEST_DIR / "GIZMO", "GIZMO")
        if restart_from is None:
            run_test(TEST_NAME, num_mpi_ranks=ranks)
        else:
            with open(f"test_{TEST_NAME}.out", "w") as out:
                subprocess.run(mpi_launch_command(ranks) + ["./GIZMO", f"{TEST_NAME}.params", "1"], stdout=out, stderr=subprocess.STDOUT, check=True)
        snaps = sorted(Path("output").resolve().glob("snapshot_*.hdf5"))
        ic = Path(f"{TEST_NAME}_ics.hdf5").resolve()
        log = Path(f"test_{TEST_NAME}.out").read_text(errors="replace")
    finally:
        os.chdir("../../../")
    return ic, snaps, log


def check_log(log, n_calls_min):
    assert len(re.findall(r"particle keys \(ID, child number\): 0 duplicates", log)) >= n_calls_min, "FOF_UNIQUE_KEY_CHECK did not report 0 duplicates in every FOF call"
    n_small = [int(x) for x in re.findall(r"the other (\d+) FOF halos are treated in parallel with serial code", log)]
    assert len(n_small) >= n_calls_min and min(n_small) >= 1, "SUBFIND did not hand any group to the serial code on another rank: %s" % n_small
    n_moved = [int(x) for x in re.findall(r"subfind_exchange: (\d+) stars with a single-star record sent to another task", log)]
    assert len(n_moved) >= 2 * n_calls_min and max(n_moved) >= 1, "no star with a record changed rank in any SUBFIND exchange: %s" % n_moved


def test_records_follow_stars_through_fof(built):
    ic, snaps, log = run("r4", 4)
    ref = records(ic)
    assert len(snaps) >= 5
    for s in snaps:
        got = records(s)
        assert len(got[0]) == len(ref[0]), s
        for a, b in zip(ref, got):
            assert a.tobytes() == b.tobytes(), s
    check_log(log, 5)


def test_records_survive_restart_after_fof(built):
    ic, _, _ = run("restart", 4, time_max=0.2)
    _, snaps, _ = run("restart", 4, time_max=0.4, restart_from=True)
    for a, b in zip(records(ic), records(snaps[-1])):
        assert a.tobytes() == b.tobytes()


def test_restartflag3_fof(built):
    d = TEST_DIR / "run_r4"
    assert (d / "output" / "snapshot_002.hdf5").exists(), "needs the 4-rank run (test_records_follow_stars_through_fof)"
    shutil.rmtree(d / "output" / "groups_002", ignore_errors=True)
    with open(d / "test_flag3.out", "w") as out:
        subprocess.run(mpi_launch_command(4) + ["./GIZMO", f"{TEST_NAME}.params", "3", "2"], cwd=d, stdout=out, stderr=subprocess.STDOUT, check=True)
    log = (d / "test_flag3.out").read_text(errors="replace")
    assert (d / "output" / "groups_002").exists()
    assert "0 duplicates" in log
