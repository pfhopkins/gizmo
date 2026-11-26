"""STL-36 / STL-431a: last_wind_log_age is written to and read from snapshots (GALSF_RESOLVEDISM_SNAPSHOT_KEEPS_LAST_WIND_AGE).

One 40 Msun star with continuous winds (TestStarInitAgeYr 3 Myr). Run A starts fresh and writes a snapshot at 25 kyr;
run B restarts from that snapshot (RestartFlag 2). Double-precision snapshot I/O: in float the age would round at ~4e-7
(checker 09), so the comparison is on the stored double. Quantity: the log age at which B's first wind interval starts
(WINDAGES log_age_prev of B's first wind) against the log age of A's last wind before the snapshot (its log_age_now).
Bar (fixed before the run): |difference| <= 1e-12. Expected without the flag: the field is not in the snapshot, B starts
its first interval at the restart's value (0, i.e. from the star's birth), so its first wind carries the composition
averaged over the whole life instead of the last interval's."""

import os
import shutil
from os.path import join

import h5py
import pytest

from audit_round1_common import HOOKS, ROUND1_FLAGS, TEST_DIR, UNIT_T_YR, binary, kv, make_box, run

ARMS = {"off": HOOKS, "on": HOOKS + ROUND1_FLAGS}
BAR = 1e-12
T_SNAP, T_END = 2.5e4, 4.5e4  # yr


def check(arm):
    exe = binary("round1_" + arm, add=ARMS[arm])
    a, b = f"lwa_A_{arm}", f"lwa_B_{arm}"
    da, db = join(TEST_DIR, "run_" + a), join(TEST_DIR, "run_" + b)
    os.makedirs(da, exist_ok=True); os.makedirs(db, exist_ok=True)
    make_box(join(da, "box_ics.hdf5"))
    star = {"TestStarMass": 40.0, "TestNumStars": 1, "TestStarInitAgeYr": 3.0e6, "TestSNTime0": 0}
    text_a = run(a, exe, {"InitCondFile": "box_ics", "TimeMax": T_SNAP / UNIT_T_YR, "TimeBetSnapshot": T_SNAP / UNIT_T_YR, **star})[1]
    snap = join(da, "output", "snapshot_001.hdf5")
    with h5py.File(snap, "r") as f:
        t_snap = float(f["Header"].attrs["Time"])
        stored = float(f["PartType4/LastWindLogAge"][0]) if "PartType4/LastWindLogAge" in f else None
    winds_a = [w for w in kv(text_a, "WINDAGES") if float(w["t"]) <= t_snap]
    assert winds_a, "run A had no wind before the snapshot"
    ref = float(winds_a[-1]["log_age_now"])
    shutil.copy(snap, join(db, "lwa_000.hdf5"))
    text_b = run(b, exe, {"InitCondFile": "lwa_000", "TimeMax": T_END / UNIT_T_YR, "TimeBetSnapshot": T_END / UNIT_T_YR, **star}, restart_flag=2)[1]
    winds_b = kv(text_b, "WINDAGES")
    assert winds_b, "run B had no wind after the restart"
    got = float(winds_b[0]["log_age_prev"])
    print(f"last wind age ({arm}): A's last wind before the snapshot ends at log age {ref:.17g}; stored in the snapshot: {stored}; "
          f"B's first wind starts at {got:.17g}: difference {got - ref:+.3e} (B's first interval {float(winds_b[0]['log_age_now']) - got:.3e} dex)")
    assert abs(got - ref) <= BAR


def test_last_wind_age_flag_on():
    check("on")


@pytest.mark.xfail(strict=True, reason="STL-36 without the flag: the snapshot restart starts the first wind interval at the star's birth")
def test_last_wind_age_flag_off():
    check("off")
