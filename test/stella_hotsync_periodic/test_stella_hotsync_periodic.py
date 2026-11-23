"""stella_hotsync_periodic: the hot-gas sync clamp measures its sphere with nearest-image distances in a periodic box.

A 12 Msun star (cell ID 1, GALSF_RESOLVEDISM_ISOLATED_FB_TEST) on the last lattice plane before the +x face of a periodic
80 x 10 x 10 pc box explodes at TestSNTime0, so its hot gas (T > 1e6 K) straddles the periodic x-boundary.
RESOLVEDISM_HOTSYNC_CLAMP puts every cell within the remnant's sphere (hot-cell centre + largest hot-cell distance
+ 3 largest hot kernel radii) on the hot gas's fine step. Measured with nearest-image distances that sphere stays at the
x-face; gas well outside it, half a box away along x, must keep its own long step.
"""

import glob
import os
import shutil
import subprocess
import sys
import tempfile
from os.path import abspath, dirname, join

import h5py
import numpy as np
import pytest

from gizmo.test import build_gizmo_for_test, mpi_launch_command, forced_sn_check
from gizmo.conservation_trace import keep_and_check_traces

TEST_NAME = "stella_hotsync_periodic"
TEST_DIR = dirname(abspath(__file__))
REPO = abspath(join(TEST_DIR, "..", ".."))
NRANKS = 4
TIME_MAX = 4.0e-6
U_HOT = 1.0e6 * 1.3807e-16 / ((2.0 / 3.0) * 0.6 * 1.6726e-24) / 1e10  # u(T = 1e6 K, mu = 0.6), code units, as the clamp

sys.path.insert(0, TEST_DIR)
from make_stella_hotsync_periodic_ics import make_stella_hotsync_periodic_ics  # noqa: E402


def nearest_image(d, box):
    return d - box * np.round(d / box)


def clamp_reach(snap):
    """Sphere of the clamp from the snapshot (nearest-image centre around the hottest cell), and the far cells' steps."""
    with h5py.File(snap, "r") as f:
        box = np.array([f["Header"].attrs["BoxSize"] * 8, f["Header"].attrs["BoxSize"], f["Header"].attrs["BoxSize"]])
        g = f["PartType0"]
        pos, u, h, dt = (g[k][:].astype(float) for k in ("Coordinates", "InternalEnergy", "KernelMaxRadius", "TimeStep"))
    hot = u > U_HOT
    if hot.sum() == 0:
        return None
    ref = pos[np.argmax(u)]
    cen = ref + nearest_image(pos[hot] - ref, box).mean(axis=0)
    r = np.sqrt((nearest_image(pos - cen, box) ** 2).sum(axis=1))
    r_sync = r[hot].max() + 3 * h[hot].max()
    far = r > 1.25 * r_sync
    dt_hot = dt[hot].max()
    return dict(n_hot=int(hot.sum()), r_sync=r_sync, n_far=int(far.sum()), dt_hot=dt_hot,
                frac_far_clamped=float(np.mean(dt[far] <= 2 * dt_hot)) if far.any() else np.nan,
                dt_far_median=float(np.median(dt[far])) if far.any() else np.nan)


@pytest.mark.skipif(bool(os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN")), reason="builds GIZMO")
def test_hot_sync_sphere_uses_nearest_image():
    cwd = os.getcwd()
    os.chdir(REPO)
    try:
        build_gizmo_for_test(TEST_NAME)
    finally:
        os.chdir(cwd)
    run = tempfile.mkdtemp(prefix=f"{TEST_NAME}_")
    os.makedirs(join(run, "output"))
    os.symlink(REPO, join(run, "gizmo"))
    make_stella_hotsync_periodic_ics(join(run, f"{TEST_NAME}_ics.hdf5"))
    shutil.copy(join(TEST_DIR, f"{TEST_NAME}.params"), run)
    with open(join(run, "run.log"), "w") as log:
        r = subprocess.run(mpi_launch_command(NRANKS) + [join(TEST_DIR, "GIZMO"), f"{TEST_NAME}.params"],
                           stdout=log, stderr=subprocess.STDOUT, cwd=run)
    out = join(TEST_DIR, "output")
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    shutil.copy(join(run, "run.log"), out)
    text = open(join(run, "run.log")).read()
    rows = []
    for snap in sorted(glob.glob(join(run, "output", "snapshot_*.hdf5"))):
        res = clamp_reach(snap)
        if res is not None:
            res["snap"] = os.path.basename(snap)
            rows.append(res)
            print(res)
    trace = keep_and_check_traces(join(run, "output"), out)
    forced = forced_sn_check(join(run, "output", "SNinfo.txt"), join(TEST_DIR, f"{TEST_NAME}.params"))
    shutil.rmtree(run, ignore_errors=True)

    assert r.returncode == 0, f"GIZMO exited {r.returncode}, see {out}/run.log"
    assert forced is None, forced
    assert f"Final time={TIME_MAX:g} reached" in text, f"run stopped before TimeMax ({out}/run.log)"
    rows = [x for x in rows if x["n_far"] > 0]
    assert rows, "no snapshot with hot gas and gas outside the clamp sphere"
    for x in rows:
        assert x["frac_far_clamped"] < 0.05, (
            f"{x['snap']}: {100 * x['frac_far_clamped']:.0f}% of the {x['n_far']} cells beyond 1.25 x the "
            f"nearest-image clamp radius ({1e3 * x['r_sync']:.1f} pc) sit on the hot step {x['dt_hot']:.3g}")
    assert trace.ok, trace.summary()
