"""ketju_task_schedule: KETJU scheduling flags against the unchanged code on 8 MPI ranks, with multi-task compute groups.

64 circular binaries in a Plummer sphere (plummer_binaries setup); KetjuRegionRadius 0.1 pc merges neighbouring binaries
into regions of 4+ members and KetjuMinParticlesPerTask 2 gives them several compute tasks. See README.md.
  - control: the unchanged code twice, bitwise equal (else nothing below can be judged);
  - KETJU_SCATTER_AFTER_ALL_REGIONS: bitwise equal to the unchanged code;
  - KETJU_ONE_TASK_PER_REGION, with and without the scatter flag: bitwise equal to each other; against the unchanged code
    only the energy error is compared (the negative half-kick of a multi-task region is summed on one task instead of
    several, a roundoff change): |dE/E|(one task) <= 2 |dE/E|(unchanged) + 1e-6, threshold fixed before the first run.
"""
import glob
import importlib.util
import re
import shutil
import subprocess
import sys
from os import path

import h5py
import numpy as np
import pytest

from gizmo.test import build_and_run_test, variant_output_dir
from gizmo.units import G_CODE

TEST_NAME = "ketju_task_schedule"
TEST_DIR = f"test/{TEST_NAME}"
IC_FILE = f"{TEST_DIR}/{TEST_NAME}_ics.hdf5"
NUM_MPI_RANKS = 8
N_BINARIES = 64
SCATTER = ("KETJU_SCATTER_AFTER_ALL_REGIONS",)
ONE = ("KETJU_ONE_TASK_PER_REGION",)
BOTH = SCATTER + ONE
CONTROL_DIR = f"{TEST_DIR}/output_control"


def _ensure_ic():
    if path.isfile(IC_FILE):
        return
    spec = importlib.util.spec_from_file_location("make_plummer_binaries_ics", "test/plummer_binaries/make_plummer_binaries_ics.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.make_plummer_binaries_ics(N_binaries=N_BINARIES, m_star=1.0, a_cluster=1.0, binary_separation_au=1000.0,
                                  boxsize=300.0, seed=42, outfile=IC_FILE)


def _snapshots(d):
    snaps = sorted(glob.glob(d + "/snapshot_*.hdf5"))
    assert len(snaps) >= 2, f"no snapshots in {d}"
    return snaps


def _compare(da, db, log, label):
    a, b = _snapshots(da), _snapshots(db)
    assert [path.basename(s) for s in a] == [path.basename(s) for s in b], f"{label}: different snapshot sets"
    res = subprocess.run([sys.executable, f"{TEST_DIR}/compare_snaps.py"] + [x for ab in zip(a, b) for x in ab], capture_output=True, text=True)
    log.write(f"== {label}\n{res.stdout}{res.stderr}")
    return res.returncode == 0, res.stdout


def _energy_error(d):
    """|E(t_end) - E(0)| / |E(0)|, direct sum on the synchronised (r, v) pairs, softening negligible (1000 AU >> 6 AU)."""
    def energy(f):
        with h5py.File(f, "r") as F:
            g = F["PartType5"]
            x = g["HermiteSyncCoordinates"][:] if "HermiteSyncCoordinates" in g else g["Coordinates"][:]
            v = g["HermiteSyncVelocities"][:] if "HermiteSyncVelocities" in g else g["Velocities"][:]
            m = g["Masses"][:]
        ke = 0.5 * np.sum(m * np.sum(v * v, axis=1))
        dx = x[:, None, :] - x[None, :, :]
        r = np.sqrt(np.sum(dx * dx, axis=2)); iu = np.triu_indices(len(m), 1)
        return ke - G_CODE * np.sum(m[iu[0]] * m[iu[1]] / r[iu])
    s = _snapshots(d)
    e0, e1 = energy(s[0]), energy(s[-1])
    return abs(e1 - e0) / abs(e0)


def test_ketju_task_schedule():
    _ensure_ic()
    build_and_run_test(TEST_NAME, NUM_MPI_RANKS, 1, ())
    out = open(f"{TEST_DIR}/test_{TEST_NAME}.out").read()
    ranges = [(int(a), int(b), int(n)) for n, a, b in re.findall(r"Region with (\d+) particles \(\d+ PN\), compute tasks (\d+)-(\d+)", out)]
    assert any(b > a for a, b, n in ranges), "no region got several compute tasks: the multi-task paths were not exercised"
    shutil.rmtree(CONTROL_DIR, ignore_errors=True)
    shutil.move(variant_output_dir(TEST_NAME, ()), CONTROL_DIR)
    for flags in [(), SCATTER, ONE, BOTH]:
        build_and_run_test(TEST_NAME, NUM_MPI_RANKS, 1, flags)
    base = variant_output_dir(TEST_NAME, ())
    with open(f"{TEST_DIR}/{TEST_NAME}_compare.txt", "w") as log:
        log.write(f"regions with several compute tasks (rank-0 roots): {sum(b > a for a, b, n in ranges)} of {len(ranges)} logged, "
                  f"largest {max(n for a, b, n in ranges)} members\n")
        ok, txt = _compare(CONTROL_DIR, base, log, "control: unchanged twice")
        assert ok, f"the unchanged code is not reproducible, nothing can be judged:\n{txt[-2000:]}"
        ok, txt = _compare(base, variant_output_dir(TEST_NAME, SCATTER), log, "KETJU_SCATTER_AFTER_ALL_REGIONS vs unchanged")
        assert ok, f"KETJU_SCATTER_AFTER_ALL_REGIONS changed the results:\n{txt[-3000:]}"
        ok, txt = _compare(variant_output_dir(TEST_NAME, ONE), variant_output_dir(TEST_NAME, BOTH), log, "both flags vs KETJU_ONE_TASK_PER_REGION")
        assert ok, f"KETJU_SCATTER_AFTER_ALL_REGIONS changed the one-task results:\n{txt[-3000:]}"
        de_base, de_one = _energy_error(base), _energy_error(variant_output_dir(TEST_NAME, ONE))
        same, _ = _compare(base, variant_output_dir(TEST_NAME, ONE), log, "KETJU_ONE_TASK_PER_REGION vs unchanged (information)")
        log.write(f"|dE/E| unchanged {de_base:.6e}  one task {de_one:.6e}  bitwise {same}\n")
    assert de_one <= 2 * de_base + 1e-6, f"KETJU_ONE_TASK_PER_REGION energy error {de_one:.3e} vs unchanged {de_base:.3e}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
