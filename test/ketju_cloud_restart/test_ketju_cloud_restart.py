"""KETJU in a gas cloud across a restart: the gmc_cooling cloud with gravity on and 16 embedded stars (4 tight binaries in
KETJU regions). A run stopped at t = 0.001 and continued from its restart file to t = 0.002 must end where the straight
run ends (stars and gas), conserve mass, and keep the regions' MSTAR energy error small."""
import glob
import os
import re
import shutil
import subprocess
from pathlib import Path

import h5py
import numpy as np

from gizmo.test import build_gizmo_for_test, run_test, download_test_files, get_cooling_tables, mpi_launch_command

TEST_NAME = "ketju_cloud_restart"
TEST_DIR = Path(__file__).parent
RANKS, THREADS = 2, 2


def _ics():
    ic = TEST_DIR / f"{TEST_NAME}_ics.hdf5"
    if not ic.exists():
        gmc = TEST_DIR.parent / "gmc_cooling" / "gmc_cooling_ics.hdf5"
        if not gmc.exists():
            cwd = os.getcwd(); os.chdir(gmc.parent)  # the download writes into the working directory
            download_test_files("gmc_cooling")
            os.chdir(cwd)
        from make_ketju_cloud_restart_ics import make_ketju_cloud_restart_ics
        make_ketju_cloud_restart_ics(str(gmc), str(ic))


def _final(outdir):
    with h5py.File(sorted(glob.glob(f"{outdir}/snapshot_*.hdf5"))[-1], "r") as f:
        r = {"t": f["Header"].attrs["Time"]}
        for pt in ("PartType0", "PartType4"):
            o = np.argsort(np.array(f[pt]["ParticleIDs"]))
            r[pt] = {k: np.array(f[pt][k])[o] for k in ("Coordinates", "Velocities", "Masses")}
    return r


def test_ketju_cloud_restart():
    _ics()
    get_cooling_tables(str(TEST_DIR))
    build_gizmo_for_test(TEST_NAME, THREADS)
    out = TEST_DIR / "output"; straight = TEST_DIR / "output_straight"
    cwd = os.getcwd(); os.chdir(TEST_DIR)  # the runs start in the test directory
    try:
        shutil.rmtree(out, ignore_errors=True)
        run_test(TEST_NAME, RANKS, THREADS)  # straight run to 0.002
        shutil.rmtree(straight, ignore_errors=True); shutil.move(str(out), str(straight))
        log_straight = Path(f"test_{TEST_NAME}.out").read_text()
        run_test(TEST_NAME, RANKS, THREADS, param_overrides={"TimeMax": 0.001})  # writes a restart file at its final time
        with open(f"test_{TEST_NAME}_restart.out", "w") as fo, open(f"test_{TEST_NAME}_restart.err", "w") as fe:
            res = subprocess.run(mpi_launch_command(RANKS, THREADS) + ["./GIZMO", f"{TEST_NAME}.params", "1"], stdout=fo, stderr=fe)
        log_restart = Path(f"test_{TEST_NAME}_restart.out").read_text()
    finally:
        os.chdir(cwd)
    assert res.returncode == 0, "restart run failed"
    a, b = _final(straight), _final(out)
    assert abs(a["t"] - 0.002) < 1e-9 and abs(b["t"] - 0.002) < 1e-9
    for pt in ("PartType0", "PartType4"):
        assert np.array_equal(a[pt]["Masses"], b[pt]["Masses"])
    m0 = a["PartType0"]["Masses"].sum() + a["PartType4"]["Masses"].sum()
    with h5py.File(TEST_DIR / f"{TEST_NAME}_ics.hdf5", "r") as f:
        mi = sum(np.array(f[p]["Masses"]).sum() for p in ("PartType0", "PartType4"))
    assert abs(m0 / mi - 1) < 1e-12, "mass not conserved"
    dx = np.abs(a["PartType4"]["Coordinates"] - b["PartType4"]["Coordinates"]).max()
    dv = np.abs(a["PartType4"]["Velocities"] - b["PartType4"]["Velocities"]).max()
    dxg = np.abs(a["PartType0"]["Coordinates"] - b["PartType0"]["Coordinates"]).max()
    print(f"restart vs straight: stars max|dx| {dx:.3e} pc, max|dv| {dv:.3e} km/s; gas max|dx| {dxg:.3e} pc")
    assert dx < 1e-6 and dv < 1e-3 and dxg < 1e-4
    stats = [l for l in (log_straight + log_restart).splitlines() if l.startswith("KETJU STATS")]
    assert stats, "no KETJU STATS line (no region was integrated)"
    worst = max(float(re.search(r"max\|dE/E\|=(\S+)", l).group(1)) for l in stats)
    print(f"{len(stats)} KETJU STATS lines, worst MSTAR |dE/E| {worst:.2e}")
    assert worst < 1e-6
