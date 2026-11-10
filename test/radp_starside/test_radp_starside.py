"""Star-side radiation pressure kick (single stars, production cooling and chemistry: SGChem net 5, CIE cooling, CO shielding; no winds,
photoionisation or tree-column direct term, so the kick is the only radiation momentum): one 40 Msun star in uniform dusty gas
(n_H = 300, solar Z, 0.8 Msun cells, no gravity). Past the pre-main-sequence the run books momentum in the radiation pressure channel
(FeedbackBudget, channel 3); the gas must receive it. Bar, set before running: radial gas momentum about the star over the booked
total within 0.8-1.1. Cooling does not change momentum directly, so the ratio stays below 1 only by gas pressure working against the kick
(upper limit 1.1 for noise, lower 0.8 from the adiabatic run, 0.895). Before the fix the kick was booked and never delivered (about -0.06)."""

import glob
import os
import subprocess
import sys

import h5py
import numpy as np

from gizmo.test import build_gizmo_for_test, mpi_launch_command

TEST_NAME = "radp_starside"
TEST_DIR = f"test/{TEST_NAME}"
MSUN_KMS_CGS = 1.989e33 * 1e5
BAR = (0.8, 1.1)


def test_radp_starside():
    subprocess.run([sys.executable, f"make_{TEST_NAME}_ics.py", f"{TEST_NAME}_ics.hdf5"], cwd=TEST_DIR, check=True)
    build_gizmo_for_test(TEST_NAME, 1)
    repo = os.getcwd(); os.chdir(TEST_DIR)
    try:
        if not os.path.islink("gizmo"):
            os.symlink(repo, "gizmo")  # the parameter file reads the stellar tables under gizmo/
        subprocess.run(["rm", "-rf", "output"]); os.makedirs("output")
        res = subprocess.run(mpi_launch_command(2, 1) + ["./GIZMO", f"{TEST_NAME}.params"], stdout=open("run.out", "w"), stderr=subprocess.STDOUT)
        assert res.returncode == 0, "run failed: " + open("run.out").read()[-400:]
        booked = sum(float(l.split()[6]) for l in open("output/FeedbackBudget.txt") if not l.startswith("#") and l.split()[1] == "3") / MSUN_KMS_CGS
        with h5py.File(sorted(glob.glob("output/snapshot_*.hdf5"))[-1], "r") as f:
            g, s, L = f["PartType0"], f["PartType4"], f["Header"].attrs["BoxSize"]
            d = g["Coordinates"][:] - s["Coordinates"][0]; d -= L * np.round(d / L)
            m = g["Masses"][:].astype(float) * 1e10
            delivered = (m * (g["Velocities"][:] * d).sum(1) / np.maximum(np.linalg.norm(d, axis=1), 1e-12)).sum()
    finally:
        os.chdir(repo)
    print(f"booked {booked:.3f} Msun km/s, delivered radial {delivered:.3f}, ratio {delivered / booked:.3f}")
    assert booked > 10
    assert BAR[0] < delivered / booked < BAR[1]
