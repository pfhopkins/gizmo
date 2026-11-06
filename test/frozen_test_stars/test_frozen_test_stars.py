"""frozen_test_stars test: with TEST_FROZEN_STARS_ZERO_VELOCITY_AND_ACCELERATION star particles keep their positions
whatever their initial velocity and the gravity of their neighbours; without it two 1e3 Msun stars 1 pc apart fall together
in 0.5 time units (pc / km/s). See README.md."""
import glob
import h5py
import numpy as np
from gizmo.test import build_and_run_test, clean_test_outputs, variant_output_dir

TEST_NAME = "frozen_test_stars"
IC_FILE = f"test/{TEST_NAME}/{TEST_NAME}_ics.hdf5"
POS0 = np.array([[49.5, 50.0, 50.0], [50.5, 50.0, 50.0]])


def write_ics():
    with h5py.File(IC_FILE, "w") as F:
        h = F.create_group("Header")
        h.attrs["NumPart_ThisFile"] = [0, 0, 0, 0, 2, 0]; h.attrs["NumPart_Total"] = [0, 0, 0, 0, 2, 0]
        h.attrs["NumPart_Total_HighWord"] = [0, 0, 0, 0, 0, 0]; h.attrs["MassTable"] = [0, 0, 0, 0, 0, 0]
        h.attrs["BoxSize"] = 100.0; h.attrs["Time"] = 0.0; h.attrs["Redshift"] = 0.0; h.attrs["NumFilesPerSnapshot"] = 1
        g = F.create_group("PartType4")
        g["Coordinates"] = POS0; g["Velocities"] = np.array([[0.0, 3.0, 0.0], [0.0, -3.0, 0.0]])
        g["Masses"] = np.full(2, 1000.0); g["ParticleIDs"] = np.array([1, 2], dtype=np.uint32)


def test_stars_do_not_move():
    """bar (written before the run): both stars end within 1e-6 pc of their start, with zero velocity"""
    write_ics()
    clean_test_outputs(TEST_NAME, ())
    build_and_run_test(TEST_NAME, 2, 0, ())
    snap = sorted(glob.glob(variant_output_dir(TEST_NAME, ()) + "/snapshot_*.hdf5"))[-1]
    with h5py.File(snap, "r") as F:
        g = F["PartType4"]; o = np.argsort(g["ParticleIDs"][()])
        pos, vel = g["Coordinates"][()][o], g["Velocities"][()][o]
        t = float(F["Header"].attrs["Time"])
    assert t >= 0.5 - 1e-9
    print("max |dx| [pc]", np.abs(pos - POS0).max(), "max |v|", np.abs(vel).max())
    assert np.abs(pos - POS0).max() < 1e-6 and np.abs(vel).max() == 0
