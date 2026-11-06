"""static_plummer_test_potential test: TEST_GRAVITY_STATIC_PLUMMER adds the fixed Plummer potential phi = -G M / sqrt(r^2 + a^2)
(M = 1e3 Msun, a = 0.5 pc) at the origin. Collisionless test particles (self-gravity off) on a circular and two eccentric orbits
keep their energy and angular momentum and follow the reference orbit. See README.md."""
import glob
import h5py
import numpy as np
from scipy.integrate import solve_ivp
from gizmo.test import build_and_run_test, clean_test_outputs, variant_output_dir

TEST_NAME = "static_plummer_test_potential"
IC_FILE = f"test/{TEST_NAME}/{TEST_NAME}_ics.hdf5"
G = 4.30091e-3  # pc (km/s)^2 / Msun
GM, A = G * 1000.0, 0.5
BOX = 100.0
C = np.zeros(3)  # the potential is centred on the origin
R0 = np.array([0.3, 1.0, 2.0])
V0 = np.array([1.0, np.sqrt(GM * 1.0 / (1.0 + A * A) ** 1.5), 0.4])  # eccentric, circular, eccentric (km/s, tangential)
T_END = 10.0


def write_ics():
    pos = C + np.stack([R0, np.zeros(3), np.zeros(3)], 1)
    vel = np.stack([np.zeros(3), V0, np.zeros(3)], 1)
    with h5py.File(IC_FILE, "w") as F:
        h = F.create_group("Header")
        h.attrs["NumPart_ThisFile"] = [0, 3, 0, 0, 0, 0]; h.attrs["NumPart_Total"] = [0, 3, 0, 0, 0, 0]
        h.attrs["NumPart_Total_HighWord"] = [0, 0, 0, 0, 0, 0]; h.attrs["MassTable"] = [0, 0, 0, 0, 0, 0]
        h.attrs["BoxSize"] = BOX; h.attrs["Time"] = 0.0; h.attrs["Redshift"] = 0.0; h.attrs["NumFilesPerSnapshot"] = 1
        g = F.create_group("PartType1")
        g["Coordinates"] = pos; g["Velocities"] = vel; g["Masses"] = np.full(3, 1e-6); g["ParticleIDs"] = np.arange(1, 4, dtype=np.uint32)


def reference(i):
    def rhs(t, y):
        r2 = y[0] ** 2 + y[1] ** 2 + y[2] ** 2
        return [y[3], y[4], y[5], *(-GM * y[:3] / (r2 + A * A) ** 1.5)]
    return solve_ivp(rhs, (0, T_END), [R0[i], 0, 0, 0, V0[i], 0], rtol=1e-11, atol=1e-12).y[:, -1]


def test_orbits_in_fixed_plummer_potential():
    """bar (written before the run): per particle the specific energy changes by less than 1e-3 of |E| and the angular momentum
    by less than 1e-3 of its value; the circular orbit keeps its radius to 1%; the final position is within 0.02 pc of the
    reference orbit (T = 10, about three circular periods)"""
    write_ics()
    clean_test_outputs(TEST_NAME, ())
    build_and_run_test(TEST_NAME, 2, 0, ())
    snap = sorted(glob.glob(variant_output_dir(TEST_NAME, ()) + "/snapshot_*.hdf5"))[-1]
    with h5py.File(snap, "r") as F:
        g = F["PartType1"]; o = np.argsort(g["ParticleIDs"][()])
        pos, vel = g["Coordinates"][()][o] - C, g["Velocities"][()][o]
        t = float(F["Header"].attrs["Time"])
    assert t >= T_END - 1e-9
    r = np.linalg.norm(pos, axis=1)
    E0 = 0.5 * V0 ** 2 - GM / np.sqrt(R0 ** 2 + A * A)
    E = 0.5 * (vel ** 2).sum(1) - GM / np.sqrt(r ** 2 + A * A)
    L0, L = R0 * V0, np.linalg.norm(np.cross(pos, vel), axis=1)
    print("dE/|E|", np.abs(E - E0) / np.abs(E0), "dL/L", np.abs(L - L0) / L0, "r circ", r[1])
    assert np.all(np.abs(E - E0) < 1e-3 * np.abs(E0)) and np.all(np.abs(L - L0) < 1e-3 * L0)
    assert abs(r[1] - 1.0) < 0.01
    for i in range(3):
        assert np.linalg.norm(pos[i] - reference(i)[:3]) < 0.02, i
