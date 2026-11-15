"""Few-body ICs of Type-4 stars (kpc, km/s, 1e10 Msun) and their Newtonian reference orbits (DOP853, rtol 1e-13);
shared by the STARS_HERMITE_DIRECT_GRAVITY and STARS_INTEGRATED_BY_RUMI tests."""
import h5py
import numpy as np
from scipy.integrate import solve_ivp

G, MSUN, AU, BOX = 6.672e-8 * 1.989e43 / (3.085678e21 * 1e10), 1e-10, 4.848137e-9, 0.3  # G as GIZMO sets it for these units
CENTRE = np.array([0.15, 0.15, 0.15])


def kepler_pair(m1, m2, a, e):
    """relative orbit at apocentre along x, velocity along y; returns positions and velocities about the pair's centre of mass"""
    ra = a * (1 + e); va = np.sqrt(G * (m1 + m2) * (1 - e) / (a * (1 + e)))
    x = np.array([[m2 / (m1 + m2) * ra, 0, 0], [-m1 / (m1 + m2) * ra, 0, 0]])
    v = np.array([[0, m2 / (m1 + m2) * va, 0], [0, -m1 / (m1 + m2) * va, 0]])
    return x, v


def binary():
    """1 + 1 Msun, a = 1000 AU, e = 0.9"""
    x, v = kepler_pair(MSUN, MSUN, 1000 * AU, 0.9)
    return CENTRE + x, v, np.full(2, MSUN)


def triple():
    """hierarchical: inner 1 + 1 Msun, a = 100 AU, e = 0.5; outer 1 Msun on a = 1000 AU, e = 0.3, inclined 30 degrees"""
    xi, vi = kepler_pair(MSUN, MSUN, 100 * AU, 0.5)
    xo, vo = kepler_pair(2 * MSUN, MSUN, 1000 * AU, 0.3)
    c, s = np.cos(np.pi / 6), np.sin(np.pi / 6); rot = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    xo, vo = xo @ rot.T, vo @ rot.T
    x = np.vstack([xi + xo[0], xo[1]]); v = np.vstack([vi + vo[0], vo[1]])
    return CENTRE + x, v, np.full(3, MSUN)


def write(name, x, v, m):
    n = len(m)
    with h5py.File(name, "w") as f:
        h = f.create_group("Header")
        h.attrs["NumPart_ThisFile"] = [0, 0, 0, 0, n, 0]; h.attrs["NumPart_Total"] = [0, 0, 0, 0, n, 0]
        h.attrs["NumPart_Total_HighWord"] = [0] * 6; h.attrs["MassTable"] = [0.0] * 6
        h.attrs["Time"] = 0.0; h.attrs["Redshift"] = 0.0; h.attrs["BoxSize"] = BOX; h.attrs["NumFilesPerSnapshot"] = 1
        h.attrs["Omega0"] = 0.0; h.attrs["OmegaLambda"] = 0.0; h.attrs["HubbleParam"] = 1.0
        for k in ("Flag_Sfr", "Flag_Cooling", "Flag_StellarAge", "Flag_Metals", "Flag_Feedback"): h.attrs[k] = 0
        g = f.create_group("PartType4")
        g["Coordinates"] = x; g["Velocities"] = v; g["Masses"] = m; g["ParticleIDs"] = np.arange(1, n + 1, dtype=np.uint64)


def energy(x, v, m):
    e = 0.5 * np.sum(m * np.sum(v * v, axis=1))
    for i in range(len(m)):
        for j in range(i):
            e -= G * m[i] * m[j] / np.linalg.norm(x[i] - x[j])
    return e


def reference(x, v, m, t):
    """Newtonian point masses from (x, v) over time t"""
    n = len(m)

    def rhs(_, y):
        p = y[:3 * n].reshape(n, 3); d = p[None, :, :] - p[:, None, :]
        r3 = np.linalg.norm(d, axis=2) ** 3; np.fill_diagonal(r3, np.inf)
        a = G * np.sum(m[None, :, None] * d / r3[:, :, None], axis=1)
        return np.concatenate([y[3 * n:], a.ravel()])

    c = np.sum(m[:, None] * x, axis=0) / np.sum(m); vc = np.sum(m[:, None] * v, axis=0) / np.sum(m)
    xs, vs = x - c, v - vc  # about the centre of mass, tolerances scaled to the orbit
    atol = np.concatenate([np.full(3 * n, 1e-13 * np.abs(xs).max()), np.full(3 * n, 1e-13 * np.abs(vs).max())])
    y = solve_ivp(rhs, (0, t), np.concatenate([xs.ravel(), vs.ravel()]), method="DOP853", rtol=1e-13, atol=atol).y[:, -1]
    return y[:3 * n].reshape(n, 3) + c + vc * t, y[3 * n:].reshape(n, 3) + vc


def elements(x, v, m):
    dr, dv = x[0] - x[1], v[0] - v[1]; r = np.linalg.norm(dr); mu = G * (m[0] + m[1])
    a = 1 / (2 / r - dv @ dv / mu); e = np.linalg.norm(np.cross(dv, np.cross(dr, dv)) / mu - dr / r)
    return a, e


def read(snapshot):
    with h5py.File(snapshot, "r") as f:
        g = f["PartType4"]; o = np.argsort(g["ParticleIDs"][:])
        xk, vk = ("HermiteSyncCoordinates", "HermiteSyncVelocities") if "HermiteSyncVelocities" in g else ("Coordinates", "Velocities")
        return g[xk][:][o], g[vk][:][o], g["Masses"][:][o], f["Header"].attrs["Time"]


if __name__ == "__main__":
    write("binary_ics.hdf5", *binary())
    write("triple_ics.hdf5", *triple())
