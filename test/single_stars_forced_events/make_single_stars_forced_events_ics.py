"""Lattice of 1 Msun gas cells at n_H = 10 cm^-3, T = 100 K, Z = 0.014, at rest (units pc, Msun, km/s), with two single
stars of drawn mass 20 Msun and 1 Myr old (far from each other, far from their table death at ~10 Myr), IDs 1 and 2; the
gas IDs follow. Written as a snapshot (start flag 2)."""
import numpy as np
import h5py

PC_KMS_IN_YR = 3.085678e13 / 3.155e7  # time unit [yr]


def make_single_stars_forced_events_ics(fname, n=24, n_H=10.0, T=100.0, z=0.014, M=20.0, age_yr=1.0e6):
    rho = n_H * 1.6726e-24 / 0.7 * 3.0857e18 ** 3 / 1.989e33
    dx = (1.0 / rho) ** (1.0 / 3.0)
    L = n * dx
    g = (np.arange(n) + 0.5) * dx
    pos = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    N = len(pos)
    u = 1.5 * 1.380649e-16 * T / (1.27 * 1.6726e-24) / 1e10
    spos = np.array([[0.25, 0.25, 0.25], [0.75, 0.75, 0.75]]) * L + 0.37 * dx
    with h5py.File(fname, "w") as F:
        h = F.create_group("Header")
        npart = np.array([N, 0, 0, 0, 2, 0])
        h.attrs["NumPart_ThisFile"] = npart; h.attrs["NumPart_Total"] = npart
        h.attrs["NumPart_Total_HighWord"] = np.zeros(6, dtype=np.uint32); h.attrs["MassTable"] = np.zeros(6)
        h.attrs["Time"] = 0.0; h.attrs["Redshift"] = 0.0; h.attrs["BoxSize"] = L; h.attrs["NumFilesPerSnapshot"] = 1
        h.attrs["Omega0"] = h.attrs["OmegaLambda"] = h.attrs["HubbleParam"] = 0.0
        h.attrs["Flag_Sfr"] = 1; h.attrs["Flag_Cooling"] = 0; h.attrs["Flag_StellarAge"] = 1; h.attrs["Flag_Metals"] = 1
        h.attrs["Flag_Feedback"] = 1; h.attrs["Flag_DoublePrecision"] = 1
        p = F.create_group("PartType0")
        p["Coordinates"] = pos; p["Velocities"] = np.zeros((N, 3)); p["ParticleIDs"] = np.arange(3, N + 3, dtype=np.uint32)
        p["Masses"] = np.ones(N); p["InternalEnergy"] = np.full(N, u); p["Metallicity"] = np.full((N, 1), z)
        p["SmoothingLength"] = np.full(N, 2.0 * dx)
        s = F.create_group("PartType4")
        s["Coordinates"] = spos; s["Velocities"] = np.zeros((2, 3)); s["ParticleIDs"] = np.array([1, 2], dtype=np.uint32)
        s["Masses"] = np.full(2, M); s["StellarFormationTime"] = np.full(2, -age_yr / PC_KMS_IN_YR)
        s["Metallicity"] = np.full((2, 1), z); s["SingleStarDrawnMass"] = np.full(2, M)
        s["SingleStarBirthMetallicity"] = np.full(2, z); s["SingleStarEventBits"] = np.zeros(2)
    return L
