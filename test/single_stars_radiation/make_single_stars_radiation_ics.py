"""Lattice of 1 Msun gas cells at n_H = 100 cm^-3, T = 100 K, Z = 0.014, at rest, with two single stars far apart
(units pc, Msun, km/s): a 35 Msun star 1 Myr old (bright, ionising), a 20 Msun star whose supernova is done (event
bit 1; a remnant, which emits nothing) and a 35 Msun star 1000 yr old (before its Kelvin-Helmholtz time: not yet ionising). Written as a snapshot (start flag 2)."""
import numpy as np
import h5py

PC_KMS_IN_YR = 3.085678e13 / 3.155e7  # time unit [yr], with the code's year (constants.h)
STAR_AGE_YR = 1.0e6


def make_single_stars_radiation_ics(fname, n=20, n_H=100.0, T=100.0, z=0.014):
    rho = n_H * 1.6726e-24 / 0.7 * 3.0857e18 ** 3 / 1.989e33
    dx = (1.0 / rho) ** (1.0 / 3.0)
    L = n * dx
    g = (np.arange(n) + 0.5) * dx
    pos = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    N = len(pos)
    u = 1.5 * 1.380649e-16 * T / (1.27 * 1.6726e-24) / 1e10
    stars = [(35.0, 35.0, STAR_AGE_YR, (0.25, 0.25, 0.25), 0), (20.0, 1.5, 2.0e7, (0.75, 0.75, 0.75), 1), (35.0, 35.0, 1.0e3, (0.75, 0.25, 0.25), 0)]
    ns = len(stars)
    with h5py.File(fname, "w") as F:
        h = F.create_group("Header")
        npart = np.array([N, 0, 0, 0, ns, 0])
        h.attrs["NumPart_ThisFile"] = npart; h.attrs["NumPart_Total"] = npart
        h.attrs["NumPart_Total_HighWord"] = np.zeros(6, dtype=np.uint32); h.attrs["MassTable"] = np.zeros(6)
        h.attrs["Time"] = 0.0; h.attrs["Redshift"] = 0.0; h.attrs["BoxSize"] = L; h.attrs["NumFilesPerSnapshot"] = 1
        h.attrs["Omega0"] = h.attrs["OmegaLambda"] = h.attrs["HubbleParam"] = 0.0
        h.attrs["Flag_Sfr"] = 1; h.attrs["Flag_Cooling"] = 0; h.attrs["Flag_StellarAge"] = 1; h.attrs["Flag_Metals"] = 1
        h.attrs["Flag_Feedback"] = 1; h.attrs["Flag_DoublePrecision"] = 1
        p = F.create_group("PartType0")
        p["Coordinates"] = pos; p["Velocities"] = np.zeros((N, 3)); p["ParticleIDs"] = np.arange(1, N + 1, dtype=np.uint32)
        p["Masses"] = np.ones(N); p["InternalEnergy"] = np.full(N, u); p["Metallicity"] = np.full((N, 1), z)
        p["SmoothingLength"] = np.full(N, 2.0 * dx)
        s = F.create_group("PartType4")
        s["Coordinates"] = np.array([np.asarray(x) * L + 0.37 * dx for _, _, _, x, _ in stars])
        s["Velocities"] = np.zeros((ns, 3)); s["ParticleIDs"] = np.arange(N + 1, N + ns + 1, dtype=np.uint32)
        s["Masses"] = np.array([m for _, m, _, _, _ in stars]); s["StellarFormationTime"] = -np.array([a for _, _, a, _, _ in stars]) / PC_KMS_IN_YR
        s["Metallicity"] = np.full((ns, 1), z); s["SingleStarDrawnMass"] = np.array([md for md, _, _, _, _ in stars])
        s["SingleStarBirthMetallicity"] = np.full(ns, z); s["SingleStarEventBits"] = np.array([b for _, _, _, _, b in stars], dtype=float)
    return L
