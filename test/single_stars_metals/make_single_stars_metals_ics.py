"""Lattice of 1 Msun gas cells at n_H = 10 cm^-3, T = 100 K, at rest, given metal slots per cell, with single stars at
rest (units pc, Msun, km/s), written as a snapshot (start flag 2 reads stellar ages, metals and star records; a fresh
start, flag 0, sets the gas metals from InitMetallicity)."""
import numpy as np
import h5py

PC_KMS_IN_YR = 3.085678e13 / 3.155e7  # time unit [yr], with the code's year (constants.h)


def make_single_stars_metals_ics(fname, metals, stars=(), n=16, n_H=10.0, T=100.0, z=0.014, gas_fields=None):
    """metals: the gas (and star) metal slots; stars: (drawn mass, mass, age [yr], position in box units, event bits[,
    log10 table age [yr] up to which the winds count as returned]); gas_fields: more gas datasets, one row per cell"""
    rho = n_H * 1.6726e-24 / 0.7 * 3.0857e18 ** 3 / 1.989e33
    dx = (1.0 / rho) ** (1.0 / 3.0)
    L = n * dx
    g = (np.arange(n) + 0.5) * dx
    pos = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    N, ns, nm = len(pos), len(stars), len(metals)
    u = 1.5 * 1.380649e-16 * T / (1.27 * 1.6726e-24) / 1e10
    with h5py.File(fname, "w") as F:
        h = F.create_group("Header")
        npart = np.array([N, 0, 0, 0, ns, 0])
        h.attrs["NumPart_ThisFile"] = npart; h.attrs["NumPart_Total"] = npart
        h.attrs["NumPart_Total_HighWord"] = np.zeros(6, dtype=np.uint32); h.attrs["MassTable"] = np.zeros(6)
        h.attrs["Time"] = 0.0; h.attrs["Redshift"] = 0.0; h.attrs["BoxSize"] = L; h.attrs["NumFilesPerSnapshot"] = 1
        h.attrs["Omega0"] = h.attrs["OmegaLambda"] = h.attrs["HubbleParam"] = 0.0
        h.attrs["Flag_Sfr"] = 1; h.attrs["Flag_Cooling"] = 1; h.attrs["Flag_StellarAge"] = 1; h.attrs["Flag_Metals"] = nm
        h.attrs["Flag_Feedback"] = 1; h.attrs["Flag_DoublePrecision"] = 1
        p = F.create_group("PartType0")
        p["Coordinates"] = pos; p["Velocities"] = np.zeros((N, 3)); p["ParticleIDs"] = np.arange(1, N + 1, dtype=np.uint32)
        p["Masses"] = np.ones(N); p["InternalEnergy"] = np.full(N, u); p["Metallicity"] = np.tile(metals, (N, 1))
        p["SmoothingLength"] = np.full(N, 2.0 * dx)
        for k, v in (gas_fields or {}).items():
            p[k] = np.tile(v, (N, 1))
        if ns:
            s = F.create_group("PartType4")
            s["Coordinates"] = np.array([np.asarray(x) * L + 0.37 * dx for _, _, _, x, *_ in stars])
            s["Velocities"] = np.zeros((ns, 3)); s["ParticleIDs"] = np.arange(N + 1, N + ns + 1, dtype=np.uint32)
            s["Masses"] = np.array([m for _, m, *_ in stars]); s["StellarFormationTime"] = -np.array([a for _, _, a, *_ in stars]) / PC_KMS_IN_YR
            s["Metallicity"] = np.tile(metals, (ns, 1)); s["SingleStarDrawnMass"] = np.array([md for md, *_ in stars])
            s["SingleStarBirthMetallicity"] = np.full(ns, z); s["SingleStarEventBits"] = np.array([x[4] for x in stars], dtype=float)
            s["SingleStarLastWindLogAge"] = np.array([x[5] if len(x) > 5 else 0.0 for x in stars])
    return L
