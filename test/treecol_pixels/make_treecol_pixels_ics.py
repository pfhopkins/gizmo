"""Initial conditions for the treecol_pixels tests, written as a snapshot (start flag 2 reads stellar ages and star
records). Units pc, Msun, km/s. Gas: given positions, 1 Msun cells, T = 100 K, solar metals (11 slots); stars: given
drawn masses, ages [yr] and positions, at rest. Optional per-cell metallicity z_gas, SGChem abundances tracabund and cosmic-ray energy e_cr."""
import numpy as np
import h5py

YR_PER_TIME_UNIT = 3.085678e13 / 3.155e7
SOLAR = np.array([0.0142, 0.2703, 2.53e-3, 7.41e-4, 6.13e-3, 1.34e-3, 7.57e-4, 7.12e-4, 3.31e-4, 6.87e-5, 1.38e-3])


def lattice(n, L, offset=0.0):
    g = (np.arange(n) + 0.5) * (L / n) + offset
    return np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)


def make_treecol_pixels_ics(fname, L, gas_pos, stars, gas_mass=1.0, z=0.0142, T=100.0, hsml=1.0, z_gas=None, tracabund=None, e_cr=None):
    """stars: list of (M_drawn [Msun], age [yr], position)"""
    N, ns = len(gas_pos), len(stars)
    u = 1.5 * 1.380649e-16 * T / (1.27 * 1.6726e-24) / 1e10
    zg = np.full(N, z) if z_gas is None else np.asarray(z_gas)
    Zg = SOLAR[None, :] * (zg / SOLAR[0])[:, None]; Zg[:, 1] = SOLAR[1]
    with h5py.File(fname, "w") as F:
        h = F.create_group("Header")
        npart = np.array([N, 0, 0, 0, ns, 0])
        h.attrs["NumPart_ThisFile"] = npart; h.attrs["NumPart_Total"] = npart
        h.attrs["NumPart_Total_HighWord"] = np.zeros(6, dtype=np.uint32); h.attrs["MassTable"] = np.zeros(6)
        h.attrs["Time"] = 0.0; h.attrs["Redshift"] = 0.0; h.attrs["BoxSize"] = L; h.attrs["NumFilesPerSnapshot"] = 1
        h.attrs["Omega0"] = h.attrs["OmegaLambda"] = h.attrs["HubbleParam"] = 0.0
        h.attrs["Flag_Sfr"] = 1; h.attrs["Flag_Cooling"] = 1; h.attrs["Flag_StellarAge"] = 1; h.attrs["Flag_Metals"] = 11
        h.attrs["Flag_Feedback"] = 1; h.attrs["Flag_DoublePrecision"] = 1
        p = F.create_group("PartType0")
        p["Coordinates"] = np.asarray(gas_pos, dtype=np.float64); p["Velocities"] = np.zeros((N, 3))
        p["ParticleIDs"] = np.arange(1, N + 1, dtype=np.uint32)
        p["Masses"] = np.full(N, float(gas_mass)); p["InternalEnergy"] = np.full(N, u); p["Metallicity"] = Zg
        p["SmoothingLength"] = np.full(N, hsml)
        if tracabund is not None:
            p["SGChemTracAbund"] = np.asarray(tracabund, dtype=np.float64)
        if e_cr is not None:  # cosmic-ray energy per cell [code units]
            p["CosmicRayEnergy"] = np.asarray(e_cr, dtype=np.float64)
        s = F.create_group("PartType4")
        s["Coordinates"] = np.array([np.asarray(x, dtype=np.float64) for _, _, x in stars]).reshape(ns, 3)
        s["Velocities"] = np.zeros((ns, 3)); s["ParticleIDs"] = np.arange(N + 1, N + ns + 1, dtype=np.uint32)
        s["Masses"] = np.array([m for m, _, _ in stars], dtype=np.float64)
        s["StellarFormationTime"] = -np.array([a for _, a, _ in stars]) / YR_PER_TIME_UNIT
        Zs = np.tile(SOLAR * z / SOLAR[0], (ns, 1)); Zs[:, 1] = SOLAR[1]
        s["Metallicity"] = Zs; s["SingleStarDrawnMass"] = s["Masses"][()]; s["SingleStarBirthMetallicity"] = np.full(ns, z)
