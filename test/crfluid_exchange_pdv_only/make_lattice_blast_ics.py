"""Lattice blast: periodic cube of 16^3 gas cells at rest, density 1, cold, with the 8 central cells hot: a Sedov-type blast whose
shock makes cells wake neighbours on shorter timesteps. Total blast energy 1 in code units (kpc, 1e10 Msun, km/s)."""
import numpy as np
import h5py


def make_lattice_blast_ics(fname, n=16, u_cold=1.0, e_blast=1.0):
    dx = 1.0 / n
    g = (np.arange(n) + 0.5) * dx
    pos = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    N = len(pos)
    u = np.full(N, u_cold)
    hot = np.all(np.abs(pos - 0.5) < 0.6 * dx, axis=1)
    u[hot] = e_blast / (hot.sum() / N)  # unit total mass: m = 1/N per cell, E = sum m u
    with h5py.File(fname, "w") as F:
        h = F.create_group("Header")
        npart = np.array([N, 0, 0, 0, 0, 0])
        h.attrs["NumPart_ThisFile"] = npart; h.attrs["NumPart_Total"] = npart
        h.attrs["NumPart_Total_HighWord"] = np.zeros(6, dtype=np.uint32); h.attrs["MassTable"] = np.zeros(6)
        h.attrs["Time"] = 0.0; h.attrs["Redshift"] = 0.0; h.attrs["BoxSize"] = 1.0; h.attrs["NumFilesPerSnapshot"] = 1
        h.attrs["Omega0"] = h.attrs["OmegaLambda"] = h.attrs["HubbleParam"] = 0.0
        h.attrs["Flag_Sfr"] = 0; h.attrs["Flag_Cooling"] = 0; h.attrs["Flag_StellarAge"] = 0; h.attrs["Flag_Metals"] = 0
        h.attrs["Flag_Feedback"] = 0; h.attrs["Flag_DoublePrecision"] = 1
        p = F.create_group("PartType0")
        p["Coordinates"] = pos; p["Velocities"] = np.zeros((N, 3)); p["ParticleIDs"] = np.arange(1, N + 1, dtype=np.uint32)
        p["Masses"] = np.full(N, 1.0 / N); p["InternalEnergy"] = u
    return N
