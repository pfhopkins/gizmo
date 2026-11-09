"""One star near the centre of a uniform periodic lattice, gas at rest, n_H = 0.1 cm^-3, T = 100 K. v_star [km/s] along x. Returns the box size."""
import h5py
import numpy as np

N_SIDE, Z = 20, 0.014
MYR_IN_CODE = 1.0 / 0.9778  # code time = pc / (km/s) = 0.9778 Myr
N_METALS = 11 + 16 + 1       # upper bound on Metallicity columns (surplus columns are ignored)


def make_ics(path, m_star, m_cell, age_myr, v_star=0.0):
    rho = 0.1 * 1.6726e-24 / 0.7 * 3.0857e18**3 / 1.989e33  # Msun/pc^3
    dx = (m_cell / rho) ** (1 / 3)
    L = N_SIDE * dx
    g = (np.arange(N_SIDE) + 0.5) * dx
    pos = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    n = len(pos)
    u = 1.5 * 1.380649e-16 * 100.0 / (1.27 * 1.6726e-24) / 1e10  # (km/s)^2
    with h5py.File(path, "w") as F:
        h = F.create_group("Header")
        npart = np.array([n, 0, 0, 0, 1, 0])
        h.attrs["NumPart_ThisFile"] = npart
        h.attrs["NumPart_Total"] = npart
        h.attrs["NumPart_Total_HighWord"] = np.zeros(6, dtype=np.uint32)
        h.attrs["MassTable"] = np.zeros(6)
        h.attrs["Time"] = 0.0
        h.attrs["Redshift"] = 0.0
        h.attrs["BoxSize"] = L
        h.attrs["NumFilesPerSnapshot"] = 1
        h.attrs["Omega0"] = h.attrs["OmegaLambda"] = h.attrs["HubbleParam"] = 0.0
        for k in ("Flag_Sfr", "Flag_StellarAge", "Flag_Feedback", "Flag_DoublePrecision"):
            h.attrs[k] = 1
        h.attrs["Flag_Metals"] = N_METALS
        h.attrs["Flag_Cooling"] = 0
        p = F.create_group("PartType0")
        met = np.zeros((n, N_METALS))
        met[:, 0] = Z
        p["Coordinates"] = pos
        p["Velocities"] = np.zeros((n, 3))
        p["ParticleIDs"] = np.arange(1, n + 1, dtype=np.uint64)
        p["Masses"] = np.full(n, m_cell)
        p["InternalEnergy"] = np.full(n, u)
        p["Metallicity"] = met
        p["SmoothingLength"] = np.full(n, 2 * dx)
        s = F.create_group("PartType4")
        s["Coordinates"] = np.array([[0.5 * L + 0.37 * dx] * 3])
        s["Velocities"] = np.array([[v_star, 0.0, 0.0]])
        s["ParticleIDs"] = np.array([n + 1], dtype=np.uint64)
        s["Masses"] = np.array([m_star])
        s["StellarFormationTime"] = np.array([-age_myr * MYR_IN_CODE])
        s["Metallicity"] = met[:1]
        s["SmoothingLength"] = np.array([2 * dx])
    return L
