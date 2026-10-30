"""Lattice of 1 Msun gas cells at n_H = 10 cm^-3, T = 100 K, Z = 0.014, at rest, with N white dwarfs of drawn mass 3 Msun, 1 Gyr
old, whose AGB death is already done (event bit 2), at random positions (units pc, Msun, km/s). Written as a snapshot (start flag 2)."""
import numpy as np
import h5py

PC_KMS_IN_YR = 3.085678e13 / 3.155e7  # time unit [yr], with the code's year (constants.h)


def make_single_stars_type_ia_rate_ics(fname, tables_file, n=24, n_H=10.0, T=100.0, z=0.014, n_wd=3000, m_drawn=3.0, age_yr=1.0e9, seed=1):
    rho = n_H * 1.6726e-24 / 0.7 * 3.0857e18 ** 3 / 1.989e33
    dx = (1.0 / rho) ** (1.0 / 3.0)
    L = n * dx
    g = (np.arange(n) + 0.5) * dx
    pos = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    N = len(pos)
    u = 1.5 * 1.380649e-16 * T / (1.27 * 1.6726e-24) / 1e10
    with h5py.File(tables_file, "r") as F:
        Zt, Mt = F["Z"][()], F["M_init"][()]
        iz, im = int(np.argmin(np.abs(Zt - z))), int(np.argmin(np.abs(Mt - m_drawn)))
        m_wd = F["remnant_mass"][()].reshape(len(Zt), len(Mt))[iz, im]
    xs = np.random.default_rng(seed).uniform(0, L, (n_wd, 3))
    with h5py.File(fname, "w") as F:
        h = F.create_group("Header")
        npart = np.array([N, 0, 0, 0, n_wd, 0])
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
        s["Coordinates"] = xs; s["Velocities"] = np.zeros((n_wd, 3)); s["ParticleIDs"] = np.arange(N + 1, N + n_wd + 1, dtype=np.uint32)
        s["Masses"] = np.full(n_wd, m_wd); s["StellarFormationTime"] = np.full(n_wd, -age_yr / PC_KMS_IN_YR)
        s["Metallicity"] = np.full((n_wd, 1), z); s["SingleStarDrawnMass"] = np.full(n_wd, m_drawn)
        s["SingleStarBirthMetallicity"] = np.full(n_wd, z); s["SingleStarEventBits"] = np.full(n_wd, 2.0)
    return L
