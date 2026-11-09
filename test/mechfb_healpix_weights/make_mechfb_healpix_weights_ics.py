"""One star in a strong density gradient along z: 1 Msun cells, stacked lattices, n_H from 1 to 1000 cm^-3 with scale height 0.5 pc,
100 cm^-3 at the star. Units pc, Msun, km/s. A snapshot (start flag 2). Event 'sn': 20 Msun star past its table lifetime; 'wind': 35 Msun
star at 1 Myr with the winds since 0.7 Myr still to return; 'slow': a 20 Msun star late in its life, wind about 100 km/s (table log age
6.9468); '_moving' after either: the star moves at 300 km/s along x through the gas at rest."""
import numpy as np
import h5py

PC_KMS_IN_YR = 3.085678e13 / 3.155e7


def density(n_H):
    return n_H * 1.6726e-24 / 0.7 * 3.0857e18 ** 3 / 1.989e33  # Msun / pc^3


def cells(L, rng):
    n_of = lambda z: 1.0 if z > 2.0 else float(np.clip(100.0 * np.exp(z / 0.5), 1.0, 1000.0))
    slabs, z = [], -0.5 * L
    while z < 0.5 * L:
        dx = density(n_of(z)) ** (-1.0 / 3.0)
        k = int(round(L / dx)); d = L / k
        g = (np.arange(k) + 0.5) * d
        X, Y = np.meshgrid(g, g, indexing="ij")
        q = np.stack([X.ravel(), Y.ravel(), np.full(k * k, z + 0.5 * dx + 0.5 * L)], -1)
        q[:, :2] += rng.uniform(0, d, 2)
        q += rng.uniform(-0.2, 0.2, q.shape) * d
        slabs.append(q); z += dx
    return np.vstack(slabs) % L


def make_ics(fname, tables_file, event, L=20.0, seed=3, z=0.014, T=100.0):
    pos = cells(L, np.random.default_rng(seed))
    i0 = int(np.argmin(np.sum((pos - 0.5 * L) ** 2, axis=1)))
    star = pos[i0].copy()
    pos = (np.delete(pos, i0, axis=0) - star + 0.5 * L) % L
    N = len(pos)
    with h5py.File(tables_file, "r") as F:
        Zt, Mt = F["Z"][()], F["M_init"][()]
        iz = int(np.argmin(np.abs(np.log10(Zt) - np.log10(z))))
        life = (F["lifetime_yr"][()] + F["t_PMS"][()]).reshape(len(Zt), len(Mt))[iz]
    if event == "sn":
        M = 20.0; age = 1.01 * np.interp(np.log10(M), np.log10(Mt), life); last = np.log10(age / 1.01)
    elif event.startswith("slow"):
        M = 20.0; age = 8.876e6; last = np.log10(age - 2.836e4) - 1e-3  # the wind owed over a short stretch of the slow phase
    else:
        M = 35.0; age = 1e6; last = np.log10(7e5)
    u = 1.5 * 1.380649e-16 * T / (1.27 * 1.6726e-24) / 1e10
    with h5py.File(fname, "w") as F:
        h = F.create_group("Header"); npart = np.array([N, 0, 0, 0, 1, 0])
        h.attrs["NumPart_ThisFile"] = npart; h.attrs["NumPart_Total"] = npart
        h.attrs["NumPart_Total_HighWord"] = np.zeros(6, dtype=np.uint32); h.attrs["MassTable"] = np.zeros(6)
        h.attrs["Time"] = 0.0; h.attrs["Redshift"] = 0.0; h.attrs["BoxSize"] = L; h.attrs["NumFilesPerSnapshot"] = 1
        h.attrs["Omega0"] = h.attrs["OmegaLambda"] = h.attrs["HubbleParam"] = 0.0
        h.attrs["Flag_Sfr"] = 1; h.attrs["Flag_Cooling"] = 0; h.attrs["Flag_StellarAge"] = 1; h.attrs["Flag_Metals"] = 1
        h.attrs["Flag_Feedback"] = 1; h.attrs["Flag_DoublePrecision"] = 1
        p = F.create_group("PartType0")
        p["Coordinates"] = pos; p["Velocities"] = np.zeros((N, 3)); p["ParticleIDs"] = np.arange(1, N + 1, dtype=np.uint32)
        p["Masses"] = np.ones(N); p["InternalEnergy"] = np.full(N, u); p["Metallicity"] = np.full((N, 1), z)
        p["SmoothingLength"] = np.full(N, 1.0)
        s = F.create_group("PartType4")
        s["Coordinates"] = np.array([[0.5 * L] * 3]); s["Velocities"] = np.array([[300.0 if event.endswith("_moving") else 0.0, 0.0, 0.0]]); s["ParticleIDs"] = np.array([N + 1], dtype=np.uint32)
        s["Masses"] = np.array([M]); s["StellarFormationTime"] = np.array([-age / PC_KMS_IN_YR]); s["Metallicity"] = np.full((1, 1), z)
        s["SingleStarDrawnMass"] = np.array([M]); s["SingleStarBirthMetallicity"] = np.full(1, z); s["SingleStarLastWindLogAge"] = np.array([last])
    return L
