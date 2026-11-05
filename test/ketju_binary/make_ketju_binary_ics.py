"""ICs for test/ketju_binary (kpc, km/s, 1e10 Msun): a 1 + 1 Msun binary, a = 1000 AU, e = 0.9, starting at apocentre, and a far 1 Msun star; the second file adds a 0.05 Msun field star 100 AU from one member."""
import h5py
import numpy as np

G, M, AU, ECC = 43007.1, 1e-10, 4.848137e-9, 0.9
A = 1000 * AU
ra = A * (1 + ECC); va = np.sqrt(G * 2 * M * (1 - ECC) / (A * (1 + ECC)))
c = np.array([0.15, 0.15, 0.15])
x = np.array([c + [0.5 * ra, 0, 0], c - [0.5 * ra, 0, 0], c + [2e-3, 0, 0]])
v = np.array([[0, 0.5 * va, 0], [0, -0.5 * va, 0], [0, 0, 0]])


def write(name, x, v, m):
    n = len(m)
    with h5py.File(name, "w") as f:
        h = f.create_group("Header")
        h.attrs["NumPart_ThisFile"] = [0, 0, 0, 0, n, 0]; h.attrs["NumPart_Total"] = [0, 0, 0, 0, n, 0]
        h.attrs["NumPart_Total_HighWord"] = [0] * 6; h.attrs["MassTable"] = [0.0] * 6
        h.attrs["Time"] = 0.0; h.attrs["Redshift"] = 0.0; h.attrs["BoxSize"] = 0.3; h.attrs["NumFilesPerSnapshot"] = 1
        h.attrs["Omega0"] = 0.0; h.attrs["OmegaLambda"] = 0.0; h.attrs["HubbleParam"] = 1.0
        for k in ("Flag_Sfr", "Flag_Cooling", "Flag_StellarAge", "Flag_Metals", "Flag_Feedback"): h.attrs[k] = 0
        g = f.create_group("PartType4")
        g["Coordinates"] = x; g["Velocities"] = v; g["Masses"] = m; g["ParticleIDs"] = np.arange(1, n + 1, dtype=np.uint64)


write("ketju_binary_ics.hdf5", x, v, np.full(3, M))
write("ketju_binary_light_ics.hdf5", np.vstack([x[:2], x[1] - [100 * AU, 0, 0]]), np.vstack([v[:2], v[1]]), np.array([M, M, 0.05 * M]))
