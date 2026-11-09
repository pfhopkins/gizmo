"""ICs for test/timeline_restart_tree (kpc, km/s, 1e10 Msun): a collapsing self-gravitating gas sphere of 1000 cells, so the steps differ."""
import h5py
import numpy as np

N, M, R, U = 1000, 1.0e-4, 0.05, 2.0
rng = np.random.default_rng(20261106)
r = R * rng.uniform(size=N) ** (1 / 1.8)
n = rng.normal(size=(N, 3)); n /= np.linalg.norm(n, axis=1)[:, None]
x = r[:, None] * n
with h5py.File("timeline_restart_tree_ics.hdf5", "w") as f:
    h = f.create_group("Header")
    h.attrs["NumPart_ThisFile"] = [N, 0, 0, 0, 0, 0]; h.attrs["NumPart_Total"] = [N, 0, 0, 0, 0, 0]
    h.attrs["NumPart_Total_HighWord"] = [0] * 6; h.attrs["MassTable"] = [0.0] * 6
    h.attrs["Time"] = 0.0; h.attrs["Redshift"] = 0.0; h.attrs["BoxSize"] = 1.0; h.attrs["NumFilesPerSnapshot"] = 1
    h.attrs["Omega0"] = 0.0; h.attrs["OmegaLambda"] = 0.0; h.attrs["HubbleParam"] = 1.0
    for k in ("Flag_Sfr", "Flag_Cooling", "Flag_StellarAge", "Flag_Metals", "Flag_Feedback"):
        h.attrs[k] = 0
    g = f.create_group("PartType0")
    g["Coordinates"] = x + 0.5; g["Velocities"] = np.zeros((N, 3)); g["Masses"] = np.full(N, M / N)
    g["ParticleIDs"] = np.arange(1, N + 1, dtype=np.uint32); g["InternalEnergy"] = np.full(N, U)
