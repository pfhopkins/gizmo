"""ICs for test/timeline_restart (kpc, km/s, 1e10 Msun): 64 equal stars in a Plummer sphere, short and long steps from close pairs."""
import h5py
import numpy as np

G, N, M_STAR, B = 43007.1, 64, 1e-10, 1.08e-3
rng = np.random.default_rng(20261104)
r = B / np.sqrt(np.clip(rng.uniform(size=N), 1e-3, 0.99) ** (-2.0 / 3.0) - 1)
n = rng.normal(size=(N, 3)); n /= np.linalg.norm(n, axis=1)[:, None]
x = r[:, None] * n
v = 0.5 * np.sqrt(G * N * M_STAR / B) * rng.normal(size=(N, 3)) / np.sqrt(3)
pair = 1000 / 206264.806 / 1e3
x[1::8] = x[0::8] + pair * n[0::8]
vc = np.sqrt(G * 2 * M_STAR / pair) * 0.5
t = np.cross(n[0::8], rng.normal(size=(len(n[0::8]), 3))); t /= np.linalg.norm(t, axis=1)[:, None]
v[1::8] = v[0::8] - vc * t; v[0::8] += vc * t
x -= x.mean(0); v -= v.mean(0)
with h5py.File("timeline_restart_ics.hdf5", "w") as f:
    h = f.create_group("Header")
    h.attrs["NumPart_ThisFile"] = [0, 0, 0, 0, N, 0]; h.attrs["NumPart_Total"] = [0, 0, 0, 0, N, 0]
    h.attrs["NumPart_Total_HighWord"] = [0] * 6; h.attrs["MassTable"] = [0.0] * 6
    h.attrs["Time"] = 0.0; h.attrs["Redshift"] = 0.0; h.attrs["BoxSize"] = 0.3; h.attrs["NumFilesPerSnapshot"] = 1
    h.attrs["Omega0"] = 0.0; h.attrs["OmegaLambda"] = 0.0; h.attrs["HubbleParam"] = 1.0
    for k in ("Flag_Sfr", "Flag_Cooling", "Flag_StellarAge", "Flag_Metals", "Flag_Feedback"): h.attrs[k] = 0
    g = f.create_group("PartType4")
    g["Coordinates"] = x; g["Velocities"] = v; g["Masses"] = np.full(N, M_STAR); g["ParticleIDs"] = np.arange(1, N + 1, dtype=np.uint64)
