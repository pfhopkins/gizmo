"""ICs for test/ketju_quiet_steps (kpc, km/s, 1e10 Msun): 4096 bodies in a Plummer sphere; 64 of them stars, far apart."""
import h5py
import numpy as np

G, N, NS, M1, B = 43007.1, 4096, 64, 1e-10, 1e-2
rng = np.random.default_rng(20261104)
r = B / np.sqrt(np.clip(rng.uniform(size=N), 1e-3, 0.99) ** (-2.0 / 3.0) - 1)
n = rng.normal(size=(N, 3)); n /= np.linalg.norm(n, axis=1)[:, None]
x = r[:, None] * n
v = np.sqrt(0.25 * G * N * M1 / B) * rng.normal(size=(N, 3)) / np.sqrt(3)
x -= x.mean(0); v -= v.mean(0)
with h5py.File("ketju_quiet_steps_ics.hdf5", "w") as f:
    h = f.create_group("Header"); c = [0, N - NS, 0, 0, NS, 0]
    h.attrs["NumPart_ThisFile"] = c; h.attrs["NumPart_Total"] = c
    h.attrs["NumPart_Total_HighWord"] = [0] * 6; h.attrs["MassTable"] = [0.0] * 6
    h.attrs["Time"] = 0.0; h.attrs["Redshift"] = 0.0; h.attrs["BoxSize"] = 3.0; h.attrs["NumFilesPerSnapshot"] = 1
    h.attrs["Omega0"] = 0.0; h.attrs["OmegaLambda"] = 0.0; h.attrs["HubbleParam"] = 1.0
    for k in ("Flag_Sfr", "Flag_Cooling", "Flag_StellarAge", "Flag_Metals", "Flag_Feedback"): h.attrs[k] = 0
    for t, a, b in ((1, 0, N - NS), (4, N - NS, N)):
        g = f.create_group(f"PartType{t}")
        g["Coordinates"] = x[a:b]; g["Velocities"] = v[a:b]; g["Masses"] = np.full(b - a, M1); g["ParticleIDs"] = np.arange(a + 1, b + 1, dtype=np.uint64)
