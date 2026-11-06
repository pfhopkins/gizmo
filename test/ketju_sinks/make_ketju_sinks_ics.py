"""Two Type 5 sinks at apocentre, e = 0.9, q = 0.1 (pc, km/s, Msun), COM frame."""
import h5py
import numpy as np

G, m1, m2, a, e = 4.300710573e-3, 1.0, 0.1, 1000 / 206264.806, 0.9
M = m1 + m2; ra = a * (1 + e); va = np.sqrt(G * M * (1 - e) / (a * (1 + e)))
x = np.array([[0.5 + m2 / M * ra, 0.5, 0.5], [0.5 - m1 / M * ra, 0.5, 0.5]])
v = np.array([[0, m2 / M * va, 0], [0, -m1 / M * va, 0]])
with h5py.File("ketju_sinks_ics.hdf5", "w") as f:
    h = f.create_group("Header")
    h.attrs["NumPart_ThisFile"] = [0, 0, 0, 0, 0, 2]; h.attrs["NumPart_Total"] = [0, 0, 0, 0, 0, 2]
    h.attrs["NumPart_Total_HighWord"] = [0] * 6; h.attrs["MassTable"] = [0.0] * 6
    h.attrs["Time"] = 0.0; h.attrs["Redshift"] = 0.0; h.attrs["BoxSize"] = 1.0; h.attrs["NumFilesPerSnapshot"] = 1
    h.attrs["Omega0"] = 0.0; h.attrs["OmegaLambda"] = 0.0; h.attrs["HubbleParam"] = 1.0
    for k in ("Flag_Sfr", "Flag_Cooling", "Flag_StellarAge", "Flag_Metals", "Flag_Feedback"): h.attrs[k] = 0
    g = f.create_group("PartType5")
    g["Coordinates"] = x; g["Velocities"] = v; g["Masses"] = np.array([m1, m2]); g["ParticleIDs"] = np.arange(1, 3, dtype=np.uint64)
    g["SinkMass"] = np.array([m1, m2])
