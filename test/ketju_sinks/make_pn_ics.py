"""two 1e6 Msun sinks on a circular orbit, a = 1e-5 pc"""
import h5py
import numpy as np

G, m, a = 4.300710573e-3, 1e6, 1e-5
v = 0.5 * np.sqrt(G * 2 * m / a)
with h5py.File("test/ketju_sinks/ketju_pn_ics.hdf5", "w") as f:
    h = f.create_group("Header")
    h.attrs["NumPart_ThisFile"] = [0, 0, 0, 0, 0, 2]; h.attrs["NumPart_Total"] = [0, 0, 0, 0, 0, 2]
    h.attrs["NumPart_Total_HighWord"] = [0] * 6; h.attrs["MassTable"] = [0.0] * 6
    h.attrs["Time"] = 0.0; h.attrs["Redshift"] = 0.0; h.attrs["BoxSize"] = 1.0; h.attrs["NumFilesPerSnapshot"] = 1
    h.attrs["Omega0"] = 0.0; h.attrs["OmegaLambda"] = 0.0; h.attrs["HubbleParam"] = 1.0
    for k in ("Flag_Sfr", "Flag_Cooling", "Flag_StellarAge", "Flag_Metals", "Flag_Feedback"): h.attrs[k] = 0
    g = f.create_group("PartType5")
    g["Coordinates"] = np.array([[0.5 + 0.5 * a, 0.5, 0.5], [0.5 - 0.5 * a, 0.5, 0.5]]); g["Velocities"] = np.array([[0, v, 0], [0, -v, 0.0]])
    g["Masses"] = np.full(2, m); g["ParticleIDs"] = np.arange(1, 3, dtype=np.uint64); g["SinkMass"] = np.full(2, m)
