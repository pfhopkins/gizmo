"""A 0.5 Msun sink at the centre of a cold uniform sphere (r = 0.02 pc, 4224 cells of 7.8e-5 Msun; pc, km/s, Msun)."""
import h5py
import numpy as np

n, r, m_gas, m_sink = 20, 0.02, 7.8125e-5, 0.5
g = (np.arange(n) + 0.5) / n * 2 * r - r
x = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
x = x[np.sum(x * x, 1) < r * r] + 0.5
u10 = 1.5 * 1.380649e-16 * 10 / (2.33 * 1.6726e-24) / 1e10
xs = np.array([[0.5 + 3e-4, 0.5 - 2e-4, 0.5 + 1e-4]])  # off the lattice, so accretion moves it
with h5py.File("tree_node_boxes_ics.hdf5", "w") as f:
    h = f.create_group("Header")
    N = [len(x), 0, 0, 0, 0, 1]
    h.attrs["NumPart_ThisFile"] = N; h.attrs["NumPart_Total"] = N
    h.attrs["NumPart_Total_HighWord"] = [0] * 6; h.attrs["MassTable"] = [0.0] * 6
    h.attrs["Time"] = 0.0; h.attrs["Redshift"] = 0.0; h.attrs["BoxSize"] = 1.0; h.attrs["NumFilesPerSnapshot"] = 1
    h.attrs["Omega0"] = 0.0; h.attrs["OmegaLambda"] = 0.0; h.attrs["HubbleParam"] = 1.0
    for k in ("Flag_Sfr", "Flag_Cooling", "Flag_StellarAge", "Flag_Metals", "Flag_Feedback"): h.attrs[k] = 0
    p = f.create_group("PartType0")
    p["Coordinates"] = x; p["Velocities"] = np.zeros_like(x); p["Masses"] = np.full(len(x), m_gas)
    p["ParticleIDs"] = np.arange(1, len(x) + 1, dtype=np.uint64); p["InternalEnergy"] = np.full(len(x), u10)
    s = f.create_group("PartType5")
    s["Coordinates"] = xs; s["Velocities"] = np.zeros((1, 3)); s["Masses"] = np.array([m_sink])
    s["ParticleIDs"] = np.array([len(x) + 1], dtype=np.uint64); s["SinkMass"] = np.array([m_sink])
print(len(x), "cells")
