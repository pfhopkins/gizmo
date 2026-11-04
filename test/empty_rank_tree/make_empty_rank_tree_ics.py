"""ICs for test/empty_rank_tree: three stars; the load balance later puts all of them on one of two ranks."""
import h5py
import numpy as np

x = np.array([[0.4928436336463462, 0.49569623403166296, 0.49780149595676176], [0.5018423123492789, 0.5023406779894755, 0.5026236975534363], [0.5013206533114901, 0.4986401858730624, 0.4968101061735308]])
v = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
m = np.array([0.766217545188308, 2.1105125558281435, 1.2078189644295287])
with h5py.File("empty_rank_tree_ics.hdf5", "w") as f:
    h = f.create_group("Header")
    h.attrs["NumPart_ThisFile"] = [0, 0, 0, 0, 3, 0]; h.attrs["NumPart_Total"] = [0, 0, 0, 0, 3, 0]
    h.attrs["NumPart_Total_HighWord"] = [0] * 6; h.attrs["MassTable"] = [0.0] * 6
    h.attrs["Time"] = 0.0; h.attrs["Redshift"] = 0.0; h.attrs["BoxSize"] = 1.0; h.attrs["NumFilesPerSnapshot"] = 1
    h.attrs["Omega0"] = 0.0; h.attrs["OmegaLambda"] = 0.0; h.attrs["HubbleParam"] = 1.0
    for k in ("Flag_Sfr", "Flag_Cooling", "Flag_StellarAge", "Flag_Metals", "Flag_Feedback"): h.attrs[k] = 0
    g = f.create_group("PartType4")
    g["Coordinates"] = x; g["Velocities"] = v; g["Masses"] = m; g["ParticleIDs"] = np.arange(1, 4, dtype=np.uint64)
