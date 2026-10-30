"""Uniform periodic box of 16^3 gas cells (1 pc, 4096 Msun, 10 K) for the stellar-evolution table start-up test."""
import numpy as np
import h5py


def make_stellar_evolution_tables_ics(fname, n=16):
    x = (np.arange(n) + 0.5) / n
    pos = np.array(np.meshgrid(x, x, x, indexing="ij")).reshape(3, -1).T
    N = len(pos)
    with h5py.File(fname, "w") as F:
        h = F.create_group("Header")
        npart = np.array([N, 0, 0, 0, 0, 0])
        h.attrs["NumPart_ThisFile"] = npart
        h.attrs["NumPart_Total"] = npart
        h.attrs["NumPart_Total_HighWord"] = np.zeros(6, dtype=int)
        h.attrs["MassTable"] = np.zeros(6)
        h.attrs["Time"] = 0.0
        h.attrs["Redshift"] = 0.0
        h.attrs["BoxSize"] = 1.0
        h.attrs["NumFilesPerSnapshot"] = 1
        h.attrs["Omega0"] = h.attrs["OmegaLambda"] = h.attrs["HubbleParam"] = 0.0
        h.attrs["Flag_Sfr"] = h.attrs["Flag_Cooling"] = h.attrs["Flag_StellarAge"] = h.attrs["Flag_Metals"] = h.attrs["Flag_Feedback"] = 0
        h.attrs["Flag_DoublePrecision"] = 1
        p = F.create_group("PartType0")
        p.create_dataset("Coordinates", data=pos)
        p.create_dataset("Velocities", data=np.zeros_like(pos))
        p.create_dataset("ParticleIDs", data=np.arange(1, N + 1, dtype=np.uint32))
        p.create_dataset("Masses", data=np.full(N, 1.0))
        p.create_dataset("InternalEnergy", data=np.full(N, 0.1))
