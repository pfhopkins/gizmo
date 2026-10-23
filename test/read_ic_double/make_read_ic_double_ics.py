"""PartType4-only HDF5 IC whose positions/velocities/masses carry full double mantissas."""
import h5py
import numpy as np


def make_ic(fname, dtype, n=64, seed=42):
    rng = np.random.default_rng(seed)
    x = 0.5 + 1e-3 * rng.standard_normal((n, 3))  # cluster-like: offsets small against |x|, as for a hard binary
    v = 3.0 * rng.standard_normal((n, 3))
    m = 1e-10 * (0.1 + rng.random(n))
    ids = np.arange(1, n + 1, dtype=np.uint64)
    with h5py.File(fname, "w") as f:
        h = f.create_group("Header")
        npart = np.array([0, 0, 0, 0, n, 0], dtype=np.uint32)
        h.attrs["NumPart_ThisFile"] = npart
        h.attrs["NumPart_Total"] = npart
        h.attrs["NumPart_Total_HighWord"] = np.zeros(6, dtype=np.uint32)
        h.attrs["MassTable"] = np.zeros(6)
        h.attrs["Time"] = 0.0
        h.attrs["Redshift"] = 0.0
        h.attrs["BoxSize"] = 1.0
        h.attrs["NumFilesPerSnapshot"] = 1
        h.attrs["Flag_DoublePrecision"] = 0
        g = f.create_group("PartType4")
        g.create_dataset("Coordinates", data=x.astype(dtype))
        g.create_dataset("Velocities", data=v.astype(dtype))
        g.create_dataset("Masses", data=m.astype(dtype))
        g.create_dataset("ParticleIDs", data=ids)
