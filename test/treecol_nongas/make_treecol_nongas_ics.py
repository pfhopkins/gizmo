"""Gas lattice, optionally with one negligible-mass star next to a gas cell."""

import h5py
import numpy as np

N_SIDE, SPACING, M_GAS = 8, 0.1, 1e-4


def make_ics(filename, with_star):
    g = (np.arange(N_SIDE) - 0.5 * (N_SIDE - 1)) * SPACING + 5.0
    x = np.array(np.meshgrid(g, g, g, indexing="ij")).reshape(3, -1).T
    x += np.random.default_rng(3).uniform(-0.1, 0.1, x.shape) * SPACING
    n = len(x)
    with h5py.File(filename, "w") as f:
        h = f.create_group("Header")
        nstar = 1 if with_star else 0
        h.attrs["NumPart_ThisFile"] = np.array([n, 0, 0, 0, nstar, 0], dtype=np.uint32)
        h.attrs["NumPart_Total"] = np.array([n, 0, 0, 0, nstar, 0], dtype=np.uint32)
        h.attrs["NumPart_Total_HighWord"] = np.zeros(6, dtype=np.uint32)
        h.attrs["MassTable"] = np.zeros(6)
        h.attrs["Time"] = 0.0
        h.attrs["BoxSize"] = 10.0
        h.attrs["NumFilesPerSnapshot"] = 1
        p = f.create_group("PartType0")
        p["Coordinates"] = x
        p["Velocities"] = np.zeros_like(x)
        p["ParticleIDs"] = np.arange(1, n + 1, dtype=np.uint32)
        p["Masses"] = np.full(n, M_GAS)
        p["InternalEnergy"] = np.full(n, 100.0)
        if with_star:  # gasless and 1e-10 of a cell's mass: it must not change any column
            s = f.create_group("PartType4")
            s["Coordinates"] = (x[n // 2] + 0.2 * SPACING)[None, :]
            s["Velocities"] = np.zeros((1, 3))
            s["ParticleIDs"] = np.array([n + 1], dtype=np.uint32)
            s["Masses"] = np.array([1e-10 * M_GAS])
