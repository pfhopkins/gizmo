"""Lattice of unit-mass gas cells in a periodic unit box, uniform internal energy, at rest. A tenth of the cells, spread over the
whole box, are heavy (mass 3, above the split threshold 2.5 in the header), so for any decomposition some of them have their
nearest neighbour on another rank. jitter: random displacement of every cell in units of the lattice spacing (0: exact lattice,
on which the direction a cell would be split along is not unique; 16 cells per side there, so that every position is exact in
binary and only the order of the neighbour sums can break the tie)."""

import h5py
import numpy as np

N_SIDE, N_SIDE_EXACT, BOX = 12, 16, 1.0


def make_ics(filename, jitter=0.15):
    rng = np.random.default_rng(11)
    ns = N_SIDE if jitter > 0 else N_SIDE_EXACT
    dx = BOX / ns
    g = (np.arange(ns) + 0.5) * dx
    x = np.array(np.meshgrid(g, g, g, indexing="ij")).reshape(3, -1).T + jitter * rng.uniform(-1, 1, (ns**3, 3)) * dx
    n = len(x)
    m = np.ones(n)
    m[rng.permutation(n)[: n // 10]] = 3.0
    with h5py.File(filename, "w") as f:
        h = f.create_group("Header")
        h.attrs["NumPart_ThisFile"] = np.array([n, 0, 0, 0, 0, 0], dtype=np.uint32)
        h.attrs["NumPart_Total"] = np.array([n, 0, 0, 0, 0, 0], dtype=np.uint32)
        h.attrs["NumPart_Total_HighWord"] = np.zeros(6, dtype=np.uint32)
        h.attrs["MassTable"] = np.zeros(6)
        h.attrs["Time"] = 0.0
        h.attrs["Redshift"] = 0.0
        h.attrs["BoxSize"] = BOX
        h.attrs["NumFilesPerSnapshot"] = 1
        h.attrs["Flag_DoublePrecision"] = 1
        h.attrs["Minimum_Mass_For_Cell_Merge"] = 0.3  # nothing merges
        h.attrs["Maximum_Mass_For_Cell_Split"] = 2.5  # only the heavy cells split
        p = f.create_group("PartType0")
        p["Coordinates"] = x
        p["Velocities"] = np.zeros((n, 3))
        p["Masses"] = m
        p["InternalEnergy"] = np.full(n, 1.0)
        p["ParticleIDs"] = np.arange(1, n + 1, dtype=np.uint32)
    return n, int((m == 3.0).sum())
