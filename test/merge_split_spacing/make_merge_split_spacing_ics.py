"""Lattice of unit-mass cells with close pairs of over-massive cells, each pair inside a lattice void."""

import h5py
import numpy as np

N_SIDE, BOX, M_CELL, M_HEAVY = 10, 1.0, 1.0, 3.0
PAIR_SEPARATION = 0.02
VOIDS = [(1, 1, 1), (3, 5, 7), (6, 2, 4), (8, 8, 8), (2, 7, 5), (5, 5, 1), (7, 3, 6), (4, 8, 2)]


def make_ics(filename):
    """Writes the IC and returns {pair id: (id_a, pos_a, id_b, pos_b)}."""
    dx = BOX / N_SIDE
    g = (np.arange(N_SIDE) + 0.5) * dx
    x = np.array(np.meshgrid(g, g, g, indexing="ij")).reshape(3, -1).T
    m = np.full(len(x), M_CELL)
    rng = np.random.default_rng(7)
    pairs = {}
    for n, v in enumerate(VOIDS):
        centre = (np.array(v) + 1.0) * dx  # a lattice void: equidistant from 8 lattice nodes
        d = rng.normal(size=3); d *= 0.5 * PAIR_SEPARATION / np.linalg.norm(d)
        a, b = centre + d, centre - d
        pairs[n] = (len(x) + 1, a, len(x) + 2, b)
        x = np.vstack([x, a, b]); m = np.append(m, [M_HEAVY, M_HEAVY])
    n = len(x)
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
        h.attrs["Minimum_Mass_For_Cell_Merge"] = 0.3 * M_CELL  # nothing merges
        h.attrs["Maximum_Mass_For_Cell_Split"] = 2.5 * M_CELL  # only the heavy cells split
        p = f.create_group("PartType0")
        p["Coordinates"] = x
        p["Velocities"] = np.zeros_like(x)
        p["ParticleIDs"] = np.arange(1, n + 1, dtype=np.uint32)
        p["Masses"] = m
        p["InternalEnergy"] = np.full(n, 100.0)
    return pairs
