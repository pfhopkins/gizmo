"""The lattice blast of make_lattice_blast_ics.py (16^3 periodic cells, 8 hot cells at the centre, total blast energy 1) plus a uniform
CR energy, 2 per unit volume (CR pressure about the gas pressure of the cold gas), and a uniform field of 2 microG along z. Written as
a snapshot (start flag 2). KernelMaxRadius: the name the reference implementation reads, so one file serves both."""
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from make_lattice_blast_ics import make_lattice_blast_ics  # noqa: E402


def make_crfluid_exchange_pdv_only_ics(fname, e_cr_density=2.0):
    n = make_lattice_blast_ics(fname)
    with h5py.File(fname, "a") as F:
        F["PartType0/CosmicRayEnergy"] = np.full(n, e_cr_density / n)  # per cell: density x volume, volume = 1/n
        F["PartType0/SmoothingLength"] = F["PartType0/KernelMaxRadius"] = np.full(n, 2.0 / 16)
        b = np.zeros((n, 3))
        b[:, 2] = 2e-6
        F["PartType0/MagneticField"] = b
    return n
