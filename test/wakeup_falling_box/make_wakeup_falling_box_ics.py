#!/usr/bin/env python3
"""Generate ICs for wakeup_falling_box: a 32^3 lattice with the totals of the sedov test ICs
(box 6, uniform density and internal energy) and the blast energy in the 8 central cells."""

import numpy as np
import h5py


def make_wakeup_falling_box_ics(output_file="wakeup_falling_box_ics.hdf5", N=32):
    L, M, u_bg, E = 6.0, 2.663383e9, 0.10153126, 3.4376716e10
    g = (np.arange(N) + 0.5) * L / N
    x = np.array(np.meshgrid(g, g, g, indexing="ij")).reshape(3, -1).T
    m = np.full(len(x), M / len(x))
    u = np.full(len(x), u_bg)
    hot = np.argsort(np.linalg.norm(x - L / 2, axis=1))[:8]
    u[hot] += E / m[hot].sum()
    with h5py.File(output_file, "w") as f:
        h = f.create_group("Header")
        npart = np.array([len(x), 0, 0, 0, 0, 0], dtype=np.int32)
        h.attrs["NumPart_ThisFile"] = npart
        h.attrs["NumPart_Total"] = npart.astype(np.uint32)
        h.attrs["NumPart_Total_HighWord"] = np.zeros(6, dtype=np.uint32)
        h.attrs["MassTable"] = np.zeros(6)
        h.attrs["Time"] = 0.0
        h.attrs["Redshift"] = 0.0
        h.attrs["BoxSize"] = L
        h.attrs["NumFilesPerSnapshot"] = 1
        h.attrs["Flag_DoublePrecision"] = 0
        p = f.create_group("PartType0")
        p["Coordinates"] = x.astype(np.float32)
        p["Velocities"] = np.zeros_like(x, dtype=np.float32)
        p["Masses"] = m.astype(np.float32)
        p["InternalEnergy"] = u.astype(np.float32)
        p["ParticleIDs"] = np.arange(1, len(x) + 1, dtype=np.uint32)


if __name__ == "__main__":
    make_wakeup_falling_box_ics()
