#!/usr/bin/env python3
"""ICs for test_ketju_pm_comoving (comoving, a = 0.5, Omega0 = 1, h = 1): a 20 Msun star at the box centre (on a PM cell
boundary, so the mesh is mirror-symmetric about it); two 1 + 1 Msun binaries (a = 50 AU, e = 0.5, each its own KETJU region) at
5000 AU physical along +x and +y on circular orbits about it, and two single 2 Msun stars at the mirror positions with mirrored
velocities. Comoving box 1e5 AU. Gadget IC convention: comoving positions, velocities = peculiar / sqrt(a)."""
import os
import sys
import h5py
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../ketju_tide"))
from make_ketju_tide_ics import kepler_state, AU, G

A0, L = 0.5, 1e5 * AU


def state():
    r, M = 5000 * AU, 20.0
    v = np.sqrt(G * (M + 2.0) / r)
    xs, vs = kepler_state(1, 1, 50 * AU, 0.5, 0.7)
    m, x, u = [M], [np.zeros(3)], [np.zeros(3)]
    for e_r, e_v in (([1, 0, 0], [0, 1, 0]), ([0, 1, 0], [-1, 0, 0])):
        e_r, e_v = np.array(e_r, float), np.array(e_v, float)
        for xi, vi in zip(xs, vs):  # binary at +r, internal orbit tilted out of the plane
            m.append(1.0); x.append(r * e_r + np.array([xi[0], 0.3 * xi[1], xi[1]])); u.append(v * e_v + np.array([vi[0], 0.3 * vi[1], vi[1]]))
        m.append(2.0); x.append(-r * e_r); u.append(-v * e_v)  # single star at the mirror position
    return np.array(m), np.array(x), np.array(u)


def main(name="ketju_pm_comoving_ics.hdf5"):
    m, x, u = state(); n = len(m)
    with h5py.File(name, "w") as f:
        h = f.create_group("Header")
        h.attrs["NumPart_ThisFile"] = [0, 0, 0, 0, n, 0]; h.attrs["NumPart_Total"] = [0, 0, 0, 0, n, 0]
        h.attrs["NumPart_Total_HighWord"] = [0] * 6; h.attrs["MassTable"] = [0.0] * 6
        h.attrs["Time"] = A0; h.attrs["Redshift"] = 1 / A0 - 1; h.attrs["BoxSize"] = L; h.attrs["NumFilesPerSnapshot"] = 1
        h.attrs["Omega0"] = 1.0; h.attrs["OmegaLambda"] = 0.0; h.attrs["HubbleParam"] = 1.0
        for k in ("Flag_Sfr", "Flag_Cooling", "Flag_StellarAge", "Flag_Metals", "Flag_Feedback"): h.attrs[k] = 0
        g = f.create_group("PartType4")
        g["Coordinates"] = x / A0 + 0.5 * L; g["Velocities"] = u / np.sqrt(A0); g["Masses"] = m / 1e10
        g["ParticleIDs"] = np.arange(1, n + 1, dtype=np.uint64)


if __name__ == "__main__":
    main()
