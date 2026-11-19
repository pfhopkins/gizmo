#!/usr/bin/env python3
"""ICs for test/ketju_pm (kpc, km/s, 1e10 Msun): a 1 + 1 Msun pair (a = 200 AU, e = 0.5, orbit tilted 0.6 rad) in a KETJU region
and a 20 Msun star on a circular orbit R_OUT = 20000 AU from it, at the centre of a periodic box of L = 2e5 AU."""
import os
import sys
import h5py
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../ketju_tide"))
from make_ketju_tide_ics import kepler_state, AU, G

L, R_OUT = 2e5 * AU, 20000 * AU


def state():
    m = np.array([1.0, 1.0, 20.0]); xs, vs = kepler_state(1, 1, 200 * AU, 0.5, 0.7)
    c, s = np.cos(0.6), np.sin(0.6); Rx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    xs = [Rx @ x for x in xs]; vs = [Rx @ v for v in vs]
    v3 = np.sqrt(G * m.sum() / R_OUT); xo, vo = np.array([R_OUT, 0, 0]), np.array([0, v3, 0]); M = m.sum()
    x = np.array([xx - m[2] / M * xo for xx in xs] + [2.0 / M * xo]); v = np.array([vv - m[2] / M * vo for vv in vs] + [2.0 / M * vo])
    return m, x, v


def write(name, m, x, v, box, time=0.0, comoving=False):
    n = len(m)
    with h5py.File(name, "w") as f:
        h = f.create_group("Header")
        h.attrs["NumPart_ThisFile"] = [0, 0, 0, 0, n, 0]; h.attrs["NumPart_Total"] = [0, 0, 0, 0, n, 0]
        h.attrs["NumPart_Total_HighWord"] = [0] * 6; h.attrs["MassTable"] = [0.0] * 6
        h.attrs["Time"] = time; h.attrs["Redshift"] = 1 / time - 1 if comoving else 0.0; h.attrs["BoxSize"] = box
        h.attrs["NumFilesPerSnapshot"] = 1; h.attrs["Omega0"] = 0.0; h.attrs["OmegaLambda"] = 0.0; h.attrs["HubbleParam"] = 1.0
        for k in ("Flag_Sfr", "Flag_Cooling", "Flag_StellarAge", "Flag_Metals", "Flag_Feedback"): h.attrs[k] = 0
        g = f.create_group("PartType4")
        g["Coordinates"] = x + 0.5 * box; g["Velocities"] = v; g["Masses"] = m / 1e10
        g["ParticleIDs"] = np.arange(1, n + 1, dtype=np.uint64)


def main(name="ketju_pm_ics.hdf5"):
    m, x, v = state(); write(name, m, x, v, L)


if __name__ == "__main__":
    main()
