#!/usr/bin/env python3
"""ICs for test/ketju_pm: the ketju_tide three-body system (1 + 1 Msun pair, a = 200 AU, in a KETJU region; 20 Msun star at
5000 AU on a circular outer orbit) at the centre of a periodic box of 50000 AU (kpc, km/s, 1e10 Msun)."""
import os
import sys
import h5py
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../ketju_tide"))
from make_ketju_tide_ics import state, AU

L = 50000 * AU


def main(name="ketju_pm_ics.hdf5"):
    m, x, v = state(); n = len(m)
    with h5py.File(name, "w") as f:
        h = f.create_group("Header")
        h.attrs["NumPart_ThisFile"] = [0, 0, 0, 0, n, 0]; h.attrs["NumPart_Total"] = [0, 0, 0, 0, n, 0]
        h.attrs["NumPart_Total_HighWord"] = [0] * 6; h.attrs["MassTable"] = [0.0] * 6
        h.attrs["Time"] = 0.0; h.attrs["Redshift"] = 0.0; h.attrs["BoxSize"] = L; h.attrs["NumFilesPerSnapshot"] = 1
        h.attrs["Omega0"] = 0.0; h.attrs["OmegaLambda"] = 0.0; h.attrs["HubbleParam"] = 1.0
        for k in ("Flag_Sfr", "Flag_Cooling", "Flag_StellarAge", "Flag_Metals", "Flag_Feedback"): h.attrs[k] = 0
        g = f.create_group("PartType4")
        g["Coordinates"] = x + 0.5 * L; g["Velocities"] = v; g["Masses"] = m / 1e10
        g["ParticleIDs"] = np.arange(1, n + 1, dtype=np.uint64)


if __name__ == "__main__":
    main()
