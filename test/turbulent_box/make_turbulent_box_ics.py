"""Uniform, static, isothermal gas on an N^3 lattice in a periodic box, for the turbulent_box test.

Units are pc - km/s - Msun. The total mass is set so the virial parameter at the target Mach number
is alpha_vir, using the box convention alpha_vir = 5 sigma_3D^2 L / (6 G M) (Federrath & Klessen 2012),
i.e. the Bertoldi & McKee sphere value with R = L/2 and sigma_1D^2 = sigma_3D^2 / 3. Defaults give
M = 3875 Msun. G comes from gizmo.units, which mirrors GIZMO's constants.h.
"""
import argparse
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python_src"))
from gizmo.units import G_CODE  # noqa: E402

GAMMA = 1.001


def total_mass(box_size, cs, mach, alpha_vir):
    sigma = mach * cs
    return 5 * sigma**2 * box_size / (6 * G_CODE * alpha_vir)


def make_ics(out, n=32, box_size=10.0, cs=0.2, mach=10.0, alpha_vir=2.0):
    x = (np.arange(n) + 0.5) * box_size / n
    pos = np.stack(np.meshgrid(x, x, x, indexing="ij"), axis=-1).reshape(-1, 3)
    ngas = len(pos)
    mtot = total_mass(box_size, cs, mach, alpha_vir)
    u = cs**2 / (GAMMA - 1)  # isothermal: P/rho = c_s^2. EOS_ENFORCE_ADIABAT overrides this anyway

    with h5py.File(out, "w") as f:
        npart = np.array([ngas, 0, 0, 0, 0, 0], dtype=np.int32)
        h = f.create_group("Header")
        h.attrs["NumPart_ThisFile"] = npart
        h.attrs["NumPart_Total"] = npart.astype(np.uint32)
        h.attrs["NumPart_Total_HighWord"] = np.zeros(6, dtype=np.uint32)
        h.attrs["MassTable"] = np.zeros(6)
        h.attrs["Time"] = 0.0
        h.attrs["Redshift"] = 0.0
        h.attrs["BoxSize"] = box_size
        h.attrs["NumFilesPerSnapshot"] = 1
        h.attrs["Omega0"] = 0.0
        h.attrs["OmegaLambda"] = 0.0
        h.attrs["HubbleParam"] = 1.0
        h.attrs["Flag_Sfr"] = 0
        h.attrs["Flag_Cooling"] = 0
        h.attrs["Flag_StellarAge"] = 0
        h.attrs["Flag_Metals"] = 0
        h.attrs["Flag_Feedback"] = 0
        h.attrs["Flag_DoublePrecision"] = 0
        g = f.create_group("PartType0")
        g.create_dataset("Coordinates", data=pos)
        g.create_dataset("Velocities", data=np.zeros((ngas, 3), dtype=np.float32))
        g.create_dataset("Masses", data=np.full(ngas, mtot / ngas, dtype=np.float32))
        g.create_dataset("ParticleIDs", data=np.arange(1, ngas + 1, dtype=np.uint32))
        g.create_dataset("InternalEnergy", data=np.full(ngas, u, dtype=np.float32))
    return mtot


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--box_size", type=float, default=10.0)
    ap.add_argument("--cs", type=float, default=0.2)
    ap.add_argument("--mach", type=float, default=10.0)
    ap.add_argument("--alpha_vir", type=float, default=2.0)
    ap.add_argument("--out", default="turbulent_box_ics.hdf5")
    a = ap.parse_args()
    m = make_ics(a.out, a.n, a.box_size, a.cs, a.mach, a.alpha_vir)
    print(f"wrote {a.out}: {a.n}^3 cells, M = {m:.6g} Msun, mean density = {m / a.box_size**3:.6g} Msun/pc^3")
