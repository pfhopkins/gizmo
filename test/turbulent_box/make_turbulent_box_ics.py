"""Uniform, static, isothermal gas on an N^3 lattice in a periodic box, for the turbulent_box problem.

Units are pc - km/s - Msun. The total mass is set so the virial parameter at the target Mach number
is alpha_vir, using the box convention alpha_vir = 5 sigma_3D^2 L / (6 G M) (Federrath & Klessen 2012),
i.e. the Bertoldi & McKee sphere value with R = L/2 and sigma_1D^2 = sigma_3D^2 / 3. Defaults give
M = 3875 Msun. A uniform B_z gives plasma beta `beta` (used only by MAGNETIC builds).
G comes from gizmo.units, which mirrors GIZMO's constants.h.
"""
import argparse
import os
import sys

import h5py
import numpy as np

# Run standalone, so the harness package (test/harness in this tree; python_src upstream) is not on
# the path the way it is under pytest.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "harness"))
from gizmo.units import G_CODE, UNIT_LENGTH_IN_CM, UNIT_MASS_IN_G, UNIT_VELOCITY_IN_CM_PER_S  # noqa: E402

GAMMA = 1.001
CS = 0.2  # isothermal sound speed, km/s: EOS_ENFORCE_ADIABAT = CS^2

UNIT_PRESSURE_IN_CGS = UNIT_MASS_IN_G * UNIT_VELOCITY_IN_CM_PER_S**2 / UNIT_LENGTH_IN_CM**3
UNIT_B_IN_GAUSS = np.sqrt(4 * np.pi * UNIT_PRESSURE_IN_CGS)  # GIZMO's internal field unit


def b_field_gauss(rho, beta):
    """Field strength in gauss giving plasma beta = P / (B^2/2) [code units] at density rho."""
    return np.sqrt(2 * CS**2 * rho**GAMMA / beta) * UNIT_B_IN_GAUSS


def write_ics(out, pos, mass, box_size, bz_gauss):
    """Gas at rest, isothermal, with a uniform B_z. Builds without MAGNETIC ignore the field."""
    ngas = len(pos)
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
        for flag in ("Flag_Sfr", "Flag_Cooling", "Flag_StellarAge", "Flag_Metals", "Flag_Feedback", "Flag_DoublePrecision"):
            h.attrs[flag] = 0
        g = f.create_group("PartType0")
        g.create_dataset("Coordinates", data=pos)
        g.create_dataset("Velocities", data=np.zeros((ngas, 3), dtype=np.float32))
        g.create_dataset("Masses", data=np.full(ngas, mass, dtype=np.float32))
        g.create_dataset("ParticleIDs", data=np.arange(1, ngas + 1, dtype=np.uint32))
        # P/rho = CS^2 up to rho^(GAMMA-1); EOS_ENFORCE_ADIABAT resets it anyway
        g.create_dataset("InternalEnergy", data=np.full(ngas, CS**2 / (GAMMA - 1), dtype=np.float32))
        g.create_dataset("MagneticField", data=np.tile([0, 0, bz_gauss], (ngas, 1)).astype(np.float32))


def make_ics(out, n=32, box_size=10.0, mach=10.0, alpha_vir=2.0, beta=1.0):
    sigma = mach * CS
    mtot = 5 * sigma**2 * box_size / (6 * G_CODE * alpha_vir)
    x = (np.arange(n) + 0.5) * box_size / n
    pos = np.stack(np.meshgrid(x, x, x, indexing="ij"), axis=-1).reshape(-1, 3)
    bz = b_field_gauss(mtot / box_size**3, beta)
    write_ics(out, pos, mtot / len(pos), box_size, bz)
    return mtot, bz


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--box_size", type=float, default=10.0)
    ap.add_argument("--mach", type=float, default=10.0)
    ap.add_argument("--alpha_vir", type=float, default=2.0)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--out", default="turbulent_box_ics.hdf5")
    a = ap.parse_args()
    m, bz = make_ics(a.out, a.n, a.box_size, a.mach, a.alpha_vir, a.beta)
    print(f"wrote {a.out}: {a.n}^3 cells, M = {m:.6g} Msun, mean density = {m / a.box_size**3:.6g} Msun/pc^3, "
          f"B_z = {bz * 1e6:.4g} uG (beta = {a.beta:g})")
