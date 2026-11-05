#!/usr/bin/env python3
"""ICs for strat_hse: the isothermal hydrostatic slab of scripts/tallbox/make_tallbox_ics.py (external stellar + DM
gravity and slab self-gravity), no turbulence, no hot floor, internal energy set for c_s,iso = CS under EOS_GAMMA."""

import sys
from pathlib import Path
import numpy as np
import h5py

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "tallbox"))
from make_tallbox_ics import vertical_profile, place_layers  # noqa: E402

L, LONG_Z, MGAS, CS, GAMMA = 1000.0, 8, 300.0, 10.0, 1.001  # pc, Msun, km/s
SIGMA_GAS, SIGMA_STAR, Z_STAR, RHO_DM, R0 = 10.0, 42.0, 245.0, 0.0064, 8000.0
Q, OMEGA = 1.0, 0.028


def profile():
    z = np.linspace(0, 0.5 * LONG_Z * L, 200001)
    return z, vertical_profile(z, SIGMA_GAS, SIGMA_STAR, Z_STAR, RHO_DM, R0, CS)


def make_strat_hse_ics(output_file="strat_hse_ics.hdf5", seed=3):
    rng = np.random.default_rng(seed)
    z, rho = profile()
    keep = rho > 1e-6 * rho[0]
    pos = place_layers(L, 0.5 * LONG_Z * L, z[keep], rho[keep], MGAS, rng)
    vel = np.zeros_like(pos)
    vel[:, 1] = -Q * OMEGA * (pos[:, 0] - 0.5 * L)
    with h5py.File(output_file, "w") as f:
        h = f.create_group("Header")
        npart = np.array([len(pos), 0, 0, 0, 0, 0], dtype=np.int32)
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
        p["Coordinates"] = pos.astype(np.float32)
        p["Velocities"] = vel.astype(np.float32)
        p["Masses"] = np.full(len(pos), MGAS, dtype=np.float32)
        p["InternalEnergy"] = np.full(len(pos), CS**2 / (GAMMA - 1), dtype=np.float32)
        p["ParticleIDs"] = np.arange(1, len(pos) + 1, dtype=np.uint32)


if __name__ == "__main__":
    make_strat_hse_ics()
