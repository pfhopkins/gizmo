#!/usr/bin/env python3
"""ICs for the shearing_box test: a uniform N^3 lattice in a unit 3D shearing box (BOX_SHEARING=3,
coordinates (r, phi, z), Omega = 1) with density 1, sound speed 1 and the equilibrium shear flow
v_phi = -q Omega X, X = x - L/2. On top of the shear:
  vr0 > 0: a uniform radial velocity (epicyclic oscillation);
  dv > 0:  a vortical shearing wave, velocity dv * (-ky, kx0, 0)/|k0| * cos(kx0 X + ky y), zero density perturbation.
"""

import numpy as np
import h5py

GAMMA = 1.000001  # EOS_GAMMA in Config.sh
CS = 1.0
L = 1.0


def make_shearing_box_ics(output_file, q, N=16, vr0=0.0, dv=0.0, kx0=0.0, ky=0.0):
    g = (np.arange(N) + 0.5) * L / N
    x = np.array(np.meshgrid(g, g, g, indexing="ij")).reshape(3, -1).T
    X = x[:, 0] - 0.5 * L
    v = np.zeros_like(x)
    v[:, 0] = vr0
    v[:, 1] = -q * X
    if dv:
        k0 = np.hypot(kx0, ky)
        c = np.cos(kx0 * X + ky * x[:, 1])
        v[:, 0] += -dv * ky / k0 * c
        v[:, 1] += dv * kx0 / k0 * c
    n = len(x)
    with h5py.File(output_file, "w") as f:
        h = f.create_group("Header")
        npart = np.array([n, 0, 0, 0, 0, 0], dtype=np.int32)
        h.attrs["NumPart_ThisFile"] = npart
        h.attrs["NumPart_Total"] = npart.astype(np.uint32)
        h.attrs["NumPart_Total_HighWord"] = np.zeros(6, dtype=np.uint32)
        h.attrs["MassTable"] = np.zeros(6)
        h.attrs["Time"] = 0.0
        h.attrs["Redshift"] = 0.0
        h.attrs["BoxSize"] = L
        h.attrs["NumFilesPerSnapshot"] = 1
        h.attrs["Flag_DoublePrecision"] = 1
        p = f.create_group("PartType0")
        p["Coordinates"] = x
        p["Velocities"] = v
        p["Masses"] = np.full(n, L**3 / n)
        p["InternalEnergy"] = np.full(n, CS**2 / (GAMMA * (GAMMA - 1)))
        p["ParticleIDs"] = np.arange(1, n + 1, dtype=np.uint32)
