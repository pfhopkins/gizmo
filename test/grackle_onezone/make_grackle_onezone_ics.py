"""Initial conditions for the grackle_onezone test: a uniform, static, periodic gas box on a cubic lattice.
Every cell is then the same one-zone thermochemistry problem. Units are the GIZMO defaults (kpc, 1e10 Msun, km/s, h=1)."""

import numpy as np
import h5py

UNIT_LENGTH, UNIT_MASS, UNIT_VEL = 3.085678e21, 1.989e43, 1e5
X_H, M_H, K_B, GAMMA = 0.76, 1.6726e-24, 1.380649e-16, 5.0 / 3.0


def make_grackle_onezone_ics(output_file, n_per_side=8, box_size=0.01, nH=10.0, temperature=1e6, mu=0.6):
    """Write the IC file; returns (density, specific internal energy) in code units. mu only sets u from temperature."""
    rho = nH * M_H / X_H / (UNIT_MASS / UNIT_LENGTH**3)
    u = K_B * temperature / ((GAMMA - 1) * mu * M_H) / UNIT_VEL**2
    N = n_per_side**3
    g = (np.arange(n_per_side) + 0.5) * box_size / n_per_side
    pos = np.array(np.meshgrid(g, g, g, indexing="ij")).reshape(3, -1).T
    with h5py.File(output_file, "w") as f:
        h = f.create_group("Header")
        npart = np.array([N, 0, 0, 0, 0, 0], dtype=np.int32)
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
        h.attrs["Flag_Cooling"] = 1
        h.attrs["Flag_StellarAge"] = 0
        h.attrs["Flag_Metals"] = 0
        h.attrs["Flag_Feedback"] = 0
        h.attrs["Flag_DoublePrecision"] = 0  # data are float64; HDF5 converts on read for single-precision input builds
        p = f.create_group("PartType0")
        p.create_dataset("Coordinates", data=pos.astype(np.float64))
        p.create_dataset("Velocities", data=np.zeros((N, 3)))
        p.create_dataset("Masses", data=np.full(N, rho * box_size**3 / N))
        p.create_dataset("ParticleIDs", data=np.arange(1, N + 1, dtype=np.uint32))
        p.create_dataset("InternalEnergy", data=np.full(N, u))
    return rho, u
