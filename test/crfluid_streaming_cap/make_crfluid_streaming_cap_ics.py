"""Uniform periodic gas (units pc, Msun, km/s), n_H about 0.03 cm^-3, 300 microG along x: Alfven speed about 3000 km/s, far above the
300 km/s CR speed. CR energy with a sinusoidal gradient along the field. Written as a snapshot (start flag 2)."""
import numpy as np
import h5py

N, L = 24, 24.0


def make_crfluid_streaming_cap_ics(fname):
    dx = L / N
    g = (np.arange(N) + 0.5) * dx
    pos = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    n = len(pos)
    m = 1e-3 * dx ** 3
    u = 1.5 * 1.380649e-16 * 1e4 / (1.27 * 1.6726e-24) / 1e10
    b = np.zeros((n, 3))
    b[:, 0] = 3e-4
    e_cr = 0.5 * m * u * (1 + 0.5 * np.sin(2 * np.pi * pos[:, 0] / L))
    with h5py.File(fname, "w") as F:
        h = F.create_group("Header")
        npart = np.array([n, 0, 0, 0, 0, 0])
        h.attrs["NumPart_ThisFile"] = npart; h.attrs["NumPart_Total"] = npart
        h.attrs["NumPart_Total_HighWord"] = np.zeros(6, dtype=np.uint32); h.attrs["MassTable"] = np.zeros(6)
        h.attrs["Time"] = 0.0; h.attrs["Redshift"] = 0.0; h.attrs["BoxSize"] = L; h.attrs["NumFilesPerSnapshot"] = 1
        h.attrs["Omega0"] = h.attrs["OmegaLambda"] = h.attrs["HubbleParam"] = 0.0
        h.attrs["Flag_Sfr"] = 0; h.attrs["Flag_Cooling"] = 0; h.attrs["Flag_StellarAge"] = 0; h.attrs["Flag_Metals"] = 0
        h.attrs["Flag_Feedback"] = 0; h.attrs["Flag_DoublePrecision"] = 1
        p = F.create_group("PartType0")
        p["Coordinates"] = pos; p["Velocities"] = np.zeros((n, 3)); p["ParticleIDs"] = np.arange(1, n + 1, dtype=np.uint32)
        p["Masses"] = np.full(n, m); p["InternalEnergy"] = np.full(n, u); p["MagneticField"] = b; p["CosmicRayEnergy"] = e_cr
        p["SmoothingLength"] = p["KernelMaxRadius"] = np.full(n, 2.0 * dx)
    return n
