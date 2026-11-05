"""Falling-Sedov IC: N^3 lattice, rho=1, L=1, cold u0, one hot cell with energy E. IDs >= 1 (the RT field skips ID 0)."""
import numpy as np
import h5py


def make_wakeup_falling_sedov_ics(fname="wakeup_falling_sedov_ics.hdf5", N=32, L=1.0, E=1.0, u0=1e-6):
    dx, m = L / N, L**3 / N**3
    ix, iy, iz = (a.ravel() for a in np.meshgrid(np.arange(N), np.arange(N), np.arange(N), indexing="ij"))
    pos = (np.stack([ix, iy, iz], 1) + 0.5) * dx
    ids = (1 + ix * N * N + iy * N + iz).astype(np.uint32)
    u = np.full(N**3, u0)
    u[(ix == N // 2) & (iy == N // 2) & (iz == N // 2)] = E / m
    with h5py.File(fname, "w") as f:
        h = f.create_group("Header")
        npart = np.array([N**3, 0, 0, 0, 0, 0], dtype=np.uint32)
        h.attrs.update({"NumPart_ThisFile": npart, "NumPart_Total": npart, "NumPart_Total_HighWord": np.zeros(6, np.uint32),
                        "MassTable": np.zeros(6), "Time": 0.0, "Redshift": 0.0, "BoxSize": L, "NumFilesPerSnapshot": 1,
                        "Omega0": 0.0, "OmegaLambda": 0.0, "HubbleParam": 1.0, "Flag_Sfr": 0, "Flag_Cooling": 0,
                        "Flag_StellarAge": 0, "Flag_Metals": 0, "Flag_Feedback": 0, "Flag_DoublePrecision": 1})
        g = f.create_group("PartType0")
        g["Coordinates"] = pos
        g["Velocities"] = np.zeros_like(pos)
        g["ParticleIDs"] = ids
        g["Masses"] = np.full(N**3, m)
        g["InternalEnergy"] = u


if __name__ == "__main__":
    make_wakeup_falling_sedov_ics()
