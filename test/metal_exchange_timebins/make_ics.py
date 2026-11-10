"""Metal box for the METALS_EXACT_PAIRWISE_EXCHANGE check: cold gas for x < L/2, hot gas for x > L/2 at equal
density, so neighbouring cells sit in different timebins and the pressure jump drives the flow and shear.
Five metal species: 0 a step (0.1 | 60 Zsun), 1 the reverse step, 2 a step into exactly zero, 3 smooth along y,
4 a per-cell random field spanning four decades with exact zeros (local extrema everywhere)."""

import h5py
import numpy as np

Z_SOLAR = 0.0142
N_SPECIES = 5


def make_ics(filename, n_side=20, box=0.5, T_cold=1e3, T_hot=1e7, seed=42):
    """Writes the IC as snapshot '<name>_000.hdf5'; returns the per-species metal mass."""
    rng = np.random.default_rng(seed)
    g = (np.arange(n_side) + 0.5) / n_side * box
    x = np.array(np.meshgrid(g, g, g, indexing="ij")).reshape(3, -1).T
    x = (x + rng.uniform(-0.05, 0.05, x.shape) * box / n_side) % box
    n = len(x)
    hot = x[:, 0] > 0.5 * box
    u = np.where(hot, T_hot, T_cold) * (206.58684 / 1e4)  # (km/s)^2 per K, mu ~ 0.6
    Z = np.zeros((n, N_SPECIES))
    Z[:, 0] = np.where(hot, 60.0, 0.1) * Z_SOLAR
    Z[:, 1] = np.where(hot, 1e-3, 1.0) * Z_SOLAR
    Z[:, 2] = np.where(hot, 1.0, 0.0) * Z_SOLAR
    Z[:, 3] = (1.0 + 0.5 * np.sin(2 * np.pi * x[:, 1] / box)) * Z_SOLAR
    Z[:, 4] = 10 ** rng.uniform(-4, 0, n) * Z_SOLAR * (rng.uniform(0, 1, n) > 0.1)
    m = np.full(n, 1e-10)
    h = (3 * 32 / (4 * np.pi)) ** (1 / 3) * box / n_side
    with h5py.File(filename, "w") as f:
        hd = f.create_group("Header")
        hd.attrs["NumPart_ThisFile"] = np.array([n, 0, 0, 0, 0, 0], dtype=np.uint32)
        hd.attrs["NumPart_Total"] = np.array([n, 0, 0, 0, 0, 0], dtype=np.uint32)
        hd.attrs["NumPart_Total_HighWord"] = np.zeros(6, dtype=np.uint32)
        hd.attrs["MassTable"] = np.zeros(6)
        hd.attrs["Time"] = 0.0
        hd.attrs["Redshift"] = 0.0
        hd.attrs["BoxSize"] = box
        hd.attrs["NumFilesPerSnapshot"] = 1
        hd.attrs["Flag_Metals"] = N_SPECIES
        hd.attrs["Flag_DoublePrecision"] = 1
        p = f.create_group("PartType0")
        p["Coordinates"] = x
        p["Velocities"] = np.zeros_like(x)
        p["ParticleIDs"] = np.arange(1, n + 1, dtype=np.uint32)
        p["Masses"] = m
        p["InternalEnergy"] = u
        p["Metallicity"] = Z
        p["KernelMaxRadius"] = np.full(n, h)
    return (m[:, None] * Z).sum(axis=0)


if __name__ == "__main__":
    import sys
    make_ics(sys.argv[1] if len(sys.argv) > 1 else "ics_000.hdf5", n_side=int(sys.argv[2]) if len(sys.argv) > 2 else 20)
