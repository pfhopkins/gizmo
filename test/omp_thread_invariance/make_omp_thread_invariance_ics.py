"""Generate the omp_thread_invariance IC: a magnetized 3D periodic box with a central
over-pressured region, a smooth shearing flow, and NUM_METAL_SPECIES passive scalars with
gradients (pairwise identical between the two halves of the species list). The blast spreads the cells over several timebins and its shock wakes
up cells ahead of it, the shear switches on turbulent metal diffusion everywhere, and every
metal species carries its own flux.

Code units are arbitrary (G does not enter: SELFGRAVITY_OFF). Heaviside-Lorentz B, so
P_mag = B^2/2.
"""

import numpy as np
import h5py

GAMMA = 5.0 / 3.0


def make_IC(N_1D=16, n_species=14, fname="omp_thread_invariance_ics.hdf5", seed=42):
    """n_species must equal NUM_METAL_SPECIES of the build (1 + STARFORGE_FEEDBACK_TRACERS here)."""
    rng = np.random.default_rng(seed)
    L = 1.0
    dx = L / N_1D
    x1 = (np.arange(N_1D) + 0.5) * dx
    x, y, z = [a.flatten() for a in np.meshgrid(x1, x1, x1, indexing="ij")]
    # jitter the lattice so no two neighbour distances tie exactly
    x = (x + rng.uniform(-0.1, 0.1, x.size) * dx) % L
    y = (y + rng.uniform(-0.1, 0.1, y.size) * dx) % L
    z = (z + rng.uniform(-0.1, 0.1, z.size) * dx) % L
    N = x.size
    pos = np.column_stack([x, y, z])

    mass = np.full(N, 1.0 / N)  # rho = 1

    r = np.sqrt(np.sum((pos - 0.5 * L) ** 2, axis=1))
    pressure = np.where(r < 0.15, 10.0, 0.01)
    u = pressure / ((GAMMA - 1.0) * 1.0)

    # ABC-like shear flow at ~0.4 of the ambient sound speed
    A = 0.05
    k = 2 * np.pi / L
    vel = A * np.column_stack([np.sin(k * y), np.sin(k * z), np.sin(k * x)])

    # plasma beta ~0.9 in the ambient medium, field oblique to the lattice
    B0 = 0.15
    B = np.tile(B0 * np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0), (N, 1))

    # periodic gradients, all positive; species s and s + n_species//2 start identical, so any
    # per-species indexing error shows up as a pair that diverges
    half = n_species // 2
    Z = np.empty((N, n_species))
    for s in range(n_species):
        p = s % half
        n_vec = np.array([1 + p % 3, (p // 3) % 3, (p // 9) % 2 + 1])
        phase = 2 * np.pi * p / half
        Z[:, s] = 0.02 * (1.0 + 0.5 * np.sin(k * pos @ n_vec + phase))

    with h5py.File(fname, "w") as F:
        h = F.create_group("Header")
        h.attrs["NumPart_ThisFile"] = np.array([N, 0, 0, 0, 0, 0], dtype=np.uint32)
        h.attrs["NumPart_Total"] = np.array([N, 0, 0, 0, 0, 0], dtype=np.uint32)
        h.attrs["NumPart_Total_HighWord"] = np.zeros(6, dtype=np.uint32)
        h.attrs["MassTable"] = np.zeros(6)
        h.attrs["Time"] = 0.0
        h.attrs["Redshift"] = 0.0
        h.attrs["BoxSize"] = L
        h.attrs["NumFilesPerSnapshot"] = 1
        h.attrs["Omega0"] = 0.0
        h.attrs["OmegaLambda"] = 0.0
        h.attrs["HubbleParam"] = 1.0
        h.attrs["Flag_Sfr"] = 0
        h.attrs["Flag_Cooling"] = 0
        h.attrs["Flag_StellarAge"] = 0
        h.attrs["Flag_Metals"] = n_species
        h.attrs["Flag_Feedback"] = 0
        h.attrs["Flag_DoublePrecision"] = 1

        p = F.create_group("PartType0")
        p.create_dataset("Coordinates", data=pos)
        p.create_dataset("Velocities", data=vel)
        p.create_dataset("ParticleIDs", data=np.arange(1, N + 1, dtype=np.uint64))
        p.create_dataset("Masses", data=mass)
        p.create_dataset("InternalEnergy", data=u)
        p.create_dataset("MagneticField", data=B)
        p.create_dataset("Metallicity", data=Z)
        p.create_dataset("KernelMaxRadius", data=np.full(N, 2.0 * dx))  # initial guess: read as a snapshot (restart flag 2)
    return fname


if __name__ == "__main__":
    make_IC()
