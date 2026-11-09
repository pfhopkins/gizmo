"""Regular lattice of n^3 gas cells of 1 Msun (units pc, Msun, km/s), uniform density and temperature, a bulk velocity plus
a divergence-free shear v_y(x); no pressure gradients in the first step. Metallicity varies across the box (a gradient in
x and a modulation in y), so a wrong donor weighting of metals shows. Written as a snapshot (the run starts from it with
start flag 2, which reads the metallicity; initial conditions proper do not carry it).
cell_id(i, j, k) gives the ID of lattice cell (i, j, k). With shuffle=True the storage order is permuted (same cells)."""
import numpy as np
import h5py

N_SIDE, SPACING, V_BULK = 24, 0.66, (1.0, 0.5, -0.25)


def cell_id(i, j, k, n=N_SIDE):
    return 1 + (i * n + j) * n + k


def make_single_stars_collect_gas_ics(fname, n=N_SIDE, shuffle=False):
    L = n * SPACING
    idx = np.arange(n)
    I, J, K = [a.ravel() for a in np.meshgrid(idx, idx, idx, indexing="ij")]
    pos = (np.stack([I, J, K], axis=1) + 0.5) * SPACING
    ids = (1 + (I * n + J) * n + K).astype(np.uint32)
    N = len(ids)
    vel = np.tile(V_BULK, (N, 1))
    vel[:, 1] += 0.3 * np.sin(2 * np.pi * (I + 0.5) / n)   # shear: depends on x only
    order = np.random.default_rng(7).permutation(N) if shuffle else np.arange(N)
    with h5py.File(fname, "w") as F:
        h = F.create_group("Header")
        npart = np.array([N, 0, 0, 0, 0, 0])
        h.attrs["NumPart_ThisFile"] = npart; h.attrs["NumPart_Total"] = npart
        h.attrs["NumPart_Total_HighWord"] = np.zeros(6, dtype=np.uint32); h.attrs["MassTable"] = np.zeros(6)
        h.attrs["Time"] = 0.0; h.attrs["Redshift"] = 0.0; h.attrs["BoxSize"] = L; h.attrs["NumFilesPerSnapshot"] = 1
        h.attrs["Omega0"] = h.attrs["OmegaLambda"] = h.attrs["HubbleParam"] = 0.0
        h.attrs["Flag_Sfr"] = 1; h.attrs["Flag_Cooling"] = 0; h.attrs["Flag_StellarAge"] = 1; h.attrs["Flag_Metals"] = 1
        h.attrs["Flag_Feedback"] = 0; h.attrs["Flag_DoublePrecision"] = 1
        p = F.create_group("PartType0")
        p.create_dataset("Coordinates", data=pos[order])
        p.create_dataset("Velocities", data=vel[order])
        p.create_dataset("ParticleIDs", data=ids[order])
        p.create_dataset("Masses", data=np.ones(N)[order])
        p.create_dataset("InternalEnergy", data=np.full(N, 1.0)[order])
        Z = 0.014 * (0.2 + 1.6 * (I + 0.5) / n) * (1.0 + 0.25 * np.cos(2 * np.pi * (J + 0.5) / n))
        p.create_dataset("Metallicity", data=Z[order].reshape(N, 1))
        p.create_dataset("SmoothingLength", data=np.full(N, 2.0 * SPACING))  # a start guess (start flag 2 reads it)
    return L
