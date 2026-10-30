"""Lattice of 16^3 gas cells of 1 Msun at n_H ~ 1e4 cm^-3 (pc, Msun, km/s) with a random velocity field at ~100 K:
every cell is above the density threshold of the star formation."""
import numpy as np
import h5py

N_SIDE, SPACING = 16, 0.16


def make_single_stars_formation_ics(fname):
    n = N_SIDE
    idx = np.arange(n); I, J, K = [x.ravel() for x in np.meshgrid(idx, idx, idx, indexing="ij")]
    rng = np.random.default_rng(11)
    pos = (np.stack([I, J, K], 1) + 0.5 + rng.uniform(-0.2, 0.2, (len(I), 3))) * SPACING
    N = len(I)
    with h5py.File(fname, "w") as F:
        h = F.create_group("Header"); npart = np.array([N, 0, 0, 0, 0, 0])
        h.attrs["NumPart_ThisFile"] = npart; h.attrs["NumPart_Total"] = npart; h.attrs["NumPart_Total_HighWord"] = np.zeros(6, dtype=np.uint32)
        h.attrs["MassTable"] = np.zeros(6); h.attrs["Time"] = 0.0; h.attrs["Redshift"] = 0.0; h.attrs["BoxSize"] = n * SPACING
        h.attrs["NumFilesPerSnapshot"] = 1; h.attrs["Omega0"] = h.attrs["OmegaLambda"] = h.attrs["HubbleParam"] = 0.0
        for k in ("Flag_Sfr", "Flag_Cooling", "Flag_StellarAge", "Flag_Metals", "Flag_Feedback"): h.attrs[k] = 0
        h.attrs["Flag_DoublePrecision"] = 1
        p = F.create_group("PartType0"); p["Coordinates"] = pos; p["Velocities"] = rng.normal(0, 0.5, (N, 3))
        p["ParticleIDs"] = np.arange(1, N + 1, dtype=np.uint32); p["Masses"] = np.ones(N); p["InternalEnergy"] = np.full(N, 0.01)
    return n * SPACING
