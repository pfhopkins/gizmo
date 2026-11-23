"""8 clumps of 4^3 gas cells (1 Msun, spacing 0.3 pc) on a 2x2x2 grid of centres in a 5.28 pc periodic box (pc, Msun, km/s), so FOF
(LINKLENGTH 0.0026 gives a linking length of 0.40 pc here: the DM density is tiny without cosmology) finds 8 groups of 64, none collective on 4 ranks (limit 0.6 N/ntask = 86), and SUBFIND moves
the particles of groups 5-8 between ranks; plus 64 single stars at random positions moving at 20-60 km/s with a drawn mass and a birth metallicity."""
import numpy as np
import h5py

N_STARS, L, DX = 64, 5.28, 0.3


def make_single_star_record_fof_ics(fname):
    c = np.array([1.32, 3.96]); centers = np.array([[a, b, d] for a in c for b in c for d in c])
    idx = np.arange(4) - 1.5; I, J, K = [x.ravel() for x in np.meshgrid(idx, idx, idx, indexing="ij")]
    off = np.stack([I, J, K], 1) * DX
    pos = np.concatenate([ctr + off for ctr in centers]); N = len(pos)
    rng = np.random.default_rng(3)
    spos = rng.uniform(0, L, (N_STARS, 3))
    d = rng.normal(size=(N_STARS, 3)); svel = d / np.linalg.norm(d, axis=1)[:, None] * rng.uniform(20, 60, (N_STARS, 1))
    with h5py.File(fname, "w") as F:
        h = F.create_group("Header"); npart = np.array([N, 0, 0, 0, N_STARS, 0])
        h.attrs["NumPart_ThisFile"] = npart; h.attrs["NumPart_Total"] = npart; h.attrs["NumPart_Total_HighWord"] = np.zeros(6, dtype=np.uint32)
        h.attrs["MassTable"] = np.zeros(6); h.attrs["Time"] = 0.0; h.attrs["Redshift"] = 0.0; h.attrs["BoxSize"] = L
        h.attrs["NumFilesPerSnapshot"] = 1; h.attrs["Omega0"] = h.attrs["OmegaLambda"] = h.attrs["HubbleParam"] = 0.0
        for k in ("Flag_Sfr", "Flag_Cooling", "Flag_StellarAge", "Flag_Metals", "Flag_Feedback"): h.attrs[k] = 0
        h.attrs["Flag_DoublePrecision"] = 1
        p = F.create_group("PartType0"); p["Coordinates"] = pos; p["Velocities"] = np.zeros_like(pos)
        p["ParticleIDs"] = np.arange(1, N + 1, dtype=np.uint32); p["Masses"] = np.ones(N); p["InternalEnergy"] = np.ones(N)
        s = F.create_group("PartType4"); s["Coordinates"] = spos; s["Velocities"] = svel
        s["ParticleIDs"] = np.arange(N + 1, N + N_STARS + 1, dtype=np.uint32); s["Masses"] = rng.uniform(0.1, 50, N_STARS)
        s["StellarFormationTime"] = np.zeros(N_STARS)
        s["SingleStarDrawnMass"] = rng.uniform(0.08, 150, N_STARS); s["SingleStarBirthMetallicity"] = rng.uniform(1e-4, 0.03, N_STARS)
    return L
