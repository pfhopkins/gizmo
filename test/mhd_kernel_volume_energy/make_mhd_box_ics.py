"""ICs of the static magnetised box (test/mhd_kernel_volume_energy, test/mhd_energy_ledger): 20^3 cells with 20 % lattice
jitter in a 70 pc periodic box, 2400 Msun, u = 77.84 (km/s)^2, at rest (code units kpc, 1e10 Msun, km/s)."""
import numpy as np
import h5py

L, N_SIDE, M_TOT, U0 = 0.07, 20, 2400e-10, 77.84


def write_ics(fn):
    rng = np.random.default_rng(1)
    d = L / N_SIDE
    g = (np.arange(N_SIDE) + 0.5) * d
    x = np.array(np.meshgrid(g, g, g, indexing="ij")).reshape(3, -1).T
    x = (x + rng.normal(0, 0.2 * d, x.shape)) % L
    n = len(x)
    with h5py.File(fn, "w") as f:
        h = f.create_group("Header").attrs
        h["NumPart_ThisFile"] = np.array([n, 0, 0, 0, 0, 0], dtype=np.int32)
        h["NumPart_Total"] = np.array([n, 0, 0, 0, 0, 0], dtype=np.uint32)
        h["NumPart_Total_HighWord"] = np.zeros(6, dtype=np.uint32)
        h["MassTable"] = np.zeros(6); h["Time"] = 0.0; h["Redshift"] = 0.0; h["BoxSize"] = L; h["NumFilesPerSnapshot"] = 1
        h["Omega0"] = 0.0; h["OmegaLambda"] = 0.0; h["HubbleParam"] = 1.0
        for k in ("Flag_Cooling", "Flag_Feedback", "Flag_Metals", "Flag_Sfr", "Flag_StellarAge"):
            h[k] = 0
        p = f.create_group("PartType0")
        p["Coordinates"] = x; p["Velocities"] = np.zeros_like(x); p["Masses"] = np.full(n, M_TOT / n)
        p["InternalEnergy"] = np.full(n, U0); p["ParticleIDs"] = np.arange(1, n + 1, dtype=np.uint32)
