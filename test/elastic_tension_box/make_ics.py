"""ICs for test/elastic_tension_box: a periodic 2D box (side 2) of the ring material (Tillotson a = b = 0, A = 1, rho_0 = 1,
shear modulus 0.22, yield strength 0.005) on a 40 x 40 square lattice, positions jittered by up to 1% of the spacing
(fixed seed), at rest, with masses that put it at density 0.95: a uniform isotropic stretch (about 2.6% in each direction), pressure -0.05 (ten
times the yield strength) and no deviatoric stress. Writes elastic_tension_box_ics.hdf5 and
elastic_tension_box_preyielded_ics.hdf5 (the same cells with RecentYieldMemory = 0.88, as if they had just flowed
plastically) in the working directory."""
import numpy as np
import h5py

N_SIDE, L, STRAIN, JITTER, SEED, MEMORY_PRE = 40, 2.0, 0.05, 0.01, 4242, 0.88

def write(path, memory=None):
    dp = L / N_SIDE
    g1 = (np.arange(N_SIDE) + 0.5) * dp
    X, Y = np.meshgrid(g1, g1, indexing="ij")
    pos = np.c_[X.ravel(), Y.ravel(), np.zeros(N_SIDE * N_SIDE)]
    pos[:, :2] += JITTER * dp * np.random.default_rng(SEED).uniform(-1, 1, (N_SIDE * N_SIDE, 2))
    pos[:, :2] %= L
    N = len(pos)
    with h5py.File(path, "w") as f:
        h = f.create_group("Header")
        for key, val in [("NumPart_ThisFile", np.array([N, 0, 0, 0, 0, 0], np.int32)), ("NumPart_Total", np.array([N, 0, 0, 0, 0, 0], np.uint32)),
                         ("NumPart_Total_HighWord", np.zeros(6, np.uint32)), ("MassTable", np.zeros(6)), ("Time", 0.0), ("Redshift", 0.0),
                         ("BoxSize", L), ("NumFilesPerSnapshot", 1), ("Omega0", 0.0), ("OmegaLambda", 0.0), ("HubbleParam", 1.0),
                         ("Flag_DoublePrecision", 1)]:
            h.attrs[key] = val
        for key in ["Flag_Sfr", "Flag_Feedback", "Flag_Cooling", "Flag_StellarAge", "Flag_Metals"]:
            h.attrs[key] = 0
        g = f.create_group("PartType0")
        g["Coordinates"] = pos
        g["Velocities"] = np.zeros_like(pos)
        g["Masses"] = np.full(N, (1 - STRAIN) * dp * dp)
        g["InternalEnergy"] = np.ones(N)
        g["ParticleIDs"] = np.arange(1, N + 1, dtype=np.uint32)
        g["CompositionType"] = np.zeros(N, np.int32)
        if memory is not None:
            g["RecentYieldMemory"] = np.full(N, memory)

def main():
    write("elastic_tension_box_ics.hdf5")
    write("elastic_tension_box_preyielded_ics.hdf5", MEMORY_PRE)

if __name__ == "__main__":
    main()
