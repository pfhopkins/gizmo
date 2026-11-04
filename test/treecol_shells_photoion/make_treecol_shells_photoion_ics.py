"""StarBench IC (units kpc, 1e10 Msun, km/s), as test/photoion_starbench: a lattice of N^3 gas cells in a 57 pc box at
n_H ~ 100 cm^-3 (the gas is held at 1000 K), with one single star of 30 Msun drawn mass at the centre, past its
pre-main-sequence phase. N = 40: m10, 9.9 Msun cells; N = 86: m1."""
import numpy as np
import h5py

L = 0.057
M_GAS = 6.3231738e-5  # total gas mass [1e10 Msun]
RUNGS = {"m10": 40, "m1": 86}


def make_ics(fname, n):
    g = (np.arange(n) + 0.5) * L / n
    pos = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    N = len(pos)
    with h5py.File(fname, "w") as F:
        h = F.create_group("Header")
        npart = np.array([N, 0, 0, 0, 1, 0])
        h.attrs["NumPart_ThisFile"] = npart; h.attrs["NumPart_Total"] = npart
        h.attrs["NumPart_Total_HighWord"] = np.zeros(6, dtype=np.uint32); h.attrs["MassTable"] = np.zeros(6)
        h.attrs["Time"] = 0.0; h.attrs["Redshift"] = 0.0; h.attrs["BoxSize"] = L; h.attrs["NumFilesPerSnapshot"] = 1
        h.attrs["Omega0"] = h.attrs["OmegaLambda"] = h.attrs["HubbleParam"] = 0.0
        h.attrs["Flag_Sfr"] = 1; h.attrs["Flag_Cooling"] = 1; h.attrs["Flag_StellarAge"] = 1; h.attrs["Flag_Metals"] = 0
        h.attrs["Flag_Feedback"] = 1; h.attrs["Flag_DoublePrecision"] = 1
        p = F.create_group("PartType0")
        p["Coordinates"] = pos; p["Velocities"] = np.zeros((N, 3)); p["ParticleIDs"] = np.arange(1, N + 1, dtype=np.uint32)
        p["Masses"] = np.full(N, M_GAS / N, dtype=np.float32); p["InternalEnergy"] = np.full(N, 1.014936, dtype=np.float32)
        s = F.create_group("PartType4")
        s["Coordinates"] = np.full((1, 3), 0.5 * L); s["Velocities"] = np.zeros((1, 3)); s["ParticleIDs"] = np.array([N + 2], dtype=np.uint32)
        s["Masses"] = np.array([1.5e-9]); s["StellarFormationTime"] = np.array([-2e-4])  # 0.2 Myr old
        s["SingleStarDrawnMass"] = np.array([30.0]); s["SingleStarBirthMetallicity"] = np.array([0.0014]); s["SingleStarBirthDensity"] = np.array([0.0])
