"""Uniform 20 pc gas lattice of 1 Msun cells (n_H = 10 cm^-3, 1e4 K, 0.1 Zsun); the cell nearest the centre has ID 1 and becomes the star."""
import h5py
import numpy as np

KPC, MSUN, M_P, K_B = 3.085678e21, 1.989e33, 1.6726e-24, 1.3807e-16
BOX_PC, N_H, T_GAS, Z_SUN, M_CELL = 20.0, 10.0, 1.0e4, 0.1, 1.0


def make_stella_fb_wakeup_ics(fname):
    x_z = Z_SUN * 0.014
    x_h = 1.0 - 0.275 - x_z
    m_gas = N_H * (BOX_PC * 3.085678e18) ** 3 * M_P / x_h / MSUN
    ng = int(round((m_gas / M_CELL) ** (1.0 / 3.0)))
    n = ng**3
    box = BOX_PC * 1e-3
    g = (np.arange(ng) + 0.5) * box / ng
    pos = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    ids = np.arange(n, dtype=np.uint32) + 2
    ids[np.argmin(np.sum((pos - 0.5 * box) ** 2, axis=1))] = 1
    ea = np.zeros((n, 27), dtype=np.float32)
    ea[:, 1] = 0.275
    for k, x in {2: 2.53e-3, 3: 7.41e-4, 4: 6.13e-3, 5: 1.34e-3, 6: 7.57e-4, 7: 7.12e-4, 8: 3.31e-4, 9: 6.87e-5, 10: 1.38e-3}.items():
        ea[:, k] = x * Z_SUN
    ea[:, 0] = 1.0 - ea.sum(axis=1)
    u = 1.5 * K_B * T_GAS / (0.6 * M_P) / 1e10
    with h5py.File(fname, "w") as f:
        h = f.create_group("Header")
        h.attrs["BoxSize"] = box
        h.attrs["NumPart_ThisFile"] = np.array([n, 0, 0, 0, 0, 0], dtype=np.int32)
        h.attrs["NumPart_Total"] = np.array([n, 0, 0, 0, 0, 0], dtype=np.uint32)
        h.attrs["NumPart_Total_HighWord"] = np.zeros(6, dtype=np.uint32)
        h.attrs["MassTable"] = np.zeros(6)
        h.attrs["Time"] = 0.0
        h.attrs["Redshift"] = 0.0
        h.attrs["NumFilesPerSnapshot"] = 1
        h.attrs["Flag_Metals"] = 27
        h.attrs["Flag_DoublePrecision"] = 0
        for k in ("Flag_Sfr", "Flag_Cooling", "Flag_StellarAge", "Flag_Feedback"):
            h.attrs[k] = 1
        p = f.create_group("PartType0")
        p["Coordinates"] = pos
        p["Velocities"] = np.zeros((n, 3), dtype=np.float32)
        p["Masses"] = np.full(n, m_gas / n * MSUN / 1.989e43, dtype=np.float32)
        p["ParticleIDs"] = ids
        p["InternalEnergy"] = np.full(n, u, dtype=np.float32)
        p["ElementAbundance"] = ea
        p["MagneticField"] = np.zeros((n, 3), dtype=np.float32)


if __name__ == "__main__":
    import sys
    make_stella_fb_wakeup_ics(sys.argv[1] if len(sys.argv) > 1 else "stella_fb_wakeup_ics.hdf5")
