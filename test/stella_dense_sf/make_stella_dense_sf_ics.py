"""A dense cold clump (n_H = 1e6 cm^-3, 10 K, 500 cells) at the centre of a 16 pc periodic box of diffuse warm gas
(n_H = 1 cm^-3, 1e4 K), all cells 1 Msun, solar metallicity. The clump forms stars on short steps while the diffuse
gas sets a long top-level step."""
import h5py
import numpy as np

MSUN, M_P, K_B, PC = 1.989e33, 1.6726e-24, 1.3807e-16, 3.085678e18
BOX_PC, N_CLUMP, N_BG, T_CLUMP, T_BG, N_CELLS_CLUMP, X_H = 16.0, 1.0e6, 1.0, 10.0, 1.0e4, 500, 0.7


def lattice(spacing, lo, hi):
    g = np.arange(lo + 0.5 * spacing, hi, spacing)
    return np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)


def make_stella_dense_sf_ics(fname):
    rho_c = N_CLUMP * M_P / X_H / MSUN * PC**3  # Msun / pc^3
    rho_b = N_BG * M_P / X_H / MSUN * PC**3
    m_cell = 1.0
    r_c = (3 * N_CELLS_CLUMP * m_cell / (4 * np.pi * rho_c)) ** (1.0 / 3.0)
    c = 0.5 * BOX_PC
    pc = lattice((m_cell / rho_c) ** (1.0 / 3.0), c - r_c, c + r_c)
    pc = pc[np.sum((pc - c) ** 2, axis=1) < r_c**2]
    pb = lattice(BOX_PC / round(BOX_PC / (m_cell / rho_b) ** (1.0 / 3.0)), 0.0, BOX_PC)
    pb = pb[np.sum((pb - c) ** 2, axis=1) > (r_c + 0.5 * (m_cell / rho_b) ** (1.0 / 3.0)) ** 2]
    pos = np.vstack([pc, pb]) * 1e-3
    n, nc = len(pos), len(pc)
    u = np.empty(n, dtype=np.float32)
    u[:nc] = 1.5 * K_B * T_CLUMP / (2.3 * M_P) / 1e10
    u[nc:] = 1.5 * K_B * T_BG / (0.6 * M_P) / 1e10
    with h5py.File(fname, "w") as f:
        h = f.create_group("Header")
        h.attrs["BoxSize"] = BOX_PC * 1e-3
        h.attrs["NumPart_ThisFile"] = np.array([n, 0, 0, 0, 0, 0], dtype=np.int32)
        h.attrs["NumPart_Total"] = np.array([n, 0, 0, 0, 0, 0], dtype=np.uint32)
        h.attrs["NumPart_Total_HighWord"] = np.zeros(6, dtype=np.uint32)
        h.attrs["MassTable"] = np.zeros(6)
        h.attrs["Time"] = 0.0
        h.attrs["Redshift"] = 0.0
        h.attrs["NumFilesPerSnapshot"] = 1
        h.attrs["Flag_Metals"] = 0
        h.attrs["Flag_DoublePrecision"] = 0
        for k in ("Flag_Sfr", "Flag_Cooling", "Flag_StellarAge", "Flag_Feedback"):
            h.attrs[k] = 1
        p = f.create_group("PartType0")
        p["Coordinates"] = pos.astype(np.float32)
        p["Velocities"] = np.zeros((n, 3), dtype=np.float32)
        p["Masses"] = np.full(n, m_cell * MSUN / 1.989e43, dtype=np.float32)
        p["ParticleIDs"] = np.arange(1, n + 1, dtype=np.uint32)
        p["InternalEnergy"] = u
        p["MagneticField"] = np.zeros((n, 3), dtype=np.float32)
    return nc


if __name__ == "__main__":
    import sys
    print(make_stella_dense_sf_ics(sys.argv[1] if len(sys.argv) > 1 else "stella_dense_sf_ics.hdf5"))
