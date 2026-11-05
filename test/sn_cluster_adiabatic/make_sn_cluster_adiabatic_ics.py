#!/usr/bin/env python3
"""ICs for sn_cluster_adiabatic: a 1e3 Msun FIRE stellar population at rest at the centre of a
uniform, static, periodic box of warm gas. The cluster is split into star particles of the gas cell
mass, as in a FIRE run, sharing a fixed total number of O stars, so every resolution has the same
supernova budget.

Units: pc, Msun, km/s. The file is read with restart flag 2 (a snapshot restart), the only way to
give an IC star particle a formation time and an O-star count: a flag-0 start zeroes both, and a
FIRE-3 population with no O stars has no core-collapse supernovae.
"""

import numpy as np
import h5py

PC_IN_CM = 3.0857e18
MSUN_IN_G = 1.989e33
M_PROTON_G = 1.6726e-24
K_BOLTZMANN = 1.3807e-16
CODE_TIME_IN_MYR = (PC_IN_CM / 1.0e5) / (1.0e6 * 365.25 * 86400.0)  # pc/(km/s) ~ 0.9778 Myr

Z_SOLAR = 0.02  # GIZMO's SolarAbundances[0] without COOL_METAL_LINES_BY_SPECIES
X_H = 0.70

# defaults of the problem; the test module imports these
M_BOX = 1.0e5  # total gas mass, Msun
N_H = 1.0  # cm^-3
T_GAS = 1.0e4  # K, fully ionized (mu = 0.59)
M_CLUSTER = 1.0e3  # Msun
AGE_CLUSTER_MYR = 3.5  # age at t=0; FIRE-3 core-collapse SNe start at 3.7 Myr
N_OSTARS = 12  # O stars in the population: the expectation 0.0115/Msun * M_CLUSTER, rounded
R_CLUSTER = 2.0  # pc
SEED = 42


def box_size(m_box=M_BOX, n_h=N_H):
    rho = n_h * M_PROTON_G / X_H * PC_IN_CM**3 / MSUN_IN_G  # Msun/pc^3
    return (m_box / rho) ** (1.0 / 3.0)


def lattice_side(m_gas, m_box=M_BOX):
    """Cells per side of the cubic lattice whose cell mass is closest to m_gas."""
    return max(4, int(round((m_box / m_gas) ** (1.0 / 3.0))))


def make_ics(output_file, m_gas=10.0, m_box=M_BOX, n_h=N_H, t_gas=T_GAS):
    L = box_size(m_box, n_h)
    n_side = lattice_side(m_gas, m_box)
    n_gas = n_side**3
    dx = L / n_side
    q = (np.arange(n_side) + 0.25) * dx  # quarter-cell offset: no cell coincides with the star at L/2
    pos = np.stack(np.meshgrid(q, q, q, indexing="ij"), axis=-1).reshape(-1, 3)
    u = 1.5 * K_BOLTZMANN * t_gas / (0.59 * M_PROTON_G) / 1.0e10  # (km/s)^2

    with h5py.File(output_file, "w") as f:
        npart = np.array([n_gas, 0, 0, 0, 1, 0], dtype=np.int32)
        h = f.create_group("Header")
        h.attrs["NumPart_ThisFile"] = npart
        h.attrs["NumPart_Total"] = npart.astype(np.uint32)
        h.attrs["NumPart_Total_HighWord"] = np.zeros(6, dtype=np.uint32)
        h.attrs["MassTable"] = np.zeros(6)
        h.attrs["Time"] = 0.0
        h.attrs["Redshift"] = 0.0
        h.attrs["BoxSize"] = L
        h.attrs["NumFilesPerSnapshot"] = 1
        h.attrs["Omega0"] = 0.0
        h.attrs["OmegaLambda"] = 0.0
        h.attrs["HubbleParam"] = 1.0
        h.attrs["Flag_Sfr"] = 1
        h.attrs["Flag_Cooling"] = 0
        h.attrs["Flag_StellarAge"] = 1
        h.attrs["Flag_Metals"] = 1
        h.attrs["Flag_Feedback"] = 1
        h.attrs["Flag_DoublePrecision"] = 0

        g = f.create_group("PartType0")
        g.create_dataset("Coordinates", data=pos)
        g.create_dataset("Velocities", data=np.zeros((n_gas, 3)))
        g.create_dataset("Masses", data=np.full(n_gas, m_box / n_gas))
        g.create_dataset("InternalEnergy", data=np.full(n_gas, u))
        g.create_dataset("ParticleIDs", data=np.arange(1, n_gas + 1, dtype=np.uint32))
        g.create_dataset("Metallicity", data=np.full(n_gas, Z_SOLAR))
        # a snapshot restart takes these as the starting guesses for the kernel iteration
        rho = m_box / L**3
        g.create_dataset("Density", data=np.full(n_gas, rho))
        g.create_dataset("KernelMaxRadius", data=np.full(n_gas, (3 * 32 / (4 * np.pi) * dx**3) ** (1.0 / 3.0)))

        rng = np.random.default_rng(SEED)
        n_star = max(1, int(round(M_CLUSTER / (m_box / n_gas))))
        x = rng.normal(size=(n_star, 3))
        x *= (R_CLUSTER * rng.random(n_star) ** (1.0 / 3.0) / np.linalg.norm(x, axis=1))[:, None]
        s = f.create_group("PartType4")
        s.create_dataset("Coordinates", data=0.5 * L + x)
        s.create_dataset("Velocities", data=np.zeros((n_star, 3)))
        s.create_dataset("Masses", data=np.full(n_star, M_CLUSTER / n_star))
        s.create_dataset("ParticleIDs", data=np.arange(n_gas + 1, n_gas + n_star + 1, dtype=np.uint32))
        s.create_dataset("Metallicity", data=np.full(n_star, Z_SOLAR))
        s.create_dataset("StellarFormationTime", data=np.full(n_star, -AGE_CLUSTER_MYR / CODE_TIME_IN_MYR))
        n_o = np.bincount(rng.integers(0, n_star, N_OSTARS), minlength=n_star)
        s.create_dataset("OStarNumber", data=n_o.astype(np.float64))
        h.attrs["NumPart_ThisFile"] = np.array([n_gas, 0, 0, 0, n_star, 0], dtype=np.int32)
        h.attrs["NumPart_Total"] = h.attrs["NumPart_ThisFile"].astype(np.uint32)
    return dict(box_size=L, n_gas=n_gas, m_gas=m_box / n_gas, u=u, n_star=n_star)


if __name__ == "__main__":
    import sys
    m = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
    print(make_ics(sys.argv[1] if len(sys.argv) > 1 else "sn_cluster_adiabatic_ics_000.hdf5", m_gas=m))
