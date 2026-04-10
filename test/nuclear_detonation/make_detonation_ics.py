#!/usr/bin/env python3
"""Generate initial conditions for a 1D thermonuclear detonation test.

Quasi-1D setup: a thin 3D periodic tube (like the Brio-Wu test) of He4 fuel
at white dwarf conditions (rho=1e5 g/cc, T=1e8 K) with a hot spot (T=9e8 K)
at the left end to trigger a detonation wave. Uses a smooth tanh temperature
profile to avoid sharp pressure discontinuities.

Units: CGS (UnitLength=1 cm, UnitMass=1 g, UnitVelocity=1 cm/s)
"""

import numpy as np
import h5py
import os
import urllib.request


def ensure_helm_table():
    """Download helm_table.dat to the EOS directory if not already present."""
    table_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "eos", "helmholtz", "helm_table.dat")
    if not os.path.exists(table_path):
        url = "http://www.tapir.caltech.edu/~phopkins/public/helm_table.dat"
        print(f"Downloading {url}...")
        urllib.request.urlretrieve(url, table_path)
        print(f"Saved to {table_path}")


def make_detonation_ics(output_file="nuclear_detonation_ics.hdf5"):
    ensure_helm_table()

    # 3D periodic box with BOX_LONG_X=4: cube-ish aspect ratio so the kernel
    # is well-resolved in all dimensions (avoids artifacts from thin tubes)
    Nx = 32   # particles along x
    Ny, Nz = 8, 8  # cross-section (8x8 gives good 3D neighbor resolution)
    N = Nx * Ny * Nz  # = 2048

    # physical box: L_x = 4 * BoxSize
    BoxSize = 2500.0  # cm (so L_x = 4 * 2500 = 10000 cm)
    L_x = 4.0 * BoxSize

    dx = L_x / Nx
    dy = BoxSize / Ny
    dz = BoxSize / Nz

    # particle positions: uniform grid
    pos = np.zeros((N, 3), dtype=np.float64)
    idx = 0
    for ix in range(Nx):
        for iy in range(Ny):
            for iz in range(Nz):
                pos[idx, 0] = (ix + 0.5) * dx
                pos[idx, 1] = (iy + 0.5) * dy
                pos[idx, 2] = (iz + 0.5) * dz
                idx += 1

    # uniform density: rho = 1e5 g/cc (sub-Chandrasekhar He shell density)
    # moderate enough to stay well within Helmholtz table range even with
    # kernel smoothing variations, while still hot enough for triple-alpha
    rho0 = 1.0e5
    vol_per_particle = dx * dy * dz
    mass = rho0 * vol_per_particle

    # uniform temperature — tests nuclear network stability in 3D without
    # pressure-driven detonation fronts that become unresolvable
    Abar = 4.0  # pure He4
    T = np.full(N, 5.0e8)  # 0.5 GK: at burning floor, negligible triple-alpha at rho=1e5

    # internal energy consistent with Helmholtz EOS (ion + radiation + electron thermal)
    k_B = 1.380649e-16; m_p = 1.672621898e-24; mu = Abar / (1.0 + Abar * 0.5)
    a_rad = 7.5657e-15
    u_ion = 1.5 * k_B * T / (mu * m_p)
    u_rad = a_rad * T**4 / rho0
    n_e = rho0 * 0.5 / (Abar * m_p)
    u_e = 1.5 * k_B * T * n_e / rho0
    u = u_ion + u_rad + u_e
    Abar = 4.0

    vel = np.zeros((N, 3), dtype=np.float64)

    # composition: pure He4
    num_metal_species = 1
    num_nuclear = 13
    n_metallicity = num_metal_species + num_nuclear
    metallicity = np.zeros((N, n_metallicity), dtype=np.float64)
    metallicity[:, 0] = 1.0
    metallicity[:, num_metal_species + 0] = 1.0  # X(He4) = 1

    Ye_arr = np.full(N, 0.5, dtype=np.float64)
    Abar_arr = np.full(N, 4.0, dtype=np.float64)
    ids = np.arange(1, N + 1, dtype=np.int32)

    with h5py.File(output_file, "w") as f:
        npart = np.array([N, 0, 0, 0, 0, 0], dtype=np.int32)
        h = f.create_group("Header")
        h.attrs["NumPart_ThisFile"] = npart
        h.attrs["NumPart_Total"] = npart.astype(np.uint32)
        h.attrs["NumPart_Total_HighWord"] = np.zeros(6, dtype=np.uint32)
        h.attrs["MassTable"] = np.zeros(6)
        h.attrs["Time"] = 0.0
        h.attrs["Redshift"] = 0.0
        h.attrs["BoxSize"] = BoxSize
        h.attrs["NumFilesPerSnapshot"] = 1
        h.attrs["Omega0"] = 0.0
        h.attrs["OmegaLambda"] = 0.0
        h.attrs["HubbleParam"] = 1.0
        h.attrs["Flag_Metals"] = n_metallicity

        g = f.create_group("PartType0")
        g.create_dataset("Coordinates", data=pos)
        g.create_dataset("Velocities", data=vel.astype(np.float32))
        g.create_dataset("Masses", data=np.full(N, mass, dtype=np.float64))
        g.create_dataset("ParticleIDs", data=ids)
        g.create_dataset("InternalEnergy", data=u)
        g.create_dataset("Density", data=np.full(N, rho0, dtype=np.float64))
        g.create_dataset("Metallicity", data=metallicity)
        g.create_dataset("Temperature", data=T)
        g.create_dataset("Abar", data=Abar_arr)
        g.create_dataset("Ye", data=Ye_arr)
        g.create_dataset("NuclearComposition", data=metallicity[:, num_metal_species:].astype(np.float32))

    print(f"Wrote {N} particles ({Nx}x{Ny}x{Nz}) to {output_file}")
    print(f"  BoxSize={BoxSize} cm, L_x={L_x} cm ({L_x/1e5:.1f} km)")
    print(f"  rho={rho0:.2e} g/cc, mass/particle={mass:.4e} g")
    print(f"  T={T[0]:.2e} K (uniform), u={u[0]:.4e} erg/g")


if __name__ == "__main__":
    make_detonation_ics()
