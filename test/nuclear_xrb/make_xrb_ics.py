#!/usr/bin/env python3
"""Generate initial conditions for an X-ray burst column test.

A 1D vertical column of pure He4 accreted onto a neutron star surface,
in hydrostatic equilibrium with gravity from an analytic point-mass potential.
The base of the column is heated to trigger thermonuclear ignition.

The test demonstrates:
- Thermonuclear runaway (He flash) when T reaches ~1 GK at the base
- Nuclear energy release and light curve
- Nucleosynthesis: He4 -> C12 -> heavier elements via alpha captures
- Hydrostatic response to the energy injection

Neutron star parameters:
  M_NS = 1.4 Msun, R_NS = 10 km
  Surface gravity: g = G*M/R^2 ~ 2.43e14 cm/s^2

Column properties:
  Height: H = 1e4 cm (100 m)
  Base density: rho_base ~ 1e6 g/cc
  Base temperature: T_base = 5e8 K (just below ignition)
  Composition: pure He4 (Ye=0.5)

Units: CGS
"""

import numpy as np
import h5py
import os
import urllib.request


def ensure_helm_table():
    """Download helm_table.dat to the EOS directory if not already present."""
    table_path = os.path.join(os.path.dirname(__file__), "..", "..", "eos", "helmholtz", "helm_table.dat")
    if not os.path.exists(table_path):
        url = "http://www.tapir.caltech.edu/~phopkins/public/helm_table.dat"
        print(f"Downloading {url}...")
        urllib.request.urlretrieve(url, table_path)
        print(f"Saved to {table_path}")


def make_xrb_ics(output_file="nuclear_xrb_ics.hdf5"):
    ensure_helm_table()
    # neutron star parameters
    G_cgs = 6.674e-8
    M_NS = 1.4 * 1.989e33  # g
    R_NS = 1.0e6            # cm (10 km)
    g_surf = G_cgs * M_NS / R_NS**2  # ~ 2.43e14 cm/s^2

    # column parameters
    H = 1.0e4  # height [cm] (100 m)
    N = 500    # number of particles
    dz = H / N

    # vertical positions (z = height above NS surface)
    z = np.linspace(0.5 * dz, H - 0.5 * dz, N)
    pos = np.zeros((N, 3), dtype=np.float64)
    pos[:, 0] = 0.5 * dz
    pos[:, 1] = 0.5 * dz
    pos[:, 2] = z  # vertical = z direction

    # hydrostatic equilibrium: dP/dz = -rho * g
    # for ideal gas: P = rho * k_B * T / (mu * m_p)
    # isothermal at T_base gives: rho(z) = rho_base * exp(-z / H_scale)
    # where H_scale = k_B * T / (mu * m_p * g)
    k_B = 1.380649e-16
    m_p = 1.672621898e-24
    Abar = 4.0
    mu = Abar / (1.0 + Abar * 0.5)  # fully ionized He4

    T_base = 5.0e8  # K — just below triple-alpha ignition
    H_scale = k_B * T_base / (mu * m_p * g_surf)  # pressure scale height

    print(f"Surface gravity: g = {g_surf:.4e} cm/s^2")
    print(f"Scale height: H_scale = {H_scale:.4e} cm")
    print(f"Column height / scale height = {H/H_scale:.2f}")

    # density profile: exponential atmosphere
    rho_base = 1.0e6  # g/cc
    rho = rho_base * np.exp(-z / H_scale)

    # temperature: slightly hotter at the base to trigger ignition
    # linear gradient from T_base at bottom to T_base * 0.5 at top
    T = T_base * (1.0 - 0.5 * z / H)

    # boost the base temperature slightly above ignition
    # the nuclear burning will do the rest
    hot_zone = z < 0.1 * H
    T[hot_zone] = 1.0e9  # 1 GK at the base — triggers triple-alpha

    # internal energy
    u = 1.5 * k_B * T / (mu * m_p)

    # mass per particle (variable, reflecting density profile)
    volume_per_particle = dz * dz * dz
    mass = rho * volume_per_particle

    # zero velocity (hydrostatic)
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
        h.attrs["BoxSize"] = H
        h.attrs["NumFilesPerSnapshot"] = 1
        h.attrs["Omega0"] = 0.0
        h.attrs["OmegaLambda"] = 0.0
        h.attrs["HubbleParam"] = 1.0

        g = f.create_group("PartType0")
        g.create_dataset("Coordinates", data=pos)
        g.create_dataset("Velocities", data=vel.astype(np.float32))
        g.create_dataset("Masses", data=mass)
        g.create_dataset("ParticleIDs", data=ids)
        g.create_dataset("InternalEnergy", data=u)
        g.create_dataset("Density", data=rho)
        g.create_dataset("Metallicity", data=metallicity.astype(np.float32))
        g.create_dataset("Temperature", data=T)
        g.create_dataset("Abar", data=Abar_arr)
        g.create_dataset("Ye", data=Ye_arr)
        g.create_dataset("NuclearComposition", data=metallicity[:, num_metal_species:].astype(np.float32))

    print(f"Wrote {N} particles to {output_file}")
    print(f"  H = {H:.2e} cm, dz = {dz:.2e} cm")
    print(f"  rho_base = {rho_base:.2e} g/cc, rho_top = {rho[-1]:.4e} g/cc")
    print(f"  T_base = {T[0]:.2e} K (hot zone), T_top = {T[-1]:.2e} K")
    print(f"  total column mass = {mass.sum():.4e} g")


if __name__ == "__main__":
    make_xrb_ics()
