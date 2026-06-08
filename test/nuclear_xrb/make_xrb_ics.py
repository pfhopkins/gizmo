#!/usr/bin/env python3
"""Generate ICs for a 1D nuclear burning test.

Uniform-density He4 box with a hot spot at one end. No gravity.
Tests that the nuclear network runs stably with the Helmholtz EOS.

Uses large cell spacing so the Courant timestep is tractable:
  dx ~ 1e4 cm → dt_Courant ~ 1e-6 s → ~1e4 steps for TimeMax=1e-2 s

Conditions: rho=1e6 g/cc, T_cold=5e8 K (no burning), T_hot=3e9 K (fast burning).
At T9=3, rho=1e6: tau_burn ~ 7000 s, so in TimeMax=0.01 s we get ~1e-6 of total
burn — enough to verify the solver runs without crashing.

Units: CGS (UnitLength=1 cm, UnitMass=1 g, UnitVelocity=1 cm/s)
"""

import numpy as np
import h5py
import os
import urllib.request


def ensure_helm_table():
    """Download helm_table.dat to the EOS directory if not already present."""
    table_path = os.path.join(os.path.dirname(__file__), "..", "..", "src", "eos", "helmholtz", "helm_table.dat")
    if not os.path.exists(table_path):
        url = "http://www.tapir.caltech.edu/~phopkins/public/helm_table.dat"
        print(f"Downloading {url}...")
        urllib.request.urlretrieve(url, table_path)
        print(f"Saved to {table_path}")


def make_xrb_ics(output_file="nuclear_xrb_ics.hdf5"):
    ensure_helm_table()

    N = 200         # particles
    BoxSize = 2.0e6  # cm (20 km)
    dx = BoxSize / N  # 1e4 cm = 100 m per cell

    # positions along x, y=z=0 (standard 1D convention)
    x = np.linspace(0.5 * dx, BoxSize - 0.5 * dx, N)
    pos = np.zeros((N, 3), dtype=np.float64)
    pos[:, 0] = x

    # uniform density
    rho0 = 1.0e5  # g/cc
    # 1D: mass = rho * dx
    mass = rho0 * dx

    # temperature: cold fuel with hot spot at left
    k_B = 1.380649e-16
    m_p = 1.672621898e-24
    Abar = 4.0
    mu = Abar / (1.0 + Abar * 0.5)

    # uniform temperature — avoids pressure gradients that create tiny timesteps
    # rho=1e5, T=1e9: tau_burn ~ 3e16 s (negligible burning), tests solver stability
    T = np.full(N, 1.0e9)

    # internal energy consistent with Helmholtz EOS (ion + radiation + electron thermal)
    a_rad = 7.5657e-15
    u_ion = 1.5 * k_B * T / (mu * m_p)
    u_rad = a_rad * T**4 / rho0
    n_e = rho0 * 0.5 / (Abar * m_p)
    u_e = 1.5 * k_B * T * n_e / rho0
    u = u_ion + u_rad + u_e

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
        h.attrs["Flag_Metals"] = n_metallicity  # must match NUM_METAL_SPECIES in compiled code

        g = f.create_group("PartType0")
        g.create_dataset("Coordinates", data=pos)
        g.create_dataset("Velocities", data=vel.astype(np.float32))
        g.create_dataset("Masses", data=np.full(N, mass, dtype=np.float64))
        g.create_dataset("ParticleIDs", data=ids)
        g.create_dataset("InternalEnergy", data=u)
        g.create_dataset("Density", data=np.full(N, rho0, dtype=np.float64))
        g.create_dataset("Metallicity", data=metallicity.astype(np.float32))
        g.create_dataset("Temperature", data=T)
        g.create_dataset("Abar", data=Abar_arr)
        g.create_dataset("Ye", data=Ye_arr)
        g.create_dataset("NuclearComposition", data=metallicity[:, num_metal_species:].astype(np.float32))

    cs_hot = 5.56e8  # approx at T9=3
    dt_courant = 0.1 * dx / cs_hot
    print(f"Wrote {N} particles to {output_file}")
    print(f"  BoxSize={BoxSize:.2e} cm, dx={dx:.2e} cm")
    print(f"  rho={rho0:.2e} g/cc, mass/particle={mass:.4e} g")
    print(f"  T={T[0]:.2e} K (uniform)")
    print(f"  dt_Courant ~ {dt_courant:.2e} s")
    print(f"  TimeMax=1e-2 s → ~{1e-2/dt_courant:.0f} Courant steps")


if __name__ == "__main__":
    make_xrb_ics()
