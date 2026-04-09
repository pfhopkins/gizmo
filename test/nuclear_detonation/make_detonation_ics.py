#!/usr/bin/env python3
"""Generate initial conditions for a 1D thermonuclear detonation test.

A tube of He4 fuel at white dwarf conditions (rho=1e7 g/cc, T=1e8 K)
with a hot spot (T=2e9 K) at the left end to trigger a detonation wave.

The aprox13 alpha-chain network burns He4 via the triple-alpha reaction.
The detonation should propagate to the right with the composition
transitioning from He4 to C12 (and some O16) behind the front.

Units: CGS (UnitLength=1 cm, UnitMass=1 g, UnitVelocity=1 cm/s)
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


def make_detonation_ics(output_file="nuclear_detonation_ics.hdf5"):
    ensure_helm_table()
    # domain: 1D tube, length L = 1e6 cm (10 km)
    L = 1.0e6
    N = 1000  # number of gas particles

    # particle positions: uniform spacing along x, centered in y,z
    dx = L / N
    x = np.linspace(0.5 * dx, L - 0.5 * dx, N)
    pos = np.zeros((N, 3), dtype=np.float64)
    pos[:, 0] = x
    pos[:, 1] = 0.5 * dx  # thin slab
    pos[:, 2] = 0.5 * dx

    # uniform density: rho = 1e7 g/cc (He white dwarf conditions)
    rho0 = 1.0e7
    volume_per_particle = dx * dx * dx  # cube with side dx
    mass = rho0 * volume_per_particle

    # temperature: cold fuel with a hot spot at the left end
    T_cold = 1.0e8    # 0.1 GK (below ignition)
    T_hot = 2.0e9      # 2 GK (triggers triple-alpha)
    hot_length = 0.05 * L  # leftmost 5% is the hot spot

    T = np.where(x < hot_length, T_hot, T_cold)

    # internal energy: u = (3/2) * k_B * T / (Abar * m_p) for ideal gas
    # with Helmholtz EOS this is approximate; the actual u will be set by the EOS
    # at initialization. For He4: Abar=4, mu ~ 4/(1+3*0.5) = 1.6
    k_B = 1.380649e-16  # erg/K
    m_p = 1.672621898e-24  # g
    Abar = 4.0
    mu = Abar / (1.0 + Abar * 0.5)  # mean molecular weight for fully ionized He4
    u = 1.5 * k_B * T / (mu * m_p)  # specific internal energy [erg/g]

    # velocities: zero
    vel = np.zeros((N, 3), dtype=np.float64)

    # composition: pure He4
    # stored in Metallicity array: first NUM_METAL_SPECIES entries are metals,
    # then NUM_ADDITIONAL_PASSIVESCALAR entries which include nuclear species.
    # For this test with NUCLEAR_NETWORK + METALS (no COOL_METAL_LINES_BY_SPECIES),
    # NUM_METAL_SPECIES = 1 (total metallicity only), and nuclear species start
    # at offset NUM_METAL_SPECIES = 1.
    # The Metallicity field has 1 + 13 = 14 entries.
    num_metal_species = 1  # total Z only
    num_nuclear = 13
    n_metallicity = num_metal_species + num_nuclear
    metallicity = np.zeros((N, n_metallicity), dtype=np.float64)
    metallicity[:, 0] = 1.0  # total metallicity = 1 (all is He4)
    # nuclear species: He4 = index 0 within the nuclear block
    metallicity[:, num_metal_species + 0] = 1.0  # X(He4) = 1.0

    # EOS fields: Ye=0.5, Abar=4.0 for pure He4
    Ye_arr = np.full(N, 0.5, dtype=np.float64)
    Abar_arr = np.full(N, 4.0, dtype=np.float64)

    # particle IDs
    ids = np.arange(1, N + 1, dtype=np.int32)

    # write HDF5 IC file
    with h5py.File(output_file, "w") as f:
        npart = np.array([N, 0, 0, 0, 0, 0], dtype=np.int32)
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
        # nuclear composition block (same data but in dedicated field)
        g.create_dataset("NuclearComposition", data=metallicity[:, num_metal_species:].astype(np.float32))

    print(f"Wrote {N} particles to {output_file}")
    print(f"  L = {L:.2e} cm, dx = {dx:.2e} cm")
    print(f"  rho = {rho0:.2e} g/cc, mass/particle = {mass:.4e} g")
    print(f"  T_cold = {T_cold:.2e} K, T_hot = {T_hot:.2e} K")
    print(f"  u_cold = {u[N//2]:.4e} erg/g, u_hot = {u[0]:.4e} erg/g")


if __name__ == "__main__":
    make_detonation_ics()
