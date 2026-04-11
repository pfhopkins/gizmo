#!/usr/bin/env python3
"""Generate Poisson-random uniform gas box ICs for ghost exchange / neighbor list testing.

Places N = 100^3 = 1,000,000 gas particles uniformly at random in a periodic
box [0, BoxSize]^3.  All particles have equal mass, zero velocity, and uniform
internal energy corresponding to T ~ 10^4 K.  Initial smoothing lengths are
set from the mean inter-particle spacing scaled to enclose ~32 neighbors.

Units match the gmc_cooling test:
  UnitLength  = 1 pc   = 3.09e18 cm
  UnitMass    = 1 Msun = 1.99e33 g
  UnitVelocity = 1 km/s = 1e5 cm/s
"""

import numpy as np
import h5py
import os


def make_poisson_box_ics(
    output_file="poisson_box_ics.hdf5",
    N_per_side=100,
    BoxSize=100.0,
    seed=42,
):
    Ngas = N_per_side**3  # 1,000,000

    rng = np.random.default_rng(seed)

    # --- Positions: uniform random in [0, BoxSize]^3 ---
    pos = rng.uniform(0.0, BoxSize, size=(Ngas, 3)).astype(np.float64)

    # --- Velocities: zero ---
    vel = np.zeros((Ngas, 3), dtype=np.float32)

    # --- Masses: equal, pick a total mass that gives a reasonable density ---
    # Target: mean n_H ~ 1 cm^-3  (low density, just for testing)
    # rho = n_H * m_p / X_H;  X_H = 0.76
    # M_total = rho * BoxSize^3  (BoxSize in pc, mass in Msun)
    # n_H = 1 /cc => rho = 1 * 1.67e-24 / 0.76 = 2.197e-24 g/cc
    # BoxSize = 100 pc = 100 * 3.09e18 cm = 3.09e20 cm
    # Volume = (3.09e20)^3 = 2.953e61 cm^3
    # M_total = 2.197e-24 * 2.953e61 / 1.99e33 = 3.26e4 Msun
    M_total = 3.26e4  # Msun
    m_gas = M_total / Ngas
    masses = np.full(Ngas, m_gas, dtype=np.float32)

    # --- Internal energy for T ~ 10^4 K ---
    # u = k_B T / (mu * m_p * (gamma-1))
    # For ionized gas at 1e4 K: mu ~ 0.6, gamma = 5/3
    # u = 1.381e-16 * 1e4 / (0.6 * 1.67e-24 * (2/3))
    #   = 1.381e-12 / 6.68e-25 = 2.068e12 erg/g
    # In code units (1 km/s)^2 = 1e10 cm^2/s^2
    # u_code = 2.068e12 / 1e10 = 206.8 (km/s)^2
    u_therm = 206.8
    internal_energy = np.full(Ngas, u_therm, dtype=np.float32)

    # --- Particle IDs ---
    ids = np.arange(1, Ngas + 1, dtype=np.uint32)

    # --- Smoothing lengths: mean inter-particle spacing * kernel radius factor ---
    # mean spacing = BoxSize / N_per_side
    # h ~ spacing * (N_ngb / (4 pi/3))^(1/3)  where N_ngb = 32
    spacing = BoxSize / N_per_side
    h_guess = spacing * (32.0 / (4.0 * np.pi / 3.0)) ** (1.0 / 3.0)
    hsml = np.full(Ngas, h_guess, dtype=np.float32)

    # --- Metallicity: solar (11-species: Ztot, He, C, N, O, Ne, Mg, Si, S, Ca, Fe) ---
    # Required because METALS + COOL_METAL_LINES_BY_SPECIES is compiled in
    Z_solar = np.array(
        [0.02, 0.28, 2.53e-3, 7.41e-4, 6.13e-3, 1.20e-3,
         5.91e-4, 6.83e-4, 4.09e-4, 6.44e-5, 1.17e-3],
        dtype=np.float32,
    )

    # --- Write HDF5 ---
    with h5py.File(output_file, "w") as f:
        # Header
        npart = np.array([Ngas, 0, 0, 0, 0, 0], dtype=np.int32)
        h = f.create_group("Header")
        h.attrs["NumPart_ThisFile"] = npart
        h.attrs["NumPart_Total"] = npart.astype(np.uint32)
        h.attrs["NumPart_Total_HighWord"] = np.zeros(6, dtype=np.uint32)
        h.attrs["MassTable"] = np.zeros(6, dtype=np.float64)
        h.attrs["Time"] = 0.0
        h.attrs["Redshift"] = 0.0
        h.attrs["BoxSize"] = BoxSize
        h.attrs["NumFilesPerSnapshot"] = 1
        h.attrs["Omega0"] = 0.0
        h.attrs["OmegaLambda"] = 0.0
        h.attrs["HubbleParam"] = 1.0
        h.attrs["Flag_Sfr"] = 0
        h.attrs["Flag_Cooling"] = 0
        h.attrs["Flag_StellarAge"] = 0
        h.attrs["Flag_Metals"] = 11
        h.attrs["Flag_Feedback"] = 0
        h.attrs["Flag_DoublePrecision"] = 0

        # PartType0 (Gas)
        g = f.create_group("PartType0")
        g.create_dataset("Coordinates", data=pos)
        g.create_dataset("Velocities", data=vel)
        g.create_dataset("Masses", data=masses)
        g.create_dataset("ParticleIDs", data=ids)
        g.create_dataset("InternalEnergy", data=internal_energy)
        g.create_dataset("SmoothingLength", data=hsml)
        g.create_dataset("Metallicity", data=np.tile(Z_solar, (Ngas, 1)))

    print(f"Wrote {output_file}")
    print(f"  N_gas        = {Ngas} ({N_per_side}^3)")
    print(f"  BoxSize      = {BoxSize}")
    print(f"  m_gas        = {m_gas:.6e} Msun")
    print(f"  M_total      = {M_total:.2e} Msun  (mean n_H ~ 1 /cc)")
    print(f"  u_therm      = {u_therm:.1f} (km/s)^2  (T ~ 1e4 K)")
    print(f"  h_guess      = {h_guess:.4f}  (mean spacing = {spacing:.2f})")


if __name__ == "__main__":
    outdir = os.path.dirname(os.path.abspath(__file__))
    make_poisson_box_ics(os.path.join(outdir, "poisson_box_ics.hdf5"))
