#!/usr/bin/env python3
"""Generate Poisson-random uniform gas box ICs for the poisson_box test problem.

Places N = N_per_side^3 gas cells uniformly at random (no lattice, no glass) in a
periodic box [0, BoxSize]^3: equal masses, zero velocity, uniform internal energy
for T ~ 10^4 K, solar metallicity, and initial kernel radii from the mean
inter-cell spacing scaled to enclose ~32 neighbours. Several resolutions can be
written at once, for scaling studies.

Usage:
  python make_poisson_box_ics.py              # 50^3 = 125k cells (the test default)
  python make_poisson_box_ics.py 30 50 80 100 # one IC per resolution
  python make_poisson_box_ics.py 80 --nH 100  # GMC-like mean density instead of 1 cm^-3
Output: poisson_box_<N_per_side>_ics.hdf5

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
    N_per_side=50,
    BoxSize=100.0,
    seed=42,
    n_H=1.0,
):
    Ngas = N_per_side**3

    rng = np.random.default_rng(seed)

    # --- Positions: uniform random in [0, BoxSize]^3 ---
    pos = rng.uniform(0.0, BoxSize, size=(Ngas, 3)).astype(np.float64)

    # --- Velocities: zero ---
    vel = np.zeros((Ngas, 3), dtype=np.float32)

    # --- Masses: equal, for a mean hydrogen number density n_H (cm^-3) ---
    # rho = n_H * m_p / X_H with X_H = 0.76; M_total = rho * BoxSize^3 (pc -> cm, g -> Msun).
    # n_H = 1 in a 100 pc box gives M_total = 3.26e4 Msun.
    rho_cgs = n_H * 1.6726e-24 / 0.76
    M_total = rho_cgs * (BoxSize * 3.0857e18) ** 3 / 1.989e33  # Msun
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
    print(f"  M_total      = {M_total:.2e} Msun  (mean n_H = {n_H:g} /cc)")
    print(f"  u_therm      = {u_therm:.1f} (km/s)^2  (T ~ 1e4 K)")
    print(f"  h_guess      = {h_guess:.4f}  (mean spacing = {spacing:.2f})")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("N_per_side", type=int, nargs="*", default=[50])
    parser.add_argument("--nH", type=float, default=1.0, help="mean n_H in cm^-3 (default 1)")
    parser.add_argument("--boxsize", type=float, default=100.0, help="box side in pc (default 100)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    outdir = os.path.dirname(os.path.abspath(__file__))
    for n in args.N_per_side:
        make_poisson_box_ics(os.path.join(outdir, f"poisson_box_{n}_ics.hdf5"), N_per_side=n,
                             BoxSize=args.boxsize, seed=args.seed, n_H=args.nH)
