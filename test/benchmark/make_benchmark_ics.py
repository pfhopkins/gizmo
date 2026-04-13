#!/usr/bin/env python3
"""Generate benchmark ICs at multiple resolutions for GPU scaling tests.

Creates uniform-density periodic box ICs at specified particle counts.
Uses the same physics setup as gmc_cooling (STARFORGE + MHD + cooling).

Usage:
  python3 make_benchmark_ics.py                    # generates all default sizes
  python3 make_benchmark_ics.py 50                 # generates 50^3 = 125k only
  python3 make_benchmark_ics.py 50 100 200         # generates specific sizes

Output files: bench_NNN_ics.hdf5 where NNN = N_per_side

Units (same as gmc_cooling):
  UnitLength  = 1 pc   = 3.09e18 cm
  UnitMass    = 1 Msun = 1.99e33 g
  UnitVelocity = 1 km/s = 1e5 cm/s
"""

import numpy as np
import h5py
import sys
import os


def make_benchmark_ic(N_per_side, outdir=".", BoxSize=100.0, seed=42):
    Ngas = N_per_side ** 3
    output_file = os.path.join(outdir, f"bench_{N_per_side}_ics.hdf5")

    rng = np.random.default_rng(seed)

    # Positions: uniform random in [0, BoxSize]^3
    pos = rng.uniform(0.0, BoxSize, size=(Ngas, 3)).astype(np.float64)

    # Velocities: zero
    vel = np.zeros((Ngas, 3), dtype=np.float32)

    # Masses: target mean n_H ~ 100 cm^-3 (GMC-like density, matches gmc_cooling)
    # rho = n_H * m_p / X_H = 100 * 1.67e-24 / 0.76 = 2.197e-22 g/cc
    # Volume = (100 pc)^3 = (3.09e20 cm)^3 = 2.953e61 cm^3
    # M_total = rho * V = 2.197e-22 * 2.953e61 / 1.99e33 = 3.26e6 Msun
    M_total = 3.26e6  # Msun (100x denser than poisson_box for GMC-like conditions)
    m_gas = M_total / Ngas
    masses = np.full(Ngas, m_gas, dtype=np.float32)

    # Internal energy for T ~ 10 K (cold GMC gas)
    # u = k_B T / (mu * m_p * (gamma-1))
    # For molecular gas at 10 K: mu ~ 2.3, gamma = 5/3
    # u = 1.381e-16 * 10 / (2.3 * 1.67e-24 * 2/3) = 5.39e-7 erg/g... too cold
    # Actually use T ~ 1e4 K like gmc_cooling (will cool quickly)
    u_therm = 206.8  # (km/s)^2, T ~ 1e4 K
    internal_energy = np.full(Ngas, u_therm, dtype=np.float32)

    # Particle IDs
    ids = np.arange(1, Ngas + 1, dtype=np.uint32)

    # Smoothing lengths from mean spacing
    spacing = BoxSize / N_per_side
    h_guess = spacing * (32.0 / (4.0 * np.pi / 3.0)) ** (1.0 / 3.0)
    hsml = np.full(Ngas, h_guess, dtype=np.float32)

    # Metallicity: solar (11-species for COOL_METAL_LINES_BY_SPECIES)
    Z_solar = np.array(
        [0.02, 0.28, 2.53e-3, 7.41e-4, 6.13e-3, 1.20e-3,
         5.91e-4, 6.83e-4, 4.09e-4, 6.44e-5, 1.17e-3],
        dtype=np.float32,
    )

    # Magnetic field: uniform weak seed field (same as gmc_cooling)
    bfield = np.zeros((Ngas, 3), dtype=np.float32)
    bfield[:, 0] = 1.0e-8
    bfield[:, 1] = 1.0e-8
    bfield[:, 2] = 1.0e-8

    with h5py.File(output_file, "w") as f:
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

        g = f.create_group("PartType0")
        g.create_dataset("Coordinates", data=pos)
        g.create_dataset("Velocities", data=vel)
        g.create_dataset("Masses", data=masses)
        g.create_dataset("ParticleIDs", data=ids)
        g.create_dataset("InternalEnergy", data=internal_energy)
        g.create_dataset("SmoothingLength", data=hsml)
        g.create_dataset("Metallicity", data=np.tile(Z_solar, (Ngas, 1)))
        g.create_dataset("MagneticField", data=bfield)

    print(f"  {output_file}: N={Ngas:,} ({N_per_side}^3), m={m_gas:.4e} Msun, h={h_guess:.4f}")
    return output_file


if __name__ == "__main__":
    outdir = os.path.dirname(os.path.abspath(__file__))

    if len(sys.argv) > 1:
        sizes = [int(x) for x in sys.argv[1:]]
    else:
        # Default: 30^3=27k, 50^3=125k, 80^3=512k, 100^3=1M
        sizes = [30, 50, 80, 100]

    print("Generating benchmark ICs:")
    for n in sizes:
        make_benchmark_ic(n, outdir=outdir)
