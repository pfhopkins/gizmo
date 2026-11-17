#!/usr/bin/env python3
"""Generate Poisson-random uniform gas box ICs for the poisson_box test problem.

Places N = N_per_side^3 gas cells uniformly at random (no lattice, no glass) in a
periodic box [0, BoxSize]^3: equal masses, zero velocity, uniform internal energy
for T ~ 10^4 K, solar metallicity, and initial kernel radii from the mean
inter-cell spacing scaled to enclose ~32 neighbours. Several resolutions can be
written at once, for scaling studies.

Large boxes are written as several files (GIZMO's multi-part IC format: set
InitCondFile to the base name, e.g. poisson_box_1600_ics, and GIZMO reads
poisson_box_1600_ics.0.hdf5, .1.hdf5, ...). By default a new file is started every
1e8 cells; --particles-per-file changes that. Every array is generated and written
in bounded chunks, so memory use does not grow with N. The cells are the same
whatever the number of files: one random stream is drawn in order across them.

Usage:
  python make_poisson_box_ics.py              # 50^3 = 125k cells (the test default)
  python make_poisson_box_ics.py 30 50 80 100 # one IC per resolution
  python make_poisson_box_ics.py 80 --nH 100  # GMC-like mean density instead of 1 cm^-3
  python make_poisson_box_ics.py 1600         # 4.1e9 cells, written as 41 files
Output: poisson_box_<N_per_side>_ics.hdf5, or poisson_box_<N_per_side>_ics.<k>.hdf5 for k = 0..files-1

Units match the gmc_cooling test:
  UnitLength  = 1 pc   = 3.09e18 cm
  UnitMass    = 1 Msun = 1.99e33 g
  UnitVelocity = 1 km/s = 1e5 cm/s
"""

import numpy as np
import h5py
import os

UINT32_MAX = 2**32 - 1
ROWS_PER_CHUNK = 10_000_000   # rows generated and written at a time, so memory stays bounded


def make_poisson_box_ics(
    output_file="poisson_box_ics.hdf5",
    N_per_side=50,
    BoxSize=100.0,
    seed=42,
    n_H=1.0,
    particles_per_file=100_000_000,
):
    Ngas = int(N_per_side) ** 3   # Python integers: exact at any size
    num_files = max(1, -(-Ngas // max(1, int(particles_per_file))))

    # --- Masses: equal, for a mean hydrogen number density n_H (cm^-3) ---
    # rho = n_H * m_p / X_H with X_H = 0.76; M_total = rho * BoxSize^3 (pc -> cm, g -> Msun).
    # n_H = 1 in a 100 pc box gives M_total = 3.26e4 Msun.
    rho_cgs = n_H * 1.6726e-24 / 0.76
    M_total = rho_cgs * (BoxSize * 3.0857e18) ** 3 / 1.989e33  # Msun
    m_gas = M_total / Ngas

    # --- Internal energy for T ~ 10^4 K ---
    # u = k_B T / (mu * m_p * (gamma-1))
    # For ionized gas at 1e4 K: mu ~ 0.6, gamma = 5/3
    # u = 1.381e-16 * 1e4 / (0.6 * 1.67e-24 * (2/3))
    #   = 1.381e-12 / 6.68e-25 = 2.068e12 erg/g
    # In code units (1 km/s)^2 = 1e10 cm^2/s^2
    # u_code = 2.068e12 / 1e10 = 206.8 (km/s)^2
    u_therm = 206.8

    # --- Smoothing lengths: mean inter-particle spacing * kernel radius factor ---
    # mean spacing = BoxSize / N_per_side
    # h ~ spacing * (N_ngb / (4 pi/3))^(1/3)  where N_ngb = 32
    spacing = BoxSize / N_per_side
    h_guess = spacing * (32.0 / (4.0 * np.pi / 3.0)) ** (1.0 / 3.0)

    # --- Metallicity: solar (11-species: Ztot, He, C, N, O, Ne, Mg, Si, S, Ca, Fe) ---
    # Required because METALS + COOL_METAL_LINES_BY_SPECIES is compiled in
    Z_solar = np.array(
        [0.02, 0.28, 2.53e-3, 7.41e-4, 6.13e-3, 1.20e-3,
         5.91e-4, 6.83e-4, 4.09e-4, 6.44e-5, 1.17e-3],
        dtype=np.float32,
    )

    # IDs run 1..Ngas; 32-bit while they fit, 64-bit beyond (GIZMO then needs LONGIDS).
    id_dtype = np.uint32 if Ngas <= UINT32_MAX else np.uint64

    # The global total, as GIZMO's header stores it: a low and a high 32-bit word.
    total = np.zeros(6, dtype=np.uint64); total[0] = Ngas
    total_low = (total & np.uint64(UINT32_MAX)).astype(np.uint32)
    total_high = (total >> np.uint64(32)).astype(np.uint32)

    if num_files == 1:
        file_names = [output_file]
    else:
        base = output_file[:-len(".hdf5")] if output_file.endswith(".hdf5") else output_file
        file_names = [f"{base}.{k}.hdf5" for k in range(num_files)]

    rng = np.random.default_rng(seed)
    first_id = 1
    for k, fname in enumerate(file_names):
        n_file = Ngas // num_files + (1 if k < Ngas % num_files else 0)
        with h5py.File(fname, "w") as f:
            npart = np.zeros(6, dtype=np.int64); npart[0] = n_file
            h = f.create_group("Header")
            h.attrs["NumPart_ThisFile"] = npart
            h.attrs["NumPart_Total"] = total_low
            h.attrs["NumPart_Total_HighWord"] = total_high
            h.attrs["MassTable"] = np.zeros(6, dtype=np.float64)
            h.attrs["Time"] = 0.0
            h.attrs["Redshift"] = 0.0
            h.attrs["BoxSize"] = BoxSize
            h.attrs["NumFilesPerSnapshot"] = num_files
            h.attrs["Omega0"] = 0.0
            h.attrs["OmegaLambda"] = 0.0
            h.attrs["HubbleParam"] = 1.0
            h.attrs["Flag_Sfr"] = 0
            h.attrs["Flag_Cooling"] = 0
            h.attrs["Flag_StellarAge"] = 0
            h.attrs["Flag_Metals"] = 11
            h.attrs["Flag_Feedback"] = 0
            h.attrs["Flag_DoublePrecision"] = 0

            # PartType0 (Gas): datasets created at full size, then filled chunk by chunk.
            g = f.create_group("PartType0")
            d_pos = g.create_dataset("Coordinates", shape=(n_file, 3), dtype=np.float64)
            d_vel = g.create_dataset("Velocities", shape=(n_file, 3), dtype=np.float32)
            d_mass = g.create_dataset("Masses", shape=(n_file,), dtype=np.float32)
            d_ids = g.create_dataset("ParticleIDs", shape=(n_file,), dtype=id_dtype)
            d_u = g.create_dataset("InternalEnergy", shape=(n_file,), dtype=np.float32)
            d_h = g.create_dataset("SmoothingLength", shape=(n_file,), dtype=np.float32)
            d_z = g.create_dataset("Metallicity", shape=(n_file, Z_solar.size), dtype=np.float32)
            for start in range(0, n_file, ROWS_PER_CHUNK):
                stop = min(start + ROWS_PER_CHUNK, n_file)
                n = stop - start
                d_pos[start:stop] = rng.uniform(0.0, BoxSize, size=(n, 3)).astype(np.float64)
                d_vel[start:stop] = np.zeros((n, 3), dtype=np.float32)
                d_mass[start:stop] = np.full(n, m_gas, dtype=np.float32)
                d_ids[start:stop] = np.arange(first_id + start, first_id + stop, dtype=np.uint64).astype(id_dtype)
                d_u[start:stop] = np.full(n, u_therm, dtype=np.float32)
                d_h[start:stop] = np.full(n, h_guess, dtype=np.float32)
                d_z[start:stop] = np.tile(Z_solar, (n, 1))
        first_id += n_file

    if num_files == 1:
        print(f"Wrote {output_file}")
    else:
        print(f"Wrote {num_files} files {file_names[0]} .. {file_names[-1]}  (InitCondFile = {os.path.basename(base)})")
    print(f"  N_gas        = {Ngas} ({N_per_side}^3)")
    if id_dtype == np.uint64:
        print("  ParticleIDs  = 64-bit (more than 2^32-1 cells): build GIZMO with LONGIDS")
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
    parser.add_argument("--particles-per-file", type=int, default=100_000_000,
                        help="start a new file after this many cells (default 1e8); 1 file below it")
    args = parser.parse_args()
    outdir = os.path.dirname(os.path.abspath(__file__))
    for n in args.N_per_side:
        make_poisson_box_ics(os.path.join(outdir, f"poisson_box_{n}_ics.hdf5"), N_per_side=n,
                             BoxSize=args.boxsize, seed=args.seed, n_H=args.nH,
                             particles_per_file=args.particles_per_file)
