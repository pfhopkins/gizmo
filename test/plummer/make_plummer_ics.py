"""Generate Plummer-sphere ICs for the plummer gravity test.

Adapted from collisionless_equilibria/plummer.py: samples radii from the
analytic Plummer mass profile, then samples velocity from the isotropic
Plummer distribution function via inverse-CDF (Aarseth, Henon & Wielen 1974).

Code units: G = 1.
"""
import argparse
import numpy as np
from scipy.optimize import brentq
import h5py


def velocity_cdf(R, target):
    """Plummer isotropic velocity CDF at fixed r, in units of v_escape."""
    return (
        2 * (
            R * np.sqrt(1 - R**2) * (-105 + 1210 * R**2 - 2104 * R**4 + 1488 * R**6 - 384 * R**8)
            + 105 * np.arcsin(R)
        )
    ) / (105.0 * np.pi) - target


PLUMMER_HALF_MASS_RADIUS = 1.305  # in units of a (analytic, G=M=a=1)


def make_plummer_ics(N, m, a, boxsize, seed, outfile, r_max=None):
    rng = np.random.default_rng(seed)

    # Cap the radial sample at 100 r_h to lop off the wildest Plummer-tail
    # outliers (untruncated Plummer can reach r ~ 1000a for N ~ 1e5) without
    # meaningfully truncating the cluster.
    if r_max is None:
        r_max = 100.0 * PLUMMER_HALF_MASS_RADIUS * a
    u_max = (r_max / a) ** 3 / (1 + (r_max / a) ** 2) ** 1.5
    u = u_max * (np.arange(N) + rng.random(N)) / N
    r = a * np.sqrt(u ** (2.0 / 3) * (1 + u ** (2.0 / 3) + u ** (4.0 / 3)) / (1 - u**2))

    phi_ang = rng.random(N) * 2 * np.pi
    cos_theta = 2.0 * rng.random(N) - 1.0
    sin_theta = np.sqrt(1.0 - cos_theta**2)
    pos = np.c_[r * np.cos(phi_ang) * sin_theta, r * np.sin(phi_ang) * sin_theta, r * cos_theta]

    potential = -m / np.sqrt(r**2 + a**2)
    v_escape = np.sqrt(-2 * potential)

    Qs = np.array([brentq(velocity_cdf, 0, 1, args=(target,)) for target in rng.random(N)])
    v_mag = v_escape * Qs

    phi_v = rng.random(N) * 2 * np.pi
    cos_tv = 2.0 * rng.random(N) - 1.0
    sin_tv = np.sqrt(1.0 - cos_tv**2)
    vel = np.c_[v_mag * np.cos(phi_v) * sin_tv, v_mag * np.sin(phi_v) * sin_tv, v_mag * cos_tv]
    # Cluster centered on origin; periodic variants will wrap coords to [0, BoxSize)
    # which splits the cluster across the boundary, exercising periodic-image handling.

    with h5py.File(outfile, "w") as F:
        F.create_group("Header")
        F["Header"].attrs["NumPart_ThisFile"] = [0, N, 0, 0, 0, 0]
        F["Header"].attrs["NumPart_Total"] = [0, N, 0, 0, 0, 0]
        F["Header"].attrs["NumPart_Total_HighWord"] = [0, 0, 0, 0, 0, 0]
        F["Header"].attrs["MassTable"] = [0, m / N, 0, 0, 0, 0]
        F["Header"].attrs["BoxSize"] = boxsize
        F["Header"].attrs["Time"] = 0.0
        F["Header"].attrs["Redshift"] = 0.0
        F["Header"].attrs["NumFilesPerSnapshot"] = 1
        g = F.create_group("PartType1")
        g.create_dataset("Coordinates", data=pos)
        g.create_dataset("Velocities", data=vel)
        g.create_dataset("Masses", data=np.full(N, m / N))
        g.create_dataset("ParticleIDs", data=np.arange(1, N + 1, dtype=np.uint32))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=16**3, help="Particle count")
    parser.add_argument("--m", type=float, default=1.0, help="Total mass")
    parser.add_argument("--a", type=float, default=1.0, help="Plummer scale radius")
    parser.add_argument("--boxsize", type=float, default=20.0, help="Box size")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--out", default="plummer_ics.hdf5", help="Output HDF5 file")
    args = parser.parse_args()
    make_plummer_ics(args.N, args.m, args.a, args.boxsize, args.seed, args.out)
