"""Plummer cluster of equal-mass circular binaries (Type 5 sinks).

Each "particle" of the Plummer sphere is replaced by a pair of equal-mass
stars on a circular orbit around their mutual center of mass. Units are
pc - km/s - Msun, so the gravitational constant in code units is
G comes from gizmo.units (GIZMO's constants.h), 4.300711e-3 pc (km/s)^2 / Msun.

Adapted from test/plummer/make_plummer_ics.py.
"""
import argparse
import numpy as np
from scipy.optimize import brentq
import h5py

import os as _os
import sys as _sys
# Run standalone (the tests invoke this as a subprocess from the test directory), so the
# harness package (test/harness in this tree; python_src upstream) is not on the path the
# way it is under pytest. G must come from gizmo.units, which mirrors
# GIZMO's constants.h: an IC built with a different G than the code integrates with is not the
# orbit it claims to be.
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  "..", "harness"))
from gizmo.units import G_CODE, AU_PER_PC  # noqa: E402


PLUMMER_HALF_MASS_RADIUS = 1.305  # in units of a (analytic, GM/a normalization)


def velocity_cdf(R, target):
    """Plummer isotropic velocity CDF at fixed r, in units of v_escape."""
    return (
        2 * (
            R * np.sqrt(1 - R**2) * (-105 + 1210 * R**2 - 2104 * R**4 + 1488 * R**6 - 384 * R**8)
            + 105 * np.arcsin(R)
        )
    ) / (105.0 * np.pi) - target


def _orthonormal_in_plane(n_hat, rng):
    """Return (e1, e2): orthonormal vectors spanning the plane perpendicular to n_hat.
    n_hat shape: (N, 3). Returns two (N, 3) arrays."""
    r = rng.normal(size=n_hat.shape)
    e1 = r - np.sum(r * n_hat, axis=1, keepdims=True) * n_hat
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = np.cross(n_hat, e1)
    return e1, e2


def make_plummer_binaries_ics(
    N_binaries,
    m_star,
    a_cluster,
    binary_separation_au,
    boxsize,
    seed,
    outfile,
    r_max=None,
):
    rng = np.random.default_rng(seed)

    if r_max is None:
        r_max = 100.0 * PLUMMER_HALF_MASS_RADIUS * a_cluster

    M_cluster = 2.0 * N_binaries * m_star  # Msun
    M_binary = 2.0 * m_star                # Msun

    # ------- Plummer COM positions (truncated at r_max) -------
    u_max = (r_max / a_cluster) ** 3 / (1 + (r_max / a_cluster) ** 2) ** 1.5
    u = u_max * (np.arange(N_binaries) + rng.random(N_binaries)) / N_binaries
    r_com = a_cluster * np.sqrt(u ** (2.0 / 3) * (1 + u ** (2.0 / 3) + u ** (4.0 / 3)) / (1 - u**2))

    phi_ang = rng.random(N_binaries) * 2 * np.pi
    cos_theta = 2.0 * rng.random(N_binaries) - 1.0
    sin_theta = np.sqrt(1.0 - cos_theta**2)
    pos_com = np.c_[
        r_com * np.cos(phi_ang) * sin_theta,
        r_com * np.sin(phi_ang) * sin_theta,
        r_com * cos_theta,
    ]

    # ------- Plummer COM velocities (inverse-CDF method) -------
    potential = -G_CODE * M_cluster / np.sqrt(r_com**2 + a_cluster**2)
    v_escape = np.sqrt(-2 * potential)
    Qs = np.array([brentq(velocity_cdf, 0, 1, args=(target,)) for target in rng.random(N_binaries)])
    v_com_mag = v_escape * Qs

    phi_v = rng.random(N_binaries) * 2 * np.pi
    cos_tv = 2.0 * rng.random(N_binaries) - 1.0
    sin_tv = np.sqrt(1.0 - cos_tv**2)
    vel_com = np.c_[
        v_com_mag * np.cos(phi_v) * sin_tv,
        v_com_mag * np.sin(phi_v) * sin_tv,
        v_com_mag * cos_tv,
    ]

    # ------- Binary internal motion -------
    binary_sep_pc = binary_separation_au / AU_PER_PC
    # circular orbit: each star moves at v_orb perpendicular to its position vector
    v_orb = 0.5 * np.sqrt(G_CODE * M_binary / binary_sep_pc)

    # Random orbital-plane normal (uniform on sphere)
    phi_n = rng.random(N_binaries) * 2 * np.pi
    cos_tn = 2.0 * rng.random(N_binaries) - 1.0
    sin_tn = np.sqrt(1.0 - cos_tn**2)
    n_hat = np.c_[np.cos(phi_n) * sin_tn, np.sin(phi_n) * sin_tn, cos_tn]
    e1, e2 = _orthonormal_in_plane(n_hat, rng)
    phase = rng.random(N_binaries) * 2 * np.pi
    cos_p = np.cos(phase)[:, None]
    sin_p = np.sin(phase)[:, None]

    r_star_rel = 0.5 * binary_sep_pc * (cos_p * e1 + sin_p * e2)       # offset of star 1
    v_star_rel = v_orb * (-sin_p * e1 + cos_p * e2)                     # velocity of star 1

    N_stars = 2 * N_binaries
    pos = np.empty((N_stars, 3))
    vel = np.empty((N_stars, 3))
    pos[0::2] = pos_com + r_star_rel
    pos[1::2] = pos_com - r_star_rel
    vel[0::2] = vel_com + v_star_rel
    vel[1::2] = vel_com - v_star_rel

    # Cluster centered on origin (periodic variants split it at box corner)

    with h5py.File(outfile, "w") as F:
        F.create_group("Header")
        F["Header"].attrs["NumPart_ThisFile"] = [0, 0, 0, 0, 0, N_stars]
        F["Header"].attrs["NumPart_Total"] = [0, 0, 0, 0, 0, N_stars]
        F["Header"].attrs["NumPart_Total_HighWord"] = [0, 0, 0, 0, 0, 0]
        F["Header"].attrs["MassTable"] = [0, 0, 0, 0, 0, m_star]
        F["Header"].attrs["BoxSize"] = boxsize
        F["Header"].attrs["Time"] = 0.0
        F["Header"].attrs["Redshift"] = 0.0
        F["Header"].attrs["NumFilesPerSnapshot"] = 1
        g = F.create_group("PartType5")
        g.create_dataset("Coordinates", data=pos)
        g.create_dataset("Velocities", data=vel)
        g.create_dataset("Masses", data=np.full(N_stars, m_star))
        # IDs 1,2 = binary 0; IDs 3,4 = binary 1; ...
        g.create_dataset("ParticleIDs", data=np.arange(1, N_stars + 1, dtype=np.uint32))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--N_binaries", type=int, default=256)
    parser.add_argument("--m_star", type=float, default=1.0, help="Msun per star")
    parser.add_argument("--a_cluster", type=float, default=1.0, help="Plummer scale radius (pc)")
    parser.add_argument("--binary_separation_au", type=float, default=1000.0,
                        help="must match BINARY_SEPARATION_AU in the test; a smaller "
                             "value silently makes a much more expensive problem")
    parser.add_argument("--boxsize", type=float, default=300.0, help="pc")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="plummer_binaries_ics.hdf5")
    args = parser.parse_args()
    make_plummer_binaries_ics(
        args.N_binaries, args.m_star, args.a_cluster, args.binary_separation_au,
        args.boxsize, args.seed, args.out,
    )
