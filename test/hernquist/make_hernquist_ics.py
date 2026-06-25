"""Generate Hernquist-sphere ICs for the hernquist gravity test.

Adapted from collisionless_equilibria/hernquist.py: samples radii from the
analytic Hernquist mass profile, then samples isotropic velocities via
vectorized rejection from the equilibrium DF (Hernquist 1990 Eq. 17 cast as
a per-radius p(v|r) ∝ v^2 f(E) form).

Code units: G = 1.
"""
import argparse
import numpy as np
import h5py


def _Fq(rv, r):
    """Hernquist isotropic v^2 f(E) ∝ Fq at fixed r, rv = v/v_escape."""
    rv2 = rv**2
    return (
        2 * r * (1 + r) ** 4 * rv2
        * (
            (
                np.sqrt(np.maximum((1 - rv2) * (r + rv2), 0.0))
                * (-1 + r + 2 * rv2)
                * (-3 - 14 * r - 3 * r**2 + 8 * (-1 + r) * rv2 + 8 * rv**4)
            )
            / (1 + r) ** 4
            + 3 * np.arcsin(np.sqrt(np.clip((1 - rv2) / (1 + r), 0.0, 1.0)))
        )
    ) / (np.pi * (r + rv2) ** 2.5)


def _sample_velocity_fractions(r_arr, rng, max_iter=10000):
    """Vectorized rejection-sample v/v_escape for each particle radius.
    Per-particle ymax is precomputed from the actual Fq peak on a 256-point grid
    in (0, 1), with a safety margin, so acceptance probability stays well above zero
    even at the truncation radius where Fq is many orders below the inner-cluster scale."""
    n = len(r_arr)
    rv_grid = np.linspace(1e-4, 1 - 1e-4, 256)
    # Fq shape (256, n) -> peak per particle
    peak = _Fq(rv_grid[:, None], r_arr[None, :]).max(axis=0)
    ymax = 1.5 * peak

    accepted = np.zeros(n)
    pending = np.ones(n, dtype=bool)
    for _ in range(max_iter):
        idx = np.flatnonzero(pending)
        if idx.size == 0:
            break
        m = idx.size
        Y = ymax[idx] * rng.random(m)
        X = rng.random(m)
        keep = Y < _Fq(X, r_arr[idx])
        accepted[idx[keep]] = X[keep]
        pending[idx[keep]] = False
    if pending.any():
        raise RuntimeError(f"Velocity rejection sampling did not converge for {pending.sum()} particles")
    return accepted


HERNQUIST_HALF_MASS_RADIUS = 2.414  # in units of a (= a / (sqrt(2) - 1))


def make_hernquist_ics(N, m, a, boxsize, seed, outfile, r_max=None):
    rng = np.random.default_rng(seed)

    # Cap the radial sample at 100 r_h to lop off the wildest tail outliers
    # without meaningfully truncating the cluster (Hernquist tail is heavy but
    # 100 r_h captures > 99% of the mass).
    if r_max is None:
        r_max = 100.0 * HERNQUIST_HALF_MASS_RADIUS * a
    u_max = (r_max / a) ** 2 / (1 + r_max / a) ** 2
    u = u_max * (np.arange(N) + rng.random(N)) / N
    r_unit = (np.sqrt(u) + u) / (1 - u)
    r = a * r_unit

    phi_ang = rng.random(N) * 2 * np.pi
    cos_theta = 2.0 * rng.random(N) - 1.0
    sin_theta = np.sqrt(1.0 - cos_theta**2)
    pos = np.c_[r * np.cos(phi_ang) * sin_theta, r * np.sin(phi_ang) * sin_theta, r * cos_theta]

    potential = -m / (r + a)
    v_escape = np.sqrt(-2 * potential)

    Qs = _sample_velocity_fractions(r_unit, rng)
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
    parser.add_argument("--N", type=int, default=16**3)
    parser.add_argument("--m", type=float, default=1.0)
    parser.add_argument("--a", type=float, default=1.0)
    parser.add_argument("--boxsize", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="hernquist_ics.hdf5")
    args = parser.parse_args()
    make_hernquist_ics(args.N, args.m, args.a, args.boxsize, args.seed, args.out)
