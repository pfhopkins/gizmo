"""ICs and analytic solution for the polytrope_n1 hydrostatic-equilibrium test.

The n = 1 polytrope (P = K rho^2, gamma = 2) has a closed-form Lane-Emden solution. In code
units G = M = R = 1, with xi = pi r / R:

    rho(r) = rho_c sin(xi) / xi,             rho_c = pi M / (4 R^3) = pi/4
    M(r)   = M [sin(xi) - xi cos(xi)] / pi
    K      = 2 G R^2 / pi = 2/pi             (R = pi sqrt(K / (2 pi G)))
    u      = K rho / (gamma - 1) = K rho
    g(r)   = -G M(r) / r^2  =  -grad(P)/rho = -2 K drho/dr   (hydrostatic balance)
    W      = -3/(5-n) G M^2/R = -3/4,   U = -W / (3 (gamma-1)) = 1/4   (virial theorem)
    c_s,c  = sqrt(gamma P_c / rho_c) = sqrt(2 K rho_c) = 1

The particles are a jittered cubic lattice cut to a sphere, then stretched radially onto the
polytrope's mass profile: the particle of radial rank i is moved to r = M^-1((i + 1/2)/N),
keeping its direction. Equal masses, zero velocities, u set from the analytic rho at the new
radius. The stretch is not isotropic: radial spacing grows faster than tangential spacing
toward the surface, where rho -> 0 (see stretch_anisotropy()).
"""
import argparse

import h5py
import numpy as np

GAMMA = 2.0
G = 1.0
TOTAL_MASS = 1.0
RADIUS = 1.0
RHO_C = np.pi * TOTAL_MASS / (4.0 * RADIUS**3)
K_POLY = 2.0 * G * RADIUS**2 / np.pi
W_EXACT = -0.75 * G * TOTAL_MASS**2 / RADIUS
U_EXACT = -W_EXACT / (3.0 * (GAMMA - 1.0))
CS_CENTRAL = np.sqrt(GAMMA * K_POLY * RHO_C)

N_DEFAULT = 2**15
JITTER = 0.02    # uniform per-axis lattice jitter amplitude, in lattice spacings
SEED = 1
IC_VERSION = 1   # bump when the construction changes, so _ensure_ic() regenerates stale files


def rho_analytic(r):
    """Density; zero outside R."""
    r = np.asarray(r, dtype=float)
    return np.where(r < RADIUS, RHO_C * np.sinc(r / RADIUS), 0.0)   # np.sinc(x) = sin(pi x)/(pi x)


def drho_dr_analytic(r):
    r = np.asarray(r, dtype=float)
    xi = np.pi * r / RADIUS
    xi_safe = np.maximum(xi, 1e-8)
    d = RHO_C * (np.pi / RADIUS) * (xi_safe * np.cos(xi_safe) - np.sin(xi_safe)) / xi_safe**2
    return np.where(r < RADIUS, d, 0.0)


def mass_enclosed(r):
    r = np.minimum(np.asarray(r, dtype=float), RADIUS)
    xi = np.pi * r / RADIUS
    return TOTAL_MASS * (np.sin(xi) - xi * np.cos(xi)) / np.pi


def g_analytic(r):
    """Radial gravitational acceleration -G M(r)/r^2 (negative = inward)."""
    r = np.asarray(r, dtype=float)
    return -G * mass_enclosed(r) / np.maximum(r, 1e-30) ** 2


def hydro_accel_analytic(r):
    """Radial -grad(P)/rho = -2 K drho/dr for P = K rho^2 (positive = outward)."""
    return -2.0 * K_POLY * drho_dr_analytic(r)


def u_analytic(r):
    return K_POLY * rho_analytic(r) / (GAMMA - 1.0)


def radius_of_mass_fraction(q, ngrid=200001):
    """Numerical inverse of M(r)/M by linear interpolation on a fine radial table (monotone)."""
    r_grid = np.linspace(0.0, RADIUS, ngrid)
    m_grid = mass_enclosed(r_grid) / TOTAL_MASS
    return np.interp(q, m_grid, r_grid)


def shell_mean_density(r_edges):
    """Exact mass-weighted shell average Delta M / Delta V, for comparing with binned particle
    densities without a bin-curvature bias."""
    r_edges = np.asarray(r_edges, dtype=float)
    dm = np.diff(mass_enclosed(r_edges))
    dv = (4.0 / 3.0) * np.pi * np.diff(r_edges**3)
    return dm / dv


def check_analytic_identities(rtol=1e-8):
    """Numerically verify the closed forms above (mass, U, W, virial, hydrostatic balance)."""
    from scipy.integrate import quad
    m = quad(lambda r: 4 * np.pi * r**2 * rho_analytic(r), 0, RADIUS)[0]
    u = quad(lambda r: 4 * np.pi * r**2 * rho_analytic(r) * u_analytic(r), 0, RADIUS)[0]
    w = quad(lambda r: g_analytic(r) * r * 4 * np.pi * r**2 * rho_analytic(r), 0, RADIUS)[0]
    r = np.linspace(0.01, 0.99, 99)
    balance = np.max(np.abs(g_analytic(r) + hydro_accel_analytic(r)) / np.abs(g_analytic(r)))
    out = {"mass": m, "U": u, "W": w, "virial": 3 * (GAMMA - 1) * u + w, "balance": balance}
    assert abs(m - TOTAL_MASS) < rtol, out
    assert abs(u - U_EXACT) < rtol, out
    assert abs(w - W_EXACT) < rtol, out
    assert abs(out["virial"]) < rtol, out
    assert balance < 1e-8, out
    assert abs(mass_enclosed(RADIUS) - TOTAL_MASS) < 1e-12, out
    return out


def stretch_anisotropy(q):
    """Radial / tangential stretch of the lattice at enclosed mass fraction q: 1 is isotropic."""
    r_old = np.asarray(q, dtype=float) ** (1.0 / 3.0)          # unit-sphere uniform: M_u = r^3
    r_new = radius_of_mass_fraction(q)
    s_t = r_new / r_old
    s_r = 3.0 * r_old**2 / (4.0 * np.pi * r_new**2 * rho_analytic(r_new))
    return s_r / s_t


def _lattice_sphere(n_part, jitter, rng):
    """The n_part points of a jittered cubic lattice closest to the origin (unit spacing)."""
    side = int(np.ceil((3.0 * n_part / (4.0 * np.pi)) ** (1.0 / 3.0))) + 3
    x = np.arange(-side, side) + 0.5      # cell-centred: no point sits on the origin
    pos = np.stack(np.meshgrid(x, x, x, indexing="ij"), axis=-1).reshape(-1, 3)
    pos += jitter * (rng.random(pos.shape) - 0.5) * 2.0
    r = np.sqrt(np.sum(pos**2, axis=1))
    keep = np.argsort(r, kind="stable")[:n_part]
    return pos[keep]


def make_polytrope_n1_ics(outfile, n_part=N_DEFAULT, seed=SEED, jitter=JITTER):
    rng = np.random.default_rng(seed)
    pos = _lattice_sphere(n_part, jitter, rng)
    pos -= pos.mean(axis=0)                      # zero centre of mass for equal masses
    r_old = np.sqrt(np.sum(pos**2, axis=1))
    rank = np.empty(n_part, dtype=np.int64)
    rank[np.argsort(r_old, kind="stable")] = np.arange(n_part)
    q = (rank + 0.5) / n_part                    # enclosed mass fraction of each particle
    r_new = radius_of_mass_fraction(q)
    pos *= (r_new / r_old)[:, None]
    pos -= pos.mean(axis=0)                      # stretch can reintroduce a tiny COM offset

    mass = np.full(n_part, TOTAL_MASS / n_part)
    u = u_analytic(np.sqrt(np.sum(pos**2, axis=1)))
    with h5py.File(outfile, "w") as F:
        F.attrs["make_polytrope_n1_ics_version"] = IC_VERSION
        h = F.create_group("Header")
        npart = np.array([n_part, 0, 0, 0, 0, 0])
        h.attrs["NumPart_ThisFile"] = npart
        h.attrs["NumPart_Total"] = npart
        h.attrs["NumPart_Total_HighWord"] = np.zeros(6, dtype=int)
        h.attrs["MassTable"] = np.zeros(6)
        h.attrs["BoxSize"] = 4.0 * RADIUS        # ignored: non-periodic, centred on the origin
        h.attrs["Time"] = 0.0
        h.attrs["Redshift"] = 0.0
        h.attrs["NumFilesPerSnapshot"] = 1
        h.attrs["Flag_DoublePrecision"] = 0
        p = F.create_group("PartType0")
        p.create_dataset("Coordinates", data=pos)
        p.create_dataset("Velocities", data=np.zeros_like(pos))
        p.create_dataset("Masses", data=mass)
        p.create_dataset("InternalEnergy", data=u)
        p.create_dataset("ParticleIDs", data=np.arange(1, n_part + 1, dtype=np.uint32))
    return outfile


def validate_ics(icfile, nbins=16):
    """Binned shell density vs the analytic shell mean over enclosed mass fractions 5-95%,
    total mass, and thermal energy. Returns a dict of the diagnostics."""
    with h5py.File(icfile, "r") as F:
        pos = F["PartType0/Coordinates"][:]
        m = F["PartType0/Masses"][:]
        u = F["PartType0/InternalEnergy"][:]
    r = np.sqrt(np.sum(pos**2, axis=1))
    edges = radius_of_mass_fraction(np.linspace(0.05, 0.95, nbins + 1))
    msum, _ = np.histogram(r, bins=edges, weights=m)
    rho_bin = msum / ((4.0 / 3.0) * np.pi * np.diff(edges**3))
    rel = rho_bin / shell_mean_density(edges) - 1.0
    return {
        "N": len(m),
        "mass": float(m.sum()),
        "U": float(np.sum(m * u)),
        "rho_maxrel": float(np.max(np.abs(rel))),
        "r_max": float(r.max()),
        "com": np.average(pos, axis=0, weights=m),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=N_DEFAULT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", default="polytrope_n1_ics.hdf5")
    args = parser.parse_args()
    print("analytic identities:", check_analytic_identities())
    make_polytrope_n1_ics(args.out, n_part=args.N, seed=args.seed)
    print("IC diagnostics:", validate_ics(args.out))
    print("radial/tangential stretch at q=0.05,0.5,0.9:", stretch_anisotropy(np.array([0.05, 0.5, 0.9])))
