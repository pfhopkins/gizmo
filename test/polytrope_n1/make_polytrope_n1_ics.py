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

Particles: N equal masses m = M/N on concentric Fibonacci-sphere shells (shell_layout()). With
the local spacing l(r) = (m/rho(r))^(1/3), shell boundaries sit at equal steps of
s(r) = int_0^r dr'/l(r'), so each shell is about one spacing thick; its particle count is fixed
by the analytic mass between its boundaries (rounded to whole particles), and the particles sit at
the shell's mass-weighted mean radius. Then the tangential spacing sqrt(4 pi r^2 / N_k) equals
the shell thickness, and the shell masses follow M(r) exactly. A single particle sits at the
centre. Each shell gets an independent seeded random rotation. Zero velocities, u from the
analytic rho at each particle's radius. An optional relaxation (relax_positions()) is off by
default: see its docstring.
"""
import argparse
import functools
import time

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
SEED = 1
Q_MAX = 1.0      # enclosed mass fraction where the regular shell walk stops (see shell_layout)
JITTER = 0.0     # uniform per-axis position jitter, in local spacings l(r)
RELAX_ITERS = 0  # relax_positions() iterations; 0 = off
N_NGB = 32.0     # DesNumNgb in polytrope_n1.params, for the offline kernel density
IC_VERSION = 2   # bump when the construction changes, so _ensure_ic() regenerates stale files

MASS_BANDS = ((0.0, 0.05), (0.05, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 0.9),
              (0.9, 0.95), (0.95, 1.0))


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


# ------------------------------------------------------------------------------------------
# particle arrangement
# ------------------------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _radial_tables(ngrid=400001):
    """r grid, M(r), and the cumulative integrals int rho^(1/3) dr and int r dM on it."""
    r = np.linspace(0.0, RADIUS, ngrid)

    def cumtrapz(f):
        return np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(r))])

    rho = rho_analytic(r)
    return r, mass_enclosed(r), cumtrapz(rho ** (1.0 / 3.0)), cumtrapz(4.0 * np.pi * r**3 * rho)


def shell_layout(n_part=N_DEFAULT, q_max=Q_MAX):
    """Shell particle counts and radii. Returns (counts, r_shell, r_inner, r_outer).

    In units of the local spacing l(r) = (m/rho)^(1/3), the shell coordinate s(r) =
    int_0^r dr'/l runs from 0 to a finite s(R) (rho -> 0 only linearly at the surface). The
    first boundary is at s0 = (3/4pi)^(1/3), which encloses one particle's mass: that particle
    sits at the centre. The remaining boundaries are evenly spaced in s up to s(r(q_max)), with
    the step rescaled (by less than 1/(2 n_shells)) so a whole number of shells fits. The
    enclosed particle count at each boundary is the analytic M(r) rounded to an integer, so each
    shell's mass interval is exact to within one particle and its radial edges are taken at
    M^-1 of those integer counts. Particles go at the mass-weighted mean radius of their shell.

    q_max < 1 stops the regular walk at that enclosed mass fraction and puts all the remaining
    mass in one final shell at its mass-weighted mean radius. The total mass and the analytic
    solution are unchanged either way; only the arrangement of the outer 1 - q_max differs.
    """
    m = TOTAL_MASS / n_part
    r_g, m_g, s_rho, i_rdm = _radial_tables()
    s_g = s_rho / m ** (1.0 / 3.0)
    s0 = (3.0 / (4.0 * np.pi)) ** (1.0 / 3.0)
    s_end = np.interp(radius_of_mass_fraction(q_max), r_g, s_g)
    n_shell = max(int(round(s_end - s0)), 1)
    r_b = np.interp(s0 + (s_end - s0) * np.arange(n_shell + 1) / n_shell, s_g, r_g)
    cum = np.rint(mass_enclosed(r_b) / m).astype(np.int64)
    cum = np.unique(np.clip(np.concatenate([[0], cum, [n_part]]), 0, n_part))
    counts = np.diff(cum)
    r_in = np.interp(cum[:-1] * m, m_g, r_g)
    r_out = np.interp(cum[1:] * m, m_g, r_g)
    r_shell = (np.interp(r_out, r_g, i_rdm) - np.interp(r_in, r_g, i_rdm)) / (counts * m)
    if counts[0] == 1:
        r_shell[0] = 0.0
    return counts, r_shell, r_in, r_out


def _fibonacci_sphere(k):
    """k nearly uniformly spaced unit vectors (golden-angle spiral, equal-area in z)."""
    i = np.arange(k) + 0.5
    z = 1.0 - 2.0 * i / k
    phi = np.pi * (3.0 - np.sqrt(5.0)) * i
    rho_xy = np.sqrt(1.0 - z * z)
    return np.stack([rho_xy * np.cos(phi), rho_xy * np.sin(phi), z], axis=1)


def _random_rotation(rng):
    """Uniformly distributed rotation matrix, from a random unit quaternion."""
    a, b, c, d = (lambda q: q / np.linalg.norm(q))(rng.normal(size=4))
    return np.array([
        [a*a + b*b - c*c - d*d, 2*(b*c - a*d), 2*(b*d + a*c)],
        [2*(b*c + a*d), a*a - b*b + c*c - d*d, 2*(c*d - a*b)],
        [2*(b*d - a*c), 2*(c*d + a*b), a*a - b*b - c*c + d*d]])


def fibonacci_shell_positions(n_part=N_DEFAULT, seed=SEED, q_max=Q_MAX, jitter=JITTER):
    rng = np.random.default_rng(seed)
    counts, r_shell, _, _ = shell_layout(n_part, q_max)
    pos = np.concatenate([
        r * (_fibonacci_sphere(k) @ _random_rotation(rng).T) if r > 0 else np.zeros((k, 3))
        for k, r in zip(counts, r_shell)])
    if jitter > 0:
        r = np.sqrt(np.sum(pos**2, axis=1))
        pos += jitter * _local_spacing(r, TOTAL_MASS / n_part)[:, None] * (2.0 * rng.random(pos.shape) - 1.0)
    return pos - pos.mean(axis=0)            # equal masses: zero centre of mass


# ------------------------------------------------------------------------------------------
# offline cubic-spline kernel density (GIZMO's definition) and relaxation
# ------------------------------------------------------------------------------------------

def _cubic_spline(u):
    """GIZMO's cubic spline with compact support at u = r/H = 1, without the 8/(pi H^3)."""
    w = np.zeros_like(u)
    a = u < 0.5
    b = ~a & (u < 1.0)
    w[a] = 1.0 - 6.0 * u[a]**2 + 6.0 * u[a]**3
    w[b] = 2.0 * (1.0 - u[b])**3
    return w


def _cubic_spline_deriv(u):
    g = np.zeros_like(u)
    a = u < 0.5
    b = ~a & (u < 1.0)
    g[a] = -12.0 * u[a] + 18.0 * u[a]**2
    g[b] = -6.0 * (1.0 - u[b])**2
    return g


def _solve_kernel_radius(d, n_ngb, iters=40):
    """Per-row bisection for H with (4 pi/3) H^3 sum_j W(d_j, H) = n_ngb (row 0 is the particle
    itself, d = 0). Returns H and a flag for rows whose neighbour list was too short."""
    lo, hi = 0.5 * d[:, 1], d[:, -1].copy()
    for _ in range(iters):
        h = 0.5 * (lo + hi)
        over = (32.0 / 3.0) * _cubic_spline(d / h[:, None]).sum(axis=1) > n_ngb
        hi = np.where(over, h, hi)
        lo = np.where(over, lo, h)
    h = 0.5 * (lo + hi)
    return h, h > 0.999 * d[:, -1]


def kernel_density(pos, mass, n_ngb=N_NGB, k=64, return_neighbours=False):
    """Density as GIZMO's density loop computes it: rho_i = sum_j m_j W(r_ij, H_i), self
    included, with H_i such that the effective neighbour number (4 pi/3) H_i^3 sum_j W = n_ngb.
    GIZMO additionally raises the neighbour number (up to 2x) where the kernel is badly
    one-sided (FaceClosureError) or ill-conditioned; that is not reproduced here, so expect
    differences near the free surface."""
    from scipy.spatial import cKDTree
    tree = cKDTree(pos)
    d, idx = tree.query(pos, k=k)
    h, short = _solve_kernel_radius(d, n_ngb)
    rho = (8.0 / np.pi) / h**3 * np.sum(mass[idx] * _cubic_spline(d / h[:, None]), axis=1)
    for i in np.flatnonzero(short):      # rare: a sparse surface particle needs a longer list
        d_i, idx_i = tree.query(pos[i], k=8 * k)
        h_i, _ = _solve_kernel_radius(d_i[None, :], n_ngb)
        h[i] = h_i[0]
        rho[i] = (8.0 / np.pi) / h[i]**3 * np.sum(mass[idx_i] * _cubic_spline(d_i / h[i]))
    if return_neighbours:
        return rho, h, d, idx
    return rho, h


def relax_positions(pos, mass, iters, step=1.0, n_ngb=N_NGB):
    """Move each particle down the SPH gradient of eps = rho_kernel/rho_analytic - 1, by
    step * l^2 * grad(eps) (l = local spacing), capped at 0.1 l per iteration; deterministic.

    On the Fibonacci shells this is not worth it, so RELAX_ITERS = 0. With N = 2^15, 10
    iterations (about 15 s) take the kernel-density rms in 5-90% enclosed mass from 0.4% to
    0.1-0.3%, almost all of it by removing the -0.3% mean kernel bias. It does that by moving
    mass radially (90% Lagrange radius -0.4%, U +0.15%), trading the exact analytic M(r) for a
    kernel-density match."""
    pos = pos.copy()
    for _ in range(iters):
        rho, h, d, idx = kernel_density(pos, mass, n_ngb, return_neighbours=True)
        r = np.sqrt(np.sum(pos**2, axis=1))
        rho_a = np.maximum(rho_analytic(r), 1e-3 * RHO_C)
        eps = rho / rho_a - 1.0
        dwdr_over_r = (8.0 / np.pi) / h[:, None]**4 * _cubic_spline_deriv(d / h[:, None]) / np.maximum(d, 1e-30)
        wgt = (mass[idx] / rho[idx]) * (eps[idx] - eps[:, None]) * dwdr_over_r
        grad = np.einsum("ij,ijk->ik", wgt, pos[:, None, :] - pos[idx])
        spacing = _local_spacing(r, mass)
        cap = 0.1 * spacing[:, None]
        pos -= np.clip(step * spacing[:, None]**2 * grad, -cap, cap)
        pos -= np.average(pos, axis=0, weights=mass)
    return pos


# ------------------------------------------------------------------------------------------
# IC file
# ------------------------------------------------------------------------------------------

def make_polytrope_n1_ics(outfile, n_part=N_DEFAULT, seed=SEED, q_max=Q_MAX, jitter=JITTER,
                          relax_iters=RELAX_ITERS):
    pos = fibonacci_shell_positions(n_part, seed, q_max, jitter)
    mass = np.full(n_part, TOTAL_MASS / n_part)
    if relax_iters > 0:
        pos = relax_positions(pos, mass, relax_iters)
    u = u_analytic(np.sqrt(np.sum(pos**2, axis=1)))
    with h5py.File(outfile, "w") as F:
        F.attrs["make_polytrope_n1_ics_version"] = IC_VERSION
        F.attrs["make_polytrope_n1_ics_args"] = (f"seed={seed} q_max={q_max} jitter={jitter} "
                                                 f"relax_iters={relax_iters}")
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


# ------------------------------------------------------------------------------------------
# truncated star in a hydrostatic ambient medium (the polytrope_n1_ambient IC)
# ------------------------------------------------------------------------------------------

Q_CUT = 0.95     # the star keeps the shells inside this enclosed mass fraction (snapped to a shell boundary)
F_AMB = 0.1      # ambient/star density contrast at the interface; ambient cells have mass F_AMB * m
BOX_AMB = 5.0    # side of the periodic box (hydro only; gravity is not periodic), star at its centre


def _cube_sphere_fraction(r, half, n_dir=20000):
    """Fraction of the sphere of radius r, centred in the cube [-half, half]^3, that lies in the cube."""
    nmax = np.max(np.abs(_fibonacci_sphere(n_dir)), axis=1)
    return np.array([np.mean(nmax * x < half) for x in np.atleast_1d(r)])


def ambient_profile(q_cut=Q_CUT, f_amb=F_AMB, box=BOX_AMB, n_part=N_DEFAULT, ngrid=4001, iters=30):
    """The truncation radius and the isothermal (constant u) atmosphere outside it.

    The star is the regular shells of shell_layout(n_part, q_max=q_cut), i.e. the standard
    construction with its step rescaled so a shell boundary falls at q_cut, without the final lumped
    shell: r_cut is that boundary and the star's mass q_cut * M is exact. At r_cut the ambient has
    rho_amb = f_amb rho(r_cut) and the star's pressure P_cut = K rho(r_cut)^2, so
    u_amb = P_cut / ((gamma-1) rho_amb); outside, hydrostatic balance at constant u gives
    dln(rho)/dr = -G M(r) / ((gamma-1) u_amb r^2), with M(r) = star + the atmosphere inside r that is
    also inside the box (spherically averaged; iterated). Interior gravity is unchanged (shell theorem),
    so the analytic star solution still holds inside r_cut."""
    counts, _, _, r_out = shell_layout(n_part, q_cut)
    n_star, r_cut = int(np.sum(counts[:-1])), float(r_out[-2])
    m_star = n_star * TOTAL_MASS / n_part
    rho_cut = float(rho_analytic(r_cut))
    p_cut = K_POLY * rho_cut**2
    rho0 = f_amb * rho_cut
    u_amb = p_cut / ((GAMMA - 1.0) * rho0)
    half = 0.5 * box
    r = np.linspace(r_cut, half * np.sqrt(3.0), ngrid)
    frac = _cube_sphere_fraction(r, half)

    def cumtrapz(f):
        return np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(r))])

    rho = np.full_like(r, rho0)
    for _ in range(iters):
        m_enc = m_star + cumtrapz(4.0 * np.pi * r**2 * frac * rho)
        rho = rho0 * np.exp(-cumtrapz(G * m_enc / r**2) / ((GAMMA - 1.0) * u_amb))
    return {"n_star": n_star, "q_cut": m_star / TOTAL_MASS, "r_cut": r_cut, "rho_cut": rho_cut, "p_cut": p_cut,
            "rho_amb0": rho0, "u_amb": u_amb, "box": box, "r": r, "rho": rho, "m_enc": m_enc,
            "m_amb": float(m_enc[-1] - m_star), "f_amb": f_amb}


def _ambient_positions(prof, m_amb, rng):
    """Fibonacci shells continuing the star's construction outside r_cut with the ambient density
    and cell mass: spacing (m_amb/rho_amb)^(1/3) equals the star's at r_cut. Shell counts follow the
    full-sphere mass; cells outside the box are dropped."""
    r, rho = prof["r"], prof["rho"]

    def cumtrapz(f):
        return np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(r))])

    s = cumtrapz((rho / m_amb) ** (1.0 / 3.0))
    m_full = cumtrapz(4.0 * np.pi * r**2 * rho)
    i_rdm = cumtrapz(4.0 * np.pi * r**3 * rho)
    n_shell = max(int(round(s[-1])), 1)
    r_b = np.interp(s[-1] * np.arange(n_shell + 1) / n_shell, s, r)
    cum = np.unique(np.rint(np.interp(r_b, r, m_full) / m_amb).astype(np.int64))
    counts = np.diff(cum)
    r_in = np.interp(cum[:-1] * m_amb, m_full, r)
    r_out = np.interp(cum[1:] * m_amb, m_full, r)
    r_shell = (np.interp(r_out, r, i_rdm) - np.interp(r_in, r, i_rdm)) / (counts * m_amb)
    pos = np.concatenate([x * (_fibonacci_sphere(k) @ _random_rotation(rng).T) for k, x in zip(counts, r_shell)])
    return pos[np.all(np.abs(pos) < 0.5 * prof["box"], axis=1)]


def make_polytrope_n1_ambient_ics(outfile, n_part=N_DEFAULT, seed=SEED, q_cut=Q_CUT, f_amb=F_AMB, box=BOX_AMB):
    """The polytrope truncated at q_cut (see ambient_profile(); same cell mass and u as the standard IC)
    inside a hydrostatic ambient medium at the truncation pressure (ambient_profile()), filling a
    periodic box of side `box` centred on the star. Star cells have ParticleIDs 1..n_star."""
    prof = ambient_profile(q_cut, f_amb, box, n_part)
    m = TOTAL_MASS / n_part
    pos_star = fibonacci_shell_positions(n_part, seed, q_max=q_cut)[:prof["n_star"]]
    pos_star -= pos_star.mean(axis=0)
    pos_amb = _ambient_positions(prof, f_amb * m, np.random.default_rng(seed + 1))
    pos = np.concatenate([pos_star, pos_amb])
    n_amb = len(pos_amb)
    mass = np.concatenate([np.full(prof["n_star"], m), np.full(n_amb, f_amb * m)])
    u = np.concatenate([u_analytic(np.sqrt(np.sum(pos_star**2, axis=1))), np.full(n_amb, prof["u_amb"])])
    n_tot = len(mass)
    with h5py.File(outfile, "w") as F:
        F.attrs["make_polytrope_n1_ics_version"] = IC_VERSION
        F.attrs["make_polytrope_n1_ics_args"] = f"ambient seed={seed} q_cut={q_cut} f_amb={f_amb} box={box}"
        for key in ("n_star", "q_cut", "r_cut", "p_cut", "rho_amb0", "u_amb", "m_amb"):
            F.attrs[key] = prof[key]
        h = F.create_group("Header")
        npart = np.array([n_tot, 0, 0, 0, 0, 0])
        h.attrs["NumPart_ThisFile"] = npart
        h.attrs["NumPart_Total"] = npart
        h.attrs["NumPart_Total_HighWord"] = np.zeros(6, dtype=int)
        h.attrs["MassTable"] = np.zeros(6)
        h.attrs["BoxSize"] = box
        h.attrs["Time"] = 0.0
        h.attrs["Redshift"] = 0.0
        h.attrs["NumFilesPerSnapshot"] = 1
        h.attrs["Flag_DoublePrecision"] = 0
        p = F.create_group("PartType0")
        p.create_dataset("Coordinates", data=pos + 0.5 * box)
        p.create_dataset("Velocities", data=np.zeros_like(pos))
        p.create_dataset("Masses", data=mass)
        p.create_dataset("InternalEnergy", data=u)
        p.create_dataset("ParticleIDs", data=np.arange(1, n_tot + 1, dtype=np.uint32))
    return prof, n_amb


# ------------------------------------------------------------------------------------------
# diagnostics
# ------------------------------------------------------------------------------------------

def smoothed_enclosed_mass(r_eval, r, mass, width):
    """Mass inside each radius r_eval, with each particle's mass spread uniformly over
    [r - width/2, r + width/2]. With width = the local spacing (m/rho)^(1/3) this removes the
    step structure of a layered (shell) arrangement, whose raw M(r) jumps by a whole shell at
    each shell radius; on a smooth profile the bias is O(width^2 M''/M'), negligible here."""
    r_eval = np.atleast_1d(np.asarray(r_eval, dtype=float))
    return np.array([np.sum(mass * np.clip((x - r) / width + 0.5, 0.0, 1.0)) for x in r_eval])


def smoothed_lagrange_radii(r, mass, width, fractions, iters=50):
    """Radii enclosing the given mass fractions of smoothed_enclosed_mass(), by bisection."""
    target = np.asarray(fractions, dtype=float) * mass.sum()
    lo, hi = np.zeros_like(target), np.full_like(target, np.max(r + width))
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        below = smoothed_enclosed_mass(mid, r, mass, width) < target
        lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
    return 0.5 * (lo + hi)


def _local_spacing(r, mass):
    return (mass / np.maximum(rho_analytic(r), 1e-3 * RHO_C)) ** (1.0 / 3.0)


def _band_masks(q, bands=MASS_BANDS):
    return [(q >= a) & ((q < b) if b < 1.0 else np.ones_like(q, dtype=bool)) for a, b in bands]


def cell_anisotropy(pos, samples_per_particle=32, seed=0):
    """Per-particle radial and tangential second moments of the Voronoi cell, by Monte Carlo:
    uniform points in the sphere of radius RADIUS are assigned to their nearest particle. For a
    box-shaped cell, sqrt(2 <x_r^2> / <x_t^2>) is the radial/tangential spacing ratio; 1 is
    isotropic. Surface cells extend to RADIUS. Returns (sum x_r^2, sum x_t^2) per particle."""
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(seed)
    n = len(pos)
    ns = samples_per_particle * n
    x = rng.normal(size=(ns, 3))
    x *= (RADIUS * rng.random(ns) ** (1.0 / 3.0) / np.sqrt(np.sum(x**2, axis=1)))[:, None]
    _, j = cKDTree(pos).query(x)
    off = x - pos[j]
    r = np.sqrt(np.sum(pos**2, axis=1))
    rhat = pos[j] / np.maximum(r[j], 1e-30)[:, None]
    xr2 = np.einsum("ij,ij->i", off, rhat) ** 2
    xt2 = np.einsum("ij,ij->i", off, off) - xr2
    return np.bincount(j, xr2, n), np.bincount(j, xt2, n)


def validate_ics(icfile, nbins=16):
    """IC diagnostics. Per enclosed-mass band (MASS_BANDS): rms and mean of the per-particle
    rho_kernel/rho_analytic - 1 (kernel_density(), what MFM sees) and the Voronoi-cell
    radial/tangential spacing ratio (cell_anisotropy()); also the binned shell density vs the
    analytic shell mean over 5-95% (rho_maxrel) and the 10/50/90% Lagrange radii vs analytic,
    both from smoothed_enclosed_mass(); total mass, thermal energy, centre of mass."""
    with h5py.File(icfile, "r") as F:
        pos = F["PartType0/Coordinates"][:].astype(float)
        m = F["PartType0/Masses"][:].astype(float)
        u = F["PartType0/InternalEnergy"][:].astype(float)
    r = np.sqrt(np.sum(pos**2, axis=1))
    width = _local_spacing(r, m)
    edges = radius_of_mass_fraction(np.linspace(0.05, 0.95, nbins + 1))
    msum = np.diff(smoothed_enclosed_mass(edges, r, m, width))
    rel = msum / ((4.0 / 3.0) * np.pi * np.diff(edges**3)) / shell_mean_density(edges) - 1.0
    lagrange_q = (0.1, 0.5, 0.9)
    lagrange_rel = smoothed_lagrange_radii(r, m, width, lagrange_q) / radius_of_mass_fraction(lagrange_q) - 1.0

    rho_k, _ = kernel_density(pos, m)
    eps = rho_k / np.maximum(rho_analytic(r), 1e-30) - 1.0
    xr2, xt2 = cell_anisotropy(pos)
    q = mass_enclosed(r) / TOTAL_MASS
    bands = []
    for (a, b), sel in zip(MASS_BANDS, _band_masks(q)):
        ok = sel.any()
        bands.append({"band": (a, b), "n": int(sel.sum()),
                      "rho_kernel_rms": float(np.sqrt(np.mean(eps[sel]**2))) if ok else np.nan,
                      "rho_kernel_mean": float(np.mean(eps[sel])) if ok else np.nan,
                      "anisotropy": float(np.sqrt(2.0 * xr2[sel].sum() / xt2[sel].sum())) if ok else np.nan})
    return {
        "N": len(m),
        "mass": float(m.sum()),
        "U": float(np.sum(m * u)),
        "U_exact": U_EXACT,
        "rho_maxrel": float(np.max(np.abs(rel))),
        "lagrange_rel": dict(zip(lagrange_q, lagrange_rel.tolist())),
        "r_max": float(r.max()),
        "com": np.average(pos, axis=0, weights=m),
        "bands": bands,
    }


def compare_kernel_density(snapshot):
    """Offline kernel_density() vs GIZMO's Density on a snapshot's own coordinates, per band:
    rms of rho_offline/rho_GIZMO - 1, and both codes' rms error against rho_analytic."""
    with h5py.File(snapshot, "r") as F:
        g = F["PartType0"]
        pos, m, rho_g = (g[k][:].astype(float) for k in ("Coordinates", "Masses", "Density"))
    pos -= np.average(pos, axis=0, weights=m)
    rho_o, _ = kernel_density(pos, m)
    r = np.sqrt(np.sum(pos**2, axis=1))
    rho_a = np.maximum(rho_analytic(r), 1e-30)
    q = mass_enclosed(r) / TOTAL_MASS
    rms = lambda x: float(np.sqrt(np.mean(x**2)))   # noqa: E731
    return [{"band": band, "offline_vs_gizmo_rms": rms(rho_o[s] / rho_g[s] - 1),
             "gizmo_rms": rms(rho_g[s] / rho_a[s] - 1), "offline_rms": rms(rho_o[s] / rho_a[s] - 1)}
            for band, s in zip(MASS_BANDS, _band_masks(q))]


def _print_validation(diag):
    print(f"N {diag['N']}, mass {diag['mass']:.12f}, U {diag['U']:.6f} (exact {diag['U_exact']}), "
          f"COM {np.array2string(diag['com'], precision=2)}, r_max {diag['r_max']:.4f}, "
          f"binned shell density max |rel| (5-95%) {diag['rho_maxrel']:.2e}, Lagrange radii rel "
          f"{ {q: round(v, 5) for q, v in diag['lagrange_rel'].items()} }")
    print("  mass band   n      rho_kernel/rho - 1: rms    mean    | cell radial/tangential")
    for b in diag["bands"]:
        lo, hi = b["band"]
        print(f"  {100 * lo:3.0f}-{100 * hi:3.0f}%  {b['n']:6d}  {b['rho_kernel_rms']:22.4f} "
              f"{b['rho_kernel_mean']:+8.4f}  | {b['anisotropy']:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=N_DEFAULT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--q-max", type=float, default=Q_MAX)
    parser.add_argument("--jitter", type=float, default=JITTER)
    parser.add_argument("--relax-iters", type=int, default=RELAX_ITERS)
    parser.add_argument("--out", default="polytrope_n1_ics.hdf5")
    parser.add_argument("--ambient-q-cut", type=float, default=None,
                        help="write the truncated star in an ambient medium instead (make_polytrope_n1_ambient_ics)")
    parser.add_argument("--f-amb", type=float, default=F_AMB)
    parser.add_argument("--box", type=float, default=BOX_AMB)
    parser.add_argument("--validate-only", action="store_true", help="diagnose an existing --out file")
    parser.add_argument("--snapshot", default=None,
                        help="also compare the offline kernel density with a GIZMO snapshot's Density")
    args = parser.parse_args()
    print("analytic identities:", check_analytic_identities())
    if args.ambient_q_cut is not None:
        prof, n_amb = make_polytrope_n1_ambient_ics(args.out, n_part=args.N, seed=args.seed, q_cut=args.ambient_q_cut,
                                                    f_amb=args.f_amb, box=args.box)
        print(f"wrote {args.out}: {prof['n_star']} star cells (q_cut {prof['q_cut']:.5f}, r_cut {prof['r_cut']:.5f}, "
              f"P_cut {prof['p_cut']:.4e}), {n_amb} ambient cells (u {prof['u_amb']:.4f}, mass {prof['m_amb']:.4f})")
        raise SystemExit(0)
    if not args.validate_only:
        t0 = time.time()
        make_polytrope_n1_ics(args.out, n_part=args.N, seed=args.seed, q_max=args.q_max,
                              jitter=args.jitter, relax_iters=args.relax_iters)
        counts, r_shell, r_in, r_out = shell_layout(args.N, args.q_max)
        ratio = (r_out - r_in)[2:-1] / np.sqrt(4 * np.pi * r_shell[2:-1]**2 / counts[2:-1])
        print(f"wrote {args.out} in {time.time() - t0:.2f} s: {len(counts)} shells, counts "
              f"{counts[:4].tolist()} ... {counts[-3:].tolist()}; shell thickness / tangential "
              f"spacing {ratio.min():.3f}-{ratio.max():.3f} (shells 2..n-1)")
    _print_validation(validate_ics(args.out))
    if args.snapshot:
        print(f"offline kernel density vs GIZMO Density in {args.snapshot}:")
        for b in compare_kernel_density(args.snapshot):
            lo, hi = b["band"]
            print(f"  {100 * lo:3.0f}-{100 * hi:3.0f}%  offline/GIZMO - 1 rms {b['offline_vs_gizmo_rms']:.1e}; "
                  f"vs analytic: GIZMO rms {b['gizmo_rms']:.4f}, offline rms {b['offline_rms']:.4f}")
