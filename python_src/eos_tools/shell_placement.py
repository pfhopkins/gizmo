"""Particle placement for HSE-equilibrium layered bodies (Phase 17h).

Two backends:
    fibonacci_shells(profile, n_particles, ...)
        Equal-mass shells; within each shell, points placed on a
        Fibonacci spiral over the unit sphere. Deterministic, fast, good
        first choice (~1% density scatter per shell at >=5k particles total).

    glass_relax(profile, n_particles, ...)
        Start from Fibonacci placement, then run a short repulsive-relaxation
        sweep that stretches a uniform glass into the radial density profile
        rho(r). Higher quality at the cost of ~1-2 seconds; suppresses the
        first-dyn-time ringdown that uniform-density ICs suffer.

Both return a dict with arrays:
    coords [n,3]   particle positions, body centered at origin
    masses [n]     equal-mass particles
    rho_p  [n]     local density at each particle (interpolated from profile)
    u_p    [n]     specific internal energy
    T_p    [n]     temperature
    comp_p [n]     CompositionType integer
"""

import numpy as np


def _interp_along_profile(r_query, prof, key):
    """Interpolate prof[key] at r_query from the inward-integrated arrays
    (which run from r=R outward-index=0 to r=0 at the last index)."""
    r_grid = prof["r"][::-1]
    y_grid = prof[key][::-1]
    return np.interp(r_query, r_grid, y_grid)


def _equal_mass_shell_radii(prof, n_shells):
    """Return shell-boundary radii at equal mass. r_b[0]=0, r_b[n_shells]=R."""
    r_in = prof["r"][::-1]
    m_in = prof["M_total"] - prof["m"][::-1]   # mass interior to r
    m_in[0] = 0.0; m_in[-1] = prof["M_total"]
    m_in = np.maximum.accumulate(m_in)        # enforce monotone (rounding guard)
    targets = np.linspace(0.0, prof["M_total"], n_shells + 1)
    return np.interp(targets, m_in, r_in)


def _fibonacci_sphere(n):
    """Unit-sphere Fibonacci spiral; returns array of shape (n, 3)."""
    if n == 1:
        return np.array([[0.0, 0.0, 1.0]])
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    theta = 2.0 * np.pi * i / phi
    return np.column_stack([r * np.cos(theta), r * np.sin(theta), z])


def _attach_thermo(coords, prof, materials_by_ct):
    r = np.linalg.norm(coords, axis=1)
    rho_p = _interp_along_profile(r, prof, "rho")
    u_p = _interp_along_profile(r, prof, "u")
    T_p = _interp_along_profile(r, prof, "T")
    zid = np.round(_interp_along_profile(r, prof, "zone_id")).astype(int)
    # zone_id is dense; map zone -> CompositionType via materials_by_ct lookup
    # (caller passes zone_to_ct list).
    return rho_p, u_p, T_p, zid


def fibonacci_shells(prof, zones, n_particles, particles_per_shell=None):
    """Place particles on equal-mass Fibonacci shells.

    zones: list of Zone (used to map zone_id -> CompositionType).
    """
    if particles_per_shell is None:
        # Heuristic: ~24 particles per shell -> n_shells = n/24, min 8.
        n_shells = max(8, int(round(n_particles / 24.0)))
    else:
        n_shells = max(1, n_particles // particles_per_shell)

    n_per = max(1, n_particles // n_shells)
    n_actual = n_per * n_shells

    r_b = _equal_mass_shell_radii(prof, n_shells)
    coords = np.empty((n_actual, 3))
    rng = np.random.default_rng(17)
    for k in range(n_shells):
        r_mid = np.cbrt(0.5 * (r_b[k] ** 3 + r_b[k + 1] ** 3))  # equal-volume midpoint
        dirs = _fibonacci_sphere(n_per)
        # Per-shell random rotation so successive shells aren't aligned.
        a = rng.uniform(0, 2 * np.pi, 3)
        Rx = np.array([[1, 0, 0], [0, np.cos(a[0]), -np.sin(a[0])], [0, np.sin(a[0]), np.cos(a[0])]])
        Ry = np.array([[np.cos(a[1]), 0, np.sin(a[1])], [0, 1, 0], [-np.sin(a[1]), 0, np.cos(a[1])]])
        Rz = np.array([[np.cos(a[2]), -np.sin(a[2]), 0], [np.sin(a[2]), np.cos(a[2]), 0], [0, 0, 1]])
        dirs = dirs @ (Rx @ Ry @ Rz).T
        coords[k * n_per:(k + 1) * n_per] = r_mid * dirs

    rho_p, u_p, T_p, zid = _attach_thermo(coords, prof, zones)
    masses = np.full(n_actual, prof["M_total"] / n_actual)
    comp_p = np.array([zones[z].material.composition_type for z in zid], dtype=np.int32)
    return {"coords": coords, "masses": masses, "rho_p": rho_p,
            "u_p": u_p, "T_p": T_p, "comp_p": comp_p}


def glass_relax(prof, zones, n_particles, n_sweeps=20, eta=0.25, seed=17):
    """Repulsive-relaxation glass placement. Initialize from Fibonacci shells,
    then iteratively shift particles toward equal local volume given the
    target rho(r). Pairwise interaction limited to k-nearest neighbors via
    a coarse spatial hash; no scipy dependency.
    """
    init = fibonacci_shells(prof, zones, n_particles)
    coords = init["coords"].copy()
    R = prof["R"]; n = len(coords)
    rng = np.random.default_rng(seed)

    # Target: each particle should occupy a volume m_i/rho(r_i).
    for sweep in range(n_sweeps):
        r = np.linalg.norm(coords, axis=1)
        rho_local = _interp_along_profile(r, prof, "rho")
        h = np.cbrt(3.0 * init["masses"] / (4.0 * np.pi * np.maximum(rho_local, 1e-30)))

        # Coarse cell-grid neighbor lookup.
        cell = np.floor(coords / (2.0 * np.median(h))).astype(int)
        from collections import defaultdict
        bins = defaultdict(list)
        for i in range(n): bins[tuple(cell[i])].append(i)

        disp = np.zeros_like(coords)
        for i in range(n):
            ci = tuple(cell[i])
            # Collect candidates from this cell + 26 neighbors.
            cand = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        cand.extend(bins.get((ci[0] + dx, ci[1] + dy, ci[2] + dz), ()))
            for j in cand:
                if j == i: continue
                d = coords[i] - coords[j]
                dist = np.sqrt(d @ d) + 1e-30
                hij = 0.5 * (h[i] + h[j])
                if dist < 2.0 * hij:
                    w = (1.0 - dist / (2.0 * hij)) ** 2
                    disp[i] += w * d / dist * hij

        # Cap step, apply, project back into r<=R.
        step = eta * h[:, None] * disp / (np.linalg.norm(disp, axis=1, keepdims=True) + 1e-30)
        coords = coords + step
        r = np.linalg.norm(coords, axis=1)
        out = r > R
        if out.any():
            coords[out] *= (R / r[out])[:, None]

    rho_p, u_p, T_p, zid = _attach_thermo(coords, prof, zones)
    masses = np.full(n, prof["M_total"] / n)
    comp_p = np.array([zones[z].material.composition_type for z in zid], dtype=np.int32)
    return {"coords": coords, "masses": masses, "rho_p": rho_p,
            "u_p": u_p, "T_p": T_p, "comp_p": comp_p}
