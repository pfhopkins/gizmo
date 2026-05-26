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
    """Return shell-boundary radii at equal mass. r_b[0]=0, r_b[n_shells]=R.
    prof['m'] is mass enclosed within r (m(R)=M_total, m(0)=0); after [::-1]
    it ascends from 0 to M_total as r ascends from 0 to R."""
    r_in = prof["r"][::-1]
    m_in = prof["m"][::-1].copy()
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


def glass_relax(prof, zones, n_particles, n_sweeps=20, seed=17,
                tol=1e-2, gradient="hybrid", **legacy_kwargs):
    """Stretched-glass placement (meshoid-backed WVT recipe).

    Builds a uniform-density particle glass in the inscribed sphere of the
    unit cube via meshoid's gradient-descent relaxer, then radial-stretches
    each particle so its enclosed mass matches the HSE profile m(r). The
    result is an SPH-density-flat IC matching the target ρ(r) — the Bern-SPH
    / Reinhardt-Stadel "stretched glass" recipe that gives ~percent-level
    rho_SPH/rho_HSE flatness, vs ~50% with the prior repulsive-force version.

    Backwards-compatible signature; n_sweeps maps to meshoid's max iteration
    count, eta (legacy) is silently dropped (meshoid uses adaptive step size).

    Args:
        prof: HSE profile dict from solve_hse().
        zones: list of Zone (used to map zone_id -> CompositionType).
        n_particles: target number of particles inside the body.
        n_sweeps: max meshoid relaxation iterations (default 20). meshoid
            converges when RMS density variation falls below `tol`.
        tol: meshoid convergence tolerance on RMS density variation
            (default 1e-2).
        gradient: meshoid gradient method, "hybrid" | "sph" | "leastsquares"
            (default "hybrid", recommended).
        seed: RNG seed for the initial QMC sample (passed via numpy state).

    Returns: dict with keys coords, masses, rho_p, u_p, T_p, comp_p — same
        schema as fibonacci_shells.
    """
    from meshoid.glass import particle_glass

    # Generate uniform glass in the unit cube. Inscribed unit-sphere fills
    # π/6 ≈ 52% of the cube — oversample so we land near n_particles after
    # rejection.
    n_cube = int(np.ceil(n_particles * 6.0 / np.pi * 1.05))
    np.random.seed(seed)
    coords_cube = particle_glass(N=n_cube, L=1.0, dim=3, tol=tol,
                                 gradient=gradient, num_steps=n_sweeps)
    # particle_glass returns coords in [0, L=1]; recenter on origin.
    x = coords_cube - 0.5
    r_unit = np.linalg.norm(x, axis=1)
    inside = r_unit < 0.5
    x = x[inside]
    r_unit = r_unit[inside]
    # Rescale unit sphere from radius 0.5 to 1.0 for readability of stretch.
    r_unit *= 2.0
    x *= 2.0

    # Trim to exactly n_particles if we got more (after rejection).
    if len(x) > n_particles:
        keep = np.argsort(r_unit)[:n_particles]  # keep innermost — preserves uniformity
        x = x[keep]
        r_unit = r_unit[keep]
    n_actual = len(x)

    # Radial stretch: a uniform-density unit sphere has M_enclosed(r) =
    # M_total * r³. Invert prof['m'](r) = r_unit³ * M_total to find new
    # radius for each particle.
    r_grid = prof["r"][::-1]
    m_grid = prof["m"][::-1].copy()
    m_grid[0] = 0.0
    m_grid[-1] = prof["M_total"]
    m_grid = np.maximum.accumulate(m_grid)
    target_M_enc = (r_unit ** 3) * prof["M_total"]
    r_new = np.interp(target_M_enc, m_grid, r_grid)

    # Apply stretch. Origin handled via tiny-r clamp.
    safe_r = np.maximum(r_unit, 1e-30)
    coords = x * (r_new / safe_r)[:, None]

    rho_p, u_p, T_p, zid = _attach_thermo(coords, prof, zones)
    masses = np.full(n_actual, prof["M_total"] / n_actual)
    comp_p = np.array([zones[z].material.composition_type for z in zid], dtype=np.int32)
    return {"coords": coords, "masses": masses, "rho_p": rho_p,
            "u_p": u_p, "T_p": T_p, "comp_p": comp_p}


def atmosphere_shell(prof, zones, n_particles,
                     rho_atmo, T_atmo,
                     R_outer=None, R_inner=None,
                     material=None, particles_per_shell=None,
                     rho_taper_power=None, equal_mass_to=None, seed=23):
    """Place a low-density "atmosphere" shell around an HSE body to cure
    the isolated-SPH-body-in-vacuum surface pathology.

    Standard recipe in the SPH-planet literature (Bern-SPH, Reinhardt &
    Stadel 2017): without an atmosphere, surface kernel sums are
    asymmetric and the outer body cells get accelerated outward by the
    pressure cliff into vacuum, ejecting them with runaway u.

    Particles are placed on equal-volume Fibonacci shells from R_inner to
    R_outer at constant rho_atmo and T_atmo. Material defaults to the
    outermost body zone's material so the runtime EOS path is identical
    to the body. Internal energy is set by `material.u_from_T(rho, T)`.

    Returns a dict with the same schema as fibonacci_shells/glass_relax:
        coords, masses, rho_p, u_p, T_p, comp_p (int32 CompositionType).

    Tunables:
        rho_atmo  — atmosphere density [g/cm^3]. Typical values are
                    ~1e-4 .. 1e-2 of the body's surface density.
        T_atmo    — atmosphere temperature [K]. Set close to the body's
                    surface T (T_outer of zones[0]).
        R_inner   — inner shell radius. Defaults to prof['R'] (body
                    surface). Set slightly outside R if you want a thin
                    gap (rarely needed).
        R_outer   — outer shell radius. Defaults to 2*prof['R'].
        material  — Material instance. Defaults to zones[0].material.
        rho_taper_power — if not None, use a power-law ρ(r) = rho_atmo *
                    (R_inner/r)**n with n = rho_taper_power. This makes
                    most of the atmosphere mass sit near the body and the
                    outer edge approach vacuum smoothly (particle masses
                    scale with the local target ρ). Recommended n=4..6
                    for SPH-isolated-body protection. Default None
                    (constant ρ) for simplicity.
        equal_mass_to — if a `placed` dict (e.g. the body), atmosphere
                    particles take its per-particle mass; rho_atmo +
                    rho_taper_power then determine particle SPACING (i.e.
                    n_particles is auto-derived from rho profile + that
                    mass). This gives a Lagrangian-mass-matched IC, which
                    SPH/MFM handle better at body↔atmo boundary than a
                    mass jump.
    """
    if material is None:
        material = zones[0].material
    if R_inner is None:
        R_inner = prof["R"]
    if R_outer is None:
        R_outer = 2.0 * prof["R"]
    if R_outer <= R_inner:
        raise ValueError(f"R_outer={R_outer} must exceed R_inner={R_inner}")

    if particles_per_shell is None:
        n_shells = max(2, int(round(n_particles / 24.0)))
    else:
        n_shells = max(1, n_particles // particles_per_shell)
    n_per = max(1, n_particles // n_shells)
    n_actual = n_per * n_shells

    # Equal-volume shell spacing across [R_inner, R_outer].
    Vmin = R_inner ** 3
    Vmax = R_outer ** 3
    V_b = np.linspace(Vmin, Vmax, n_shells + 1)
    r_b = np.cbrt(V_b)

    coords = np.empty((n_actual, 3))
    rng = np.random.default_rng(seed)
    for k in range(n_shells):
        r_mid = np.cbrt(0.5 * (r_b[k] ** 3 + r_b[k + 1] ** 3))
        dirs = _fibonacci_sphere(n_per)
        a = rng.uniform(0, 2 * np.pi, 3)
        Rx = np.array([[1, 0, 0], [0, np.cos(a[0]), -np.sin(a[0])], [0, np.sin(a[0]), np.cos(a[0])]])
        Ry = np.array([[np.cos(a[1]), 0, np.sin(a[1])], [0, 1, 0], [-np.sin(a[1]), 0, np.cos(a[1])]])
        Rz = np.array([[np.cos(a[2]), -np.sin(a[2]), 0], [np.sin(a[2]), np.cos(a[2]), 0], [0, 0, 1]])
        dirs = dirs @ (Rx @ Ry @ Rz).T
        coords[k * n_per:(k + 1) * n_per] = r_mid * dirs

    # Per-particle thermo. ρ either constant or power-law tapered toward
    # vacuum at the outer edge. T fixed (no thermal taper — the body's
    # gravitational scale height for an isothermal rocky atmosphere is
    # tiny compared to R_body; physical atmosphere is ~negligible. The
    # shell exists to give body cells SPH neighbors, not to be physical).
    r_p = np.linalg.norm(coords, axis=1)
    if rho_taper_power is None:
        rho_p = np.full(n_actual, rho_atmo, dtype=np.float64)
    else:
        rho_p = rho_atmo * (R_inner / np.maximum(r_p, R_inner)) ** float(rho_taper_power)
    T_p   = np.full(n_actual, T_atmo, dtype=np.float64)
    u_p   = np.array([material.u_from_T(rho_p[i], T_atmo) for i in range(n_actual)],
                     dtype=np.float64)
    comp_p = np.full(n_actual, material.composition_type, dtype=np.int32)

    # Mass: by default, each particle gets a mass proportional to its
    # local target ρ times its share of the shell volume (equal-volume
    # shells → equal volume per particle). For constant-ρ this is
    # uniform; for tapered ρ, outer particles are lighter → their
    # ejection contributes negligibly to total momentum.
    #
    # equal_mass_to=body forces atmo particles to share the body's mass
    # per particle, eliminating the body↔atmo mass discontinuity that
    # otherwise breaks SPH/MFM closure at the surface.
    if equal_mass_to is None:
        V_per_particle = ((4.0 / 3.0) * np.pi * (R_outer ** 3 - R_inner ** 3)) / n_actual
        masses = rho_p * V_per_particle
    else:
        m_body = float(np.mean(equal_mass_to["masses"]))
        masses = np.full(n_actual, m_body, dtype=np.float64)

    return {"coords": coords, "masses": masses, "rho_p": rho_p,
            "u_p": u_p, "T_p": T_p, "comp_p": comp_p}


def merge_placements(*placements):
    """Concatenate placements (e.g. body + atmosphere) into one dict with
    the same schema. Particle order is the order of arguments."""
    keys = ("coords", "masses", "rho_p", "u_p", "T_p", "comp_p")
    out = {k: np.concatenate([p[k] for p in placements], axis=0) for k in keys}
    return out
