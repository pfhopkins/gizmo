"""Particle placement for HSE-equilibrium layered bodies.

seagen_shells(profile, zones, n_particles, mass_ratio="spacing")
    SEAGen shells (needs the optional seagen package), one material per zone, with the particle mass of each
    zone chosen so the particle spacing matches across each material interface. The recommended placement
    for tabulated-EOS planets.

Two further backends:
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


def shells_of(r, rel_gap=1e-3):
    """shell index (0 = innermost) of each particle of one material: SEAGen places a material's particles on spheres, so their
    radii cluster (to within the small offset between the shells' centre and the body's centre of mass, far below the spacing)"""
    o = np.argsort(r); rs = r[o]; cut = np.concatenate([[0], np.cumsum(np.diff(rs) / rs[1:] > rel_gap)])
    idx = np.empty(len(r), int); idx[o] = cut
    return idx


def _zone_profiles(prof):
    """outward (ascending r, r > 0) profile arrays and the zone index of each sample"""
    r = prof["r"][::-1]; keep = r > 0
    return r[keep], prof["rho"][::-1][keep], prof["zone_id"][::-1][keep]


def _placement_domains(zones):
    """placement domain of each zone (outer -> inner): consecutive zones with the same (not None) placement_group share one"""
    dom = np.zeros(len(zones), int)
    for z in range(1, len(zones)):
        g = getattr(zones[z], "placement_group", None)
        dom[z] = dom[z - 1] + (0 if (g is not None and g == getattr(zones[z - 1], "placement_group", None)) else 1)
    return dom


def _seagen_realise(prof, dom, n_particles, m_rel, seed):
    """one SEAGen sphere with relative particle mass m_rel[d] per placement domain d (outer -> inner); returns the domain of each particle"""
    import seagen
    r, rho, zid = _zone_profiles(prof)
    d_prof = dom[zid]; nd = dom.max() + 1
    M_d = np.array([np.sum(np.diff(np.r_[0.0, prof["m"][::-1][prof["r"][::-1] > 0]])[d_prof == d]) for d in range(nd)])
    m_rel = np.asarray(m_rel, float) / m_rel[-1]               # SEAGen counts the desired number in units of the innermost domain's particle mass
    m0 = np.sum(M_d / m_rel) / n_particles
    s = seagen.GenSphere(int(round(prof["M_total"] / m0)), r, rho, A1_mat_prof=d_prof, A1_m_rel_prof=m_rel[d_prof],
                         verbosity=0, seed=seed)
    return np.c_[s.A1_x, s.A1_y, s.A1_z], np.asarray(s.A1_m, float), np.asarray(s.A1_mat, int)


def _interface_spacing_ratio(prof, dom, coords, masses, dom_p, d_outer):
    """particle spacing (m/rho)^(1/3) of the innermost shell of domain d_outer over that of the outermost shell of domain d_outer + 1,
    each with its own profile density"""
    r = np.linalg.norm(coords, axis=1); rp, rhop, zp = _zone_profiles(prof); dp = dom[zp]
    def spacing(d, pick):
        s = dom_p == d; rs = r[s]; shell = np.isclose(rs, pick(rs), rtol=1e-6)
        rho_d = np.interp(pick(rs), rp[dp == d], rhop[dp == d])
        return np.median((masses[s][shell] / rho_d) ** (1.0 / 3.0))
    return spacing(d_outer + 1, np.max) / spacing(d_outer, np.min)


def _zones_by_shell(prof, zones, dom, rr, masses, dom_p):
    """the zone of each particle: its domain's single zone, or, in a domain of several zones (composition bins), the zone whose profile
    mass interval holds the middle of the particle's WHOLE shell (enclosed mass of all particles, scaled to the profile's total), so no
    shell is split between zones; returns (zone, realised mass of each zone)"""
    _, _, zid_prof = _zone_profiles(prof); sampled = np.isin(np.arange(len(zones)), zid_prof)   # zones the profile holds samples of
    order = np.argsort(rr); m_enc = np.empty(len(rr)); m_enc[order] = np.cumsum(masses[order]) - masses[order]   # mass strictly inside
    scale = prof["M_total"] / masses.sum()
    inner = np.array([z.inner_mass_frac for z in zones]) * prof["M_total"]; outer = np.r_[prof["M_total"], inner[:-1]]
    zone = np.empty(len(rr), int)
    for d in range(dom.max() + 1):
        members = np.where(dom == d)[0]; s = np.where(dom_p == d)[0]
        if len(members) == 1: zone[s] = members[0]; continue
        members = members[sampled[members]]
        sh = shells_of(rr[s])
        for k in np.unique(sh):
            q = s[sh == k]; mid = (m_enc[q].min() + 0.5 * masses[q].sum()) * scale
            hit = members[(inner[members] <= mid) & (mid < outer[members])]
            zone[q] = hit[0] if len(hit) else members[np.argmin(np.minimum(abs(inner[members] - mid), abs(outer[members] - mid)))]   # nearest
    M_real = np.array([masses[zone == z].sum() for z in range(len(zones))]) * scale
    return zone, M_real


def seagen_shells(prof, zones, n_particles, mass_ratio="spacing", seed=17, verbose=True, scan_contrast=1.3):
    """Place particles on SEAGen shells (Kegerreis et al. 2019), one material per placement domain.

    A placement domain is one zone, or several consecutive zones sharing a placement_group (the composition bins of a graded interior):
    a domain is laid out as one set of shells with one particle mass, following the profile's density (bin steps included), and each
    WHOLE shell is then assigned to the zone whose mass interval holds its middle; the realised zone masses are returned (zone_mass) and
    printed beside the profile's. Without placement groups every zone is its own domain.

    mass_ratio sets the particle mass of each domain relative to the domain outside it (one value per interface between domains,
    outer -> inner, or a single value for a two-domain body): 1.0 = equal masses; "spacing" (the default)
    = for each interface, the realised ratio whose particle spacing ratio across that interface is nearest 1
    (equal particle volume on both sides). SEAGen fits whole shells, so a requested ratio is realised only
    approximately and the admissible ratios are discrete: the scan tries requested ratios from 0.85 to 1.75 times
    the interface density ratio (inner over outer) and keeps the realisation itself. Interfaces whose density ratio
    is within scan_contrast of 1 take the density ratio itself, without the scan of 19 realisations: at a small contrast the shell
    quantisation moves the spacing ratio far less. Positions are not relaxed
    afterwards (relaxing toward the profile density disorders the shells and worsens the discrete force balance).
    Needs the optional package seagen (pip install seagen)."""
    try:
        import seagen  # noqa: F401
    except ImportError:
        raise ImportError("seagen_shells needs the seagen package: pip install seagen")
    nz = len(zones); dom = _placement_domains(zones); nd = dom.max() + 1
    r, rho, zid = _zone_profiles(prof); dprof = dom[zid]
    for d in range(nd):
        if not np.any(dprof == d):
            raise ValueError("seagen_shells: zones " + ", ".join(zones[z].material.name for z in np.where(dom == d)[0])
                             + " hold no sample of the hydrostatic profile (too little mass for its resolution, let alone a particle shell)")
    m_rel = np.ones(nd)   # outer domain = 1; inner domains relative to it
    realise = lambda m_rel: _seagen_realise(prof, dom, n_particles, m_rel, seed)
    for d_outer in reversed(range(nd - 1)):          # interfaces from the centre outward; inner ones already fixed
        rho_out = rho[dprof == d_outer][np.argmin(r[dprof == d_outer])]   # densities either side of the interface
        rho_in = rho[dprof == d_outer + 1][np.argmax(r[dprof == d_outer + 1])]
        if mass_ratio == "spacing" and max(rho_in / rho_out, rho_out / rho_in) < scan_contrast:
            m_rel[d_outer + 1:] = m_rel[d_outer + 1:] / m_rel[d_outer + 1] * (m_rel[d_outer] * rho_in / rho_out)
        elif mass_ratio == "spacing":
            best = None
            for f in np.arange(0.85, 1.7501, 0.05):
                trial = m_rel.copy(); trial[d_outer + 1:] = m_rel[d_outer + 1:] / m_rel[d_outer + 1] * (m_rel[d_outer] * f * rho_in / rho_out)
                coords, masses, dom_p = realise(trial)
                sr = _interface_spacing_ratio(prof, dom, coords, masses, dom_p, d_outer)
                if best is None or abs(sr - 1) < abs(best[1] - 1): best = (trial, sr, f)
            m_rel = best[0]
        else:
            q = np.atleast_1d(mass_ratio)[d_outer if np.ndim(mass_ratio) else 0]
            m_rel[d_outer + 1:] = m_rel[d_outer + 1:] / m_rel[d_outer + 1] * (m_rel[d_outer] * q)
    coords, masses, dom_p = realise(m_rel)
    rr = np.linalg.norm(coords, axis=1)
    zone, M_real = _zones_by_shell(prof, zones, dom, rr, masses, dom_p)
    rp, rhop, zp = _zone_profiles(prof); uu = prof["u"][::-1][prof["r"][::-1] > 0]; TT = prof["T"][::-1][prof["r"][::-1] > 0]
    rho_p = np.empty(len(rr)); u_p = np.empty(len(rr)); T_p = np.empty(len(rr))
    for z in range(nz):   # each particle reads its own material's branch of the profile
        s = zone == z; on = zp == z
        if not s.any(): continue
        x = np.clip(rr[s], rp[on].min(), rp[on].max())
        rho_p[s] = np.interp(x, rp[on], rhop[on]); u_p[s] = np.interp(x, rp[on], uu[on]); T_p[s] = np.interp(x, rp[on], TT[on])
    inner = np.array([z.inner_mass_frac for z in zones]) * prof["M_total"]; M_prof = np.r_[prof["M_total"], inner[:-1]] - inner
    if verbose:
        for d_outer in range(nd - 1):
            a, b = np.where(dom == d_outer)[0][-1], np.where(dom == d_outer + 1)[0][0]
            q_got = np.median(masses[dom_p == d_outer + 1]) / np.median(masses[dom_p == d_outer])
            print(f"  seagen_shells: interface {zones[a].material.name}/{zones[b].material.name}: "
                  f"particle mass ratio {q_got:.3f}, spacing ratio {_interface_spacing_ratio(prof, dom, coords, masses, dom_p, d_outer):.3f}")
        print(f"  seagen_shells: N {len(rr)} (" + ", ".join(f"{zones[z].material.name} {np.sum(zone == z)}" for z in range(nz))
              + f"), total mass / M {masses.sum() / prof['M_total']:.6f}")
        if nd < nz:
            print("  seagen_shells: zones placed together, assigned by whole shells -- realised / profile mass - 1: "
                  + ", ".join(f"{zones[z].material.name} " + (f"{M_real[z] / M_prof[z] - 1:+.3e}" if M_prof[z] > 0 else ("no shell" if M_real[z] == 0 else f"{M_real[z]:.3e} (profile 0)"))
                              for z in range(nz) if np.sum(dom == dom[z]) > 1))
    comp_p = np.array([zones[z].material.composition_type for z in zone], dtype=np.int32)
    return {"coords": coords, "masses": masses, "rho_p": rho_p, "u_p": u_p, "T_p": T_p, "comp_p": comp_p, "zone_mass": M_real}


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
