"""Top-level IC builder: zone spec -> HSE solve -> particle
placement -> HDF5 output. Designed to replace the uniform-density rejection-
sampled-sphere generators in test/aneos_giant_impact, test/jutzi_crater,
test/cd21_hhe_compression once validated.

For a tabulated-EOS (ANEOS) planet, build_planet() below does every step in one call (SEAGen shells with
spacing-matched masses, GIZMO's own kernel density, u calibrated at it). The steps one by one, for an
Earth-like body:

    from initial_conditions.eos_tools.eos_dispatch import Material
    from initial_conditions.eos_tools.hse_solver   import Zone, solve_hse
    from initial_conditions.eos_tools.shell_placement import fibonacci_shells
    from initial_conditions.eos_tools.build_layered_body import write_hdf5

    iron      = Material.tillotson("iron")
    forsterite= Material.tillotson("olivine")  # Tillotson olivine ~= forsterite
    zones = [
        Zone(forsterite, inner_mass_frac=0.32, T_profile="adiabatic"),  # mantle (68%)
        Zone(iron,       inner_mass_frac=0.00, T_profile="adiabatic"),  # core (32%)
    ]
    prof = solve_hse(M_total=5.972e27, T_surface=300.0, P_surface=1e6,
                     zones=zones, verbose=True)
    placed = fibonacci_shells(prof, zones, n_particles=5000)
    write_hdf5("earth_ics.hdf5", placed, prof, box_size=10*prof["R"])
"""

import numpy as np
import h5py


def write_hdf5(path, placed, prof, box_size, center_offset=None):
    """Write IC file matching the schema used by the existing planet test
    IC generators (PartType0 with Coordinates/Velocities/Masses/InternalEnergy/
    ParticleIDs/CompositionType)."""
    coords = placed["coords"]
    n = len(coords)
    if center_offset is None:
        center_offset = np.array([0.5 * box_size] * 3)
    coords_box = coords + center_offset

    with h5py.File(path, "w") as f:
        h = f.create_group("Header")
        h.attrs["NumPart_ThisFile"] = np.array([n, 0, 0, 0, 0, 0], dtype=np.int32)
        h.attrs["NumPart_Total"] = np.array([n, 0, 0, 0, 0, 0], dtype=np.uint32)
        h.attrs["NumPart_Total_HighWord"] = np.array([0] * 6, dtype=np.uint32)
        h.attrs["MassTable"] = np.zeros(6, dtype=np.float64)
        h.attrs["Time"] = 0.0
        h.attrs["Redshift"] = 0.0
        h.attrs["BoxSize"] = float(box_size)
        h.attrs["NumFilesPerSnapshot"] = 1
        h.attrs["Omega0"] = 0.0
        h.attrs["OmegaLambda"] = 0.0
        h.attrs["HubbleParam"] = 1.0
        h.attrs["Flag_Sfr"] = 0
        h.attrs["Flag_Feedback"] = 0
        h.attrs["Flag_Cooling"] = 0
        h.attrs["Flag_StellarAge"] = 0
        h.attrs["Flag_Metals"] = 0
        h.attrs["Flag_DoublePrecision"] = 1

        g = f.create_group("PartType0")
        g.create_dataset("Coordinates", data=coords_box)
        g.create_dataset("Velocities", data=placed["vel"] if "vel" in placed else np.zeros((n, 3)))
        g.create_dataset("Masses", data=placed["masses"])
        g.create_dataset("InternalEnergy", data=placed["u_p"])
        g.create_dataset("ParticleIDs", data=np.arange(1, n + 1, dtype=np.uint32))
        g.create_dataset("CompositionType", data=placed["comp_p"].astype(np.int32))
        # Bookkeeping: record solved profile R + central P/T as attributes.
        g.attrs["HSE_Radius"] = float(prof["R"])
        g.attrs["HSE_M_total"] = float(prof["M_total"])
        g.attrs["HSE_P_central"] = float(prof["P"][-1])
        g.attrs["HSE_T_central"] = float(prof["T"][-1])
        if "kernel" in placed:   # the run's kernel and neighbour number the densities were calibrated for
            g.attrs["KERNEL_FUNCTION"] = int(placed["kernel"]); g.attrs["DesNumNgb"] = float(placed["des_num_ngb"])

    print(f"Wrote {path}: n={n}  R={prof['R']:.3e} cm  "
          f"P_c={prof['P'][-1]:.3e} dyn/cm^2  T_c={prof['T'][-1]:.3e} K")


def shells_of(r, rel_gap=1e-3):
    """shell index (0 = innermost) of each particle of one material: SEAGen places a material's particles on spheres, so their
    radii cluster (to within the small offset between the shells' centre and the body's centre of mass, far below the spacing)"""
    o = np.argsort(r); rs = r[o]; cut = np.concatenate([[0], np.cumsum(np.diff(rs) / rs[1:] > rel_gap)])
    idx = np.empty(len(r), int); idx[o] = cut
    return idx


def hydrostatic_pressure(prof, r_particles, m_particles, P_surface):
    """P(r) on the profile's radial grid for the body as placed: dP/dr = -G M(r) rho(r) / r^2 integrated inward from P_surface at
    the profile radius, with M(r) the mass of the particles inside r and rho(r) the profile density (what calibrated masses give
    the run). Returns a copy of prof with "P" replaced."""
    from .hse_solver import G_CGS
    r = prof["r"]                                   # outer -> inner, r[0] = R
    o = np.argsort(r_particles); M_in = np.searchsorted(r_particles[o], r, side="right")
    M = np.concatenate([[0.0], np.cumsum(m_particles[o])])[M_in]
    g = np.where(r > 0, G_CGS * M * prof["rho"] / np.maximum(r * r, 1e-300), 0.0)
    P = P_surface + np.concatenate([[0.0], np.cumsum(0.5 * (g[1:] + g[:-1]) * (r[:-1] - r[1:]))])
    return dict(prof, P=P)


def calibrate_planet(placed, prof, zones, P_surface, rho_kernel, h, calibrate_masses=False, min_shell_particles=64, verbose=True):
    """Set u, T per particle so that EOS(rho, u) = the hydrostatic pressure at its radius, with rho the run's density (rho_kernel:
    GIZMO's zero-time snapshot densities, or sph_density.estimate_sph_density with the run's kernel and DesNumNgb).
    Particles with no valid EOS state at their target pressure keep the closest valid one and are reported (count and fraction per
    material, worst layer fraction, largest contiguous group, pressure miss), unless they form a defect of the IC, which stops the
    build: (a) more than half of a layer -- a layer is one material's SEAGen shell, pooled outward with its neighbours until it holds
    min_shell_particles; (b) a contiguous group of at least min_shell_particles/2 -- particles of one material linked within that
    material's median kernel radius. Neither applies inside the outermost material's free-surface layer (the layers within one kernel
    support of the surface): there a planet's outer boundary is generally unresolved (a 1-bar H/He surface has a pressure scale height
    of ~1e-3 of the kernel radius at any feasible resolution), and the run's surface treatment, not the builder, sets the density;
    its invalid particles are reported separately.
    calibrate_masses=True (an explicit option, off by default) also sets the particle masses so that the run's density equals the
    profile density, one mass per SEAGen shell (shells with fewer than min_shell_particles are pooled with their outer neighbours;
    free-surface shells keep their masses); the run's number density depends on the positions only, so one pass sets the density, and
    the hydrostatic pressure is then recomputed for the body as placed (its enclosed mass changes). The requested and realised masses
    are printed. Updates placed in place (masses, u_p, T_p, rho_sph, noroot) and returns the profile used."""
    from .sph_density import calibrate_u, _material_branch_profile
    x = placed["coords"]; m = placed["masses"]; comp = placed["comp_p"]
    com = (m[:, None] * x).sum(0) / m.sum(); r = np.linalg.norm(x - com, axis=1); R = r.max()
    zone_by_ct = {z.material.composition_type: iz for iz, z in enumerate(zones)}
    zone_of = np.array([zone_by_ct[c] for c in comp])
    n_num = rho_kernel / m                                         # the run's number density at these positions
    want = _material_branch_profile(r, prof, zone_of, "rho") / n_num
    m_new = m.copy(); free_surface = np.zeros(len(m), bool); layer = np.zeros(len(m), int); n_layers = 0
    for iz in range(len(zones)):
        s = np.where(zone_of == iz)[0]
        if s.size == 0: continue
        sh = shells_of(r[s]); counts = np.bincount(sh)
        group = np.zeros(counts.size, int); g, filled = 0, 0          # pool small shells (the centre) with their outer neighbours
        for k in range(counts.size):
            group[k] = g; filled += counts[k]
            if filled >= min_shell_particles: g += 1; filled = 0
        if filled and g > 0: group[group == g] = g - 1               # a small outermost remainder joins the shell group below it
        for grp in np.unique(group):
            q = group[sh] == grp; rk = r[s][q].mean(); layer[s[q]] = n_layers; n_layers += 1
            if iz == 0 and R - rk < np.median(h[s][q]): free_surface[s[q]] = True; continue   # the outermost material's surface layer
            if calibrate_masses: m_new[s[q]] = np.mean(want[s][q])
    if calibrate_masses:                                           # the body as placed now: its centre, radii and enclosed mass follow m_new
        com = (m_new[:, None] * x).sum(0) / m_new.sum(); r = np.linalg.norm(x - com, axis=1); R = r.max()
        prof_cal = dict(hydrostatic_pressure(prof, r, m_new, P_surface), M_total=m_new.sum())
    else:
        prof_cal = prof
    rho_cal = m_new * n_num
    u, T, noroot = calibrate_u(x, m_new, comp, rho_cal, prof_cal, zones, verbose=verbose)
    # the IC defects that stop the build (docstring): (a) more than half of a layer; (b) a contiguous group of one material; both
    # outside the outermost material's free-surface layer. A scattered few keep the closest valid state (a 16k Earth with 25 such iron
    # particles holds ten dynamical times to 2e-5 in energy)
    surface = free_surface & (zone_of == 0)
    frac = np.bincount(layer[noroot & ~surface], minlength=n_layers) / np.maximum(np.bincount(layer, minlength=n_layers), 1)
    cluster = np.zeros(len(m), int)                       # size of each invalid particle's contiguous group within its material
    if (noroot & ~surface).any():
        from scipy.spatial import cKDTree
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        for z in np.unique(zone_of[noroot & ~surface]):
            ii = np.where(noroot & ~surface & (zone_of == z))[0]
            pairs = cKDTree(x[ii]).query_pairs(np.median(h[ii]), output_type="ndarray")
            _, lab = connected_components(coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(ii.size, ii.size)), directed=False)
            cluster[ii] = np.bincount(lab)[lab]
    bad = noroot & ~surface & ((frac[layer] > 0.5) | (cluster >= min_shell_particles // 2))
    if bad.any():
        raise ValueError(f"calibrate_planet: {bad.sum()} particles have no valid EOS state at their target pressure in more than half of a layer "
                         f"(worst {frac.max():.2f}) or a contiguous group (largest {cluster.max()}) (materials " + ", ".join(sorted({zones[z].material.name for z in zone_of[bad]}))
                         + f", r/R {r[bad].min() / R:.3f}..{r[bad].max() / R:.3f}): the run's density there exceeds what the material can have at "
                         "that pressure (a cold body near its table's lowest valid temperature tolerates ~1%); use more particles or neighbours"
                         + ("" if calibrate_masses else ", or calibrate_masses=True"))
    if verbose:
        if noroot.any():
            P_kept = np.empty(noroot.sum()); ii = np.where(noroot)[0]
            for z in np.unique(zone_of[ii]):
                s = zone_of[ii] == z; mat = zones[z].material
                P_kept[s] = mat._backend.pressure_v(rho_cal[ii][s], T[ii][s]) if mat.kind == "sesame" else mat.pressure(rho_cal[ii][s], u[ii][s], "u")
            miss = np.abs(P_kept / np.interp(r[ii], prof_cal["r"][::-1], prof_cal["P"][::-1]) - 1)
            print(f"  calibrate_planet: *** {noroot.sum()} particles have no valid EOS state at their target pressure and keep the closest one: "
                  + ", ".join(f"{zones[z].material.name} {np.sum(noroot & (zone_of == z))} ({np.sum(noroot & (zone_of == z)) / np.sum(zone_of == z):.1e} of it"
                              + (f", {np.sum(noroot & surface & (zone_of == z))} in the free-surface layer" if z == 0 else "") + ")"
                              for z in np.unique(zone_of[ii]))
                  + f"; r/R {r[ii].min() / R:.3f}..{r[ii].max() / R:.3f}; below the surface layer: worst layer fraction {frac.max():.3f}, largest "
                  f"contiguous group {cluster.max()}; pressure miss max {miss.max():.2e} median {np.median(miss):.2e} ***")
        M_req = prof["M_total"]
        dev = (m * n_num / _material_branch_profile(r, prof, zone_of, "rho") - 1)[~free_surface]
        if dev.size:
            print(f"  calibrate_planet: the run's density / profile - 1 below the free-surface layer (masses as placed): median "
                  f"{np.median(dev):+.4f}, p1/p99 {np.percentile(dev, 1):+.4f}/{np.percentile(dev, 99):+.4f}")
        print(f"  calibrate_planet: mass {'CALIBRATED' if calibrate_masses else 'as placed'}: requested {M_req:.6e} realised {m_new.sum():.6e} ({m_new.sum() / M_req - 1:+.3e}); "
              f"R {R:.4e} (profile {prof['R']:.4e}); P_c {prof['P'][-1]:.4e} -> {prof_cal['P'][-1]:.4e}")
        for iz, z in enumerate(zones):
            s = zone_of == iz
            if s.any():
                print(f"    {z.material.name}: mass {m[s].sum():.6e} -> {m_new[s].sum():.6e} ({m_new[s].sum() / m[s].sum() - 1:+.3e}), "
                      f"particle mass range {m_new[s].min() / np.median(m_new[s]):.3f}-{m_new[s].max() / np.median(m_new[s]):.3f} of the median")
    placed["masses"] = m_new; placed["u_p"] = u; placed["rho_sph"] = rho_cal; placed["noroot"] = noroot
    placed["T_p"] = np.where(np.isfinite(T), T, placed["T_p"])
    return prof_cal


def build_planet(zones, M_total, T_surface, P_surface, n_particles, out, kernel=3, des_num_ngb=32, mass_ratio="spacing",
                 calibrate_masses=False, box_size=None, seed=17, n_steps=4000):
    """Layered body in hydrostatic equilibrium, in one call: HSE profile -> SEAGen shells with the particle mass
    matched to the spacing across each material interface -> GIZMO's kernel density at the placed positions (the run's
    KERNEL_FUNCTION `kernel` and DesNumNgb) -> u per particle so that EOS(rho, u) = P_HSE, stopping if a particle below the
    free-surface layer has no valid EOS state (calibrate_planet; calibrate_masses=True also sets one mass per shell so that the
    run's density equals the profile's, printing the requested and realised masses) -> HDF5 IC at `out`, which records the
    kernel and neighbour number it was built for. zones are ordered outer -> inner (see hse_solver.Zone). The run's own zero-time
    densities (its h band and, in EOS_ELASTIC builds, its free-surface correction) can be passed to calibrate_planet instead.
    Returns (placed, prof)."""
    from .hse_solver import solve_hse
    from .shell_placement import seagen_shells
    from .sph_density import estimate_sph_density
    prof = solve_hse(M_total=M_total, T_surface=T_surface, P_surface=P_surface, zones=zones, n_steps=n_steps)
    print(f"  HSE profile: R = {prof['R']:.4e} cm, P_c = {prof['P'][-1]:.3e}, T_c = {prof['T'][-1]:.3e}")
    placed = seagen_shells(prof, zones, n_particles, mass_ratio=mass_ratio, seed=seed)
    rho, h = estimate_sph_density(placed["coords"], placed["masses"], des_num_ngb=des_num_ngb, kernel=kernel)
    prof = calibrate_planet(placed, prof, zones, P_surface, rho, h, calibrate_masses=calibrate_masses)
    placed["kernel"], placed["des_num_ngb"] = kernel, des_num_ngb
    write_hdf5(out, placed, prof, box_size=box_size if box_size else 10.0 * prof["R"])
    return placed, prof


def place_collision(target, impactor, v_contact_over_v_esc, impact_parameter, separation_in_contact_radii=2.0, G=6.674e-8):
    """Two bodies (placed dicts from build_planet/seagen_shells, each centred on its own centre of mass) set on the two-body
    orbit that brings them into contact (centre separation R_t + R_i, each the outermost particle's distance from its body's centre
    of mass) at relative speed v_contact_over_v_esc x the mutual escape speed sqrt(2 G (M_t + M_i) / (R_t + R_i)), with impact parameter
    b = sin(angle between the relative velocity and the line of centres at contact), started at a centre separation of
    separation_in_contact_radii x (R_t + R_i). The orbit is traced back exactly (energy and angular momentum of the
    relative Kepler orbit); the impactor starts on +x, approaching, with its angular momentum along +z. Positions and
    velocities are in the centre-of-mass frame; the impactor's particle IDs follow the target's. Returns the merged placed
    dict (with "vel"), the starting separation, the relative velocity there, and the escape speed."""
    M_t, M_i = target["masses"].sum(), impactor["masses"].sum(); mu = G * (M_t + M_i)
    com = lambda body: (body["masses"][:, None] * body["coords"]).sum(0) / body["masses"].sum()
    R_t = np.max(np.linalg.norm(target["coords"] - com(target), axis=1)); R_i = np.max(np.linalg.norm(impactor["coords"] - com(impactor), axis=1))
    r_c = R_t + R_i; r0 = separation_in_contact_radii * r_c
    v_esc = np.sqrt(2.0 * mu / r_c); v_contact = v_contact_over_v_esc * v_esc
    E = 0.5 * v_contact ** 2 - mu / r_c                 # specific orbital energy of the relative orbit
    L = r_c * v_contact * impact_parameter              # specific angular momentum
    v0 = np.sqrt(2.0 * (E + mu / r0)); vt0 = L / r0
    if vt0 > v0:
        raise ValueError("this orbit does not reach the requested starting separation (it turns around inside it)")
    vr0 = -np.sqrt(v0 ** 2 - vt0 ** 2)                  # approaching
    rel_x = np.array([r0, 0.0, 0.0]); rel_v = np.array([vr0, vt0, 0.0])
    f_t, f_i = M_i / (M_t + M_i), M_t / (M_t + M_i)    # centre-of-mass frame
    out = {}
    for k in ("masses", "rho_p", "u_p", "T_p", "comp_p"):
        out[k] = np.concatenate([target[k], impactor[k]])
    out["coords"] = np.concatenate([target["coords"] - com(target) - f_t * rel_x, impactor["coords"] - com(impactor) + f_i * rel_x])
    out["vel"] = np.concatenate([np.tile(-f_t * rel_v, (len(target["masses"]), 1)), np.tile(f_i * rel_v, (len(impactor["masses"]), 1))])
    print(f"  collision: M_t {M_t:.4e} M_i {M_i:.4e}  R_t {R_t:.4e} R_i {R_i:.4e}  v_contact {v_contact:.4e} ({v_contact / v_esc:.3f} v_esc)"
          f"  b {impact_parameter:.3f}  start: separation {r0:.4e}, relative speed {v0:.4e} (radial {vr0:.4e}, tangential {vt0:.4e})")
    return out, r0, rel_v, v_esc


def hse_diagnostics(prof):
    """Print residual of dP/dr + G m rho/r^2 along the profile (RMS / max).
    Excludes points within ±2 cells of any zone boundary, where the T-restart
    of the new adiabat produces a one-cell numerical artifact."""
    G = 6.674e-8
    r = prof["r"][1:-1]; P = prof["P"]; m = prof["m"]; rho = prof["rho"]
    dPdr = (P[2:] - P[:-2]) / (prof["r"][2:] - prof["r"][:-2])
    rhs = -G * m[1:-1] * rho[1:-1] / np.maximum(r * r, 1e-30)
    denom = np.maximum(np.abs(rhs), np.abs(dPdr).max() * 1e-6)
    rel = np.abs(dPdr - rhs) / denom

    # Mask zone-boundary cells AND the innermost ~1% of points (1/r^2 blowup).
    zid = prof["zone_id"][1:-1]
    boundary = np.zeros_like(zid, dtype=bool)
    for k in range(1, len(zid) - 1):
        if zid[k] != zid[k - 1] or zid[k] != zid[k + 1]:
            boundary[max(0, k - 2):k + 3] = True
    n_inner_skip = max(1, len(zid) // 100)
    boundary[-n_inner_skip:] = True
    rel_clean = rel[~boundary]
    print(f"  HSE residual (zone-boundary cells excluded): "
          f"rms={np.sqrt(np.mean(rel_clean**2)):.2e}  max={rel_clean.max():.2e}  "
          f"(masked {boundary.sum()} of {len(rel)} cells)")
    return rel_clean
