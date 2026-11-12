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

import os

import numpy as np
import h5py

from .shell_placement import shells_of, _placement_domains


def write_hdf5(path, placed, prof, box_size, center_offset=None, id_offset=0):
    """Write IC file matching the schema used by the existing planet test
    IC generators (PartType0 with Coordinates/Velocities/Masses/InternalEnergy/
    ParticleIDs/CompositionType). ParticleIDs are id_offset + 1 .. id_offset + n: give each body of a
    multi-body run its final, disjoint range when it is built (tensile-flaw thresholds are drawn from the IDs,
    so a body must never be renumbered after it has been settled)."""
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
        g.create_dataset("ParticleIDs", data=np.arange(id_offset + 1, id_offset + n + 1, dtype=np.uint64 if id_offset + n >= 2**32 else np.uint32))
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


def calibrate_planet(placed, prof, zones, P_surface, rho_kernel, h, calibrate_masses=False, min_shell_particles=64, verbose=True,
                     max_mass_variation=0.3):
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
    calibrate_masses (an explicit option, off by default) also sets the particle masses so that the run's density equals the
    profile density: the run's (MFM) density is the particle's mass times a number density that depends on the positions only, so
    m = rho_profile / n sets it in one pass. "shell" (or True): one mass per SEAGen shell (shells with fewer than min_shell_particles
    pooled with their outer neighbours), so the shell's MEAN density matches; "particle": each particle its own mass, so every particle's
    density matches -- needed where the material sits close to its cold compression curve (a cold rock or iron core: a few percent of
    density excess there has no state at any temperature), at the price of masses varying by a few percent within a shell; a particle
    whose mass would leave [1 / (1 + max_mass_variation), 1 + max_mass_variation] of its shell group's median stops the build. Free-
    surface shells keep their masses. The hydrostatic pressure is then recomputed for the body as placed (its enclosed mass changes).
    The requested and realised masses, and the particle-mass percentiles within the shell groups, are printed. Updates placed in place (masses, u_p, T_p, rho_sph, noroot; free_surface = the outermost material's free-surface
    layer, rho_profile = each particle's profile density) and returns the profile used."""
    from .sph_density import calibrate_u, _material_branch_profile
    if calibrate_masses is True: calibrate_masses = "shell"
    if calibrate_masses not in (False, None, "shell", "particle"):
        raise ValueError(f"calibrate_masses={calibrate_masses!r} not in {{False, 'shell' (= True), 'particle'}}")
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
            if calibrate_masses == "shell": m_new[s[q]] = np.mean(want[s][q])
            elif calibrate_masses == "particle": m_new[s[q]] = want[s][q]
    if calibrate_masses == "particle":                              # each particle's mass relative to its shell group's median
        rel = np.ones(len(m))
        for k in np.unique(layer[~free_surface]):
            q = (layer == k) & ~free_surface; rel[q] = m_new[q] / np.median(m_new[q])
        too = ~free_surface & ((rel > 1.0 + max_mass_variation) | (rel < 1.0 / (1.0 + max_mass_variation)))
        if too.any():
            raise ValueError(f"calibrate_planet: {too.sum()} particles would need a mass outside 1 / {1 + max_mass_variation:g} .. {1 + max_mass_variation:g} "
                             f"of their shell's median (range {rel.min():.3f}..{rel.max():.3f}, r/R {r[too].min() / R:.3f}..{r[too].max() / R:.3f}): "
                             "the placement's density scatter is too large to calibrate away; use more particles or neighbours")
        if verbose:
            for iz, z in enumerate(zones):
                q = (zone_of == iz) & ~free_surface
                if q.any():
                    print(f"  calibrate_planet: {z.material.name}: particle mass / its shell's median: p1 {np.percentile(rel[q], 1):.4f}, "
                          f"p50 {np.percentile(rel[q], 50):.4f}, p99 {np.percentile(rel[q], 99):.4f} (min {rel[q].min():.4f}, max {rel[q].max():.4f})")
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
                         + ("" if calibrate_masses == "particle" else ", or calibrate_masses='particle'"))
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
        print(f"  calibrate_planet: mass {('CALIBRATED per ' + calibrate_masses) if calibrate_masses else 'as placed'}: requested {M_req:.6e} realised {m_new.sum():.6e} ({m_new.sum() / M_req - 1:+.3e}); "
              f"R {R:.4e} (profile {prof['R']:.4e}); P_c {prof['P'][-1]:.4e} -> {prof_cal['P'][-1]:.4e}")
        for iz, z in enumerate(zones):
            s = zone_of == iz
            if s.any():
                print(f"    {z.material.name}: mass {m[s].sum():.6e} -> {m_new[s].sum():.6e} ({m_new[s].sum() / m[s].sum() - 1:+.3e}), "
                      f"particle mass range {m_new[s].min() / np.median(m_new[s]):.3f}-{m_new[s].max() / np.median(m_new[s]):.3f} of the median")
    placed["masses"] = m_new; placed["u_p"] = u; placed["rho_sph"] = rho_cal; placed["noroot"] = noroot
    placed["free_surface"] = free_surface; placed["rho_profile"] = _material_branch_profile(r, prof_cal, zone_of, "rho")
    placed["T_p"] = np.where(np.isfinite(T), T, placed["T_p"])
    return prof_cal


def _align_grouped_zones(zones, prof, placed, M_total, T_surface, P_surface, n_particles, mass_ratio, seed, n_steps, adiabat_route,
                         tol, max_passes, allow_unaligned=False):
    """zones placed together (a placement group) get whole shells, so their profile masses and the shells' disagree: move each boundary
    INSIDE a placement domain to the realised masses (the domain's total and every boundary between domains stay as requested), drop
    zones that received no shell, solve the profile again and place again, until no grouped zone is empty and every one's realised mass is
    within tol of its profile mass (a new profile gives new shells, so one pass is not enough). A pass that finds empty zones always drops
    them and places again, also beyond max_passes (each such pass removes at least one zone). Not within tol after max_passes: refused,
    unless allow_unaligned (diagnostics only: a warning). A placement domain that receives no particles is refused. Prints the residual."""
    import copy
    from .hse_solver import solve_hse
    from .shell_placement import seagen_shells
    zones = [copy.copy(z) for z in zones]; npass = 0
    while True:
        npass += 1
        dom = _placement_domains(zones); M_real = placed["zone_mass"]
        inner = np.array([z.inner_mass_frac for z in zones]); outer = np.r_[1.0, inner[:-1]]
        grouped = np.array([getattr(z, "placement_group", None) is not None for z in zones])   # also a group's last surviving zone
        for d in np.unique(dom[grouped]):
            if M_real[dom == d].sum() <= 0:
                raise ValueError("build_planet: the placement group of zones " + ", ".join(zones[z].material.name for z in np.where(dom == d)[0])
                                 + " received no particles (its mass is below one particle shell): give it more mass or the body more particles")
        M_prof = (outer - inner) * M_total
        live = grouped & (M_prof > 0)
        resid = np.abs(M_real[live] / M_prof[live] - 1).max() if live.any() else 0.0
        empty = grouped & (M_real == 0)
        msg = (f"build_planet: zones placed together after {npass} pass(es): largest |realised / profile mass - 1| {resid:.2e} "
               f"(tolerance {tol:.0e}) over {int(live.sum())} zones holding shells")
        if not empty.any() and resid <= tol: print("  " + msg); break
        if not empty.any() and npass >= max_passes:
            if not allow_unaligned:
                raise ValueError(msg + ": not reached in max_align_passes (raise it, use more particles, or allow_unaligned=True for diagnostics)")
            print("  WARNING " + msg + ": not reached in max_align_passes (allow_unaligned)"); break
        for d in np.unique(dom[grouped]):
            members = np.where(dom == d)[0]
            top, bottom = outer[members[0]], inner[members[-1]]          # the domain's requested outer and inner mass fractions
            m = M_real[members] / M_real[members].sum() * (top - bottom)  # its zones' realised masses, scaled to the domain's requested total
            for k, z in enumerate(members[:-1]): zones[z].inner_mass_frac = float(bottom + m[k + 1:].sum())
        zones = [z for z, e in zip(zones, empty) if not e]
        prof = solve_hse(M_total=M_total, T_surface=T_surface, P_surface=P_surface, zones=zones, n_steps=n_steps, adiabat_route=adiabat_route)
        print(f"  HSE profile with shell-aligned zone masses (pass {npass}, {np.sum(empty)} zones without a shell dropped): R = {prof['R']:.4e} cm, "
              f"P_c = {prof['P'][-1]:.3e}, T_c = {prof['T'][-1]:.3e}")
        placed = seagen_shells(prof, zones, n_particles, mass_ratio=mass_ratio, seed=seed)
    return zones, prof, placed


def build_planet(zones, M_total, T_surface, P_surface, n_particles, out, kernel=3, des_num_ngb=32, mass_ratio="spacing",
                 calibrate_masses=False, box_size=None, seed=17, n_steps=4000, adiabat_route="first_law", id_offset=0,
                 align_tol=1e-2, max_align_passes=5, allow_unaligned=False):
    """Layered body in hydrostatic equilibrium, in one call: HSE profile -> SEAGen shells with the particle mass
    matched to the spacing across each material interface -> GIZMO's kernel density at the placed positions (the run's
    KERNEL_FUNCTION `kernel` and DesNumNgb) -> u per particle so that EOS(rho, u) = P_HSE, stopping if a particle below the
    free-surface layer has no valid EOS state (calibrate_planet; calibrate_masses="shell" (or True) also sets one mass per shell,
    "particle" one mass per particle, so that the run's density equals the profile's, printing the requested and realised masses) -> HDF5 IC at `out`, which records the
    kernel and neighbour number it was built for. zones are ordered outer -> inner (see hse_solver.Zone). The run's own zero-time
    densities (its h band and, in EOS_ELASTIC builds, its free-surface correction) can be passed to calibrate_planet instead.
    adiabat_route: see hse_solver; id_offset: the body's ParticleIDs start after it (write_hdf5). Zones sharing a placement_group (the
    composition bins of a graded interior) are placed as one domain and get whole shells; the profile is then solved again with their
    shell-aligned masses and the body placed again, until every such zone's realised mass is within align_tol of its profile mass (at most
    max_align_passes, else the build is refused unless allow_unaligned; zones left without a shell are always dropped), so each bin's
    equation of state applies where its shells sit to that tolerance. Boundaries between placement domains keep the requested masses.
    Returns (placed, prof)."""
    from .hse_solver import solve_hse
    from .shell_placement import seagen_shells
    from .sph_density import estimate_sph_density
    prof = solve_hse(M_total=M_total, T_surface=T_surface, P_surface=P_surface, zones=zones, n_steps=n_steps, adiabat_route=adiabat_route)
    print(f"  HSE profile: R = {prof['R']:.4e} cm, P_c = {prof['P'][-1]:.3e}, T_c = {prof['T'][-1]:.3e}")
    placed = seagen_shells(prof, zones, n_particles, mass_ratio=mass_ratio, seed=seed)
    if any(getattr(z, "placement_group", None) is not None for z in zones):
        zones, prof, placed = _align_grouped_zones(zones, prof, placed, M_total, T_surface, P_surface, n_particles, mass_ratio, seed,
                                                   n_steps, adiabat_route, align_tol, max_align_passes, allow_unaligned)
    rho, h = estimate_sph_density(placed["coords"], placed["masses"], des_num_ngb=des_num_ngb, kernel=kernel)
    prof = calibrate_planet(placed, prof, zones, P_surface, rho, h, calibrate_masses=calibrate_masses)
    placed["kernel"], placed["des_num_ngb"] = kernel, des_num_ngb
    write_hdf5(out, placed, prof, box_size=box_size if box_size else 10.0 * prof["R"], id_offset=id_offset)
    return placed, prof


def collision_orbit(M_t, M_i, R_t, R_i, impact_parameter, v_contact_over_v_esc=None, v_inf=None,
                    separation_in_contact_radii=2.0, time_before_contact=None, G=6.674e-8):
    """The relative two-body orbit that brings two bodies of masses M_t, M_i into contact, i.e. to a centre separation
    R_t + R_i (each a reproducible surface of its body: its truncation surface or its reported 1-bar radius), at speed
    v_contact with impact parameter b = sin(angle between the relative velocity and the line of centres at contact).
    Give v_contact_over_v_esc (in units of the mutual escape speed sqrt(2 G (M_t + M_i) / (R_t + R_i))) or v_inf (the
    speed at infinity, v_contact^2 = v_inf^2 + v_esc^2). The orbit is traced back exactly (energy and angular momentum)
    to a start at separation_in_contact_radii x (R_t + R_i), or, with time_before_contact, to the separation reached that
    long before contact (numerical integration of the relative orbit). The impactor starts on +x, approaching, with its
    angular momentum along +z. Returns (rel_x, rel_v, info): the impactor's position and velocity relative to the target
    at the start, and a dict of the orbit's numbers (also printed), incl. b_inf = b (R_t + R_i) v_contact / v_inf, the
    asymptotic impact parameter (a distance) of an unbound orbit."""
    if (v_contact_over_v_esc is None) == (v_inf is None):
        raise ValueError("give exactly one of v_contact_over_v_esc and v_inf")
    if impact_parameter is None or not 0.0 <= impact_parameter <= 1.0:
        raise ValueError(f"impact_parameter b = sin(theta) at contact must lie in [0, 1] (got {impact_parameter})")
    mu = G * (M_t + M_i); r_c = R_t + R_i; v_esc = np.sqrt(2.0 * mu / r_c)
    v_contact = v_contact_over_v_esc * v_esc if v_inf is None else np.sqrt(v_inf ** 2 + v_esc ** 2)
    E = 0.5 * v_contact ** 2 - mu / r_c                 # specific orbital energy of the relative orbit
    L = r_c * v_contact * impact_parameter              # specific angular momentum
    if time_before_contact is None:
        r0 = separation_in_contact_radii * r_c
        if not E + mu / r0 > 0:
            raise ValueError("the starting separation lies beyond this bound orbit's apocentre: start closer, or faster")
        v0 = np.sqrt(2.0 * (E + mu / r0)); vt0 = L / r0
        if vt0 > v0:
            raise ValueError("this orbit does not reach the requested starting separation (it turns around inside it)")
        rel_x = np.array([r0, 0.0, 0.0]); rel_v = np.array([-np.sqrt(v0 ** 2 - vt0 ** 2), vt0, 0.0])   # approaching
    else:
        from scipy.integrate import solve_ivp
        # in units of the contact distance and the orbital speed sqrt(mu / r_c) (mu = 1), so one tolerance fits positions and
        # velocities alike
        v_u = np.sqrt(mu / r_c); t_u = r_c / v_u
        cos_t = np.sqrt(max(0.0, 1.0 - impact_parameter ** 2))
        y_c = np.array([1.0, 0.0, -cos_t, impact_parameter]) * np.array([1.0, 1.0, v_contact / v_u, v_contact / v_u])
        rhs = lambda t, y: [y[2], y[3], -y[0] / np.hypot(y[0], y[1]) ** 3, -y[1] / np.hypot(y[0], y[1]) ** 3]
        inside = lambda t, y: np.hypot(y[0], y[1]) - 1.0      # back inside contact: a bound orbit traced past apocentre
        inside.terminal = True; inside.direction = -1
        sol = solve_ivp(rhs, (0.0, -float(time_before_contact) / t_u), y_c, method="DOP853", rtol=1e-12, atol=1e-13, events=inside)
        if not sol.success:
            raise ValueError(f"tracing the orbit back failed: {sol.message}")
        if sol.t_events[0].size:
            raise ValueError(f"this long before contact the bodies would already have touched (the orbit is bound and was inside "
                             f"contact {-sol.t_events[0][0] * t_u:.4e} s before the intended contact): start later")
        x0, y0, vx, vy = sol.y[:, -1] * np.array([r_c, r_c, v_u, v_u])
        r0 = np.hypot(x0, y0); c, s_ = x0 / r0, -y0 / r0     # rotate the start onto +x
        if x0 * vx + y0 * vy > 0:
            raise ValueError("at the requested start the bodies are moving apart (past apocentre of a bound orbit): start later")
        rel_x = np.array([r0, 0.0, 0.0]); rel_v = np.array([c * vx - s_ * vy, s_ * vx + c * vy, 0.0])
    info = dict(v_esc=v_esc, v_contact=v_contact, b=impact_parameter, r_contact=r_c, r_start=float(np.linalg.norm(rel_x)),
                v_start=float(np.linalg.norm(rel_v)), v_inf=np.nan, b_inf=np.nan)
    if E > 1e-10 * mu / r_c:                            # unbound (a parabolic orbit, v_contact = v_esc, has no v_inf)
        info["v_inf"] = np.sqrt(2.0 * E); info["b_inf"] = impact_parameter * r_c * v_contact / info["v_inf"]
    print(f"  collision orbit: M_t {M_t:.4e} M_i {M_i:.4e}  contact radii {R_t:.4e} + {R_i:.4e}  v_contact {v_contact:.4e} "
          f"({v_contact / v_esc:.4f} v_esc; v_inf {info['v_inf']:.4e})  b {impact_parameter:.3f} (b_inf {info['b_inf']:.4e} cm = "
          f"{info['b_inf'] / r_c:.4f} contact distances)  start: separation {info['r_start']:.4e}, relative speed {info['v_start']:.4e}"
          + (f", {time_before_contact:.4e} s before contact" if time_before_contact is not None else ""))
    return rel_x, rel_v, info


def place_collision(target, impactor, v_contact_over_v_esc=None, impact_parameter=None, separation_in_contact_radii=2.0, G=6.674e-8,
                    v_inf=None, R_target=None, R_impactor=None, time_before_contact=None):
    """Two bodies (placed dicts from build_planet/seagen_shells) set on collision_orbit, in the centre-of-mass frame. The contact
    radii R_target, R_impactor should be reproducible surfaces (the truncation surface prof["R"], or the reported 1-bar radius);
    without them each is the outermost particle's distance from its body's centre of mass. The impactor's particle IDs follow the
    target's in the order written. Returns the merged placed dict (with "vel"), the starting separation, the relative velocity
    there, and the escape speed."""
    M_t, M_i = target["masses"].sum(), impactor["masses"].sum()
    com = lambda body: (body["masses"][:, None] * body["coords"]).sum(0) / body["masses"].sum()
    R_t = R_target if R_target is not None else np.max(np.linalg.norm(target["coords"] - com(target), axis=1))
    R_i = R_impactor if R_impactor is not None else np.max(np.linalg.norm(impactor["coords"] - com(impactor), axis=1))
    rel_x, rel_v, info = collision_orbit(M_t, M_i, R_t, R_i, impact_parameter, v_contact_over_v_esc, v_inf,
                                         separation_in_contact_radii, time_before_contact, G)
    f_t, f_i = M_i / (M_t + M_i), M_t / (M_t + M_i)    # centre-of-mass frame
    out = {}
    for k in ("masses", "rho_p", "u_p", "T_p", "comp_p"):
        out[k] = np.concatenate([target[k], impactor[k]])
    out["coords"] = np.concatenate([target["coords"] - com(target) - f_t * rel_x, impactor["coords"] - com(impactor) + f_i * rel_x])
    out["vel"] = np.concatenate([np.tile(-f_t * rel_v, (len(target["masses"]), 1)), np.tile(f_i * rel_v, (len(impactor["masses"]), 1))])
    return out, info["r_start"], rel_v, info["v_esc"]


# every snapshot written by GIZMO carries these; a reassembled IC also needs the evolved state of the build (e.g. the solid
# fields: StressTensor, RecentYieldMemory, StrengthFraction, Damage, Distention, ActiveCracks, TensileFlawsPerMass), passed as
# `required` by the caller, who knows its build
SNAPSHOT_CORE_DATASETS = ("Coordinates", "Velocities", "Masses", "ParticleIDs", "InternalEnergy", "Density", "CompositionType")


# header attributes that legitimately differ between two bodies' snapshots (see assemble_collision)
_HEADER_PER_BODY = ("NumPart_ThisFile", "NumPart_Total", "NumPart_Total_HighWord", "Time", "Redshift",
                    "Minimum_Mass_For_Cell_Merge", "Maximum_Mass_For_Cell_Split")


def read_snapshot(path):
    with h5py.File(path, "r") as f:
        return {k: f["PartType0"][k][:] for k in f["PartType0"].keys()}, dict(f["Header"].attrs)


def validate_snapshot(path, required=()):
    """Refuse (ValueError) a snapshot / IC that lacks any of SNAPSHOT_CORE_DATASETS or `required`, has non-unique ParticleIDs,
    datasets of different lengths, or non-finite values. Returns the number of particles."""
    d, _ = read_snapshot(path)
    bad = []
    missing = [k for k in tuple(SNAPSHOT_CORE_DATASETS) + tuple(required) if k not in d]
    if missing: bad.append(f"missing datasets {missing}")
    ids = d.get("ParticleIDs", np.array([])); n = len(ids)
    if len(np.unique(ids)) != n: bad.append(f"{n - len(np.unique(ids))} duplicate ParticleIDs")
    for k, v in d.items():
        if len(v) != n: bad.append(f"{k} has {len(v)} rows, ParticleIDs {n}")
        elif v.dtype.kind == "f" and not np.all(np.isfinite(v)): bad.append(f"{k} has non-finite values")
    if bad:
        raise ValueError(f"validate_snapshot {path}: " + "; ".join(bad))
    return n


SPIN_POLICIES = ("refuse", "remove", "keep")


def _centre_body(d):
    """recentre a body on its centre of mass and remove its bulk velocity (in place), and fit its rigid spin (omega = I^-1 L);
    returns the masses, the removed bulk velocity, the fitted angular velocity and the intrinsic angular momentum about the centre"""
    m = d["Masses"].astype(float); x = d["Coordinates"].astype(float); v = d["Velocities"].astype(float)
    xc = x - (m[:, None] * x).sum(0) / m.sum(); v_bulk = (m[:, None] * v).sum(0) / m.sum(); vc = v - v_bulk
    L = (m[:, None] * np.cross(xc, vc)).sum(0)
    I = np.einsum("i,ij,ik->jk", m, xc, xc); I = np.trace(I) * np.eye(3) - I
    w = np.linalg.solve(I, L)
    d["Coordinates"], d["Velocities"] = xc, vc
    return m, v_bulk, w, L


def assemble_collision(target_snapshot, impactor_snapshot, out, R_target, R_impactor, impact_parameter, v_contact_over_v_esc=None,
                       v_inf=None, separation_in_contact_radii=2.0, time_before_contact=None, required=(), box_size=None, G=6.674e-8,
                       allow_header_differences=(), spin_policy="refuse", spin_tolerance=1e-3):
    """A collision IC from two bodies settled in isolation (GIZMO snapshots): every PartType0 dataset of both is carried (a
    dataset present in only one body is refused), each body is recentred and its bulk velocity removed, and its rigid spin is
    handled by spin_policy, measured by the rotation speed of the fitted rigid spin at the body's contact radius in units of its own
    escape speed: "refuse" (default) removes a spin up to spin_tolerance and stops on a larger one (a spin that matters is a choice
    of the setup, never deleted silently; a body settled in isolation keeps a numerical residual of a few 1e-4 that grows with
    settling time, so a caller whose bodies are meant not to spin says so with "remove"); "remove" removes it whatever its size; "keep" keeps
    every centred internal velocity unchanged (for a body spun on purpose). Per body (target, impactor) the removed bulk velocity,
    the fitted angular velocity, the spin measure, whether the spin was removed and the intrinsic angular momentum about the body's
    centre of mass before and after are printed, returned in the info under "spin" and written as PartType0 attributes of the output
    (target_* / impactor_*, and spin_policy). The bodies are set on collision_orbit (R_target, R_impactor = their reproducible contact surfaces) in the
    centre-of-mass frame, and the result is written with Time 0 so that a RestartFlag 2 start continues every evolved field
    (a fresh start would recompute the integrated density and, through it, reset the stress). ParticleIDs are kept as they are
    (they must already be disjoint: give each body its range when it is built); bodies are not rotated (a rotated body would need
    its StressTensor rotated too). Both inputs and the output pass validate_snapshot(required).
    The two snapshots must come from compatible runs: every header attribute must agree except the particle counts, the time and
    redshift, and the merge / split mass limits (GIZMO sets those from each run's own particle masses and reads them back from an
    IC: the combined file carries the lower merge and the higher split limit, as GIZMO's rule would give over both bodies);
    allow_header_differences names any further attribute allowed to differ (the target's value is kept), e.g. "BoxSize" when
    box_size is given. Returns the orbit info, with the per-body spin record under "spin"."""
    if spin_policy not in SPIN_POLICIES:
        raise ValueError(f"assemble_collision: spin_policy must be one of {SPIN_POLICIES}, not {spin_policy!r}")
    if not (np.isfinite(spin_tolerance) and spin_tolerance >= 0):
        raise ValueError(f"assemble_collision: spin_tolerance {spin_tolerance!r} is not a finite non-negative number")
    bodies, spin_record = [], {}
    for name, path, R_body in (("target", target_snapshot, R_target), ("impactor", impactor_snapshot, R_impactor)):
        if not (np.isfinite(R_body) and R_body > 0):
            raise ValueError(f"assemble_collision: the {name}'s contact radius {R_body!r} is not a positive number")
        validate_snapshot(path, required)
        d, hdr = read_snapshot(path)
        m, v_b, w, L_in = _centre_body(d)
        spin = float(np.linalg.norm(w)) * R_body / np.sqrt(2.0 * G * m.sum() / R_body)
        if spin_policy == "refuse" and spin > spin_tolerance:
            raise ValueError(f"assemble_collision: the {name} {path} spins (rotation speed at its contact radius {spin:.3e} of its escape speed > "
                             f"{spin_tolerance:g}); give spin_policy='keep' to keep it or 'remove' to remove it deliberately")
        removed = spin_policy != "keep"
        if removed:
            d["Velocities"] = d["Velocities"] - np.cross(w, d["Coordinates"])
            L_out = (m[:, None] * np.cross(d["Coordinates"], d["Velocities"])).sum(0)
        else:
            L_out = L_in
        print(f"  {name} {path}: removed bulk speed {np.linalg.norm(v_b):.4e} cm/s; rigid spin |omega| {np.linalg.norm(w):.4e} rad/s "
              f"(rotation speed at the contact radius {spin:.2e} of its escape speed) {'removed' if removed else 'kept'}; |L| about its centre "
              f"{np.linalg.norm(L_in):.4e} -> {np.linalg.norm(L_out):.4e}")
        bodies.append((d, hdr))
        spin_record[name] = dict(bulk_velocity_removed=[float(c) for c in v_b], angular_velocity=[float(c) for c in w], spin_over_escape_speed=spin,
                                 spin_removed=removed, spin_angular_momentum_in=[float(c) for c in L_in], spin_angular_momentum_out=[float(c) for c in L_out])
    (dt, hdr), (di, hdr_i) = bodies
    differing = []
    for k in sorted(set(hdr) | set(hdr_i)):
        if k in _HEADER_PER_BODY or k in allow_header_differences: continue
        a_, b_ = np.asarray(hdr.get(k, np.nan)), np.asarray(hdr_i.get(k, np.nan))
        if k not in hdr or k not in hdr_i or a_.shape != b_.shape or not np.array_equal(a_, b_, equal_nan=a_.dtype.kind == "f"):
            differing.append(k)
    if differing:
        raise ValueError(f"assemble_collision: the two snapshots' headers differ in {differing}: they come from runs with different "
                         "builds, units or material tables (name an attribute in allow_header_differences only if it is harmless)")
    merge_min = min(float(h["Minimum_Mass_For_Cell_Merge"]) for h in (hdr, hdr_i) if "Minimum_Mass_For_Cell_Merge" in h) \
        if any("Minimum_Mass_For_Cell_Merge" in h for h in (hdr, hdr_i)) else None
    split_max = max(float(h["Maximum_Mass_For_Cell_Split"]) for h in (hdr, hdr_i) if "Maximum_Mass_For_Cell_Split" in h) \
        if any("Maximum_Mass_For_Cell_Split" in h for h in (hdr, hdr_i)) else None
    only = sorted(set(dt) ^ set(di))
    if only:
        raise ValueError(f"assemble_collision: datasets present in only one body: {only}")
    if np.intersect1d(dt["ParticleIDs"], di["ParticleIDs"]).size:
        raise ValueError("assemble_collision: the two bodies share ParticleIDs (build them with disjoint id_offset ranges)")
    M_t, M_i = dt["Masses"].sum(), di["Masses"].sum()
    rel_x, rel_v, info = collision_orbit(M_t, M_i, R_target, R_impactor, impact_parameter, v_contact_over_v_esc, v_inf,
                                         separation_in_contact_radii, time_before_contact, G)
    f_t, f_i = M_i / (M_t + M_i), M_t / (M_t + M_i)
    box = box_size if box_size else float(hdr["BoxSize"])
    centre = np.full(3, 0.5 * box)
    dt["Coordinates"] = dt["Coordinates"] - f_t * rel_x + centre; dt["Velocities"] = dt["Velocities"] - f_t * rel_v
    di["Coordinates"] = di["Coordinates"] + f_i * rel_x + centre; di["Velocities"] = di["Velocities"] + f_i * rel_v
    comb = {k: np.concatenate([dt[k], di[k]]) for k in dt}
    n = len(comb["ParticleIDs"])
    if np.any(comb["Coordinates"] < 0) or np.any(comb["Coordinates"] >= box):
        raise ValueError(f"assemble_collision: the placed bodies extend outside the box [0, {box:.4e}) (span "
                         f"{comb['Coordinates'].min():.4e}..{comb['Coordinates'].max():.4e}): pass a larger box_size")
    with h5py.File(out, "w") as f:
        h = f.create_group("Header")
        for k, v in hdr.items(): h.attrs[k] = v
        npt = np.zeros(6, dtype=np.uint32); npt[0] = n
        h.attrs["NumPart_ThisFile"] = npt; h.attrs["NumPart_Total"] = npt; h.attrs["NumPart_Total_HighWord"] = np.zeros(6, dtype=np.uint32)
        h.attrs["NumFilesPerSnapshot"] = 1; h.attrs["Time"] = 0.0; h.attrs["BoxSize"] = box
        if merge_min is not None: h.attrs["Minimum_Mass_For_Cell_Merge"] = merge_min
        if split_max is not None: h.attrs["Maximum_Mass_For_Cell_Split"] = split_max
        g = f.create_group("PartType0")
        for k, v in comb.items(): g.create_dataset(k, data=v)
        g.attrs["N_target"] = len(dt["ParticleIDs"])
        for k in ("b", "v_contact", "v_inf", "b_inf", "v_esc", "r_contact", "r_start"): g.attrs["collision_" + k] = float(info[k])
        g.attrs["spin_policy"] = spin_policy; g.attrs["spin_tolerance"] = float(spin_tolerance)
        for name, rec in spin_record.items():
            for k, val in rec.items(): g.attrs[f"{name}_{k}"] = np.asarray(val, dtype=float) if isinstance(val, list) else val
    try:
        validate_snapshot(out, required)
    except Exception:
        os.remove(out); raise
    info = dict(info, spin=dict(policy=spin_policy, tolerance=spin_tolerance, **spin_record))
    print(f"  assemble_collision -> {out}: {n} particles ({len(dt['ParticleIDs'])} target + {len(di['ParticleIDs'])} impactor), "
          f"{len(comb)} datasets, box {box:.4e}; merge / split mass limits {merge_min} / {split_max}")
    return info


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
