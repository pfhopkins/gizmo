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

    print(f"Wrote {path}: n={n}  R={prof['R']:.3e} cm  "
          f"P_c={prof['P'][-1]:.3e} dyn/cm^2  T_c={prof['T'][-1]:.3e} K")


def build_planet(zones, M_total, T_surface, P_surface, n_particles, out, des_num_ngb=32, mass_ratio="spacing",
                 box_size=None, seed=17, n_steps=4000):
    """Layered body in hydrostatic equilibrium, in one call: HSE profile -> SEAGen shells with the particle mass
    matched to the spacing across each material interface -> GIZMO's own kernel density at the placed positions
    (cubic spline, des_num_ngb) -> u per particle so that EOS(rho_GIZMO, u) = P_HSE(r) -> HDF5 IC at `out`.
    zones are ordered outer -> inner (see hse_solver.Zone). Particles whose target pressure no valid EOS state
    reaches (e.g. below a table's lowest valid temperature) keep the closest valid state and are reported.
    For the exact densities GIZMO will use, run GIZMO on `out` for zero time and recalibrate with
    sph_density.calibrate_u on its snapshot_000 densities (the two agree to ~1e-4 for DesNumNgb 32,
    MaxNumNgbDeviation 0.1). Returns (placed, prof)."""
    from .hse_solver import solve_hse
    from .shell_placement import seagen_shells
    from .sph_density import correct_u_for_sph_density
    prof = solve_hse(M_total=M_total, T_surface=T_surface, P_surface=P_surface, zones=zones, n_steps=n_steps)
    print(f"  HSE profile: R = {prof['R']:.4e} cm, P_c = {prof['P'][-1]:.3e}, T_c = {prof['T'][-1]:.3e}")
    placed = seagen_shells(prof, zones, n_particles, mass_ratio=mass_ratio, seed=seed)
    correct_u_for_sph_density(placed, prof, zones, des_num_ngb=des_num_ngb, verbose=True)
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
