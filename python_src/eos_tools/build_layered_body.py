"""Top-level Phase 17h IC builder: zone spec -> HSE solve -> particle
placement -> HDF5 output. Designed to replace the uniform-density rejection-
sampled-sphere generators in test/aneos_giant_impact, test/jutzi_crater,
test/cd21_hhe_compression once validated.

Example usage (Earth-like body):

    from python_src.eos_tools.eos_dispatch import Material
    from python_src.eos_tools.hse_solver   import Zone, solve_hse
    from python_src.eos_tools.shell_placement import fibonacci_shells
    from python_src.eos_tools.build_layered_body import write_hdf5

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
        g.create_dataset("Velocities", data=np.zeros((n, 3)))
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
