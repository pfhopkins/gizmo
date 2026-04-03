"""Generate ICs for ANEOS giant impact test.

Creates a uniform-density differentiated sphere:
  - Iron core (CompositionType=1): inner 50% by radius
  - Forsterite mantle (CompositionType=0): outer shell
  - ~5000 particles total, glass-like random placement
  - At rest, with thermal energy set to ~2000 K

Uses code units where G=1 and the sphere has radius=1, mass~1.
"""

import numpy as np
import h5py


def generate_sphere_particles(n_target, radius=1.0):
    """Generate approximately n_target particles in a uniform sphere using rejection sampling."""
    # Over-generate in a cube, then reject outside sphere
    n_gen = int(n_target * 6 / np.pi * 1.1)  # ~10% extra
    rng = np.random.default_rng(42)

    coords = rng.uniform(-radius, radius, (n_gen, 3))
    r = np.linalg.norm(coords, axis=1)
    mask = r < radius
    coords = coords[mask]

    # Trim to target
    if len(coords) > n_target:
        coords = coords[:n_target]

    return coords


def generate_ics(fname="aneos_giant_impact_ics.hdf5", n_particles=5000):
    radius = 1.0
    core_frac = 0.5  # core radius = 50% of total radius

    # Material densities in CGS (for setting thermal energy)
    rho_forsterite = 3.2  # g/cm^3
    rho_iron = 7.87       # g/cm^3

    coords = generate_sphere_particles(n_particles, radius)
    n = len(coords)
    r = np.linalg.norm(coords, axis=1)

    # Assign materials: core (iron) vs mantle (forsterite)
    is_core = r < core_frac * radius
    comp_type = np.where(is_core, 1, 0).astype(np.int32)

    # Uniform density in code units
    volume = 4.0 / 3.0 * np.pi * radius**3
    total_mass = 1.0
    particle_mass = total_mass / n
    masses = np.full(n, particle_mass)

    # Velocities: at rest
    velocities = np.zeros((n, 3))

    # Internal energy: set to give ~2000 K
    # For forsterite at rho~3.2 g/cm^3, Cv ~ 1e7 erg/g/K -> u = Cv*T ~ 2e10 erg/g
    # For iron at rho~7.87 g/cm^3, Cv ~ 4.5e6 erg/g/K -> u = Cv*T ~ 9e9 erg/g
    # In code units we need u such that the EOS gives reasonable results.
    # Since the ANEOS dispatch converts to CGS using UNIT_*_IN_CGS,
    # and we'll set UnitLength=1cm, UnitMass=1g, UnitVelocity=1cm/s (CGS=code),
    # we can use CGS values directly.
    u_forsterite = 1.0e10  # erg/g, ~ 1000 K for forsterite
    u_iron = 5.0e9         # erg/g, ~ 1000 K for iron
    internal_energy = np.where(is_core, u_iron, u_forsterite)

    # Write HDF5
    with h5py.File(fname, "w") as f:
        # Header
        h = f.create_group("Header")
        h.attrs["NumPart_ThisFile"] = np.array([n, 0, 0, 0, 0, 0], dtype=np.int32)
        h.attrs["NumPart_Total"] = np.array([n, 0, 0, 0, 0, 0], dtype=np.uint32)
        h.attrs["NumPart_Total_HighWord"] = np.array([0, 0, 0, 0, 0, 0], dtype=np.uint32)
        h.attrs["MassTable"] = np.array([0, 0, 0, 0, 0, 0], dtype=np.float64)
        h.attrs["Time"] = 0.0
        h.attrs["Redshift"] = 0.0
        h.attrs["BoxSize"] = 10.0  # large enough box
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

        # PartType0
        g = f.create_group("PartType0")
        g.create_dataset("Coordinates", data=coords + 5.0)  # center in box
        g.create_dataset("Velocities", data=velocities)
        g.create_dataset("Masses", data=masses)
        g.create_dataset("InternalEnergy", data=internal_energy)
        g.create_dataset("ParticleIDs", data=np.arange(1, n + 1, dtype=np.uint32))
        g.create_dataset("CompositionType", data=comp_type)

    n_core = np.sum(is_core)
    n_mantle = n - n_core
    print(f"Generated {fname}: {n} particles ({n_mantle} forsterite + {n_core} iron)")


if __name__ == "__main__":
    generate_ics()
