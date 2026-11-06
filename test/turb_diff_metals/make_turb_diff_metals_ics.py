"""Metal step box for the TURB_DIFF_METALS conservation test.

Periodic box of side L: cold, metal-poor gas for x < L/2 and hot, metal-rich gas for x > L/2, at equal density.
The temperature contrast puts the two halves in different timebins, and the pressure jump drives the shear
the Smagorinsky diffusivity needs. A spread of cell masses, with merge/split thresholds written to the header,
makes cells merge and split during the run.
"""

import h5py
import numpy as np

Z_SOLAR = 0.0142


def make_ics(filename, n_side=20, box=0.5, T_cold=1e4, T_hot=1e6, Z_cold=0.1, Z_hot=60.0, mass_spread=0.0, seed=42, B_z=0.0, time=0.0):
    """Writes the IC. Z_cold/Z_hot in solar units; mass_spread>0 draws cell masses uniformly from
    m*(1 -/+ mass_spread) and sets the merge/split thresholds inside that range. B_z>0 adds a uniform field
    along z, in units of the run's UnitMagneticField_in_gauss. time is the header time (the scale factor, for a comoving run)."""
    rng = np.random.default_rng(seed)
    g = (np.arange(n_side) + 0.5) / n_side * box
    x = np.array(np.meshgrid(g, g, g, indexing="ij")).reshape(3, -1).T
    x = (x + rng.uniform(-0.05, 0.05, x.shape) * box / n_side) % box
    n = len(x)
    hot = x[:, 0] > 0.5 * box
    u_per_K = 206.58684 / 1e4  # (km/s)^2 per K, for mu ~ 0.6
    u = np.where(hot, T_hot, T_cold) * u_per_K
    Z = np.where(hot, Z_hot, Z_cold) * Z_SOLAR
    m0 = 1e-10
    m = m0 * (1 + rng.uniform(-mass_spread, mass_spread, n)) if mass_spread > 0 else np.full(n, m0)
    h = (3 * 32 / (4 * np.pi)) ** (1 / 3) * box / n_side
    with h5py.File(filename, "w") as f:
        hd = f.create_group("Header")
        hd.attrs["NumPart_ThisFile"] = np.array([n, 0, 0, 0, 0, 0], dtype=np.uint32)
        hd.attrs["NumPart_Total"] = np.array([n, 0, 0, 0, 0, 0], dtype=np.uint32)
        hd.attrs["NumPart_Total_HighWord"] = np.zeros(6, dtype=np.uint32)
        hd.attrs["MassTable"] = np.zeros(6)
        hd.attrs["Time"] = time
        hd.attrs["Redshift"] = 0.0
        hd.attrs["BoxSize"] = box
        hd.attrs["NumFilesPerSnapshot"] = 1
        hd.attrs["Flag_Metals"] = 1
        hd.attrs["Flag_DoublePrecision"] = 1
        if mass_spread > 0:
            hd.attrs["Minimum_Mass_For_Cell_Merge"] = m0 * (1 - 0.6 * mass_spread)
            hd.attrs["Maximum_Mass_For_Cell_Split"] = m0 * (1 + 0.6 * mass_spread)
        p = f.create_group("PartType0")
        p["Coordinates"] = x
        p["Velocities"] = np.zeros_like(x)
        p["ParticleIDs"] = np.arange(1, n + 1, dtype=np.uint32)
        p["Masses"] = m
        p["InternalEnergy"] = u
        p["Metallicity"] = Z
        p["KernelMaxRadius"] = np.full(n, h)
        if B_z > 0:
            B = np.zeros_like(x)
            B[:, 2] = B_z
            p["MagneticField"] = B
    return (m * Z).sum()


if __name__ == "__main__":
    make_ics("turb_diff_metals_ics.hdf5")
