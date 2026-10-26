"""2D periodic box of uniform, static, cold gas on a jittered lattice with a Gaussian cosmic-ray energy bump.
The gas is cold (u = 5e-7) so that its own pressure work does not swamp the CR energy budget the test checks."""
import sys
import h5py
import numpy as np


def make_ics(fn, n_side=32, box=1.0, jitter=0.2, rho=1.0, u=5e-7, ecr0=1e-6, sigma=0.1, seed=7):
    rng = np.random.default_rng(seed)
    g = (np.arange(n_side) + 0.5) / n_side * box
    xy = np.array(np.meshgrid(g, g, indexing="ij")).reshape(2, -1).T
    xy = (xy + rng.uniform(-jitter, jitter, xy.shape) * box / n_side) % box
    n = len(xy)
    x = np.zeros((n, 3))
    x[:, :2] = xy
    m = np.full(n, rho * box**2 / n)
    r2 = ((xy - 0.5 * box) ** 2).sum(1)
    Ecr = ecr0 * (1e-3 + np.exp(-r2 / (2 * sigma**2))) * (m / rho)
    h = np.sqrt(20 / np.pi) * box / n_side  # initial guess for DesNumNgb=20 in 2D
    with h5py.File(fn, "w") as f:
        hd = f.create_group("Header")
        hd.attrs["NumPart_ThisFile"] = np.array([n, 0, 0, 0, 0, 0], dtype=np.uint32)
        hd.attrs["NumPart_Total"] = np.array([n, 0, 0, 0, 0, 0], dtype=np.uint32)
        hd.attrs["NumPart_Total_HighWord"] = np.zeros(6, dtype=np.uint32)
        hd.attrs["MassTable"] = np.zeros(6)
        hd.attrs["Time"] = 0.0
        hd.attrs["Redshift"] = 0.0
        hd.attrs["BoxSize"] = box
        hd.attrs["NumFilesPerSnapshot"] = 1
        hd.attrs["Flag_DoublePrecision"] = 1
        p = f.create_group("PartType0")
        p["Coordinates"] = x
        p["Velocities"] = np.zeros_like(x)
        p["ParticleIDs"] = np.arange(1, n + 1, dtype=np.uint32)
        p["Masses"] = m
        p["InternalEnergy"] = np.full(n, u)
        p["CosmicRayEnergy"] = Ecr
        p["KernelMaxRadius"] = np.full(n, h)
    return Ecr.sum()


if __name__ == "__main__":
    print(make_ics(sys.argv[1] if len(sys.argv) > 1 else "cr_flux_conservation_ics.hdf5"))
