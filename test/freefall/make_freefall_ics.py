"""Generate ICs for the freefall test: a cold, static, uniform-density gas sphere.

Positions are a cubic lattice with a small random jitter, cut to a sphere. A lattice rather than
random sampling because the test compares forces against the analytic field of a uniform sphere:
Poisson density noise would put O(N^-1/2) per-particle force fluctuations on top of the tree error
it is meant to measure. The jitter breaks the lattice's exact symmetries so that the tree is not
evaluated on a degenerate configuration (many particles at identical node offsets).

The N particles nearest the origin are kept, so N is exact and the sphere radius is R to within
a fraction of the spacing. The spacing d is set from N so that the lattice density m/d^3 equals
the nominal 3M/(4 pi R^3): the interior field is then -G M r/R^3 exactly, whatever the surface
rounding does to the outermost shell.

Code units G = M = R = 1.
"""
import argparse

import h5py
import numpy as np

DES_NUM_NGB = 32  # matches DesNumNgb in freefall.params; only sets the initial kernel-radius guess


def lattice_spacing(N, R=1.0):
    """Spacing at which a cubic lattice holding N points in a sphere of radius R has density N/V."""
    return (4.0 * np.pi * R**3 / (3.0 * N)) ** (1.0 / 3.0)


def make_freefall_ics(outfile, N=2**15, M=1.0, R=1.0, u0=1e-5, jitter=0.1, seed=42, boxsize=4.0):
    """Write a uniform sphere of N equal-mass gas cells at rest, centred on the origin.

    jitter is the per-axis displacement amplitude (uniform in [-jitter, +jitter]) in units of the
    lattice spacing. u0 is the specific internal energy; see freefall.params for why it is tiny.
    boxsize is written to the header only (the run is non-periodic).
    """
    rng = np.random.default_rng(seed)
    d = lattice_spacing(N, R)
    n = int(np.ceil(R / d)) + 2
    # half-integer offsets keep the origin empty: no cell sits where the analytic field vanishes
    ax = (np.arange(-n, n) + 0.5) * d
    pos = np.stack(np.meshgrid(ax, ax, ax, indexing="ij"), axis=-1).reshape(-1, 3)
    pos += jitter * d * rng.uniform(-1.0, 1.0, pos.shape)
    r = np.sqrt(np.sum(pos**2, axis=1))
    order = np.argsort(r)
    if r[order[N - 1]] > 1.2 * R:
        raise RuntimeError("lattice block too small for the requested N")
    pos = pos[order[:N]]

    mass = np.full(N, M / N)
    vel = np.zeros((N, 3))
    u = np.full(N, u0)
    # kernel support radius enclosing ~DES_NUM_NGB lattice cells; GIZMO iterates it to convergence
    hsml = np.full(N, d * (3.0 * DES_NUM_NGB / (4.0 * np.pi)) ** (1.0 / 3.0))

    with h5py.File(outfile, "w") as F:
        h = F.create_group("Header")
        npart = np.array([N, 0, 0, 0, 0, 0], dtype=np.int32)
        h.attrs["NumPart_ThisFile"] = npart
        h.attrs["NumPart_Total"] = npart.astype(np.uint32)
        h.attrs["NumPart_Total_HighWord"] = np.zeros(6, dtype=np.uint32)
        h.attrs["MassTable"] = np.zeros(6)  # per-cell Masses dataset is authoritative
        h.attrs["Time"] = 0.0
        h.attrs["Redshift"] = 0.0
        h.attrs["BoxSize"] = boxsize
        h.attrs["NumFilesPerSnapshot"] = 1
        # 0 = the build's precision; datasets are float64 and HDF5 converts on read. Set to 1 only
        # with INPUT_IN_DOUBLEPRECISION, else GIZMO aborts (read_ic.cc).
        h.attrs["Flag_DoublePrecision"] = 0
        for flag in ("Flag_Sfr", "Flag_Cooling", "Flag_StellarAge", "Flag_Metals", "Flag_Feedback"):
            h.attrs[flag] = 0
        g = F.create_group("PartType0")
        g.create_dataset("Coordinates", data=pos)
        g.create_dataset("Velocities", data=vel)
        g.create_dataset("Masses", data=mass)
        g.create_dataset("InternalEnergy", data=u)
        g.create_dataset("ParticleIDs", data=np.arange(1, N + 1, dtype=np.uint32))
        g.create_dataset("SmoothingLength", data=hsml)  # read only with INPUT_READ_KERNELRADIUS
    return pos, mass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=2**15)
    parser.add_argument("--u0", type=float, default=1e-5)
    parser.add_argument("--jitter", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="freefall_ics.hdf5")
    args = parser.parse_args()
    make_freefall_ics(args.out, N=args.N, u0=args.u0, jitter=args.jitter, seed=args.seed)
