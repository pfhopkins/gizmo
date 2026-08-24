"""Initial conditions for the fewbody suite: many independent cold-collapse few-body problems.

Each problem is a separate IC file holding a single system of n stars, n drawn in [n_min, n_max],
with masses from a Salpeter IMF, dropped at rest at random positions inside a sphere of radius
r_sys. Starting cold means the system falls together on its free-fall time and passes through a
deep, near-radial collapse, which is the point: it manufactures the close encounters and the large
dynamic range in timestep that stress the integrator, without needing a contrived initial orbit.
Drawing the masses rather than making them equal is what gives the encounters their teeth -- an
unequal-mass system segregates, preferentially ejects its lightest members, and hardens a binary
of its heaviest, so the suite spans a wide range of encounter speeds and mass ratios.

One problem per file, rather than many systems packed into one box, so that each run contains
exactly one isolated N-body problem. That is what makes the energy budget per problem exact:
nothing else is in the box to exchange energy with, no neighbour tugs on it, and an ejected star
cannot wander into someone else's system. The suite is then run as many independent GIZMO
invocations in parallel -- see test_fewbody.py.

Every draw comes from one seeded generator consumed in a fixed order, so the whole suite is a pure
function of (seed, n_problems) and re-running reproduces it exactly.

Units are pc - km/s - Msun, so G = 4.302e-3 pc (km/s)^2 / Msun, matching plummer_binaries.
"""
import argparse
from os import makedirs, path

import numpy as np
import h5py

G_CODE = 4.300917270e-3  # pc (km/s)^2 / Msun
SALPETER_ALPHA = -2.35   # dN/dm ~ m^alpha


def sample_salpeter(rng, size, m_min, m_max, alpha=SALPETER_ALPHA):
    """Inverse-CDF draw from dN/dm ~ m^alpha on [m_min, m_max].

    For a power law the CDF integrates in closed form, so this needs no rejection step and the
    draw stays a pure function of the RNG state -- which is what keeps the IC reproducible.
    """
    b = alpha + 1.0
    if abs(b) < 1e-12:  # alpha == -1 would make the integral logarithmic
        return m_min * (m_max / m_min) ** rng.random(size)
    lo, hi = m_min**b, m_max**b
    return (lo + (hi - lo) * rng.random(size)) ** (1.0 / b)


def freefall_time(m_system, r_sys):
    """Collapse timescale of a uniform sphere, used to set each problem's TimeMax.

    Each problem gets its own TimeMax in units of ITS free-fall time, so a 3-body system of
    0.5 Msun stars and a 10-body system containing a 40 Msun star are followed for the same
    number of dynamical times instead of the same absolute time.
    """
    return np.sqrt(r_sys**3 / (G_CODE * m_system))


def make_one_problem(pos, masses, boxsize, outfile):
    """Write a single-system IC. The system is centred in the box."""
    n = len(masses)
    with h5py.File(outfile, "w") as F:
        F.create_group("Header")
        F["Header"].attrs["NumPart_ThisFile"] = [0, 0, 0, 0, 0, n]
        F["Header"].attrs["NumPart_Total"] = [0, 0, 0, 0, 0, n]
        F["Header"].attrs["NumPart_Total_HighWord"] = [0, 0, 0, 0, 0, 0]
        # zero, not the mean: a nonzero MassTable entry tells GIZMO every particle of that type
        # shares one mass and the per-particle Masses dataset may be ignored
        F["Header"].attrs["MassTable"] = [0, 0, 0, 0, 0, 0]
        F["Header"].attrs["BoxSize"] = boxsize
        F["Header"].attrs["Time"] = 0.0
        F["Header"].attrs["Redshift"] = 0.0
        F["Header"].attrs["NumFilesPerSnapshot"] = 1
        g = F.create_group("PartType5")
        g.create_dataset("Coordinates", data=pos)
        g.create_dataset("Velocities", data=np.zeros_like(pos))  # cold
        g.create_dataset("Masses", data=masses)
        g.create_dataset("ParticleIDs", data=np.arange(1, n + 1, dtype=np.uint32))


def make_fewbody_suite(
    n_problems=48,
    n_min=3,
    n_max=10,
    m_min=0.5,
    m_max=50.0,
    r_sys=0.01,
    boxsize=1.0,
    seed=42,
    outdir="ics",
    prefix="fewbody",
):
    """Generate the whole suite. Returns a list of dicts, one per problem.

    r_sys is 0.01 pc = ~2060 AU against a 3.117e-5 pc (~6.4 AU) softening, so the collapse has
    ~300x of dynamic range to work through before the softening floors the acceleration -- deep
    enough to be a real test, bounded enough that the shortest timestep stays finite.
    """
    rng = np.random.default_rng(seed)
    makedirs(outdir, exist_ok=True)
    problems = []
    for k in range(n_problems):
        n = int(rng.integers(n_min, n_max + 1))
        masses = sample_salpeter(rng, n, m_min, m_max)
        # uniform in the sphere: cube-root of a uniform gives a flat radial mass distribution
        u = rng.random(n) ** (1.0 / 3.0)
        d = rng.normal(size=(n, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        pos = r_sys * u[:, None] * d
        # put the system's centre of mass at rest at the box centre, so it neither drifts across
        # the box nor sits on the origin where a coordinate singularity could flatter the result
        pos -= np.average(pos, axis=0, weights=masses)
        pos += 0.5 * boxsize
        name = f"{prefix}_p{k:03d}"
        ic = path.join(outdir, name + "_ics.hdf5")
        make_one_problem(pos, masses, boxsize, ic)
        problems.append({
            "index": k, "name": name, "ic": ic, "n": n,
            "m_total": float(masses.sum()), "m_min": float(masses.min()),
            "m_max": float(masses.max()),
            "t_ff": float(freefall_time(masses.sum(), r_sys)),
        })
    return problems


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_problems", type=int, default=48)
    ap.add_argument("--n_min", type=int, default=3)
    ap.add_argument("--n_max", type=int, default=10)
    ap.add_argument("--m_min", type=float, default=0.5, help="Msun")
    ap.add_argument("--m_max", type=float, default=50.0, help="Msun")
    ap.add_argument("--r_sys", type=float, default=0.01, help="pc")
    ap.add_argument("--boxsize", type=float, default=1.0, help="pc")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="ics")
    a = ap.parse_args()
    pr = make_fewbody_suite(a.n_problems, a.n_min, a.n_max, a.m_min, a.m_max,
                            a.r_sys, a.boxsize, a.seed, a.outdir)
    ns = np.array([p["n"] for p in pr]); tf = np.array([p["t_ff"] for p in pr])
    mt = np.array([p["m_total"] for p in pr])
    print(f"{len(pr)} problems, N = {ns.min()}-{ns.max()} (total {ns.sum()} stars)")
    print(f"system mass : {mt.min():.2f} - {mt.max():.2f} Msun")
    print(f"t_ff        : {tf.min():.3e} - {tf.max():.3e} code time units")
