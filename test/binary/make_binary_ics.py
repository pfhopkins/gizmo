"""Two-body Kepler orbit: e = 0.9, q = m2/m1 = 0.1, as Type 5 sinks.

A deliberately hard case for a block-timestep integrator, and one with an exact answer. The
eccentricity swings the required timestep by (1+e)^3/(1-e)^3 ~ 6900 between apocentre and
pericentre, so the pair moves through many timebins per orbit; the mass ratio means the two
components want DIFFERENT bins, since each one's timestep responds to its own acceleration.
That combination -- one component changing bin while the other does not -- is exactly the
configuration that produces secular momentum non-conservation in a hierarchical scheme.

Everything the test measures is exactly known at t=0:
  * total momentum is identically zero (the ICs are constructed in the COM frame), so any
    non-zero P(t) is pure integration error, with no subtraction of a baseline;
  * total energy is -G*m1*m2/(2a), so the semi-major axis can be recovered from the energy
    at any time and compared against its initial value.

Softening is set far below pericentre in binary.params so the orbit is Newtonian throughout
and the test measures the integrator, not the force kernel.

Units are pc - km/s - Msun: G = 4.300917e-3 pc (km/s)^2 / Msun.
"""
import argparse

import h5py
import numpy as np

G_CODE = 4.300917270e-3   # pc (km/s)^2 / Msun
AU_PER_PC = 206264.806


def kepler_ic(m1, m2, a, ecc, boxsize):
    """Return (pos, vel, mass) for the two bodies, started at APOCENTRE in the COM frame.

    Apocentre is the natural start: it is the slowest point, so the initial timestep is the
    largest of the orbit and every subsequent pericentre passage forces the scheme to refine
    and then coarsen again -- which is the behaviour under test. Starting at pericentre would
    instead hand the integrator its easiest possible first step.
    """
    M = m1 + m2
    r_apo = a * (1.0 + ecc)
    # vis-viva at apocentre: v^2 = G*M*(2/r - 1/a), which at r = a(1+e) reduces to this
    v_apo = np.sqrt(G_CODE * M / a * (1.0 - ecc) / (1.0 + ecc))

    # separation along +x, relative velocity along +y -> orbit in the z=0 plane
    r_vec = np.array([r_apo, 0.0, 0.0])
    v_vec = np.array([0.0, v_apo, 0.0])

    # split about the centre of mass. Writing it this way makes sum(m*x) and sum(m*v)
    # identically zero up to round-off rather than approximately zero.
    x1 = -(m2 / M) * r_vec
    x2 = +(m1 / M) * r_vec
    v1 = -(m2 / M) * v_vec
    v2 = +(m1 / M) * v_vec

    centre = np.full(3, 0.5 * boxsize)
    pos = np.array([x1, x2]) + centre
    vel = np.array([v1, v2])
    mass = np.array([m1, m2])
    return pos, vel, mass


def make_binary_ics(m1, q, a_au, ecc, boxsize, outfile):
    m2 = q * m1
    a = a_au / AU_PER_PC
    pos, vel, mass = kepler_ic(m1, m2, a, ecc, boxsize)

    M = m1 + m2
    period = 2.0 * np.pi * np.sqrt(a ** 3 / (G_CODE * M))
    energy = -G_CODE * m1 * m2 / (2.0 * a)
    print(f"  m1={m1:g}  m2={m2:g}  q={q:g}  M={M:g} Msun")
    print(f"  a={a:.6e} pc ({a_au:g} AU)   e={ecc:g}")
    print(f"  pericentre={a*(1-ecc):.6e} pc   apocentre={a*(1+ecc):.6e} pc")
    print(f"  period={period:.6e} code time   E_exact={energy:.9e}")
    print(f"  v_orb=sqrt(GM/a)={np.sqrt(G_CODE*M/a):.6g} km/s")
    print(f"  |sum m*v| = {np.linalg.norm((mass[:, None]*vel).sum(0)):.3e} (should be ~0)")

    with h5py.File(outfile, "w") as F:
        F.create_group("Header")
        F["Header"].attrs["NumPart_ThisFile"] = [0, 0, 0, 0, 0, 2]
        F["Header"].attrs["NumPart_Total"] = [0, 0, 0, 0, 0, 2]
        F["Header"].attrs["NumPart_Total_HighWord"] = [0, 0, 0, 0, 0, 0]
        F["Header"].attrs["MassTable"] = [0, 0, 0, 0, 0, 0]   # per-particle masses differ
        F["Header"].attrs["BoxSize"] = boxsize
        F["Header"].attrs["Time"] = 0.0
        F["Header"].attrs["Redshift"] = 0.0
        F["Header"].attrs["NumFilesPerSnapshot"] = 1
        g = F.create_group("PartType5")
        g.create_dataset("Coordinates", data=pos)
        g.create_dataset("Velocities", data=vel)
        g.create_dataset("Masses", data=mass)
        g.create_dataset("ParticleIDs", data=np.array([1, 2], dtype=np.uint32))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--m1", type=float, default=1.0, help="primary mass, Msun")
    p.add_argument("--q", type=float, default=0.1, help="mass ratio m2/m1")
    p.add_argument("--a_au", type=float, default=100.0, help="semi-major axis, AU")
    p.add_argument("--ecc", type=float, default=0.9)
    p.add_argument("--boxsize", type=float, default=1.0, help="pc")
    p.add_argument("--out", default="binary_ics.hdf5")
    a = p.parse_args()
    make_binary_ics(a.m1, a.q, a.a_au, a.ecc, a.boxsize, a.out)
