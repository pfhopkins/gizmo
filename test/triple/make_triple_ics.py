"""Hierarchical triple: equal-mass inner binary + distant tertiary, as Type 5 sinks.

The many-timebin configuration test/binary cannot produce: a bound pair splits by at most ~1
bin (the symmetric 2-body criterion binds both members), but a hierarchy separates cleanly --
the tertiary's timestep is set by the slow outer orbit, the inner stars' by the fast inner one,
and the period ratio (a_out/a_in)^1.5 * sqrt(M_in/M_tot) puts them log2 of that apart. Defaults
give ratio 88, ~6.5 bins.

Design choices that make the diagnostics clean:
  * equal inner masses -> the inner pair shares a bin by symmetry and cannot leak momentum
    against itself; any secular COM drift is the inner<->outer channel under test;
  * both orbits circular -> no pericentre refinement churn; the bin structure is static, so a
    measured leak cannot be blamed on bin transitions;
  * constructed in the exact COM frame -> P == 0 at t=0 with no baseline subtraction;
  * inner orbit started perpendicular to the outer separation so the hierarchies are not
    degenerate at t=0.

Units are pc - km/s - Msun: G = 4.300917e-3.
"""
import argparse

import h5py
import numpy as np

G_CODE = 4.300917270e-3
AU_PER_PC = 206264.806


def make_triple_ics(m_in, m3, a_in_au, a_out_au, boxsize, outfile):
    a_in = a_in_au / AU_PER_PC
    a_out = a_out_au / AU_PER_PC
    m_binary = 2.0 * m_in
    mtot = m_binary + m3

    p_in = 2 * np.pi * np.sqrt(a_in ** 3 / (G_CODE * m_binary))
    p_out = 2 * np.pi * np.sqrt(a_out ** 3 / (G_CODE * mtot))

    # outer orbit: inner-binary COM vs tertiary, circular, in the z=0 plane
    v_rel_out = np.sqrt(G_CODE * mtot / a_out)
    x_bc, x_3 = -(m3 / mtot) * a_out, +(m_binary / mtot) * a_out
    vy_bc, vy_3 = -(m3 / mtot) * v_rel_out, +(m_binary / mtot) * v_rel_out

    # inner orbit: circular, split about the inner COM, separation along y
    v_rel_in = np.sqrt(G_CODE * m_binary / a_in)
    pos = np.array([[x_bc, +0.5 * a_in, 0.0],
                    [x_bc, -0.5 * a_in, 0.0],
                    [x_3, 0.0, 0.0]])
    vel = np.array([[-0.5 * v_rel_in, vy_bc, 0.0],
                    [+0.5 * v_rel_in, vy_bc, 0.0],
                    [0.0, vy_3, 0.0]])
    mass = np.array([m_in, m_in, m3])

    pos += 0.5 * boxsize
    vel -= (mass[:, None] * vel).sum(0) / mass.sum()      # exact COM frame

    print(f"  m_in={m_in:g}+{m_in:g}  m3={m3:g}  a_in={a_in_au:g} AU  a_out={a_out_au:g} AU")
    print(f"  P_in={p_in:.6e}  P_out={p_out:.6e}  ratio={p_out/p_in:.1f} "
          f"(~{np.log2(p_out/p_in):.1f} timebins)")
    print(f"  |sum m*v| = {np.linalg.norm((mass[:, None]*vel).sum(0)):.3e} (should be ~0)")

    with h5py.File(outfile, "w") as F:
        F.create_group("Header")
        F["Header"].attrs["NumPart_ThisFile"] = [0, 0, 0, 0, 0, 3]
        F["Header"].attrs["NumPart_Total"] = [0, 0, 0, 0, 0, 3]
        F["Header"].attrs["NumPart_Total_HighWord"] = [0] * 6
        F["Header"].attrs["MassTable"] = [0.0] * 6
        F["Header"].attrs["BoxSize"] = boxsize
        F["Header"].attrs["Time"] = 0.0
        F["Header"].attrs["Redshift"] = 0.0
        F["Header"].attrs["NumFilesPerSnapshot"] = 1
        g = F.create_group("PartType5")
        g.create_dataset("Coordinates", data=pos)
        g.create_dataset("Velocities", data=vel)
        g.create_dataset("Masses", data=mass)
        g.create_dataset("ParticleIDs", data=np.array([1, 2, 3], dtype=np.uint32))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--m_in", type=float, default=0.5, help="each inner component, Msun")
    p.add_argument("--m3", type=float, default=1.0, help="tertiary, Msun")
    p.add_argument("--a_in_au", type=float, default=4.0)
    p.add_argument("--a_out_au", type=float, default=100.0)
    p.add_argument("--boxsize", type=float, default=1.0, help="pc")
    p.add_argument("--out", default="triple_ics.hdf5")
    a = p.parse_args()
    make_triple_ics(a.m_in, a.m3, a.a_in_au, a.a_out_au, a.boxsize, a.out)
