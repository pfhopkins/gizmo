"""Hierarchical triple: unequal-mass eccentric inner binary + inclined tertiary, as Type 5 sinks.

The many-timebin configuration test/binary cannot produce: a bound pair splits by at most ~1
bin (the symmetric 2-body criterion binds both members), but a hierarchy separates cleanly --
the tertiary's timestep is set by the slow outer orbit, the inner stars' by the fast inner one,
and the period ratio (a_out/a_in)^1.5 * sqrt(M_in/M_tot) puts them log2 of that apart. Defaults
give ratio 88, ~6.5 bins. Period depends only on a, so eccentricity does not change that.

Design choices that make the diagnostics clean:
  * UNEQUAL inner masses -- see below;
  * constructed in the exact COM frame -> P == 0 at t=0 with no baseline subtraction;
  * GENERIC orientation and phase -- see below.

UNEQUAL INNER MASSES. dt_tidal is set by the COMPANION mass, so 0.8+0.2 puts the two components
one bin apart -- the same asymmetry the momentum leak was traced to. This matters because the
inner binary carries 86% of the total energy: at equal masses it shares a bin by symmetry and is
structurally blind to the source prediction, making the energy diagnostic a 6:1 mixture dominated
by a component that cannot respond to what is under test. The cost is that COM drift is no longer
unambiguously the inner<->outer channel, since the pair can now leak against itself.

ECCENTRIC, INCLINED, RANDOMLY ORIENTED, so that no part of the configuration is a special case.
Circular coplanar orbits are measure-zero: the separation never varies, so the source prediction
is probed at a single staleness, and with all motion in one plane the z components of the
momentum error are structurally unlike x and y. Eccentricity also sweeps dt through the orbit
(5.2x, 2.4 bins at e_in=0.5), exercising bin TRANSITIONS that a static hierarchy never reaches.

The price is that ENERGY error becomes useless as a convergence probe here -- it oscillates by
several times its own accumulated value within an outer orbit. test_triple.py measures order from
trajectory error instead; see the sweep block there.

STAYING NON-CHAOTIC is the constraint that bounds those choices, because a chaotic system has no
convergence order to measure:
  * hierarchy -- checked against Mardling & Aarseth (2001) at IC time and asserted, not assumed.
    Defaults sit ~6x inside the stability boundary;
  * mutual inclination below the Kozai-Lidov critical angle, cos^2 i = 3/5 -> 39.2 deg. Inside
    the 39.2-140.8 window e_in and i librate on a t ~ P_out^2/P_in timescale, which would move
    the inner pericentre -- and therefore the bin structure -- under the diagnostics. At 25 deg
    there is no libration, only regular apsidal and nodal precession.
Randomness is therefore in the ORIENTATION only: the whole system gets one uniformly random
rigid rotation at a fixed seed, which removes the coordinate-axis alignment without touching the
mutual inclination that stability depends on.

Units are pc - km/s - Msun. G comes from gizmo.units (GIZMO's constants.h), 4.300711e-3.
"""
import argparse

import h5py
import numpy as np

import os as _os
import sys as _sys
# Run standalone (the tests invoke this as a subprocess from the test directory), so the
# harness package (test/harness in this tree; python_src upstream) is not on the path the
# way it is under pytest. G must come from gizmo.units, which mirrors
# GIZMO's constants.h: an IC built with a different G than the code integrates with is not the
# orbit it claims to be.
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  "..", "harness"))
from gizmo.units import G_CODE, AU_PER_PC  # noqa: E402

KOZAI_CRIT_DEG = np.degrees(np.arccos(np.sqrt(3.0 / 5.0)))     # 39.23


def _kepler_state(mu, a, e, f_deg, inc_deg, node_deg, argp_deg):
    """Relative (r, v) for a Keplerian orbit from its elements. f is the true anomaly."""
    f, inc, node, argp = np.radians([f_deg, inc_deg, node_deg, argp_deg])
    p = a * (1.0 - e ** 2)
    r = p / (1.0 + e * np.cos(f))
    r_pf = np.array([r * np.cos(f), r * np.sin(f), 0.0])
    v_pf = np.sqrt(mu / p) * np.array([-np.sin(f), e + np.cos(f), 0.0])
    cO, sO, ci, si, cw, sw = (np.cos(node), np.sin(node), np.cos(inc),
                              np.sin(inc), np.cos(argp), np.sin(argp))
    # R_z(node) R_x(inc) R_z(argp)
    R = np.array([
        [cO * cw - sO * sw * ci, -cO * sw - sO * cw * ci,  sO * si],
        [sO * cw + cO * sw * ci, -sO * sw + cO * cw * ci, -cO * si],
        [sw * si,                 cw * si,                 ci],
    ])
    return R @ r_pf, R @ v_pf


def _random_rotation(seed):
    """Uniformly random 3D rotation (Shoemake's quaternion method); no scipy dependency."""
    u1, u2, u3 = np.random.default_rng(seed).random(3)
    q = np.array([np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
                  np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
                  np.sqrt(u1) * np.sin(2 * np.pi * u3),
                  np.sqrt(u1) * np.cos(2 * np.pi * u3)])
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def _stability_margin(a_in, a_out, e_in, e_out, m_binary, m3, i_mut_deg):
    """Mardling & Aarseth (2001) critical outer/inner semi-major axis ratio, and the margin."""
    q = m3 / m_binary
    crit = 2.8 * ((1.0 + q) * (1.0 + e_out) / np.sqrt(1.0 - e_out)) ** 0.4
    crit *= 1.0 - 0.3 * i_mut_deg / 180.0
    return crit, (a_out / a_in) / crit


def make_triple_ics(m1, m2, m3, a_in_au, a_out_au, e_in, e_out, i_mut, seed, boxsize, outfile):
    a_in = a_in_au / AU_PER_PC
    a_out = a_out_au / AU_PER_PC
    m_binary = m1 + m2
    mtot = m_binary + m3

    p_in = 2 * np.pi * np.sqrt(a_in ** 3 / (G_CODE * m_binary))
    p_out = 2 * np.pi * np.sqrt(a_out ** 3 / (G_CODE * mtot))

    # --- non-chaotic checks, asserted rather than assumed ---------------------------------
    crit, margin = _stability_margin(a_in, a_out, e_in, e_out, m_binary, m3, i_mut)
    assert margin > 2.0, (
        f"a_out/a_in = {a_out/a_in:.1f} is only {margin:.2f}x the Mardling-Aarseth critical "
        f"ratio {crit:.2f}; this triple is not safely hierarchical and its convergence order "
        f"would be meaningless")
    assert i_mut < KOZAI_CRIT_DEG - 5.0 or i_mut > 180.0 - KOZAI_CRIT_DEG + 5.0, (
        f"mutual inclination {i_mut:g} deg is inside the Kozai-Lidov window "
        f"({KOZAI_CRIT_DEG:.1f}-{180-KOZAI_CRIT_DEG:.1f} deg); e_in would librate and move the "
        f"inner pericentre, and with it the bin structure, under the diagnostics")

    # Inner orbit in its own plane; outer inclined by i_mut about the line of nodes. Phases are
    # arbitrary but fixed, and chosen off pericentre/apocentre so t=0 is not a special point.
    r_in, v_in = _kepler_state(G_CODE * m_binary, a_in, e_in, f_deg=63.0,
                               inc_deg=0.0, node_deg=0.0, argp_deg=0.0)
    r_out, v_out = _kepler_state(G_CODE * mtot, a_out, e_out, f_deg=137.0,
                                 inc_deg=i_mut, node_deg=0.0, argp_deg=41.0)

    # Split each relative orbit about its own centre of mass.
    f1, f2 = m2 / m_binary, m1 / m_binary        # split about the inner COM by mass ratio
    pos = np.array([+f1 * r_in, -f2 * r_in, np.zeros(3)])
    vel = np.array([+f1 * v_in, -f2 * v_in, np.zeros(3)])
    pos[:2] += -(m3 / mtot) * r_out
    vel[:2] += -(m3 / mtot) * v_out
    pos[2] = +(m_binary / mtot) * r_out
    vel[2] = +(m_binary / mtot) * v_out
    mass = np.array([m1, m2, m3])

    R = _random_rotation(seed)                    # one rigid rotation: orientation only
    pos, vel = pos @ R.T, vel @ R.T

    vel -= (mass[:, None] * vel).sum(0) / mass.sum()      # exact COM frame
    pos -= (mass[:, None] * pos).sum(0) / mass.sum()
    pos += 0.5 * boxsize

    r_peri_in, r_apo_in = a_in_au * (1 - e_in), a_in_au * (1 + e_in)
    r_peri_out = a_out_au * (1 - e_out)
    print(f"  m_in={m1:g}+{m2:g}  m3={m3:g}  a_in={a_in_au:g} AU  a_out={a_out_au:g} AU")
    print(f"  inner mass ratio {m1/m2:.1f} -> dt_tidal differs by sqrt(m1/m2) = "
          f"{np.sqrt(m1/m2):.2f} ({np.log2(np.sqrt(m1/m2)):.1f} bins) INSIDE the pair")
    print(f"  e_in={e_in:g}  e_out={e_out:g}  i_mut={i_mut:g} deg  (Kozai window "
          f"{KOZAI_CRIT_DEG:.1f}-{180-KOZAI_CRIT_DEG:.1f}, avoided)  seed={seed}")
    print(f"  P_in={p_in:.6e}  P_out={p_out:.6e}  ratio={p_out/p_in:.1f} "
          f"(~{np.log2(p_out/p_in):.1f} timebins)")
    print(f"  stability: a_out/a_in = {a_out/a_in:.1f} vs M&A critical {crit:.2f} "
          f"-> {margin:.1f}x margin")
    print(f"  separations: inner {r_peri_in:.2f}-{r_apo_in:.2f} AU, outer pericentre "
          f"{r_peri_out:.1f} AU ({r_peri_out/r_apo_in:.1f}x the inner apocentre)")
    # dt ~ sqrt(r^3/GM), so the inner timestep swings by (r_apo/r_peri)^1.5 over an orbit --
    # this is the bin-transition exercise the circular version could not provide.
    print(f"  inner dt varies {((1+e_in)/(1-e_in))**1.5:.1f}x over an orbit "
          f"({np.log2(((1+e_in)/(1-e_in))**1.5):.1f} bins)")
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
    p.add_argument("--m1", type=float, default=0.8, help="primary inner component, Msun")
    p.add_argument("--m2", type=float, default=0.2, help="secondary inner component, Msun")
    p.add_argument("--m3", type=float, default=1.0, help="tertiary, Msun")
    p.add_argument("--a_in_au", type=float, default=4.0)
    p.add_argument("--a_out_au", type=float, default=100.0)
    p.add_argument("--e_in", type=float, default=0.5, help="inner eccentricity")
    p.add_argument("--e_out", type=float, default=0.3, help="outer eccentricity")
    p.add_argument("--i_mut", type=float, default=25.0,
                   help="mutual inclination, deg; must avoid the Kozai window")
    p.add_argument("--seed", type=int, default=20250825, help="orientation seed")
    p.add_argument("--boxsize", type=float, default=1.0, help="pc")
    p.add_argument("--out", default="triple_ics.hdf5")
    a = p.parse_args()
    make_triple_ics(a.m1, a.m2, a.m3, a.a_in_au, a.a_out_au, a.e_in, a.e_out, a.i_mut,
                    a.seed, a.boxsize, a.out)
