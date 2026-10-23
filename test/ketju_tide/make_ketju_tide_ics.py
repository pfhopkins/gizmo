#!/usr/bin/env python3
"""ICs for test/ketju_tide (kpc, km/s, 1e10 Msun): a 1 + 1 Msun pair (a = 200 AU, e = 0.5, P ~ 2000 yr) in a KETJU region
and a 20 Msun star on a circular orbit 5000 AU from it (P_out ~ 0.07 Myr), whose tide reaches the pair. Centred on the
CoM at the origin."""
import os
import sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../ketju_star_collision"))
from make_ketju_star_collision_ics import write, AU, G

def kepler_state(m1, m2, a, e, f):
    mu = G * (m1 + m2); p = a * (1 - e * e); r = p / (1 + e * np.cos(f))
    x = r * np.array([np.cos(f), np.sin(f), 0]); v = np.sqrt(mu / p) * np.array([-np.sin(f), e + np.cos(f), 0])
    M = m1 + m2; return [-m2 / M * x, m1 / M * x], [-m2 / M * v, m1 / M * v]

def state():
    m = np.array([1.0, 1.0, 20.0]); xs, vs = kepler_state(1, 1, 200 * AU, 0.5, 0.7)
    # tilt the inner orbit out of the outer plane so the tide changes e and a
    c, s = np.cos(0.6), np.sin(0.6); Rx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    xs = [Rx @ x for x in xs]; vs = [Rx @ v for v in vs]
    r3 = 5000 * AU; v3 = np.sqrt(G * m.sum() / r3)
    xo, vo = np.array([r3, 0, 0]), np.array([0, v3, 0]); M = m.sum()
    x = np.array([xx - m[2] / M * xo for xx in xs] + [2.0 / M * xo]); v = np.array([vv - m[2] / M * vo for vv in vs] + [2.0 / M * vo])
    return m, x, v

def main():
    m, x, v = state(); write("ketju_tide_ics.hdf5", m, x / AU, v)

if __name__ == "__main__":
    main()
