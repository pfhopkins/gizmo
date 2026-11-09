#!/usr/bin/env python3
"""ICs for test/ketju_snapshot_midstep: an isolated 1.2 + 0.8 Msun binary (a = 50 AU, e = 0.6, P ~ 250 yr) and a
0.5 Msun star at 1.45 pc (kpc, km/s, 1e10 Msun; centred on the origin)."""
import sys
import numpy as np
sys.path.insert(0, "../ketju_star_collision")
from make_ketju_star_collision_ics import write, AU, G

def kepler_state(m1, m2, a, e, f):
    mu = G * (m1 + m2); p = a * (1 - e * e); r = p / (1 + e * np.cos(f))
    x = r * np.array([np.cos(f), np.sin(f), 0]); v = np.sqrt(mu / p) * np.array([-np.sin(f), e + np.cos(f), 0])
    M = m1 + m2; return [-m2 / M * x, m1 / M * x], [-m2 / M * v, m1 / M * v]

M1, M2, A_AU, E, F0 = 1.2, 0.8, 50.0, 0.6, 0.3

def main():
    xs, vs = kepler_state(M1, M2, A_AU * AU, E, F0)
    write("ketju_snapshot_midstep_ics.hdf5", [M1, M2, 0.5], [xx / AU for xx in xs] + [np.array([3e5, 0, 0])], list(vs) + [np.zeros(3)])

if __name__ == "__main__":
    main()
