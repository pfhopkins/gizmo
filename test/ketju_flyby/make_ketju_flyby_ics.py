#!/usr/bin/env python3
"""ICs for test/ketju_flyby (kpc, km/s, 1e10 Msun): a bound 1 + 1 Msun pair (a = 200 AU, e = 0.5) and a 1 Msun star
20000 AU away moving at 30 km/s with impact parameter b (1500 AU: outside the pair's region; 600 AU: through it).
Writes ketju_flyby_ics.hdf5 (b = 1500) and ketju_flyby_b600_ics.hdf5. Centred on the CoM at the origin."""
import sys
import numpy as np
sys.path.insert(0, "../ketju_star_collision")
from make_ketju_star_collision_ics import write, AU, G

def kepler_state(m1, m2, a, e, f):
    mu = G * (m1 + m2); p = a * (1 - e * e); r = p / (1 + e * np.cos(f))
    x = r * np.array([np.cos(f), np.sin(f), 0]); v = np.sqrt(mu / p) * np.array([-np.sin(f), e + np.cos(f), 0])
    M = m1 + m2; return [-m2 / M * x, m1 / M * x], [-m2 / M * v, m1 / M * v]

def state(b_au):
    m = np.array([1.0, 1.0, 1.0]); xs, vs = kepler_state(1, 1, 200 * AU, 0.5, 0.7)
    x = np.array(list(xs) + [np.array([-20000 * AU, b_au * AU, 300 * AU])]); v = np.array(list(vs) + [np.array([30.0, 0, 0])])
    x -= (m[:, None] * x).sum(0) / m.sum(); v -= (m[:, None] * v).sum(0) / m.sum()
    return m, x, v

def main():
    for b, name in ((1500.0, "ketju_flyby_ics.hdf5"), (600.0, "ketju_flyby_b600_ics.hdf5")):
        m, x, v = state(b); write(name, m, x / AU, v)

if __name__ == "__main__":
    main()
