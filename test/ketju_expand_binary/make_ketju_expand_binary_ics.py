#!/usr/bin/env python3
"""ICs for test/ketju_expand_binary: a hard 1.2 + 0.8 Msun binary (a = 2 AU, e = 0.6, P ~ 2 yr) and a 1 Msun star 900 AU away on
a circular orbit about it, all in one KETJU region (radius 1000 AU), plus a 0.5 Msun star at 1.45 pc (kpc, km/s, 1e10 Msun)."""
import sys
import numpy as np
sys.path.insert(0, "../ketju_star_collision")
from make_ketju_star_collision_ics import write, AU, G
sys.path.insert(0, "../ketju_snapshot_midstep")
from make_ketju_snapshot_midstep_ics import kepler_state

M1, M2, M3, A_AU, E, F0, R3 = 1.2, 0.8, 1.0, 2.0, 0.6, 0.3, 900.0


def main():
    xs, vs = kepler_state(M1, M2, A_AU * AU, E, F0)
    M = M1 + M2 + M3; v3 = np.sqrt(G * M / (R3 * AU))
    xo, vo = np.array([R3, 0, 0]), np.array([0, v3, 0])
    x = [xx / AU - M3 / M * xo for xx in xs] + [(M1 + M2) / M * xo, np.array([3e5, 0, 0])]
    v = [vv - M3 / M * vo for vv in vs] + [(M1 + M2) / M * vo, np.zeros(3)]
    write("ketju_expand_binary_ics.hdf5", [M1, M2, M3, 0.5], x, v)


if __name__ == "__main__":
    main()
