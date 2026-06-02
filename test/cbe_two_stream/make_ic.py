#!/usr/bin/env python
"""IC for cbe_two_stream — 1D counter-streaming CBE test.

Mirrors python_harness/cbe1d init_two_stream (rest frame): N equal-mass
Type=1 particles on a uniform 1D periodic mesh, each carrying a 2-basis
internal velocity distribution (+v_stream, -v_stream) with equal mass
split. Bulk velocity is zero, so each stream should advect at +/-v_stream
while total density stays uniform.

Usage: python make_ic.py [N [v_stream]]
  defaults: N=64, v_stream=1.0
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from cbe_vlasov_common import write_cbe_ic

HERE = os.path.dirname(os.path.abspath(__file__))
DIM = 1
BOX = 1.0

N        = int(sys.argv[1]) if len(sys.argv) > 1 else 64
v_stream = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0


def main():
    x = (np.arange(N) + 0.5) * BOX / N
    pos = np.zeros((N, 3)); pos[:, 0] = x          # y=z=0 for 1D
    m_cell = 1.0 / N
    per_particle_bases = []
    for _ in range(N):
        per_particle_bases.append([
            (0.5 * m_cell, [+v_stream, 0.0, 0.0]),
            (0.5 * m_cell, [-v_stream, 0.0, 0.0]),
        ])
    write_cbe_ic(os.path.join(HERE, "cbe_two_stream_ics.hdf5"),
                 pos, per_particle_bases, DIM, BOX)


if __name__ == "__main__":
    main()
