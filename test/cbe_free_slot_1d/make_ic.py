#!/usr/bin/env python
"""IC for cbe_free_slot_1d — 1D free-slot injection test.

Mirrors python_harness/tests/test_free_slot make_ic. Background 4-basis
distribution v = (+1, 0, -1, -2) with mass fractions ~(0.49, 0.02, 0.49,
~0). A Gaussian-localized perturbation at x=0.5 flips basis-3's velocity
to v=+2 (mass fraction up to perturb_amp), redistributing the other three.
Neighbours have NO v=+2 slot, so propagating the +2 stream forces the
free-slot pairing fallback. Type=1, 1D periodic, no gravity.

Usage: python make_ic.py [N [sigma [perturb_amp]]]
  defaults: N=48, sigma=0.05, perturb_amp=0.5
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from cbe_vlasov_common import write_cbe_ic

HERE = os.path.dirname(os.path.abspath(__file__))
DIM = 1
BOX = 1.0

N           = int(sys.argv[1]) if len(sys.argv) > 1 else 48
sigma       = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
perturb_amp = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5

V_BKG    = np.array([1.0, 0.0, -1.0, -2.0])
FRAC_BKG = np.array([0.49, 0.02, 0.49, 1e-8])
FRAC_BKG = FRAC_BKG / FRAC_BKG.sum()


def free_slot_bases_1d(x, box, m_cell):
    """Per-particle list of [(mass, [vx,0,0]) x4], with the localized v=+2
    perturbation at x=0.5 (same construction as the harness)."""
    dxc = np.abs(x - 0.5 * box)
    dxc = np.minimum(dxc, box - dxc)          # periodic min-image to centre
    pert = perturb_amp * np.exp(-(dxc / sigma) ** 2)
    out = []
    for a in range(len(x)):
        v_arr = V_BKG.copy()
        if pert[a] > 1e-6:
            v_arr[3] = 2.0
            mf = np.empty(4)
            mf[3] = pert[a]
            mf[:3] = (1.0 - pert[a]) * FRAC_BKG[:3] / FRAC_BKG[:3].sum()
        else:
            mf = FRAC_BKG.copy()
        m_arr = m_cell * mf
        out.append([(float(m_arr[k]), [float(v_arr[k]), 0.0, 0.0]) for k in range(4)])
    return out


def main():
    x = (np.arange(N) + 0.5) * BOX / N
    pos = np.zeros((N, 3)); pos[:, 0] = x
    m_cell = 1.0 / N
    per_particle_bases = free_slot_bases_1d(x, BOX, m_cell)
    write_cbe_ic(os.path.join(HERE, "cbe_free_slot_1d_ics.hdf5"),
                 pos, per_particle_bases, DIM, BOX)


if __name__ == "__main__":
    main()
