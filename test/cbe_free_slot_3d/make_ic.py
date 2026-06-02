#!/usr/bin/env python
"""IC for cbe_free_slot_3d — 3D version of the free-slot injection test.

Same physics as cbe_free_slot_1d (4 streams +1/0/-1/-2 along x, a Gaussian
+2 perturbation at x=0.5) but on a 3D slab: long in x, thin in y,z, so the
+2 stream propagates along x while the transverse directions carry only a
few cells. All velocities lie along x (p_y=p_z=0 per basis).

Box: BoxSize sets the TRANSVERSE (y,z) extent; BOX_LONG_X stretches x.
With Config BOX_LONG_X=16 and BoxSize=0.0625 the box is [1.0 x 0.0625 x
0.0625] and the lattice spacing is uniform (1/64 on every axis).

Usage: python make_ic.py [Nx [Ny [sigma [perturb_amp]]]]
  defaults: Nx=64, Ny=Nz=4, sigma=0.05, perturb_amp=0.5
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from cbe_vlasov_common import write_cbe_ic

HERE = os.path.dirname(os.path.abspath(__file__))
DIM = 3

# Must match Config.sh BOX_LONG_X and the params BoxSize.
BOX_LONG_X = 16
BOXSIZE    = 0.0625                 # transverse (y,z) extent
LX = BOXSIZE * BOX_LONG_X           # = 1.0
LY = LZ = BOXSIZE                   # = 0.0625

Nx          = int(sys.argv[1]) if len(sys.argv) > 1 else 64
Ny          = int(sys.argv[2]) if len(sys.argv) > 2 else 4
Nz          = Ny
sigma       = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05
perturb_amp = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5

V_BKG    = np.array([1.0, 0.0, -1.0, -2.0])
FRAC_BKG = np.array([0.49, 0.02, 0.49, 1e-8])
FRAC_BKG = FRAC_BKG / FRAC_BKG.sum()


def main():
    xs = (np.arange(Nx) + 0.5) * LX / Nx
    ys = (np.arange(Ny) + 0.5) * LY / Ny
    zs = (np.arange(Nz) + 0.5) * LZ / Nz
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    pos = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    N = pos.shape[0]
    m_cell = 1.0 / N

    # Perturbation localized in x only (slab); periodic min-image along x.
    xp = pos[:, 0]
    dxc = np.abs(xp - 0.5 * LX)
    dxc = np.minimum(dxc, LX - dxc)
    pert = perturb_amp * np.exp(-(dxc / sigma) ** 2)

    per_particle_bases = []
    for a in range(N):
        v_arr = V_BKG.copy()
        if pert[a] > 1e-6:
            v_arr[3] = 2.0
            mf = np.empty(4)
            mf[3] = pert[a]
            mf[:3] = (1.0 - pert[a]) * FRAC_BKG[:3] / FRAC_BKG[:3].sum()
        else:
            mf = FRAC_BKG.copy()
        m_arr = m_cell * mf
        # velocities along x only -> [vx, 0, 0]
        per_particle_bases.append(
            [(float(m_arr[k]), [float(v_arr[k]), 0.0, 0.0]) for k in range(4)])

    write_cbe_ic(os.path.join(HERE, "cbe_free_slot_3d_ics.hdf5"),
                 pos, per_particle_bases, DIM, BOXSIZE,
                 box_long=(BOX_LONG_X, 1.0, 1.0))


if __name__ == "__main__":
    main()
