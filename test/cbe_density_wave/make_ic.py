#!/usr/bin/env python
"""IC for cbe_density_wave — 1D density-modulated counter-streaming test.

Mirrors python_harness/cbe1d init_two_stream_density_wave (rest frame):
equal-mass Type=1 particles whose POSITIONS are inverse-CDF sampled from
rho(x) = 1 + eps cos(2 pi x / L), each carrying the same 2-basis (+v, -v)
internal distribution. The total density advects as a wave at the stream
speed while each per-stream density tracks +/-v_stream.

Usage: python make_ic.py [N [eps [v_stream [sigma]]]]
  defaults: N=64, eps=0.2, v_stream=1.0, sigma=0.0
  sigma>0 selects SECONDMOMENT layout (NMOMENTS=3 in 1D) with isotropic
  relative-frame per-basis dispersion T_rel_aa = m*(v_rel_a^2 + sigma^2).
  The collisionless warm density-wave decays via Landau / phase-mixing:
      rho_+/-(x,t) = 0.5*[1 + eps*exp(-0.5*(k*sigma*t)^2)*cos(k*(x -/+ v0*t))]
  Output filename changes to cbe_density_wave_warm_ics.hdf5 when sigma>0
  so the cold-regression IC artifact stays untouched.
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
eps      = float(sys.argv[2]) if len(sys.argv) > 2 else 0.2
v_stream = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
sigma    = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0


def inverse_cdf_positions(N, eps, box):
    """Newton-iterate x so the cumulative mass of rho=1+eps cos(2 pi x/L)
    is uniform in particle index — same scheme as the harness."""
    targets = (np.arange(N) + 0.5) / N
    x = box * targets.copy()
    twopi_L = 2.0 * np.pi / box
    for _ in range(100):
        cdf  = (x + eps * box / (2 * np.pi) * np.sin(twopi_L * x)) / box
        dcdf = (1.0 + eps * np.cos(twopi_L * x)) / box
        dx = (cdf - targets) / dcdf
        x -= dx
        if np.max(np.abs(dx)) < 1e-13:
            break
    return np.sort(x % box)


def main():
    x = inverse_cdf_positions(N, eps, BOX)
    pos = np.zeros((N, 3)); pos[:, 0] = x
    m_cell = 1.0 / N
    per_particle_bases = []
    for _ in range(N):
        per_particle_bases.append([
            (0.5 * m_cell, [+v_stream, 0.0, 0.0]),
            (0.5 * m_cell, [-v_stream, 0.0, 0.0]),
        ])
    if sigma > 0.0:
        out = os.path.join(HERE, "cbe_density_wave_warm_ics.hdf5")
    else:
        out = os.path.join(HERE, "cbe_density_wave_ics.hdf5")
    write_cbe_ic(out, pos, per_particle_bases, DIM, BOX, sigma=sigma)


if __name__ == "__main__":
    main()
