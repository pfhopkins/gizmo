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
import numpy as np
import h5py


# ----------------------------------------------------------------------------
# Inlined CBE Vlasov-moment IC builder (RELATIVE-FRAME storage convention).
#   basis_p_stored[α] = m_α * (v_phys[α] − P.Vel)
#   v_phys[α]         = basis_p_stored[α] / m_α + P.Vel
# Per-basis momenta are stored relative to the mesh-generating-point velocity
# P.Vel (mass-weighted-mean basis velocity).  Σ_α basis_p_stored = 0 by IC
# closure; do_cbe_initialization() enforces the same condition at runtime.
# ----------------------------------------------------------------------------
def _n_stress_for_dim(dim): return dim * (dim + 1) // 2


def _n_moments_for_dim(dim, secondmoment=False):
    return dim + 1 + (_n_stress_for_dim(dim) if secondmoment else 0)


def _stress_slot(a, b, dim):
    if a > b: a, b = b, a
    if a == b: return 1 + dim + a
    off = a * (2 * dim - a - 1) // 2 + (b - a - 1)
    return 1 + 2 * dim + off


def _basis_moment_row(mass, vel_rel_xyz, dim, sigma=0.0):
    secondmoment = (sigma > 0.0)
    row = np.zeros(_n_moments_for_dim(dim, secondmoment=secondmoment))
    row[0] = mass
    for k in range(dim):
        row[1 + k] = mass * vel_rel_xyz[k]
    if secondmoment:
        s2 = sigma * sigma
        for a in range(dim):
            row[_stress_slot(a, a, dim)] = mass * (vel_rel_xyz[a]**2 + s2)
            for b in range(a + 1, dim):
                row[_stress_slot(a, b, dim)] = mass * vel_rel_xyz[a] * vel_rel_xyz[b]
    return row


def _bulk_velocity(per_particle_bases):
    """Mass-weighted-mean velocity per particle (N,3)."""
    N = len(per_particle_bases)
    vbulk = np.zeros((N, 3))
    for i, bases in enumerate(per_particle_bases):
        mtot = sum(m for (m, v) in bases)
        if mtot > 0:
            p = np.zeros(3)
            for (m, v) in bases:
                p += m * np.asarray(v, dtype=float)
            vbulk[i] = p / mtot
    return vbulk


def _build_vlasov_moments(per_particle_bases, dim, vel_bulk_per_particle, sigma=0.0):
    """(N, NBASIS*NMOMENTS) basis-major flat; relative-frame momentum slots."""
    N = len(per_particle_bases)
    nbasis = len(per_particle_bases[0])
    nm = _n_moments_for_dim(dim, secondmoment=(sigma > 0.0))
    out = np.zeros((N, nbasis * nm))
    for i, bases in enumerate(per_particle_bases):
        vbulk_i = np.asarray(vel_bulk_per_particle[i])
        for b, (m, v) in enumerate(bases):
            v_rel = np.asarray(v, dtype=float) - vbulk_i
            out[i, nm * b:nm * (b + 1)] = _basis_moment_row(m, v_rel, dim, sigma=sigma)
    return out


def write_cbe_ic(fname, pos, per_particle_bases, dim, box_size,
                 box_long=(1.0, 1.0, 1.0), sigma=0.0):
    """Write a CBE Type=1 IC HDF5 file.  For 1D problems pass pos with y=z=0;
    GIZMO wraps only the active axes."""
    pos = np.asarray(pos, dtype=float)
    N = pos.shape[0]
    masses = np.array([sum(m for (m, v) in bases) for bases in per_particle_bases])
    vel = _bulk_velocity(per_particle_bases)
    vmoments = _build_vlasov_moments(per_particle_bases, dim, vel, sigma=sigma)
    ids = np.arange(1, N + 1, dtype=np.uint32)
    with h5py.File(fname, "w") as F:
        h = F.create_group("Header")
        h.attrs["NumPart_ThisFile"]       = [0, N, 0, 0, 0, 0]
        h.attrs["NumPart_Total"]          = [0, N, 0, 0, 0, 0]
        h.attrs["NumPart_Total_HighWord"] = [0, 0, 0, 0, 0, 0]
        h.attrs["MassTable"]              = [0.0, 0.0, 0, 0, 0, 0]
        h.attrs["BoxSize"]                = box_size
        h.attrs["Time"]                   = 0.0
        h.attrs["Redshift"]               = 0.0
        h.attrs["NumFilesPerSnapshot"]    = 1
        h.attrs["Omega0"]                 = 0.0
        h.attrs["OmegaLambda"]            = 0.0
        h.attrs["HubbleParam"]            = 1.0
        h.attrs["Flag_Sfr"]               = 0
        h.attrs["Flag_Cooling"]           = 0
        h.attrs["Flag_StellarAge"]        = 0
        h.attrs["Flag_Metals"]            = 0
        h.attrs["Flag_Feedback"]          = 0
        h.attrs["Flag_DoublePrecision"]   = 1
        g = F.create_group("PartType1")
        g.create_dataset("Coordinates",   data=pos.astype(np.float64))
        g.create_dataset("Velocities",    data=vel.astype(np.float64))
        g.create_dataset("Masses",        data=masses.astype(np.float64))
        g.create_dataset("ParticleIDs",   data=ids)
        g.create_dataset("VlasovMoments", data=vmoments.astype(np.float64))
    nbasis = len(per_particle_bases[0])
    nm = _n_moments_for_dim(dim, secondmoment=(sigma > 0.0))
    print(f"Wrote {fname}: N={N} Type=1, dim={dim}, NBASIS={nbasis}, "
          f"NMOMENTS={nm}{' (SECONDMOMENT, sigma=%g)' % sigma if sigma > 0 else ''}, "
          f"VlasovMoments shape ({N},{nbasis * nm}), BoxSize={box_size}, "
          f"box_long={box_long}, Mtot={masses.sum():.6g}")
    return fname

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
