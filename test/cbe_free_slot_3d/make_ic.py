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
