#!/usr/bin/env python
"""IC for cbe_harmonic_1d — 1D hot Gaussian in a fixed harmonic potential.

A single hot velocity basis per Type=1 particle (bulk velocity zero, per-basis
dispersion sigma_v). Particles are placed at equal-mass quantiles of a Gaussian
spatial density of width sigma_x, centered at the box midpoint. With the
external harmonic potential a = -Omega^2 (x - x_c) (Omega=1) and sigma_x =
sigma_v / Omega, this is the EXACT stationary collisionless equilibrium
f ~ exp[-(v^2 + Omega^2 (x-x_c)^2)/(2 sigma_v^2)] — it should sit still forever.

Set xwidth_factor != 1 to seed a width mismatch (sigma_x = xwidth_factor *
sigma_v/Omega): the analytic response is an UNDAMPED width oscillation at 2*Omega
(the "breathing" test). Any numerical damping/collapse shows up there.

Usage: python make_ic.py [N [sigma_v [xwidth_factor]]]
  defaults: N=256, sigma_v=1.0, xwidth_factor=1.0 (stationary)
Moments are stored in the RELATIVE-FRAME convention (same as cbe_two_stream):
  basis_p_stored = m*(v_phys - P.Vel);  T_rel_xx = m*(v_rel_x^2 + sigma_v^2).
"""
import os
import sys
import numpy as np
import h5py
from scipy.special import erfinv

HERE = os.path.dirname(os.path.abspath(__file__))
DIM = 1
OMEGA = 1.0          # must match GravAccel_Harmonic1D in analytic_gravity.h
BOX = 12.0           # ~ +/-6 sigma_x about the center; tails ~ e^-18 at edges
XC = 0.5 * BOX       # equilibrium center = box midpoint


# --- relative-frame Vlasov-moment row (1D + SECONDMOMENT => NMOMENTS=3) -------
def _basis_moment_row_1d(mass, v_rel_x, sigma_v):
    s2 = sigma_v * sigma_v
    return np.array([mass, mass * v_rel_x, mass * (v_rel_x * v_rel_x + s2)])


def write_ic(fname, x, masses, vmoments, box_size):
    N = x.shape[0]
    pos = np.zeros((N, 3)); pos[:, 0] = x
    vel = np.zeros((N, 3))                       # bulk velocity zero everywhere
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
        g.create_dataset("Coordinates", data=pos.astype(np.float64))
        g.create_dataset("Velocities",  data=vel.astype(np.float64))
        g.create_dataset("Masses",      data=masses.astype(np.float64))
        g.create_dataset("ParticleIDs", data=ids)
        g.create_dataset("VlasovMoments", data=vmoments.astype(np.float64))


def main():
    N       = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    sigma_v = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    xwidth_factor = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    nbasis  = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    mode    = sys.argv[5] if len(sys.argv) > 5 else ""   # "twin" => 2 near-identical half-mass bases
    sigma_x = xwidth_factor * sigma_v / OMEGA

    # equal-mass quantile placement of a Gaussian: x_i = XC + sigma_x * Phi^-1(p_i)
    p = (np.arange(N) + 0.5) / N
    x = XC + sigma_x * np.sqrt(2.0) * erfinv(2.0 * p - 1.0)

    # split each particle's hot Maxwellian into nbasis bases: velocities at the
    # equal-mass quantiles z_k of N(0,1) scaled by sigma_v, residual variance
    # carried as a common sub-dispersion sigma_sub so the bases reconstruct the
    # full second moment exactly. K=1 -> z=0, sigma_sub=sigma_v (single basis).
    pk = (np.arange(nbasis) + 0.5) / nbasis
    zk = np.sqrt(2.0) * erfinv(2.0 * pk - 1.0)          # symmetric => sum=0 => V=0
    vk = sigma_v * zk
    sigma_sub = sigma_v * np.sqrt(max(0.0, 1.0 - np.mean(zk * zk)))

    # "twin" mode (Phil): instead of the quantile split (two SEPARATED half-mass
    # bases), make two VERY SIMILAR half-mass bases — same near-zero velocity
    # (tiny eps split so they are distinct but degenerate) each carrying the FULL
    # sigma_v. Tests whether a redundant well-represented second basis still trips
    # the free-slot/pairing pathology. Requires nbasis==2.
    if mode == "twin":
        assert nbasis == 2, "twin mode requires nbasis=2"
        eps = 1.0e-3
        vk = sigma_v * np.array([-eps, +eps])
        sigma_sub = sigma_v                              # each basis = full-width half-mass copy

    m_cell = 1.0 / N                               # equal mass, Mtot = 1
    masses = np.full(N, m_cell)
    vmoments = np.zeros((N, 3 * nbasis))           # NBASIS=nbasis, NMOMENTS=3

    if mode == "empty":
        # The intended "1 massive + 1 empty basis" multibasis IC (canonical
        # OPEN_cbe_method_CANONICAL S.O.4b). basis0 = the normal single-basis
        # hot Gaussian (mass m_cell, v_rel=0, sigma_v). basis1 = a FLOOR-MASS,
        # COLD slot, given a small velocity OPPOSITE to its imminent infall
        # (the populated basis has zero mean vel, so the sign is set by which
        # side of x_c the cell sits on: at x>x_c infall is toward -x, so the
        # empty slot gets +v_eps; v_empty = v_eps*sign(x-x_c)). Documented
        # parameter choices (to confirm w/ Phil+codex): floor_frac below the
        # free-slot eps_rho gate so the empty basis as a SOURCE is floor-gated.
        assert nbasis == 2, "empty mode requires nbasis=2"
        floor_frac = 1.0e-10                              # empty-basis mass = floor_frac*m_cell
        sigma_empty = 0.05 * sigma_v                      # cold
        v_eps = 0.1 * sigma_v                             # small
        m_big = m_cell                                    # basis0 = exact 1-basis Gaussian
        m_emp = floor_frac * m_cell
        masses = np.full(N, m_big + m_emp)               # P.Mass = sum of basis masses
        for i in range(N):
            v_empty = v_eps * np.sign(x[i] - XC)
            vmoments[i, 0:3] = _basis_moment_row_1d(m_big, 0.0, sigma_v)
            vmoments[i, 3:6] = _basis_moment_row_1d(m_emp, v_empty, sigma_empty)
    else:
        for i in range(N):
            for b in range(nbasis):
                vmoments[i, 3 * b:3 * b + 3] = _basis_moment_row_1d(m_cell / nbasis, vk[b], sigma_sub)

    btag = "" if nbasis == 1 else f"_nb{nbasis}"
    if mode == "twin": btag = "_twin"
    if mode == "empty": btag = "_empty"
    xtag = "" if abs(xwidth_factor - 1.0) < 1e-12 else "_breathe"
    out = os.path.join(HERE, f"cbe_harmonic_1d{xtag}{btag}_ics.hdf5")
    write_ic(out, x, masses, vmoments, BOX)
    print(f"Wrote {out}: N={N}, sigma_v={sigma_v}, sigma_x={sigma_x} "
          f"(xwidth_factor={xwidth_factor}), nbasis={nbasis}, vk={np.round(vk,3)}, "
          f"sigma_sub={sigma_sub:.4f}, Omega={OMEGA}, BoxSize={BOX}, "
          f"x in [{x.min():.4f},{x.max():.4f}], Mtot={masses.sum():.6g}")


if __name__ == "__main__":
    main()
