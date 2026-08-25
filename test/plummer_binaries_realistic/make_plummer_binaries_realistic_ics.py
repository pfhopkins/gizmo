"""Plummer cluster with a REALISTIC binary population (Type 5 sinks).

test/plummer_binaries puts every star in an identical 1000 AU equal-mass circular binary. That
is a clean control, but it exercises exactly one point in the parameter space the sink
integrator has to survive. Here the population is drawn from the observed distributions
instead, so a single run spans several decades in binary hardness, mass ratio and eccentricity
simultaneously -- which is what a production STARFORGE cluster actually contains.

Sampled:
  IMF               Kroupa (2001) broken power law, dN/dm ~ m^-1.3 (0.08-0.5) and m^-2.3
                    (0.5-m_max). Sampled by exact inverse-CDF, not rejection, so the star count
                    is deterministic for a given seed.
  binary fraction   f(M1) rising with primary mass, interpolated in log M from Duchene & Kraus
                    (2013): ~22% at 0.1 Msun to ~80% above 16 Msun. This is the piece the
                    equal-mass test cannot represent at all -- there, f = 1 by construction.
  mass ratio        q = m2/m1 uniform on [q_min, 1] (Duchene & Kraus find gamma ~ 0 for
                    solar-type; the low-q cut is observational incompleteness, and here also
                    keeps the secondary above the IMF floor).
  period            log10(P/days) ~ Normal(5.03, 2.28), Raghavan et al. (2010) -- equivalently
                    Duquennoy & Mayor (1991). TRUNCATED, see below.
  eccentricity      thermal, f(e) = 2e, for P > P_CIRC; circular below it, which stands in for
                    tidal circularisation of the short-period tail.

TRUNCATION, and why it is not optional. The log-normal period distribution has a substantial
tail below one day: sampled raw, a few systems land at a << 1 AU and their orbital periods set
the global timestep, so the run cost is dominated by a handful of binaries that say nothing the
wider ones do not. The semi-major axis is therefore clipped to [A_MIN_AU, A_MAX_AU], and the
PERICENTRE a(1-e) is clipped separately -- eccentricity, not a, is what actually sets the
smallest lengthscale the integrator sees. Both clips are reported in the summary so the
truncation is visible rather than silent. The surviving range still spans ~3 decades in a,
against the single value the equal-mass test uses.

Wide binaries are left alone: they are cheap, and a cluster-embedded wide binary is precisely
the configuration where perturbation by neighbours competes with the internal orbit.

Units are pc - km/s - Msun; G comes from gizmo.units (GIZMO's constants.h), not astropy.
"""
import argparse

import astropy.units as u
import h5py
import numpy as np
from scipy.optimize import brentq

import os as _os
import sys as _sys
# Run standalone (the test invokes this as a subprocess from the test directory), so python_src
# is not on the path the way it is under pytest.
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  "..", "..", "python_src"))
from gizmo.units import G_CODE, AU_PER_PC  # noqa: E402

# G comes from gizmo.units, which mirrors GIZMO's constants.h -- NOT from astropy. An IC built
# with a different G than the code integrates with is not the orbit it claims to be: Kepler's
# third law below converts a sampled PERIOD into a semi-major axis, so a 4.8e-5 error in G is a
# 3.2e-5 error in every a, systematic across the population.
#
# The code-time conversion is astropy's, and that is the right call for it: it has no dynamical
# role, only setting which periods get sampled. Writing it by hand as 365.25 * 977.79222 dropped
# a factor of 1000 here, inflating every a by 1000^(2/3) = 100x and pinning the whole population
# against the wide-end clip -- an error that yields a plausible-looking IC rather than a crash.
DAY_PER_CODE_TIME = (1 * (u.pc / (u.km / u.s))).to(u.day).value      # 1 code time = 977792 yr

PLUMMER_HALF_MASS_RADIUS = 1.305   # in units of a (analytic, GM/a normalization)

# --- Kroupa (2001) IMF ---
IMF_BREAK = 0.5
IMF_SLOPE_LO, IMF_SLOPE_HI = -1.3, -2.3

# --- Duchene & Kraus (2013) multiplicity fraction vs primary mass ---
DK13_LOGM = np.log10([0.10, 0.30, 0.70, 1.25, 3.0, 8.0, 16.0])
DK13_FBIN = np.array([0.22, 0.26, 0.44, 0.50, 0.60, 0.70, 0.80])

# --- Raghavan et al. (2010) period distribution ---
LOGP_MEAN, LOGP_SIGMA = 5.03, 2.28   # log10(P / days)
P_CIRC_DAYS = 10.0                   # below this, treat as tidally circularised


def velocity_cdf(R, target):
    """Plummer isotropic velocity CDF at fixed r, in units of v_escape."""
    return (
        2 * (
            R * np.sqrt(1 - R**2) * (-105 + 1210 * R**2 - 2104 * R**4 + 1488 * R**6 - 384 * R**8)
            + 105 * np.arcsin(R)
        )
    ) / (105.0 * np.pi) - target


def _orthonormal_in_plane(n_hat, rng):
    """(e1, e2) orthonormal and spanning the plane perpendicular to each n_hat (N,3)."""
    r = rng.normal(size=n_hat.shape)
    e1 = r - np.sum(r * n_hat, axis=1, keepdims=True) * n_hat
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    return e1, np.cross(n_hat, e1)


def sample_kroupa(n, m_min, m_max, rng):
    """Exact inverse-CDF sample of the Kroupa (2001) two-segment power law."""
    def _seg_int(a, b, slope):                      # integral of m^slope
        p = slope + 1.0
        return (b**p - a**p) / p
    lo_hi = min(max(IMF_BREAK, m_min), m_max)
    w_lo = _seg_int(m_min, lo_hi, IMF_SLOPE_LO) if m_min < IMF_BREAK else 0.0
    # continuity at the break: the high segment is scaled so the pdf is continuous
    k_hi = IMF_BREAK ** (IMF_SLOPE_LO - IMF_SLOPE_HI)
    w_hi = k_hi * _seg_int(lo_hi, m_max, IMF_SLOPE_HI) if m_max > IMF_BREAK else 0.0
    u = rng.random(n) * (w_lo + w_hi)
    m = np.empty(n)
    in_lo = u < w_lo
    p = IMF_SLOPE_LO + 1.0
    m[in_lo] = (m_min**p + u[in_lo] * p) ** (1.0 / p)
    p = IMF_SLOPE_HI + 1.0
    m[~in_lo] = (lo_hi**p + (u[~in_lo] - w_lo) / k_hi * p) ** (1.0 / p)
    return m


def binary_fraction(m1):
    """Multiplicity fraction vs primary mass, log-interpolated from Duchene & Kraus (2013)."""
    return np.clip(np.interp(np.log10(m1), DK13_LOGM, DK13_FBIN), 0.0, 1.0)


def make_realistic_ics(n_systems, a_cluster, boxsize, seed, outfile,
                       m_min=0.08, m_max=30.0, q_min=0.1,
                       a_min_au=100.0, a_max_au=5.0e3, r_max=None):
    rng = np.random.default_rng(seed)
    if r_max is None:
        r_max = 100.0 * PLUMMER_HALF_MASS_RADIUS * a_cluster

    # ---------------- the stellar population ----------------
    m1 = sample_kroupa(n_systems, m_min, m_max, rng)
    m1 = np.sort(m1)[::-1]                       # descending: IDs then run massive-first
    is_bin = rng.random(n_systems) < binary_fraction(m1)
    q = np.where(is_bin, q_min + (1.0 - q_min) * rng.random(n_systems), 0.0)
    m2 = np.where(is_bin, np.maximum(q * m1, m_min), 0.0)
    m_sys = m1 + m2

    # ---------------- orbits, with the truncation made explicit ----------------
    logP = rng.normal(LOGP_MEAN, LOGP_SIGMA, n_systems)          # log10(P / days)
    P_code = 10.0 ** logP / DAY_PER_CODE_TIME
    a_raw = (G_CODE * m_sys * (P_code / (2 * np.pi)) ** 2) ** (1.0 / 3.0)   # Kepler
    a_min, a_max = a_min_au / AU_PER_PC, a_max_au / AU_PER_PC
    a = np.clip(a_raw, a_min, a_max)
    n_clip_a = int(is_bin.sum() and ((a_raw < a_min) | (a_raw > a_max))[is_bin].sum())

    # eccentricity: thermal, circularised at short period
    ecc = np.sqrt(rng.random(n_systems))                          # f(e) = 2e
    ecc[10.0 ** logP < P_CIRC_DAYS] = 0.0

    # PERICENTRE is what sets the smallest lengthscale the integrator sees, so it -- not a --
    # carries the floor. Redraw the (period, e) PAIR for systems that violate it rather than
    # clipping e: clipping piles every offender up at one eccentricity, replacing the sampled
    # f(e) with an artificial spike. Redrawing gives the correctly TRUNCATED joint
    # distribution, which is a statement about what this test covers rather than a distortion
    # of the physics inside that coverage.
    peri_min = a_min
    n_redraw = 0
    for _ in range(64):                       # bounded; the accept region is large
        bad = is_bin & (a * (1.0 - ecc) < peri_min)
        if not bad.any():
            break
        nb_ = int(bad.sum())
        n_redraw += nb_
        lp = rng.normal(LOGP_MEAN, LOGP_SIGMA, nb_)
        Pc = 10.0 ** lp / DAY_PER_CODE_TIME
        a[bad] = np.clip((G_CODE * m_sys[bad] * (Pc / (2 * np.pi)) ** 2) ** (1.0 / 3.0),
                         a_min, a_max)
        e_ = np.sqrt(rng.random(nb_))
        e_[10.0 ** lp < P_CIRC_DAYS] = 0.0
        ecc[bad] = e_
    else:
        # never converged: fall back to circularising the stragglers so the IC is still valid
        bad = is_bin & (a * (1.0 - ecc) < peri_min)
        ecc[bad] = np.maximum(0.0, 1.0 - peri_min / a[bad])
    n_clip_e = n_redraw

    # ---------------- Plummer COM positions (truncated at r_max) ----------------
    u_max = (r_max / a_cluster) ** 3 / (1 + (r_max / a_cluster) ** 2) ** 1.5
    # Stratified (one sample per 1/N bin) to cut shot noise in the radial profile, then SHUFFLED.
    # Without the shuffle this is monotonic in index, and m1 is sorted descending above, so the
    # most massive system lands innermost and the cluster is fully mass-segregated by
    # construction -- measured Spearman(mass, radius) = -0.918. That makes it far more bound than
    # the Plummer velocity sampling below assumes, leaving it sub-virial (2KE/|PE| = 0.66) so it
    # collapses instead of holding equilibrium. The equal-mass sibling never showed this because
    # every mass is identical there.
    u = u_max * (np.arange(n_systems) + rng.random(n_systems)) / n_systems
    rng.shuffle(u)
    r_com = a_cluster * np.sqrt(u ** (2.0 / 3) * (1 + u ** (2.0 / 3) + u ** (4.0 / 3)) / (1 - u**2))
    phi_ang = rng.random(n_systems) * 2 * np.pi
    cos_th = 2.0 * rng.random(n_systems) - 1.0
    sin_th = np.sqrt(1.0 - cos_th**2)
    pos_com = np.c_[r_com * np.cos(phi_ang) * sin_th,
                    r_com * np.sin(phi_ang) * sin_th,
                    r_com * cos_th]

    # ---------------- Plummer COM velocities ----------------
    M_cluster = m_sys.sum()
    potential = -G_CODE * M_cluster / np.sqrt(r_com**2 + a_cluster**2)
    v_escape = np.sqrt(-2 * potential)
    Qs = np.array([brentq(velocity_cdf, 0, 1, args=(t,)) for t in rng.random(n_systems)])
    v_com_mag = v_escape * Qs
    phi_v = rng.random(n_systems) * 2 * np.pi
    cos_tv = 2.0 * rng.random(n_systems) - 1.0
    sin_tv = np.sqrt(1.0 - cos_tv**2)
    vel_com = np.c_[v_com_mag * np.cos(phi_v) * sin_tv,
                    v_com_mag * np.sin(phi_v) * sin_tv,
                    v_com_mag * cos_tv]

    # ---------------- internal orbits, started at APOCENTRE ----------------
    # Apocentre is the slowest point, so the first step is the largest of the orbit and every
    # subsequent pericentre forces the scheme to refine and coarsen again -- the behaviour under
    # test. Starting at pericentre would hand the integrator its easiest first step.
    phi_n = rng.random(n_systems) * 2 * np.pi
    cos_tn = 2.0 * rng.random(n_systems) - 1.0
    sin_tn = np.sqrt(1.0 - cos_tn**2)
    n_hat = np.c_[np.cos(phi_n) * sin_tn, np.sin(phi_n) * sin_tn, cos_tn]
    e1, e2 = _orthonormal_in_plane(n_hat, rng)

    r_apo = a * (1.0 + ecc)
    with np.errstate(divide="ignore", invalid="ignore"):
        v_apo = np.sqrt(G_CODE * m_sys / a * (1.0 - ecc) / (1.0 + ecc))
    v_apo = np.where(is_bin, v_apo, 0.0)
    r_rel = r_apo[:, None] * e1                    # separation along e1 at apocentre
    v_rel = v_apo[:, None] * e2                    # relative velocity perpendicular to it

    # split about each system's COM so sum(m*x) and sum(m*v) vanish system by system
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(is_bin, m2 / m_sys, 0.0)[:, None]
        f2 = np.where(is_bin, m1 / m_sys, 0.0)[:, None]

    n_stars = int(n_systems + is_bin.sum())
    pos = np.empty((n_stars, 3))
    vel = np.empty((n_stars, 3))
    mass = np.empty(n_stars)
    ids = np.empty(n_stars, dtype=np.uint32)
    k = 0
    prim_of = np.full(n_systems, -1, dtype=int)    # star index of each system's primary
    sec_of = np.full(n_systems, -1, dtype=int)     # ... and secondary, -1 if single
    for i in range(n_systems):
        prim_of[i] = k
        pos[k] = pos_com[i] - f1[i] * r_rel[i]
        vel[k] = vel_com[i] - f1[i] * v_rel[i]
        mass[k] = m1[i]; ids[k] = k + 1; k += 1
        if is_bin[i]:
            sec_of[i] = k
            pos[k] = pos_com[i] + f2[i] * r_rel[i]
            vel[k] = vel_com[i] + f2[i] * v_rel[i]
            mass[k] = m2[i]; ids[k] = k + 1; k += 1

    pos += 0.5 * boxsize
    vel -= (mass[:, None] * vel).sum(0) / mass.sum()      # exact global COM frame

    nb = int(is_bin.sum())
    ab = a[is_bin] * AU_PER_PC
    print(f"  systems {n_systems}  ->  stars {n_stars}   binaries {nb} "
          f"(f_bin = {nb/n_systems:.2f})")
    print(f"  M1   {m1.min():.3f} - {m1.max():.2f} Msun   M_cluster = {mass.sum():.1f} Msun")
    if nb:
        print(f"  a    {ab.min():.2f} - {ab.max():.1f} AU   (median {np.median(ab):.1f})")
        print(f"  peri {(a[is_bin]*(1-ecc[is_bin])*AU_PER_PC).min():.2f} AU (min)   "
              f"e median {np.median(ecc[is_bin]):.2f}")
        print(f"  TRUNCATION: {n_clip_a} clipped in a, {n_clip_e} (period,e) redraws for the "
              f"{a_min_au:g} AU pericentre floor")
    print(f"  |sum m*v| = {np.linalg.norm((mass[:, None]*vel).sum(0)):.3e} (should be ~0)")

    with h5py.File(outfile, "w") as F:
        F.create_group("Header")
        F["Header"].attrs["NumPart_ThisFile"] = [0, 0, 0, 0, 0, n_stars]
        F["Header"].attrs["NumPart_Total"] = [0, 0, 0, 0, 0, n_stars]
        F["Header"].attrs["NumPart_Total_HighWord"] = [0] * 6
        F["Header"].attrs["MassTable"] = [0.0] * 6          # per-particle masses differ
        F["Header"].attrs["BoxSize"] = boxsize
        F["Header"].attrs["Time"] = 0.0
        F["Header"].attrs["Redshift"] = 0.0
        F["Header"].attrs["NumFilesPerSnapshot"] = 1
        g = F.create_group("PartType5")
        g.create_dataset("Coordinates", data=pos)
        g.create_dataset("Velocities", data=vel)
        g.create_dataset("Masses", data=mass)
        g.create_dataset("ParticleIDs", data=ids)
        # the pairing, so the test can measure per-binary elements without re-deriving it
        prim = np.flatnonzero(is_bin)
        F.create_group("BinaryCatalog")
        F["BinaryCatalog"].attrs["n_systems"] = n_systems
        F["BinaryCatalog"].create_dataset("PrimaryID", data=ids[prim_of[prim]])
        F["BinaryCatalog"].create_dataset("SecondaryID", data=ids[sec_of[prim]])
        F["BinaryCatalog"].create_dataset("SemiMajorAxis_AU", data=a[prim] * AU_PER_PC)
        F["BinaryCatalog"].create_dataset("Eccentricity", data=ecc[prim])
        F["BinaryCatalog"].create_dataset("MassRatio", data=(m2 / np.maximum(m1, 1e-30))[prim])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n_systems", type=int, default=256)
    p.add_argument("--a_cluster", type=float, default=1.0, help="Plummer scale radius (pc)")
    p.add_argument("--boxsize", type=float, default=300.0, help="pc")
    p.add_argument("--m_min", type=float, default=0.08)
    p.add_argument("--m_max", type=float, default=30.0)
    p.add_argument("--q_min", type=float, default=0.1)
    p.add_argument("--a_min_au", type=float, default=100.0,
                   help="pericentre floor in AU; SETS THE RUNTIME, see README.md")
    p.add_argument("--a_max_au", type=float, default=5.0e3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="plummer_binaries_realistic_ics.hdf5")
    a = p.parse_args()
    make_realistic_ics(a.n_systems, a.a_cluster, a.boxsize, a.seed, a.out,
                       m_min=a.m_min, m_max=a.m_max, q_min=a.q_min,
                       a_min_au=a.a_min_au, a_max_au=a.a_max_au)
