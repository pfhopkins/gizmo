"""shearing_box: 3D shear-periodic boundaries (BOX_SHEARING=3), for q = 3/2 (Keplerian) and q = 1 (flat rotation curve).

Units: Omega = 1, L = 1, c_s = 1, density 1. X = x - L/2, eta = v_phi + q Omega X (velocity relative to the shear flow).
Each q is built once and runs three problems:

  equilibrium  uniform gas on a lattice with the shear flow v_phi = -q Omega X. On a lattice every x-plane slides
               rigidly, so all particles in a plane see identical neighbourhoods and the discrete equilibrium is
               exact; the boundary preserves it only if the remap across x = 0/L (phi shift and v_phi offset) is
               consistent with the interior. Every particle must stay at v_r = eta = v_z = 0, uniform density.
  epicycle     the same plus a uniform v_r = vr0. The exact (nonlinear) solution stays uniform:
               v_r = vr0 cos(kappa t), eta = -(2-q) Omega vr0 sin(kappa t) / kappa, kappa^2 = 2 (2-q) Omega^2.
               The radial excursion vr0/kappa carries several lattice planes through the x-boundary each half
               period. Checks: the spread between particles (any inconsistent remap, in-code or in the snapshot
               writer, shows up as particles off the common solution), the frequency and amplitudes, and the
               synchronised final state.
  shearing_wave  a linear vortical shearing wave exp(i(kx(t) X + ky phi)), kx(t) = kx0 + q Omega ky t, swinging from
               leading to trailing (Johnson & Gammie 2005). Its phase is continuous across x = 0/L only through the
               time-dependent phi offset, the part of the boundary that axisymmetric tests cannot see. Compared with
               the compressible linear equations integrated here, and checked for conservation of the linearised
               potential vorticity i(kx eta - ky v_r) - (2-q) Omega delta.

For q = 3/2 only, an integrator check:

  cold_epicycle  a uniform epicycle with c_s = 1e-3, vr0 = 0.1 c_s, so the timestep is set by MaxSizeTimestep rather
               than the Courant condition. The epicycle energy E = v_r^2 + 2 eta^2 / (2-q) is conserved exactly; a
               symplectic second-order integration keeps its error bounded at O((kappa dt)^2). Gas over 100 epicycles,
               and collisionless particles over 10.

Velocities in intermediate snapshots are half-kicked (written mid-step); the synchronised final snapshot is used where
that matters.
"""

from pathlib import Path
import glob
import os
import shutil
import sys

import h5py
import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gizmo.test import build_gizmo_for_test, run_test, default_mpi_ranks
from make_shearing_box_ics import make_shearing_box_ics

TEST_NAME = "shearing_box"
TEST_DIR = Path(__file__).parent
NUM_MPI_RANKS = default_mpi_ranks(max_ranks=4)

N_UNIFORM, T_UNIFORM = 16, 6 * np.pi  # three orbits
VR0 = 0.3
N_WAVE, DV, KX0, KY = 32, 1e-3, -4 * np.pi, 2 * np.pi

# Exact lattice equilibria hold to round-off (<5e-13 measured); a wrong phi shift or v_phi offset at the boundary
# shows up at 5e-4 or more.
TOL_SPREAD = 1e-6
# Epicycle mean motion: second-order time-integration error only. The frequency is high by 3e-4 (converging as dt^2),
# i.e. a phase error up to 8e-3 rad after the 4.2 epicycles of q = 1 (3.6e-3 vr0 measured at the final time).
TOL_EPI_FINAL = 1e-2
TOL_EPI_FREQ = 2e-3
TOL_EPI_AMP = 5e-3
# Shearing wave, ~14 particles per wavelength: <=3.6% measured (5.7-7.2% at 24^3, ~12% at 16^3: second order).
TOL_WAVE = 0.08

EPI_COLD_CS, EPI_COLD_MAXDT = 1e-3, 0.1
# A symplectic integrator keeps |dE/E| ~ (kappa dt)^2 bounded (1.5e-3 at this dt for the symplectic epicycle
# integrator of Rein & Tremaine 2011); a non-symplectic one drifts without bound.
TOL_EPI_ENERGY = 1e-2


def kappa(q):
    return np.sqrt(2 * (2 - q))


def cases(q):
    """(tag, IC kwargs, TimeMax, snapshot interval, MaxSizeTimestep or None) for every run of this q."""
    t_wave = (2.25 * 2 * np.pi / KY) / q  # stop just after the swing, at kx = +ky/4
    c = [("equilibrium", dict(N=N_UNIFORM), T_UNIFORM, 0.5, None),
         ("epicycle", dict(N=N_UNIFORM, vr0=VR0), T_UNIFORM, 0.5, None),
         ("shearing_wave", dict(N=N_WAVE, dv=DV, kx0=KX0, ky=KY), t_wave, t_wave / 12, None)]
    if q != 1.5:
        return c
    t_epi = 2 * np.pi / kappa(q)
    vr0 = 0.1 * EPI_COLD_CS
    c.append(("cold_epicycle", dict(N=8, cs=EPI_COLD_CS, vr0=vr0), 100 * t_epi, t_epi, EPI_COLD_MAXDT))
    c.append(("cold_epicycle_collisionless", dict(N=8, cs=EPI_COLD_CS, vr0=vr0, n_collisionless=4), 10 * t_epi,
              t_epi / 4, EPI_COLD_MAXDT))
    return c


@pytest.fixture(scope="module", params=[1.5, 1.0], ids=["q1.5", "q1"])
def runs(request):
    """Builds for one q and runs its problems; returns q and {case: (output dir, TimeMax)}."""
    q = request.param
    build_gizmo_for_test(TEST_NAME, 0, (f"BOX_SHEARING_Q=({q!r})",))
    out = {}
    cwd = os.getcwd()
    os.chdir(TEST_DIR)
    try:
        for case, ic_kwargs, t_max, dt_snap, max_dt in cases(q):
            tag = f"{case}_q{q}"
            make_shearing_box_ics(f"{tag}_ics.hdf5", q, **ic_kwargs)
            outdir = f"output_{tag}"
            if not os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
                shutil.rmtree(outdir, ignore_errors=True)
            overrides = {"InitCondFile": f"{tag}_ics", "OutputDir": outdir, "TimeMax": f"{t_max:.12g}",
                         "TimeBetSnapshot": f"{dt_snap:.12g}"}
            if max_dt:
                overrides["MaxSizeTimestep"] = f"{max_dt:.12g}"
            run_test(TEST_NAME, NUM_MPI_RANKS, 0, param_overrides=overrides)
            out[case] = (str(TEST_DIR / outdir), t_max)
    finally:
        os.chdir(cwd)
    return q, out


def load_snapshots(outdir, t_max, ptype=0):
    snaps = []
    for s in sorted(glob.glob(f"{outdir}/snapshot_*.hdf5")):
        with h5py.File(s, "r") as f:
            p = f[f"PartType{ptype}"]
            o = np.argsort(p["ParticleIDs"][:])
            snaps.append(dict(t=float(f["Header"].attrs["Time"]), x=p["Coordinates"][:][o].astype(float),
                              v=p["Velocities"][:][o].astype(float)))
            if ptype == 0:
                snaps[-1]["rho"] = p["Density"][:][o].astype(float)
    assert snaps, f"no snapshots in {outdir}"
    assert abs(snaps[-1]["t"] / t_max - 1) < 1e-6, f"{outdir}: last snapshot at t={snaps[-1]['t']}, not {t_max}"
    return snaps


def eta(s, q):
    return s["v"][:, 1] + q * (s["x"][:, 0] - 0.5)


def spreads(snaps, q):
    """Largest particle-to-particle spread (rms about the mean) of v_r, eta, v_z and rho over all snapshots."""
    worst = {}
    for s in snaps:
        for name, a in (("v_r", s["v"][:, 0]), ("eta", eta(s, q)), ("v_z", s["v"][:, 2]), ("rho", s["rho"])):
            r = np.std(a) / (np.mean(a) if name == "rho" else 1.0)
            if r > worst.get(name, (0,))[0]:
                worst[name] = (r, s["t"])
    return worst


def test_equilibrium(runs):
    q, out = runs
    snaps = load_snapshots(*out["equilibrium"])
    print(f"q={q} equilibrium: max deviation", {k: f"{r:.2e}" for k, (r, _) in spreads(snaps, q).items()})
    for s in snaps:
        dev = max(np.abs(s["v"][:, 0]).max(), np.abs(eta(s, q)).max(), np.abs(s["v"][:, 2]).max())
        drho = np.abs(s["rho"] / s["rho"].mean() - 1).max()
        assert dev < TOL_SPREAD and drho < TOL_SPREAD, (
            f"q={q}: equilibrium shear flow broken at t={s['t']:.3g}: max |v - v_eq| = {dev:.3e}, max |drho/rho| = {drho:.3e}")


def test_epicycle(runs):
    q, out = runs
    snaps = load_snapshots(*out["epicycle"])
    k = kappa(q)

    # the boundary must actually be crossed: count particles whose x jumped by more than L/2 between snapshots
    crossed = np.zeros(len(snaps[0]["x"]), dtype=bool)
    for a, b in zip(snaps[:-1], snaps[1:]):
        crossed |= np.abs(b["x"][:, 0] - a["x"][:, 0]) > 0.5
    assert crossed.mean() > 0.25, f"only {crossed.mean():.1%} of particles crossed the x-boundary"

    worst = spreads(snaps, q)
    print(f"q={q} epicycle: crossed {crossed.mean():.0%}; max spread", {k: f"{r:.2e}" for k, (r, _) in worst.items()})
    for name, (r, t) in worst.items():
        assert r < TOL_SPREAD, f"q={q}: particles departed from the uniform epicycle: rms spread of {name} = {r:.3e} at t={t:.4g}"

    t = np.array([s["t"] for s in snaps])
    vr = np.array([s["v"][:, 0].mean() for s in snaps])
    et = np.array([eta(s, q).mean() for s in snaps])
    # final snapshot is synchronised: compare directly
    err_vr = abs(vr[-1] - VR0 * np.cos(k * t[-1])) / VR0
    err_eta = abs(et[-1] + (2 - q) * VR0 * np.sin(k * t[-1]) / k) / VR0
    print(f"q={q} epicycle: final error {err_vr:.2e} (v_r), {err_eta:.2e} (eta) of vr0")
    assert max(err_vr, err_eta) < TOL_EPI_FINAL, f"q={q}: final mean velocity off the epicycle by {err_vr:.2e} (v_r), {err_eta:.2e} (eta) of vr0"

    # half-kicked intermediate velocities shift the phase slightly, not the frequency or amplitude
    (A, w, _), _ = curve_fit(lambda t, A, w, p: A * np.cos(w * t + p), t, vr, p0=(VR0, k, 0))
    (B, w2, _), _ = curve_fit(lambda t, B, w, p: B * np.sin(w * t + p), t, et, p0=(-(2 - q) * VR0 / k, k, 0))
    B_exact = (2 - q) * VR0 / k
    print(f"q={q} epicycle: omega/kappa-1 = {w / k - 1:.2e}, {w2 / k - 1:.2e}; |A|/vr0-1 = {abs(A) / VR0 - 1:.2e}; |B|/B_exact-1 = {abs(B) / B_exact - 1:.2e}")
    assert abs(w / k - 1) < TOL_EPI_FREQ and abs(w2 / k - 1) < TOL_EPI_FREQ, f"q={q}: epicyclic frequency {w:.5f}, {w2:.5f} vs kappa = {k:.5f}"
    assert abs(abs(A) / VR0 - 1) < TOL_EPI_AMP, f"q={q}: v_r amplitude {abs(A):.5f} vs {VR0}"
    assert abs(abs(B) / B_exact - 1) < TOL_EPI_AMP, f"q={q}: eta amplitude {abs(B):.5f} vs {B_exact:.5f}"


def linear_shearing_wave(q, t_eval):
    """Compressible isothermal shearing wave (delta, v_r, eta) amplitudes, from the IC's vortical wave."""
    def rhs(t, y):
        d, ux, uy = y[0] + 1j * y[1], y[2] + 1j * y[3], y[4] + 1j * y[5]
        kx = KX0 + q * KY * t
        r = (-1j * (kx * ux + KY * uy), -1j * kx * d + 2 * uy, -1j * KY * d - (2 - q) * ux)
        return [f(z) for z in r for f in (np.real, np.imag)]
    k0 = np.hypot(KX0, KY)
    y0 = [0, 0, -DV * KY / k0, 0, DV * KX0 / k0, 0]
    sol = solve_ivp(rhs, (0, t_eval[-1]), y0, t_eval=t_eval, rtol=1e-10, atol=1e-14)
    return sol.y[0::2] + 1j * sol.y[1::2]


def test_shearing_wave(runs):
    q, out = runs
    snaps = load_snapshots(*out["shearing_wave"])
    t = np.array([s["t"] for s in snaps])
    meas = []
    for s in snaps:
        X = s["x"][:, 0] - 0.5
        ph = np.exp(-1j * ((KX0 + q * KY * s["t"]) * X + KY * s["x"][:, 1]))
        # equal-mass particles: the number-density mode is the Lagrangian density perturbation
        meas.append([2 * np.mean(ph), 2 * np.mean(s["v"][:, 0] * ph), 2 * np.mean(eta(s, q) * ph)])
    meas = np.array(meas).T
    lin = linear_shearing_wave(q, t)
    assert np.abs(lin[1]).max() > 3 * np.abs(lin[1, 0]), "the wave no longer swing-amplifies: setup changed"
    for i, name in ((1, "v_r"), (2, "eta")):
        err = np.abs(meas[i] - lin[i]).max() / np.abs(lin[i]).max()
        print(f"q={q} shearing wave: {name} error {err:.3f} of peak")
        assert err < TOL_WAVE, f"q={q}: shearing-wave {name} amplitude deviates from linear theory by {err:.3f} of its peak"
    kx = KX0 + q * KY * t
    pv = 1j * (kx * meas[2] - KY * meas[1]) - (2 - q) * meas[0]
    err = np.abs(pv / pv[0] - 1).max()
    print(f"q={q} shearing wave: potential vorticity change {err:.3f}")
    assert err < TOL_WAVE, f"q={q}: linear potential vorticity not conserved: max relative change {err:.3f}"


def cold_runs(runs):
    q, out = runs
    if "cold_epicycle" not in out:
        pytest.skip("cold-gas cases run for q = 3/2 only")
    return q, out


def epicycle_energy(s, q):
    return np.mean(s["v"][:, 0]) ** 2 + 2 * np.mean(eta(s, q)) ** 2 / (2 - q)


def test_cold_epicycle_energy(runs):
    q, out = cold_runs(runs)
    snaps = load_snapshots(*out["cold_epicycle"])
    E = np.array([epicycle_energy(s, q) for s in snaps])
    E0 = (0.1 * EPI_COLD_CS) ** 2
    t = np.array([s["t"] for s in snaps])
    rate = np.polyfit(t, np.log(E / E0), 1)[0]
    print(f"cold epicycle (gas): E(T)/E0 - 1 = {E[-1] / E0 - 1:.3e}, d ln E / dt = {rate:.3e}")
    assert abs(E[-1] / E0 - 1) < TOL_EPI_ENERGY, f"gas epicycle energy drifted by {E[-1] / E0 - 1:.3e} over 100 epicycles"


def test_cold_epicycle_energy_collisionless(runs):
    q, out = cold_runs(runs)
    snaps = load_snapshots(*out["cold_epicycle_collisionless"], ptype=1)
    E = epicycle_energy(snaps[-1], q) / (0.1 * EPI_COLD_CS) ** 2
    print(f"cold epicycle (collisionless): E(T)/E0 - 1 = {E - 1:.3e}")
    assert abs(E - 1) < TOL_EPI_ENERGY, f"collisionless epicycle energy drifted by {E - 1:.3e} over 10 epicycles"
