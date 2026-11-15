"""Type-4 stars on the 4th-order Hermite integrator with exact direct star-star gravity (STARS_HERMITE_DIRECT_GRAVITY):
an eccentric binary and a hierarchical triple, convergence in the Aarseth step parameter eta against Newtonian reference
orbits. Bars from theory, set before the runs: see README.md."""

import glob
import importlib.util

import numpy as np
from scipy.integrate import solve_ivp

from gizmo.test import build_and_run_test, variant_output_dir

TEST_NAME = "stars_hermite_direct_gravity"
TEST_DIR = f"test/{TEST_NAME}"
_spec = importlib.util.spec_from_file_location("fewbody_reference", f"{TEST_DIR}/fewbody_reference.py")
ref = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(ref)

ERRTOL = 0.049  # largest GIZMO accepts; the acceleration step stays above the Aarseth step for the ladders below
ORDER = (3.5, 4.5)  # 4th order; next-order terms ~ dt/tau ~ sqrt(eta) move the measured order by < 0.5
LIFT = 1e-9  # snapshots this fraction of an interval after a full sync point, where the start-of-step state is the corrected one
ORIGIN = np.full(3, 2e-5)  # centre of mass near the origin: coordinate roundoff 1e4 below that at 0.15 kpc
DEV = ("DEVELOPER_MODE",)  # ErrTolIntAccuracy read from the parameters


def period(m, a):
    return 2 * np.pi * np.sqrt(a ** 3 / (ref.G * m))


P_BIN = period(2 * ref.MSUN, 1000 * ref.AU)
P_OUT = period(3 * ref.MSUN, 1000 * ref.AU)
SYSTEMS = {"binary": dict(ic=ref.binary, tmax=8 * P_BIN, every=P_BIN, maxstep=P_BIN / 4,  # snapshots after orbits 1..7
                       etas=(0.00125, 3.125e-4, 7.8125e-5)),  # eta x1/4 halves the Aarseth step, so the block steps shift by one bin
           "triple": dict(ic=ref.triple, tmax=2 * P_OUT, every=P_OUT, maxstep=P_OUT / 4,  # snapshot after one outer orbit
                       etas=(0.005, 0.00125, 3.125e-4))}
RUNS = {}


def run(system, eta):
    """initial state and the snapshots at k x every (+LIFT), k >= 1, as (k, x, v, t)"""
    key = (system, eta)
    if key in RUNS:
        return RUNS[key]
    for name, sy in SYSTEMS.items():  # every IC on disk: the harness checks the params file's InitCondFile before the override
        xi, vi, mi = sy["ic"]()
        ref.write(f"{TEST_DIR}/hermite_{name}_ics.hdf5", xi - ref.CENTRE + ORIGIN, vi, mi)
    s = SYSTEMS[system]
    x, v, m = s["ic"]()
    x = x - ref.CENTRE + ORIGIN
    flags = (f"STARS_HERMITE_AARSETH_ETA={eta}",) + DEV
    build_and_run_test(TEST_NAME, 1, 1, extra_config_flags=flags,
                       param_overrides={"InitCondFile": f"hermite_{system}_ics", "TimeMax": s["tmax"], "TimeOfFirstSnapshot": s["every"] * (1 + LIFT),
                                        "TimeBetSnapshot": s["every"], "TimeBetStatistics": s["tmax"], "MaxSizeTimestep": s["maxstep"],
                                        "ErrTolIntAccuracy": ERRTOL})
    out = []
    for f in sorted(glob.glob(variant_output_dir(TEST_NAME, flags) + "/snapshot_*.hdf5")):
        x1, v1, _, t1 = ref.read(f)
        k = round(t1 / s["every"] - LIFT)
        if k >= 1 and abs(t1 - s["every"] * (k + LIFT)) < 0.5 * LIFT * s["every"]:
            out.append((k, x1, v1, t1))
    RUNS[key] = (x, v, m, out)
    return RUNS[key]


def errors(x0, v0, m, x1, v1, t1):
    """signed relative energy error, largest offset of the relative positions from the reference in inner semi-major axes"""
    xr, _ = ref.reference(x0, v0, m, t1)
    de = ref.energy(x1, v1, m) / ref.energy(x0, v0, m) - 1
    a0 = ref.elements(x0[:2], v0[:2], m[:2])[0]
    dx = np.max(np.linalg.norm((x1 - x1[0]) - (xr - xr[0]), axis=1)) / a0
    return de, dx


def truncation_bound(x0, v0, m, t_end, eta, maxstep):
    """|dE/E| bound of the 4th-order Hermite step on the reference orbit, no cancellation between steps: per step and body the
    corrector's quadrature error (dt^5/720 of the 4th and 3rd derivatives of the acceleration) plus the predictor error
    (snap dt^4/24 in position, dt^3/6 in velocity, own and source's) through the tidal tensor, at the Aarseth step
    sqrt(eta (|a||s| + |j|^2) / (|j||c| + |s|^2)) of the exact derivatives (Makino & Aarseth 1992)"""
    n, G = len(m), ref.G
    c0, vc = (m[:, None] * x0).sum(0) / m.sum(), (m[:, None] * v0).sum(0) / m.sum()
    y0 = np.concatenate([(x0 - c0).ravel(), (v0 - vc).ravel()])

    def rhs(_, y):
        p = y[:3 * n].reshape(n, 3); d = p[None, :, :] - p[:, None, :]
        r3 = np.linalg.norm(d, axis=2) ** 3; np.fill_diagonal(r3, np.inf)
        return np.concatenate([y[3 * n:], (G * np.sum(m[None, :, None] * d / r3[:, :, None], axis=1)).ravel()])

    atol = np.concatenate([np.full(3 * n, 1e-13 * np.abs(y0[:3 * n]).max()), np.full(3 * n, 1e-13 * np.abs(y0[3 * n:]).max())])
    fwd = solve_ivp(rhs, (0, 1.02 * t_end), y0, method="DOP853", rtol=1e-13, atol=atol, dense_output=True)
    bwd = solve_ivp(rhs, (0, -0.02 * t_end), y0, method="DOP853", rtol=1e-13, atol=atol, dense_output=True)

    def state(t):
        y = np.where(t >= 0, fwd.sol(np.maximum(t, 0)), bwd.sol(np.minimum(t, 0)))
        return y[:3 * n].T.reshape(-1, n, 3), y[3 * n:].T.reshape(-1, n, 3)

    def acc_jerk(t):
        x, v = state(t)
        d, dv = x[:, None, :, :] - x[:, :, None, :], v[:, None, :, :] - v[:, :, None, :]  # [t, i, k]: k relative to i
        r = np.linalg.norm(d, axis=3); r[:, np.arange(n), np.arange(n)] = np.inf
        rv = np.sum(d * dv, axis=3)
        a = G * np.sum(m[None, None, :, None] * d / r[..., None] ** 3, axis=2)
        j = G * np.sum(m[None, None, :, None] * (dv / r[..., None] ** 3 - 3 * rv[..., None] * d / r[..., None] ** 5), axis=2)
        return x, v, a, j, r, np.linalg.norm(dv, axis=3)

    t = np.unique(np.concatenate([np.linspace(fwd.t[i], fwd.t[i + 1], 9)[:-1] for i in range(len(fwd.t) - 1)] + [[t_end]]))
    t = t[t <= t_end]
    x, v, a, j, r, w = acc_jerk(t)
    nrm = lambda q: np.linalg.norm(q, axis=-1)
    h = 0.02 * np.min(nrm(a) / nrm(j), axis=1)
    jp, jm, jp2, jm2 = (acc_jerk(t + f * h)[3] for f in (1, -1, 2, -2))
    hh = h[:, None, None]
    s, c, a4 = (jp - jm) / (2 * hh), (jp - 2 * j + jm) / hh ** 2, (jp2 - 2 * jp + 2 * jm - jm2) / (2 * hh ** 3)
    A, J, S, C, A4 = nrm(a), nrm(j), nrm(s), nrm(c), nrm(a4)
    dt = np.minimum(np.sqrt(eta * (A * S + J * J) / (J * C + S * S)), maxstep)
    px, pv = S * dt ** 4 / 24, S * dt ** 3 / 6
    T = 2 * G * m[None, None, :] / r ** 3
    da = np.sum(T * (px[:, :, None] + px[:, None, :]), axis=2)
    dj = np.sum(T * (pv[:, :, None] + pv[:, None, :]) + 3 * w / r * T * (px[:, :, None] + px[:, None, :]), axis=2)
    dv_ = A4 * dt ** 5 / 720 + da * dt / 2 + dj * dt ** 2 / 12
    dx_ = C * dt ** 5 / 720 + dv_ * dt / 2 + da * dt ** 2 / 12
    rate = np.sum(m[None, :] * (nrm(v) * dv_ + A * dx_) / dt, axis=1)
    return getattr(np, "trapezoid", getattr(np, "trapz", None))(rate, t) / abs(ref.energy(x0, v0, m))


def ladder(system, etas):
    """per eta: signed dE/E, offset and the truncation bound at the last snapshot"""
    s, rows = SYSTEMS[system], []
    for eta in etas:
        x0, v0, m, out = run(system, eta)
        k, x1, v1, t1 = out[-1]
        de, dx = errors(x0, v0, m, x1, v1, t1)
        rows.append((eta, k, de, dx, truncation_bound(x0, v0, m, t1, eta, s["maxstep"])))
        print(f"{system} eta {eta}: after {k} x {s['every']:.4g}: dE/E {de:.3e} (bound {rows[-1][4]:.3e}), offset {dx:.3e}")
    return rows


def orders(rows, col):
    q = np.abs([r[col] for r in rows])
    return np.log2(q[:-1] / q[1:])  # step ratio 2 per rung


def check(rows, energy_order):
    for eta, k, de, dx, bound in rows:
        assert abs(de) <= bound, (eta, de, bound)
    po, pe = orders(rows, 3), orders(rows, 2)
    print(f"measured order: energy {pe}, offset {po}")
    assert np.all((po >= ORDER[0]) & (po <= ORDER[1])), po
    if energy_order:
        assert np.all((pe >= ORDER[0]) & (pe <= ORDER[1])), pe


def test_binary():
    check(ladder("binary", SYSTEMS["binary"]["etas"]), True)


def test_triple():
    check(ladder("triple", SYSTEMS["triple"]["etas"]), True)

