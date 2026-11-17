"""Porous shock compaction: the P-alpha porosity model (Jutzi et al. 2008) in a full hydro run, against the exact
Rankine-Hugoniot state.

Two halves of a 1D periodic box of porous basalt (matrix density 2.7 g/cc, distention alpha_0 = 1.275, Tillotson
matrix EOS, crush pressures P_e = 1e8 and P_s = 7e10 dyn/cm^2) collide at 2 x 3 km/s. Two shocks run out from the
collision plane and leave the material at rest, compacted. (Where the halves separate, at the periodic boundary, the
undamaged solid is pulled into strong tension; that region is kept out of the checks, which use |x - L/2| < 60.) The reference state behind them is the Rankine-Hugoniot
jump of the porous material: mass, momentum and energy conservation across the shock, the bulk pressure
P = P_matrix(alpha rho, u) / alpha, and alpha on the crush curve at the matrix pressure alpha P (compaction is
irreversible, so alpha never rises). It is solved below to machine precision.

Validates:
  1. The run reaches its end time with mass conserved.
  2. Material ahead of the shocks (matrix pressure below P_e) keeps alpha = alpha_0 exactly (no spurious compaction).
  3. The shocked plateau's pressure, density and distention match the Rankine-Hugoniot state (1%, 1% and 2% of the
     compaction alpha_0 - alpha), and alpha there lies on the crush curve at the cell's own matrix pressure.
  4. The shocks have swept the mass the Rankine-Hugoniot shock speed predicts (3%).
"""

import glob

import h5py
import numpy as np
import pytest
from scipy.optimize import brentq

from gizmo.test import (
    build_and_run_test,
    clean_test_outputs,
    default_mpi_ranks,
    default_omp_threads,
    assert_final_time,
)

TEST_NAME = "palpha_shock"

# must match make_ics.py and palpha_shock.params
N, L, RHO_MATRIX, ALPHA_0, U_P, U_INIT = 1600, 200.0, 2.7, 1.275, 3.0e4, 1.0e6
P_E, P_S = 1.0e8, 7.0e10
TIL = dict(a=0.5, b=1.5, u0=4.87e12, rho0=2.7, A=2.67e11, B=2.67e11, us=4.72e10, usp=1.82e11, al=5.0, be=5.0)


def tillotson_pressure(rho, u, p=TIL):
    """Tillotson pressure as evaluated in GIZMO (declarations/cell_data.h)."""
    eta = rho / p["rho0"]; mu = eta - 1; w = 1 + u / (p["u0"] * eta * eta); p0 = u * rho; z = 1 / eta - 1
    Pc = (p["a"] + p["b"] / w) * p0 + p["A"] * mu + p["B"] * mu * mu
    Pe = p["a"] * p0 + (p["b"] / w * p0 + p["A"] * mu * np.exp(-p["be"] * z)) * np.exp(-p["al"] * z * z)
    if u <= p["us"]:
        return Pc
    if u >= p["usp"]:
        return Pe
    return (Pe * (u - p["us"]) + Pc * (p["usp"] - u)) / (p["usp"] - p["us"])


def crush_curve(Pm):
    """Jutzi et al. 2008 eq. 8: equilibrium distention at matrix pressure Pm."""
    if Pm <= P_E:
        return ALPHA_0
    if Pm >= P_S:
        return 1.0
    r = (P_S - Pm) / (P_S - P_E)
    return 1 + (ALPHA_0 - 1) * r * r


def porous_state(rho, u, alpha_prev):
    """bulk pressure and distention: alpha = min(alpha_prev, crush(alpha P)), P = P_matrix(alpha rho, u) / alpha"""
    f = lambda a: a - min(alpha_prev, crush_curve(tillotson_pressure(a * rho, u)))
    if f(alpha_prev) <= 0:
        a = alpha_prev
    elif f(1.0) >= 0:
        a = 1.0
    else:
        a = brentq(f, 1.0, alpha_prev, xtol=1e-14)
    return tillotson_pressure(a * rho, u) / a, a


def rankine_hugoniot():
    """shocked state of the symmetric collision (material brought to rest)"""
    rho1 = RHO_MATRIX / ALPHA_0
    P1, a1 = porous_state(rho1, U_INIT, ALPHA_0)
    def jump(rho2):
        P2 = P1 + rho1 * U_P * U_P * rho2 / (rho2 - rho1)
        u2 = U_INIT + 0.5 * (P1 + P2) * (1 / rho1 - 1 / rho2)
        return P2, u2
    rho2 = brentq(lambda r: porous_state(r, jump(r)[1], a1)[0] - jump(r)[0], rho1 * (1 + 1e-6), rho1 * ALPHA_0 * 1.6, xtol=1e-14 * rho1)
    P2, u2 = jump(rho2)
    return dict(rho=rho2, P=P2, u=u2, alpha=porous_state(rho2, u2, a1)[1], Us=rho1 * U_P / (rho2 - rho1))


def _make_ics():
    import subprocess, sys
    subprocess.run([sys.executable, "make_ics.py"], check=True, cwd=f"test/{TEST_NAME}")


@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(),))
@pytest.mark.parametrize("num_omp_threads", (default_omp_threads(),))
def test_palpha_shock(num_mpi_ranks, num_omp_threads):
    _make_ics()
    clean_test_outputs(TEST_NAME)
    build_and_run_test(TEST_NAME, num_mpi_ranks, num_omp_threads)
    snaps = sorted(glob.glob(f"test/{TEST_NAME}/output/snapshot_*.hdf5"))
    assert snaps, "no snapshots: GIZMO did not run"
    assert_final_time(snaps[-1], TEST_NAME)
    with h5py.File(snaps[0], "r") as f:
        m0 = f["PartType0/Masses"][:].sum()
    with h5py.File(snaps[-1], "r") as f:
        g = f["PartType0"]; t = f["Header"].attrs["Time"]
        m, x, v, P, rho, alpha = (g[k][:] for k in ("Masses", "Coordinates", "Velocities", "Pressure", "Density", "Distention"))
    win = np.abs(x[:, 0] - 0.5 * L) < 60.0
    v, P, rho, alpha = v[win, 0], P[win], rho[win], alpha[win]
    rh = rankine_hugoniot()

    assert abs(m.sum() / m0 - 1) < 1e-12, "mass not conserved"

    ahead = (np.abs(np.abs(v) / U_P - 1) < 0.02) & (alpha * P < P_E)
    assert ahead.sum() > 100, "too few unshocked cells left to check"
    assert np.abs(alpha[ahead] - ALPHA_0).max() < 1e-6, "unshocked porous material compacted"

    rest = np.abs(v) < 0.05 * U_P
    assert rest.sum() > 100, "no shocked plateau"
    P_sim, rho_sim, a_sim = np.median(P[rest]), np.median(rho[rest]), np.median(alpha[rest])
    assert abs(P_sim / rh["P"] - 1) < 0.01, f"plateau pressure {P_sim:.4e} vs Rankine-Hugoniot {rh['P']:.4e}"
    assert abs(rho_sim / rh["rho"] - 1) < 0.01, f"plateau density {rho_sim:.5f} vs Rankine-Hugoniot {rh['rho']:.5f}"
    assert abs(a_sim - rh["alpha"]) < 0.02 * (ALPHA_0 - rh["alpha"]), f"plateau distention {a_sim:.5f} vs Rankine-Hugoniot {rh['alpha']:.5f}"
    on_curve = np.array([a / crush_curve(a * p) - 1 for a, p in zip(alpha[rest], P[rest])])
    assert np.median(np.abs(on_curve)) < 1e-3, "plateau distention off the crush curve at its own matrix pressure"

    swept = np.sum(np.abs(v) < 0.5 * U_P)                       # cells behind either shock front
    expected = 2 * (rh["Us"] + U_P) * t / (L / N)              # each shock sweeps rho1 (Us + u_p) t of mass per unit area
    assert abs(swept / expected - 1) < 0.03, f"shocked cells {swept} vs {expected:.1f} from the Rankine-Hugoniot shock speed"
