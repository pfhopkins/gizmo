"""Standing longitudinal wave in a cold elastic solid, through tension: phase speed and damping against linear theory.

A 1D periodic box of compact Tillotson basalt (2.7 g/cc, u = 1e6 erg/g) carries a standing wave v = A cos(w t) sin(k x)
of amplitude 30 cm/s and 50 cells per wavelength, followed for 8 periods. The material is cold, so the pressure swings
from compression into tension (about -2e7 to +3e7 dyn/cm^2) every half period: the run exercises the elastic solver's
treatment of negative pressure (the tensile-instability correction) in the simplest geometry. In 1D the stress
deviator vanishes (dS/dt = 2 mu (strain rate - trace)), so linear theory gives w = c k with c^2 = dP/drho at constant
entropy from the Tillotson EOS at the initial state.

Validates:
  1. The run reaches its end time with mass conserved and the total momentum unchanged.
  2. The wave's phase speed is within 5% of linear theory (the tensile correction slows it by about 3% at this
     amplitude and resolution; without it the error is below 0.1%).
  3. The amplitude falls by less than 15% per period (measured: about 10%), and does not grow (by more than 2%,
     the fit's noise): growth would be an instability.
"""

import glob

import h5py
import numpy as np
import pytest
from scipy.optimize import curve_fit

from gizmo.test import (
    build_and_run_test,
    clean_test_outputs,
    default_mpi_ranks,
    default_omp_threads,
    assert_final_time,
)

TEST_NAME = "elastic_tension_wave"

# must match make_ics.py and elastic_tension_wave.params
N, L, RHO_0, U_INIT, AMPLITUDE, N_WAVES = 400, 200.0, 2.7, 1.0e6, 30.0, 8
TIL = dict(a=0.5, b=1.5, u0=4.87e12, rho0=2.7, A=2.67e11, B=2.67e11)


def tillotson_pressure_compressed(rho, u, p=TIL):
    """Tillotson pressure on its compressed/cold branch (u below u_s), as evaluated in GIZMO."""
    eta = rho / p["rho0"]; mu = eta - 1; w = 1 + u / (p["u0"] * eta * eta)
    return (p["a"] + p["b"] / w) * rho * u + p["A"] * mu + p["B"] * mu * mu


def linear_sound_speed():
    """c^2 = dP/drho along the adiabat du = P/rho^2 drho, at the initial state."""
    drho = 1e-7 * RHO_0
    P0 = tillotson_pressure_compressed(RHO_0, U_INIT)
    du = P0 / RHO_0**2 * drho
    return np.sqrt((tillotson_pressure_compressed(RHO_0 + drho, U_INIT + du) - tillotson_pressure_compressed(RHO_0 - drho, U_INIT - du)) / (2 * drho))


def _make_ics():
    import subprocess, sys
    subprocess.run([sys.executable, "make_ics.py"], check=True, cwd=f"test/{TEST_NAME}")


@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(),))
@pytest.mark.parametrize("num_omp_threads", (default_omp_threads(),))
def test_elastic_tension_wave(num_mpi_ranks, num_omp_threads):
    _make_ics()
    clean_test_outputs(TEST_NAME)
    build_and_run_test(TEST_NAME, num_mpi_ranks, num_omp_threads)
    snaps = sorted(glob.glob(f"test/{TEST_NAME}/output/snapshot_*.hdf5"))
    assert snaps, "no snapshots: GIZMO did not run"
    assert_final_time(snaps[-1], TEST_NAME)

    k = 2 * np.pi * N_WAVES / L
    t, mode, mass, momentum, p_min = [], [], [], [], []
    for snap in snaps:
        with h5py.File(snap, "r") as f:
            g = f["PartType0"]
            x, v, m = g["Coordinates"][:, 0], g["Velocities"][:, 0], g["Masses"][:]
            t.append(f["Header"].attrs["Time"]); p_min.append(g["Pressure"][:].min())
        mode.append(abs(2 * np.mean(v * np.exp(-1j * k * x))))   # = A |cos(w t)| for the standing wave
        mass.append(m.sum()); momentum.append(np.sum(m * v))
    t, mode = np.array(t), np.array(mode)

    assert abs(mass[-1] / mass[0] - 1) < 1e-12, "mass not conserved"
    assert abs(momentum[-1] - momentum[0]) < 1e-6 * AMPLITUDE * mass[0], "momentum not conserved"
    assert min(p_min) < 0, "the wave never put the material into tension"

    c = linear_sound_speed()
    (a0, gamma, w), _ = curve_fit(lambda tt, a0, g, ww: a0 * np.exp(-g * tt) * np.abs(np.cos(ww * tt)), t, mode, p0=[AMPLITUDE, 0.0, c * k], maxfev=20000)
    speed = abs(w) / k
    loss_per_period = 1 - np.exp(-gamma * 2 * np.pi / abs(w))
    print(f"phase speed {speed:.5e} vs linear theory {c:.5e} (ratio {speed / c:.4f}); amplitude loss per period {loss_per_period:.4f}")
    assert abs(speed / c - 1) < 0.05, f"phase speed {speed:.4e} vs linear theory {c:.4e}"
    assert -0.02 < loss_per_period < 0.15, f"amplitude changes by {-loss_per_period:+.3f} per period (growth means instability)"
