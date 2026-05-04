"""Biermann battery growth-rate test (sign-anchor for MHD_BATTERY_MECHANISMS).

Lays down a 2D periodic box with orthogonal grad(n_e) and grad(T_e) and B=0,
then checks that the simulation grows a nonzero B_z field with the analytic
Biermann sign and (within an order of magnitude) the analytic rate.

The analytic Biermann source at the unperturbed origin is
    dB_z/dt = -(c k_B / e) * a * b * T_0 * k_phys**2 [Gauss / s]
where a, b are the relative amplitudes of rho and u and k_phys = 2 pi / L_phys.
n_e cancels out, so the prediction is independent of the mean density.

This is the *sign-anchor* test — Squire-Hopkins 2018 streaming-instability
gives the gold-standard signed-growth-rate cross-check, but Biermann growth in
this controlled IC is enough to catch sign bugs and order-of-magnitude
prefactor errors (which is what we need for commit 7/N).
"""

import os
import pytest
import numpy as np
import h5py

from gizmo.test import (
    build_and_run_test,
    default_mpi_ranks,
    default_omp_threads,
    get_final_snapshot,
    assert_final_time,
)

# Constants/unit conversions (must match make_biermann_growth_ics.py).
C_LIGHT_CGS = 2.99792458e10
BOLTZMANN_CGS = 1.38065e-16
ELECTRON_CHARGE_ESU = 4.80320425e-10
UnitLength_cm = 1.0
UnitVel_cms = 1.18e7


def _ensure_ic():
    """Generate the IC if it doesn't already exist (idempotent)."""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    ic_path = os.path.join(test_dir, "biermann_growth_ics.hdf5")
    if not os.path.isfile(ic_path):
        # Run the IC generator in-place.
        cwd = os.getcwd()
        os.chdir(test_dir)
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "make_biermann_growth_ics",
                os.path.join(test_dir, "make_biermann_growth_ics.py"),
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            mod.make_IC()
        finally:
            os.chdir(cwd)


@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(2),))
@pytest.mark.parametrize("num_omp_threads", (default_omp_threads(),))
def test_biermann_growth(num_mpi_ranks, num_omp_threads):
    test_name = "biermann_growth"
    _ensure_ic()
    build_and_run_test(test_name, num_mpi_ranks, num_omp_threads)
    final_snap = get_final_snapshot(test_name)
    assert_final_time(final_snap, test_name)

    with h5py.File(final_snap, "r") as F:
        coords = F["PartType0/Coordinates"][:]
        B = F["PartType0/MagneticField"][:]
        t_final = float(F["Header"].attrs["Time"])
        boxsize = float(F["Header"].attrs["BoxSize"])

    # Test parameters (must match the IC generator).
    amp_rho = 1e-4
    amp_u = 1e-4
    T0_K = 1.0e6
    Lbox_code = boxsize
    L_phys_cgs = Lbox_code * UnitLength_cm
    k_phys = 2.0 * np.pi / L_phys_cgs

    # Analytic prediction at the box center: pick a small region around the
    # peak of cos(k x) cos(k y) to average a few particles.
    dBz_dt_analytic = -(C_LIGHT_CGS * BOLTZMANN_CGS / ELECTRON_CHARGE_ESU) \
                      * amp_rho * amp_u * T0_K * k_phys**2  # Gauss/s

    # Convert code time (= L/V) to seconds. UnitMagneticField_in_gauss=1 so
    # code Gauss == cgs Gauss.
    UnitTime_s = UnitLength_cm / UnitVel_cms
    t_final_s = t_final * UnitTime_s
    Bz_predicted_G = dBz_dt_analytic * t_final_s

    # Pick particles near (x,y) ~ (0, 0): the corner of the lattice (we used
    # cell-centered placement starting at half a cell width). The Biermann
    # source maximum is at cos(k x) cos(k y) = 1, i.e. at x=y=0 in our IC.
    r2 = (coords[:, 0] - 0.5*Lbox_code)**2 + (coords[:, 1] - 0.5*Lbox_code)**2
    near_center = r2 < (0.05 * Lbox_code) ** 2
    Bz_simulated = float(np.mean(B[near_center, 2]))  # already in code Gauss

    # Sanity prints (visible with -s).
    print(f"  t_final (code) = {t_final}")
    print(f"  t_final (s)    = {t_final_s:.3e}")
    print(f"  Bz_predicted   = {Bz_predicted_G:.3e} Gauss")
    print(f"  Bz_simulated   = {Bz_simulated:.3e} Gauss (mean over {near_center.sum()} cells near center)")

    # Sign check: Biermann at (0,0) on rho=rho_0(1+a sin kx), u=u_0(1+b sin ky)
    # grad(n_e) = a*n_0*k*cos(kx)*x-hat, grad(T_e) = b*T_0*k*cos(ky)*y-hat.
    # At (x=y=0): both cos = 1, so cross product is in +z. dB/dt has overall
    # minus from the formula => Bz < 0. (assert_final consumes this.)
    assert Bz_simulated * Bz_predicted_G > 0, (
        f"Biermann sign mismatch: simulated Bz={Bz_simulated:.3e} G, "
        f"predicted={Bz_predicted_G:.3e} G"
    )
    # Magnitude within a factor of ~5 of the analytic prediction. Loose because
    # the simulation also evolves rho/u via pressure gradients on the sound-
    # crossing time, which slightly shifts the effective gradients.
    ratio = Bz_simulated / Bz_predicted_G
    assert 0.2 < ratio < 5.0, (
        f"Biermann magnitude off: simulated/analytic = {ratio:.3f}"
    )
