"""Special-relativistic shock tube with general-relativistic hydrodynamics (GENERAL_RELATIVITY, Minkowski metric; Lupi 2023).

Marti & Mueller problem 1: rest-mass density 10 | 1, pressure 40/3 | 1e-6, gas at rest, adiabatic index 5/3, a mildly
relativistic shell (v = 0.714 c) behind the shock. Run with the HLL solver for both meshless methods (MFM and MFV), and
compared at t = 0.3 with the exact solution in the window 0.5 < x < 1.5 (discontinuity at x = 1).

Downloaded: grhd_shocktube_ics.hdf5 (2200 equal-mass cells, 2000 + 200 split 10:1 across the discontinuity) and
grhd_shocktube_exact.txt (the exact Riemann solution at t = 0.3, columns x, rho0, P, v_x; computed with the srrp package).
"""

import pytest
import numpy as np
from os import path
from matplotlib import pyplot as plt
import h5py
from gizmo.test import (
    build_and_run_test,
    assert_final_time,
    default_mpi_ranks,
    default_omp_threads,
    get_final_snapshot,
    variant_output_dir,
)

TEST_NAME = "grhd_shocktube"
GAMMA = 5.0 / 3.0  # EOS_GAMMA of Config.sh

# the meshless methods run by this test, and the mean absolute error in the window each must stay below
# (measured at 2 and 4 ranks alike: MFM rho0 0.024, P 0.018, v_x 0.0022; MFV 0.050, 0.032, 0.0053)
VARIANTS = (("HYDRO_MESHLESS_FINITE_MASS",), ("HYDRO_MESHLESS_FINITE_VOLUME",))
TOLERANCE = {"HYDRO_MESHLESS_FINITE_MASS": {"rho0": 0.035, "P": 0.027, "v": 0.0035},
             "HYDRO_MESHLESS_FINITE_VOLUME": {"rho0": 0.075, "P": 0.048, "v": 0.008}}


@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(),))
@pytest.mark.parametrize("num_omp_threads", (default_omp_threads(),))
@pytest.mark.parametrize("extra_config_flags", VARIANTS, ids=["MFM", "MFV"])
def test_grhd_shocktube(num_mpi_ranks, num_omp_threads, extra_config_flags):
    build_and_run_test(TEST_NAME, num_mpi_ranks, num_omp_threads, extra_config_flags)
    final_snap = get_final_snapshot(TEST_NAME, extra_config_flags)
    assert_final_time(final_snap, TEST_NAME)

    with h5py.File(final_snap, "r") as F:
        p = F["PartType0"]
        x = p["Coordinates"][:, 0]
        rho0 = p["RestMassDensity"][:]
        P = (GAMMA - 1.0) * rho0 * p["InternalEnergy"][:]
        v = p["Velocities"][:, 0]
    order = np.argsort(x)
    window = order[(x[order] > 0.5) & (x[order] < 1.5)]
    x, values = x[window], {"rho0": rho0[window], "P": P[window], "v": v[window]}
    xe, rho_e, P_e, v_e = np.loadtxt(f"test/{TEST_NAME}/{TEST_NAME}_exact.txt", unpack=True)
    exact = {"rho0": rho_e, "P": P_e, "v": v_e}
    errors = {q: np.mean(np.abs(values[q] - np.interp(x, xe, exact[q]))) for q in values}

    fig, axes = plt.subplots(3, 1, figsize=(6, 9), sharex=True)
    for ax, q, label in zip(axes, ("rho0", "P", "v"), (r"$\rho_0$", r"$P$", r"$v_x$")):
        ax.plot(xe, exact[q], color="lightgrey", linewidth=4, label="exact")
        ax.plot(x, values[q], color="blue", linewidth=1.5, label=extra_config_flags[0])
        ax.set_ylabel(label)
    axes[0].legend(frameon=False)
    axes[-1].set_xlabel(r"$x$")
    fig.savefig(path.join(variant_output_dir(TEST_NAME, extra_config_flags), f"{TEST_NAME}.pdf"), bbox_inches="tight")
    plt.close(fig)

    tol = TOLERANCE[extra_config_flags[0]]
    print(f"{extra_config_flags[0]}: mean absolute errors " + ", ".join(f"{q} {errors[q]:.4f} (tolerance {tol[q]})" for q in errors))
    for q in errors:
        assert errors[q] < tol[q], f"{q} error {errors[q]:.4f} exceeds {tol[q]}"
