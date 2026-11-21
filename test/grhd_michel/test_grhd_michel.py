"""Michel (relativistic Bondi) accretion onto a black hole with general-relativistic hydrodynamics (GENERAL_RELATIVITY,
Kerr metric with zero spin in Kerr-Schild coordinates, excision inside 1.05 horizon radii; Lupi 2023).

Run with the HLL solver for both meshless methods (MFM and MFV) and compared with the analytic Michel solution.

This is a short regression run (TimeMax = 30 GM/c^3). The initial conditions start empty inside r ~ 4 GM/c^2, and at
t = 30 the flow inside r ~ 6 is still relaxing: only 6 < r < 30 is compared, and the plot shades the region that is not.
For the science comparison of Lupi (2023), which covers the inner flow down to the excision radius, set TimeMax = 200 in
the parameter file (about 7 times the cost of this test).

Downloaded: grhd_michel_ics.hdf5 (2.7e5 equal-mass cells with the Michel density between 4 and 100 GM/c^2) and
grhd_michel_exact.txt, the Michel solution: columns r, D, rho0, v_r (in Schwarzschild time), u.
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

TEST_NAME = "grhd_michel"
M = 1.0  # GR_Mass of the parameter file
R_SETTLED, R_OUTER = 6.0, 30.0  # the compared radii (GM/c^2) at TimeMax = 30
RADIAL_BINS = np.geomspace(R_SETTLED, R_OUTER, 7)

# the meshless methods run by this test, and their bounds over R_SETTLED < r < R_OUTER: median |log10 rho0/rho0_Michel|, the same
# median in the worst of 6 logarithmic radial bins, median |v_r - v_r^Michel| (in c) and median |log10 u/u_Michel|
# (measured at 2 and 4 ranks alike: MFM 0.0035 dex, 0.014 dex, 0.0023, 0.0008 dex; MFV 0.0036 dex, 0.016 dex, 0.0024, 0.0008 dex)
VARIANTS = (("HYDRO_MESHLESS_FINITE_MASS",), ("HYDRO_MESHLESS_FINITE_VOLUME",))
TOLERANCE = {"HYDRO_MESHLESS_FINITE_MASS": {"rho0_dex": 0.0055, "worst_bin_dex": 0.021, "v_r": 0.0035, "u_dex": 0.0012},
             "HYDRO_MESHLESS_FINITE_VOLUME": {"rho0_dex": 0.0055, "worst_bin_dex": 0.025, "v_r": 0.0035, "u_dex": 0.0012}}


@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(),))
@pytest.mark.parametrize("num_omp_threads", (default_omp_threads(),))
@pytest.mark.parametrize("extra_config_flags", VARIANTS, ids=["MFM", "MFV"])
def test_grhd_michel(num_mpi_ranks, num_omp_threads, extra_config_flags):
    build_and_run_test(TEST_NAME, num_mpi_ranks, num_omp_threads, extra_config_flags)
    final_snap = get_final_snapshot(TEST_NAME, extra_config_flags)
    assert_final_time(final_snap, TEST_NAME)

    with h5py.File(final_snap, "r") as F:
        p = F["PartType0"]
        pos = p["Coordinates"][:]
        r = np.linalg.norm(pos, axis=1)
        rho0 = p["RestMassDensity"][:]
        v_r = np.sum(pos * p["Velocities"][:], axis=1) / r
        u = p["InternalEnergy"][:]
    r_s, _, rho_s, v_s, u_s = np.loadtxt(f"test/{TEST_NAME}/{TEST_NAME}_exact.txt", unpack=True)
    v_s = v_s / (1.0 + 2.0 * M * v_s / (r_s - 2.0 * M))  # radial velocity in Schwarzschild time -> Kerr-Schild time

    compared = (r > R_SETTLED) & (r < R_OUTER)
    rho_dex = np.abs(np.log10(rho0 / np.interp(r, r_s, rho_s)))
    bins = [rho_dex[(r > a) & (r < b)] for a, b in zip(RADIAL_BINS[:-1], RADIAL_BINS[1:])]
    assert all(b.size > 0 and np.all(np.isfinite(b)) for b in bins), "a compared radial bin is empty or not finite"
    errors = {"rho0_dex": np.median(rho_dex[compared]),
              "worst_bin_dex": max(np.median(b) for b in bins),
              "v_r": np.median(np.abs(v_r[compared] - np.interp(r[compared], r_s, v_s))),
              "u_dex": np.median(np.abs(np.log10(u[compared] / np.interp(r[compared], r_s, u_s))))}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, q, qs, label, log in zip(axes, (rho0, v_r, u), (rho_s, v_s, u_s), (r"$\rho_0$", r"$dr/dt$", r"$u$"), (True, False, True)):
        ax.axvspan(2.0, R_SETTLED, facecolor="black", alpha=0.25, edgecolor="none")  # not yet relaxed at TimeMax = 30: not compared
        ax.plot(r_s, qs, color="lightgrey", linewidth=4, label="Michel")
        ax.plot(r, q, ",", color="blue", label=extra_config_flags[0])
        ax.set_xscale("log")
        if log:
            ax.set_yscale("log")
        ax.set_xlim(2.0, R_OUTER)
        ax.set_xlabel(r"$r$  [$GM/c^2$]")
        ax.set_ylabel(label)
    axes[0].legend(frameon=False)
    fig.savefig(path.join(variant_output_dir(TEST_NAME, extra_config_flags), f"{TEST_NAME}.pdf"), bbox_inches="tight")
    plt.close(fig)

    tol = TOLERANCE[extra_config_flags[0]]
    print(f"{extra_config_flags[0]}, {R_SETTLED} < r < {R_OUTER}: " + ", ".join(f"{q} {errors[q]:.4f} (tolerance {tol[q]})" for q in errors))
    for q in errors:
        assert errors[q] < tol[q], f"{q} = {errors[q]:.4f} exceeds {tol[q]}"
