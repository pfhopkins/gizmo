"""TOV star in its own tabulated static metric with general-relativistic hydrodynamics (GENERAL_RELATIVITY, GR_METRIC < 0; Lupi 2023).

A polytrope (K = 1, Gamma = 2, central rest-mass density 0.129285) in hydrostatic equilibrium in the metric of the
tabulated TOV solution (areal radius). Run with the HLL solver for both meshless methods (MFM and MFV) for about 7
dynamical times: the star must stay in equilibrium, i.e. the central density stays within about 8% of the tabulated value
(its first dip, near t ~ 3, is the largest) and the final density profile matches the TOV solution.

Downloaded: grhd_tov_ics.hdf5 (31103 equal-mass cells on a cubic lattice, moved radially to the TOV rest-mass profile) and
grhd_tov_exact.txt, the TOV solution: header "# N log", then rows r, M(r), -g_tt, rho0(r), P(r). GIZMO reads it as the
tabulated metric (GR_MetricTableFile); the test compares the density with its rho0 column.
"""

import pytest
import numpy as np
from glob import glob
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

TEST_NAME = "grhd_tov"
R_CORE = 0.1      # the central density is the median over r < R_CORE
R_COMPARED = 0.7  # the profile is compared inside this radius (the low-density surface layer is poorly resolved)
PROFILE_BINS = np.arange(0.0, R_COMPARED + 0.05, 0.1)

# the meshless methods run by this test, and their bounds: the central density over the table's central density at every
# snapshot, the median |log10 rho0/rho0_TOV| of the final profile over r < R_COMPARED, and the same median in the worst
# radial bin of width 0.1 (measured at 2 and 4 ranks alike: MFM rho_c/rho_TOV(0) 0.932-1.005, 0.0054 dex, worst bin 0.010 dex;
# MFV 0.922-1.007, 0.0144 dex, 0.023 dex)
VARIANTS = (("HYDRO_MESHLESS_FINITE_MASS",), ("HYDRO_MESHLESS_FINITE_VOLUME",))
TOLERANCE = {"HYDRO_MESHLESS_FINITE_MASS": {"rho_c_range": (0.90, 1.03), "profile_dex": 0.008, "worst_bin_dex": 0.015},
             "HYDRO_MESHLESS_FINITE_VOLUME": {"rho_c_range": (0.89, 1.03), "profile_dex": 0.022, "worst_bin_dex": 0.035}}


@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(),))
@pytest.mark.parametrize("num_omp_threads", (default_omp_threads(),))
@pytest.mark.parametrize("extra_config_flags", VARIANTS, ids=["MFM", "MFV"])
def test_grhd_tov(num_mpi_ranks, num_omp_threads, extra_config_flags):
    build_and_run_test(TEST_NAME, num_mpi_ranks, num_omp_threads, extra_config_flags)
    final_snap = get_final_snapshot(TEST_NAME, extra_config_flags)
    assert_final_time(final_snap, TEST_NAME)

    table = np.loadtxt(f"test/{TEST_NAME}/{TEST_NAME}_exact.txt", comments="#")
    r_tab, rho_tab = table[:, 0], table[:, 3]
    outdir = variant_output_dir(TEST_NAME, extra_config_flags)
    times, rho_c = [], []
    for snap in sorted(glob(path.join(outdir, "snapshot_*.hdf5"))):
        with h5py.File(snap, "r") as F:
            r = np.linalg.norm(F["PartType0/Coordinates"][:], axis=1)
            rho0 = F["PartType0/RestMassDensity"][:]
            times.append(F["Header"].attrs["Time"])
        rho_c.append(np.median(rho0[r < R_CORE]) / rho_tab[0])
    times, rho_c = np.array(times), np.array(rho_c)  # r and rho0 are now those of the final snapshot

    inner = r < R_COMPARED
    r_in, dex = r[inner], np.abs(np.log10(rho0[inner] / np.interp(r[inner], r_tab, rho_tab)))
    profile_dex = np.median(dex)
    bins = [dex[(r_in >= a) & (r_in < b)] for a, b in zip(PROFILE_BINS[:-1], PROFILE_BINS[1:])]
    assert all(b.size > 0 and np.all(np.isfinite(b)) for b in bins), "a radial bin of the compared profile is empty or not finite"
    worst_bin_dex = max(np.median(b) for b in bins)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(times, rho_c, color="blue", linewidth=1.5)
    axes[0].axhline(1.0, color="grey", linestyle=":")
    axes[0].set_xlabel(r"$t$")
    axes[0].set_ylabel(r"$\rho_{0,c}(t)/\rho_{0,c}^{\rm TOV}$")
    axes[1].plot(r_tab, rho_tab, color="lightgrey", linewidth=4, label="TOV")
    axes[1].plot(r, rho0, ",", color="blue", label=extra_config_flags[0])
    axes[1].set_xlim(0, 1.1)
    axes[1].set_xlabel(r"$r$")
    axes[1].set_ylabel(r"$\rho_0$")
    axes[1].legend(frameon=False)
    fig.savefig(path.join(outdir, f"{TEST_NAME}.pdf"), bbox_inches="tight")
    plt.close(fig)

    tol = TOLERANCE[extra_config_flags[0]]
    lo, hi = tol["rho_c_range"]
    print(f"{extra_config_flags[0]}: rho_c/rho_TOV(0) in [{rho_c.min():.3f}, {rho_c.max():.3f}] (allowed [{lo}, {hi}]); final profile "
          f"{profile_dex:.4f} dex (tolerance {tol['profile_dex']}), worst bin {worst_bin_dex:.4f} dex (tolerance {tol['worst_bin_dex']})")
    assert len(times) > 10, "expected the snapshot series of the whole run"
    assert lo < rho_c.min() and rho_c.max() < hi, f"central density left [{lo}, {hi}]: [{rho_c.min():.3f}, {rho_c.max():.3f}]"
    assert profile_dex < tol["profile_dex"], f"final profile error {profile_dex:.4f} dex exceeds {tol['profile_dex']}"
    assert worst_bin_dex < tol["worst_bin_dex"], f"worst radial bin error {worst_bin_dex:.4f} dex exceeds {tol['worst_bin_dex']}"
