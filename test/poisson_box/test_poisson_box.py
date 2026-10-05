"""Poisson box: hydro + MHD + cooling on Poisson-random cell positions.

A periodic 100 pc box of 50^3 gas cells at uniformly random positions (no lattice, no
glass), at T ~ 1e4 K and mean n_H = 1 cm^-3, with a weak uniform magnetic field. A random
point set is the least regular arrangement the neighbour search, the kernel-radius
iteration and the gradient/MHD estimators can be handed: local cell counts fluctuate by
O(1) on the kernel scale, so every kernel radius must converge from a poor initial guess
and every cell sees a different neighbour geometry. The run is a few all-active steps.

Checks:
  - the run reaches its final time;
  - total gas mass is conserved to round-off (MFM: no mass flux);
  - every cell has a finite, positive density and kernel radius;
  - the magnetic energy stays at its initial value to 3e-4 (a uniform field in a
    medium at rest and uniform pressure has no source; growth means a divergence or
    gradient error).
The IC is generated locally by make_poisson_box_ics.py (50^3 by default).
"""

import importlib.util
import os

import h5py
import numpy as np
import pytest

from gizmo.test import (
    assert_final_time,
    build_and_run_test,
    default_mpi_ranks,
    default_omp_threads,
    get_cooling_tables,
    get_final_snapshot,
)

TEST_NAME = "poisson_box"
N_PER_SIDE = 50

# Calibrated from a recorded run; the bound is 10x the measured value.
MAGNETIC_ENERGY_TOL = 3e-4  # measured 3.0e-5 (np=2, two steps)


def _ensure_ic():
    test_dir = os.path.dirname(os.path.abspath(__file__))
    ic_path = os.path.join(test_dir, f"poisson_box_{N_PER_SIDE}_ics.hdf5")
    if os.path.isfile(ic_path):
        # reuse an existing file only if it is this test's IC (N, box, mean density n_H = 1)
        with h5py.File(ic_path, "r") as F:
            n = int(F["Header"].attrs["NumPart_Total"][0])
            box = float(F["Header"].attrs["BoxSize"])
            mtot = float(np.sum(F["PartType0/Masses"][:]))
        if n == N_PER_SIDE**3 and box == 100.0 and abs(mtot / 3.26e4 - 1) < 0.01:
            return
    spec = importlib.util.spec_from_file_location(
        "make_poisson_box_ics", os.path.join(test_dir, "make_poisson_box_ics.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.make_poisson_box_ics(ic_path, N_per_side=N_PER_SIDE)


def _load(snapfile):
    with h5py.File(snapfile, "r") as F:
        g = F["PartType0"]
        out = {k: g[k][:] for k in ("Masses", "Density", "KernelMaxRadius", "MagneticField")
               if k in g}
        out["BoxSize"] = float(F["Header"].attrs["BoxSize"])
    return out


@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(2),))
@pytest.mark.parametrize("num_omp_threads", (default_omp_threads(),))
def test_poisson_box(num_mpi_ranks, num_omp_threads):
    _ensure_ic()
    get_cooling_tables(f"test/{TEST_NAME}")
    build_and_run_test(TEST_NAME, num_mpi_ranks, num_omp_threads)
    final_snap = get_final_snapshot(TEST_NAME)
    assert_final_time(final_snap, TEST_NAME)
    first_snap = sorted(os.path.join(os.path.dirname(final_snap), f)
                        for f in os.listdir(os.path.dirname(final_snap)) if f.startswith("snapshot_"))[0]

    s0, s1 = _load(first_snap), _load(final_snap)
    m, rho, h = s1["Masses"], s1["Density"], s1["KernelMaxRadius"]

    dm = abs(m.sum() - s0["Masses"].sum()) / s0["Masses"].sum()
    assert dm < 1e-10, f"gas mass not conserved: relative change {dm:.3e}"

    assert np.all(np.isfinite(rho)) and np.all(rho > 0), "non-finite or non-positive density"
    assert np.all(np.isfinite(h)) and np.all(h > 0), "non-finite or non-positive kernel radius"

    assert final_snap != first_snap, "only one snapshot was written: nothing to compare"
    rho_mean = m.sum() / s1["BoxSize"] ** 3
    B0 = np.sum(np.sum(s0["MagneticField"] ** 2, axis=1) * s0["Masses"] / s0["Density"])
    B1 = np.sum(np.sum(s1["MagneticField"] ** 2, axis=1) * m / rho)
    dEB = abs(B1 / B0 - 1)
    print(f"[poisson_box] N={len(m)} magnetic energy change {dEB:.3e}, rho std/mean {np.std(rho) / rho_mean:.3f}")

    assert dEB < MAGNETIC_ENERGY_TOL, f"magnetic energy changed by {dEB:.3e} from a uniform field"
