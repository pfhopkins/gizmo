"""Ion-neutral ambipolar drag relaxation test (HYDRO_MULTIFLUID_IONNEUTRAL).

Two interleaved Type=0 fluids (FluidType=FLUID_DEFAULT neutrals,
FluidType=FLUID_ION ions) start with a uniform relative velocity
Δv0 = v_ion - v_neutral = 1e-4 along x. Ambipolar drag damps the
relative velocity exponentially:
    d(Δv)/dt = -γ_AD (ρ_i + ρ_n) Δv  =>  Δv(t) = Δv0 · exp(-t / t_damp)
Units chosen so t_damp = 1 in code time (see ionneutral_drag.params).

Validation:
  (1) Mean relative velocity decays exponentially with rate within ~15%
      of the analytic 1.0/code-time (small drift expected from kernel-
      density sampling and timestep integrator).
  (2) Neutrals retain B = 0 throughout (corridor skip keeps it so).
"""

import pytest
import numpy as np
from matplotlib import pyplot as plt
import h5py
from os import path, chdir
from gizmo.test import (
    build_gizmo_for_test, run_test, clean_test_outputs,
    assert_final_time, get_final_snapshot,
    default_mpi_ranks, default_omp_threads,
)

# FluidID values must match declarations/multifluid_helpers.h
FLUID_DEFAULT = 0
FLUID_ION     = 2
DV0           = 1.0e-4
T_DAMP_CODE   = 1.0   # analytic damping time with our unit choice


@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(max_ranks=2),))
@pytest.mark.parametrize("num_omp_threads", (default_omp_threads(),))
def test_ionneutral_drag(num_mpi_ranks, num_omp_threads):
    test_name = "ionneutral_drag"
    clean_test_outputs(test_name)
    build_gizmo_for_test(test_name, num_omp_threads)
    chdir(f"test/{test_name}/")

    # Generate the IC locally (no website dependency); idempotent.
    ic_file = "ionneutral_drag_ics.hdf5"
    if not path.isfile(ic_file):
        import subprocess, sys
        subprocess.check_call([sys.executable, "make_ionneutral_drag_ics.py"])

    run_test(test_name, num_mpi_ranks, num_omp_threads)
    chdir("../../")

    outputdir = f"test/{test_name}/output"
    snap_files = sorted([
        path.join(outputdir, f) for f in ["snapshot_%03d.hdf5" % i for i in range(11)]
        if path.isfile(path.join(outputdir, f))
    ])
    if len(snap_files) < 6:
        raise RuntimeError("GIZMO did not produce enough snapshots.")
    assert_final_time(get_final_snapshot(test_name), test_name)

    # ---- gather Δv(t) from each snapshot ----
    times = []
    dvs   = []
    bmax_neutral_per_snap = []
    for sf in snap_files:
        with h5py.File(sf, "r") as F:
            t = float(F["Header"].attrs["Time"])
            ft = F["PartType0/FluidType"][:]
            v  = F["PartType0/Velocities"][:, 0]
            b  = F["PartType0/MagneticField"][:]
            mask_ion = (ft == FLUID_ION)
            mask_n   = (ft == FLUID_DEFAULT)
            dv  = float(np.mean(v[mask_ion]) - np.mean(v[mask_n]))
            bmax_neutral = float(np.max(np.abs(b[mask_n])))
            times.append(t); dvs.append(dv); bmax_neutral_per_snap.append(bmax_neutral)

    times = np.asarray(times); dvs = np.asarray(dvs)
    bmax_neutral_per_snap = np.asarray(bmax_neutral_per_snap)

    # ---- analytic comparison ----
    dv_analytic = DV0 * np.exp(-times / T_DAMP_CODE)

    # Plot
    plt.figure()
    plt.semilogy(times, np.abs(dvs),       "ko-", label=r"$\Delta v$ (GIZMO)")
    plt.semilogy(times, np.abs(dv_analytic), "r--", label=r"$\Delta v_0\, e^{-t/t_{\rm damp}}$")
    plt.xlabel("t [code]"); plt.ylabel("|v_ion - v_neutral|")
    plt.legend()
    plt.savefig(f"test/{test_name}/dv_decay.png")
    plt.close()

    # ---- assertions ----
    # (1) Exponential decay fit to a linear log-Δv vs t. Skip t=0 and any
    #     snapshots that have dropped below 1e-4 of the initial signal (noise floor).
    keep = (np.abs(dvs) > 1e-4 * DV0) & (times > 0)
    if keep.sum() < 4:
        raise AssertionError(f"Too few usable snapshots for decay fit: {keep.sum()}")
    slope, _ = np.polyfit(times[keep], np.log(np.abs(dvs[keep])), 1)
    measured_rate = -slope
    analytic_rate = 1.0 / T_DAMP_CODE
    rel_err = abs(measured_rate - analytic_rate) / analytic_rate
    assert rel_err < 0.15, (
        f"Damping rate mismatch: measured={measured_rate:.3f}, analytic={analytic_rate:.3f}, "
        f"rel_err={rel_err:.3f} exceeds 0.15"
    )

    # (2) Neutrals retain B=0 (corridor skip keeps cross-fluid MHD pairs out).
    #     A small numerical floor is acceptable but should not grow appreciably.
    assert bmax_neutral_per_snap.max() < 1e-12, (
        f"Neutrals picked up nonzero B-field: max|B_neutral|={bmax_neutral_per_snap.max():.3e}"
    )
