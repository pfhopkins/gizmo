"""Toth MHD shock tube test (Toth 2000; Hopkins & Raives 2016)

Colliding supersonic flows in an oblique field: two fast shocks, two slow shocks and a contact.
The 2D run is binned onto x and compared with a converged 1D Athena++ solution,
test/toth/toth_exact.txt (see test/mhd_reference_athena/). See README.md for the setup, the
comparison window and what each assertion is sensitive to.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

from gizmo.test import (build_and_run_test, assert_final_time, assert_energy_conserved,
                        default_mpi_ranks, default_omp_threads, get_final_snapshot)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import mhd_1d_reference as ref1d  # noqa: E402

# Regional L1 over the comparison window, normalised by the reference field's range there.
# A 2nd-order grid code (Athena++ at the run's 896 cells across x) scores
# rho 0.007, vx 0.003, vy 0.011, By 0.008. vx is small because its range (20) is dominated by
# the two fast shocks; vy carries the weak slow-shock / rotational structure.
L1_TOL = {  # PROVISIONAL: untuned, not yet run
    "rho": 0.03,
    "vx": 0.02,
    "vy": 0.05,
    "By": 0.03,
}

# Constant states between waves: (label, field, x inside the plateau). Reference values:
# 2.680, 3.839, 0.357, 4.038, 3.748, -0.388, 5.427. The plateau between the two
# slow shocks is narrow (~0.11, ~12 bins) and straddles the contact, across which vy and By are
# continuous; the density sub-plateaus either side of the contact are too narrow to test.
PLATEAUS = (
    ("left-post-fast-shock", "rho", 1.80),
    ("left-post-fast-shock", "By", 1.80),
    ("between-slow-shocks", "vy", 2.025),
    ("between-slow-shocks", "By", 2.025),
    ("right-post-fast-shock", "rho", 2.25),
    ("right-post-fast-shock", "vy", 2.25),
    ("right-post-fast-shock", "By", 2.25),
)
# Relative error of the binned plateau mean. The 896-cell grid solution is within 0.3% of the
# reference on every plateau (worst: between-slow-shocks vy).
PLATEAU_TOL = 0.02  # PROVISIONAL: untuned, not yet run

# |divB| h/|B| over the comparison window, h = (m/rho)^(1/2). OpenGadget3 asserts rms < 0.05 on
# Brio-Wu; same bounds here.
DIVB_RMS_TOL = 0.05  # PROVISIONAL: untuned, not yet run
DIVB_P99_TOL = 0.25  # PROVISIONAL: untuned, not yet run

# Closed periodic box: total (thermal + kinetic + magnetic) energy is conserved up to the
# non-conservative divB source terms and Dedner damping.
ENERGY_TOL = 0.01  # PROVISIONAL: untuned, not yet run


# Known failure (measured 2026-11-05 against the Athena++ reference; details in README.md): B_y behind
# the fast shocks is +1.4% (left) and +4.5% (right), the v_y plateaus +11-12%. The excess is created as
# the fast shocks form (present at t=0.02, constant after) and does not converge (identical at 2x
# resolution). It is the Powell 8-wave induction source, DtB -= divB * v (hydro/hydro_toplevel.cc),
# which is not conservative: at the fast shocks divB spikes to 0.05-0.1 |B|/h and multiplies
# particle-scale v_y spikes of +-1, and the integral of -divB v_y along the gas path through each
# shock reproduces the observed excess to 80-95%. This is what Toth (2000) built the problem to
# expose. The code is Galilean invariant (boosts by vx+10 and vy+5 reproduce the plateaus to 4
# digits), so deleting or smoothing the term is not a fix. MHD_CONSTRAINED_GRADIENT halves divB and
# brings B_y within 1.7% but still misses the v_y plateaus by 5%. Strict xfail so an improvement is noticed.
TOTH_XFAIL = pytest.mark.xfail(strict=True, reason="Powell 8-wave induction source at the strong fast shocks "
                               "(Toth 2000): B_y +1.4/+4.5% and v_y plateaus +11-12% vs Athena++; see README")


@TOTH_XFAIL
@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(),))
@pytest.mark.parametrize("num_omp_threads", (default_omp_threads(),))
def test_toth(num_mpi_ranks, num_omp_threads):
    test_name = "toth"
    build_and_run_test(test_name, num_mpi_ranks, num_omp_threads)

    final_snap = get_final_snapshot(test_name)
    assert_final_time(final_snap, test_name)

    problem = ref1d.PROBLEMS[test_name]
    ref = ref1d.load_reference(test_name)
    snap = ref1d.read_snapshot(final_snap, problem["gamma"])
    cmp = ref1d.compare_snapshot(snap, ref, plateaus=PLATEAUS)
    ref1d.plot_comparison(snap, ref, cmp, f"test/{test_name}")

    print(f"[{test_name}] window x in [{cmp['window'][0]:.3f}, {cmp['window'][1]:.3f}], bin {cmp['dx_bin']:.4g}")
    print(f"[{test_name}] L1/range: " + " ".join(f"{f}={v:.4f}" for f, v in cmp["L1"].items()))
    for key, (sim, val, err, n) in cmp["plateaus"].items():
        print(f"[{test_name}] plateau {key}: GIZMO {sim:.5g} reference {val:.5g} rel err {err:.3g} ({n} bins)")
    if "divB_rms" in cmp:
        print(f"[{test_name}] |divB| h/|B|: rms {cmp['divB_rms']:.4g} p99 {cmp['divB_p99']:.4g} "
              f"(whole box rms {cmp['divB_rms_global']:.4g})")

    for f in ref1d.COMPARED_FIELDS:
        assert np.all(np.isfinite(snap[f])), f"non-finite {f}"
    assert np.all(snap["rho"] > 0), "non-positive density"
    # MagneticField must be in the internal units (magnetic pressure B^2/2) to compare with the
    # reference and for the magnetic energy below
    if "B_unit_ratio" in snap:
        assert abs(snap["B_unit_ratio"] - 1) < 1e-6, f"snapshot B unit is {snap['B_unit_ratio']} x internal"
    assert cmp["empty_bin_fraction"] < 0.02, (  # PROVISIONAL: untuned, not yet run
        f"{cmp['empty_bins_in_window']} empty x-bins in the window ({cmp['empty_bin_fraction']:.1%}): "
        "particles stretched past the bin width; L1 skips empty bins but too many means the binning is wrong")

    for f, tol in L1_TOL.items():
        assert cmp["L1"][f] < tol, f"{f}: L1/range {cmp['L1'][f]:.4f} vs reference exceeds {tol}"
    for key, (sim, val, err, n) in cmp["plateaus"].items():
        assert n >= 4, f"plateau {key} spans only {n} bins after trimming; move its anchor"
        assert err < PLATEAU_TOL, f"plateau {key}: GIZMO {sim:.5g} vs reference {val:.5g} (rel err {err:.3g})"
    assert "divB_rms" in cmp, "DivergenceOfMagneticField missing; OUTPUT_BFIELD_DIVCLEAN_INFO not in Config.sh?"
    assert cmp["divB_rms"] < DIVB_RMS_TOL, f"rms |divB| h/|B| = {cmp['divB_rms']:.4g}"
    assert cmp["divB_p99"] < DIVB_P99_TOL, f"99th percentile |divB| h/|B| = {cmp['divB_p99']:.4g}"

    assert_energy_conserved(test_name, tol=ENERGY_TOL, include_magnetic=True)
