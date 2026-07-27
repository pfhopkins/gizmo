"""HII_region subcycled variants: verify that TRANSPORT_SUBCYCLE (RT operator subcycling)
and TRANSPORT_SUBCYCLE + TRANSPORT_SUBCYCLE_COOLING reproduce the non-subcycled baseline.

Same physical setup as HII_region; the test *procedure* (profiles, temperature check,
subcycle comparison, plots) is imported from HII_region/hii_region_procedure.py so it
stays the source of truth."""

import sys
import pytest
from pathlib import Path
from gizmo.test import default_omp_threads, default_mpi_ranks

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "HII_region"))
import hii_region_procedure as hii  # noqa: E402

TEST_NAME = "HII_region_subcycle"
TEST_DIR = Path(__file__).parent

_baseline = {}
_all_profiles = {}
_all_ifront = {}


@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(),))
@pytest.mark.parametrize("num_omp_threads", (default_omp_threads(),))
@pytest.mark.parametrize(
    "extra_config_flags",
    [
        (),
        ("TRANSPORT_SUBCYCLE=10",),
        ("TRANSPORT_SUBCYCLE=10", "TRANSPORT_SUBCYCLE_COOLING"),
    ],
    ids=["baseline", "subcycle_rt", "subcycle_rt_cooling"],
)
def test_HII_region_subcycle(num_mpi_ranks, num_omp_threads, extra_config_flags):
    profiles, label = hii.run_variant(TEST_NAME, TEST_DIR, num_mpi_ranks, num_omp_threads, extra_config_flags)
    hii.assert_hii_temperature(profiles)

    _all_profiles[label] = profiles
    _all_ifront[label] = hii.compute_ifront_evolution(TEST_NAME, extra_config_flags)
    final_snap = hii.get_final_snapshot(TEST_NAME, extra_config_flags)
    rif = hii.compute_ifront_radius(final_snap)[1]

    if not extra_config_flags:
        _baseline["profiles"] = profiles
        _baseline["rif"] = rif
    else:
        base = _baseline.get("profiles")
        if base is None:
            pytest.skip("baseline must run first")
        # Subcycling must reproduce the baseline: interior ionization state + front
        # position (robust to the evacuated interior and the single fuzzy front bin).
        hii.assert_subcycle_matches_baseline(base, profiles, _baseline["rif"], rif)

    # After the last variant, generate the cross-variant comparison plots
    if len(_all_profiles) == 3:
        hii.make_comparison_plots(_all_profiles, TEST_DIR)
        hii.make_ifront_evolution_plot(_all_ifront, TEST_DIR)
