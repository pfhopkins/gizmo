"""HII_region_simple: minimal HII-region test with only the ionizing band and radiation
pressure disabled (RT_CHEM_PHOTOION + RT_DISABLE_RAD_PRESSURE, no OPTICAL_NIR/NUV/
PHOTOELECTRIC/INFRARED bands). Isolates the Stromgren-sphere ionization and the
photoheating/cooling balance from the reprocessed-radiation and radiation-pressure physics.

Same setup and analysis procedure as HII_region (imported from
HII_region/hii_region_procedure.py, the source of truth); only Config.sh differs."""

import sys
import pytest
from pathlib import Path
from gizmo.test import default_omp_threads, default_mpi_ranks

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "HII_region"))
import hii_region_procedure as hii  # noqa: E402

TEST_NAME = "HII_region_simple"
TEST_DIR = Path(__file__).parent


@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(),))
@pytest.mark.parametrize("num_omp_threads", (default_omp_threads(),))
# jaco_rt: the jaco model of the standard module and its coupling to the ionizing band, whose photons it solves (the RT
# kick only transports them), held to the baseline's I-front radius over time
@pytest.mark.parametrize("extra_config_flags", [(), ("JACO=starforge_legacy_RT_EUV",)], ids=["baseline", "jaco_rt"])
def test_HII_region_simple(num_mpi_ranks, num_omp_threads, extra_config_flags):
    profiles, label, evo = hii.run_variant(TEST_NAME, TEST_DIR, num_mpi_ranks, num_omp_threads, extra_config_flags)
    hii.assert_hii_temperature(profiles)
    if extra_config_flags:
        ref = hii.compute_ifront_evolution(TEST_NAME)
        if not len(ref[0]):
            pytest.skip("baseline must run first")
        hii.make_ifront_evolution_plot({"baseline": ref, label: evo}, TEST_DIR)
        hii.assert_ifront_matches_reference(evo, ref)
