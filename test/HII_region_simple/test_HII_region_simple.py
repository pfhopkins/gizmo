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
def test_HII_region_simple(num_mpi_ranks, num_omp_threads):
    profiles, _, _ = hii.run_variant(TEST_NAME, TEST_DIR, num_mpi_ranks, num_omp_threads)
    hii.assert_hii_temperature(profiles)
