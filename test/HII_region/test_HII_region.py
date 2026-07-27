"""HII region test: 10 Msun ZAMS star in a uniform gas box with mean density 100 H/cc (20 pc).

Baseline of the HII_region test family and the source of truth for the test *procedure*
(see hii_region_procedure.py). Checks that an ionization front forms and that the
photoionized gas settles at a physical HII-region temperature (~1e4 K). The subcycled
variants live in ../HII_region_subcycle and the ionizing-band-only variant in
../HII_region_simple; both import the procedure from here."""

import pytest
from pathlib import Path
from gizmo.test import default_omp_threads, default_mpi_ranks
import hii_region_procedure as hii

TEST_NAME = "HII_region"
TEST_DIR = Path(__file__).parent


@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(),))
@pytest.mark.parametrize("num_omp_threads", (default_omp_threads(),))
def test_HII_region(num_mpi_ranks, num_omp_threads):
    profiles, _ = hii.run_variant(TEST_NAME, TEST_DIR, num_mpi_ranks, num_omp_threads)
    hii.assert_hii_temperature(profiles)
