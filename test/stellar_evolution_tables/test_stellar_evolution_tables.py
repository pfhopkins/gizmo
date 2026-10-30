"""stellar_evolution_tables test: start-up check of the externally installed single-star tables (GALSF_SINGLE_STARS_EVOLUTION_TABLES).
The tables are installed once by scripts/get_stella_tables.sh (via the harness); StellarTablesFile names the main table,
the wind table is read from the same directory; GIZMO checks the content hashes of both at start-up.
  1. the installed files: GIZMO reports both ok and runs to TimeMax;
  2. a directory without tables: GIZMO stops at start-up, naming the file and the install script;
  3. a file with the right name but other content: same, naming its hash.
"""
import os
from pathlib import Path
import pytest
from gizmo.test import build_gizmo_for_test, run_test, get_stellar_evolution_tables, default_mpi_ranks

TEST_NAME = "stellar_evolution_tables"
TEST_DIR = Path(__file__).parent


@pytest.fixture(scope="module")
def built():
    tables = get_stellar_evolution_tables()
    build_gizmo_for_test(TEST_NAME)
    ic = TEST_DIR / f"{TEST_NAME}_ics.hdf5"
    if not ic.exists():
        os.chdir(TEST_DIR)
        from make_stellar_evolution_tables_ics import make_stellar_evolution_tables_ics
        make_stellar_evolution_tables_ics(str(ic))
        os.chdir("../../")
    return tables


def run(overrides, allow_fail):
    os.chdir(TEST_DIR)
    try:
        run_test(TEST_NAME, num_mpi_ranks=min(2, default_mpi_ranks()), param_overrides=overrides, allow_nonzero_exit=allow_fail)
        log = Path(f"test_{TEST_NAME}.out").read_text() + Path(f"test_{TEST_NAME}.err").read_text()
    finally:
        os.chdir("../../")
    return log


def test_installed_tables_pass(built):
    log = run(built, allow_fail=False)
    assert log.count(" ok\n") >= 2 and "stellar_wind_cumulative.hdf5 ok" in log


def test_missing_table_stops(built):
    log = run({"StellarTablesFile": "/nonexistent/stellar_tables_unified.hdf5"}, allow_fail=True)
    assert "/nonexistent/stellar_tables_unified.hdf5 cannot be read" in log
    assert "get_stella_tables.sh" in log
    assert "Final time" not in log


def test_wrong_table_stops(built, tmp_path):
    import h5py
    fake = tmp_path / "stellar_tables_unified.hdf5"
    with h5py.File(fake, "w") as F:
        F["lifetime_yr"] = [1.0, 2.0]
    log = run({"StellarTablesFile": str(fake)}, allow_fail=True)
    assert f"{fake} has hash" in log and "get_stella_tables.sh" in log
    assert "Final time" not in log
