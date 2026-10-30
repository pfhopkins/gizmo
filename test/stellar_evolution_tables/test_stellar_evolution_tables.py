"""stellar_evolution_tables test: start-up check of the externally installed single-star tables (GALSF_SINGLE_STARS_EVOLUTION_TABLES).
The tables are installed once by scripts/get_stella_tables.sh (via the harness); StellarTablesFile names the main table,
the wind table is read from the same directory; GIZMO checks the content hashes of both at start-up.
  1. the installed files: GIZMO reports both ok and runs to TimeMax;
  2. a directory without tables: GIZMO stops at start-up, naming the file and the install script;
  3. a file with the right name but other content: same, naming its hash;
  4. the lookups (node-shared tables, standalone build in unit/): exact at the grid nodes, closure of the ejecta,
     lifetime monotone, agreement with the reference implementation to 1e-12, same on both ranks.
"""
import subprocess
import os
from pathlib import Path
import pytest
from gizmo.test import build_gizmo_for_test, run_test, get_stellar_evolution_tables, default_mpi_ranks, mpi_launch_command

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


def test_lookups(built):
    """the lookups, evaluated on a fixed grid of 64 masses x 8 metallicities x 7 ages (on, between and off the grid
    nodes, past the end of life) by a standalone build (unit/), on 2 ranks:
      - at the grid nodes the 2D tables are returned exactly (lifetime, remnant mass, PMS time, SN and AGB
        element ejecta, compared with the HDF5 datasets);
      - closure at the nodes: pre-SN mass (HDF5) = remnant + sum of the SN element ejecta (exploding stars) or of the AGB
        element ejecta (white dwarfs), to 1e-12 relative;
      - lifetime decreases with mass above 2 Msun at every metallicity;
      - every value agrees with the reference implementation to 1e-12 relative (a stored sample of 3719 values; the
        two codes interpolate alike, so on one machine they agree bitwise; across compilers only rounding differs);
      - both ranks get the same values;
      - the dead white-dwarf-progenitor fraction that normalises the Type Ia hazard (8 metallicities x 40 ages, past both ends of its
        table) agrees with the reference implementation to 1e-12 relative (all 320 values), lies in [0, 1], never falls with age and
        is the same on both ranks."""
    import h5py
    import numpy as np
    unit = TEST_DIR / "unit"
    exe, out = unit / "lookup_grid", unit / "lookups"
    src = TEST_DIR.parent.parent / "src" / "galaxy_sf"
    subprocess.run(["bash", str(unit / "build_lookup_grid.sh"), str(src / "stellar_evolution_tables.cc"), str(src / "stellar_evolution_tables.h"),
                    str(exe), "-DGALSF_SINGLE_STARS_EVOLUTION_TABLES", "-DGALSF_SINGLE_STARS_TYPE_IA", "-DLOAD_TABLES=stellar_evolution_tables_load"], check=True)
    subprocess.run(mpi_launch_command(2) + [str(exe), built["StellarTablesFile"], str(out)], check=True, capture_output=True)
    v = np.fromfile(f"{out}.0")
    assert np.fromfile(f"{out}.1").tobytes() == v.tobytes()
    ref = np.loadtxt(unit / "lookup_reference.txt")
    idx, val = ref[:, 0].astype(int), ref[:, 1]
    assert len(v) == 226843
    assert np.all(np.abs(v[idx] - val) <= 1e-12 * np.abs(val) + 1e-300)
    with h5py.File(built["StellarTablesFile"], "r") as F:
        NZ, NM, NE = len(F["log_Z"]), len(F["M_init"]), 27
        nodes = np.fromfile(f"{out}.nodes.0").reshape(NZ, NM, 3 + 2 * NE)
        for j, name in enumerate(["lifetime_yr", "remnant_mass", "t_PMS"]):
            assert np.array_equal(nodes[:, :, j], F[name][()].reshape(NZ, NM)), name
        assert np.array_equal(nodes[:, :, 3:3 + NE], F["elem_ej_SN_mass"][()].reshape(NZ, NM, NE))
        assert np.array_equal(nodes[:, :, 3 + NE:], F["elem_ej_AGB_mass"][()].reshape(NZ, NM, NE))
        wd, M, pre = F["remnant_type"][()].reshape(NZ, NM) == 0, F["M_init"][()], F["M_preSN"][()].reshape(NZ, NM)
    ej = np.where(wd, nodes[:, :, 3 + NE:].sum(-1), nodes[:, :, 3:3 + NE].sum(-1))
    assert np.all(np.abs(pre - nodes[:, :, 1] - ej) <= 1e-12 * pre)
    assert np.all(np.diff(nodes[:, M >= 2, 0], axis=1) < 0)
    ia, ref_ia = np.fromfile(f"{out}.ia.0"), np.loadtxt(unit / "lookup_reference_ia.txt")
    assert np.fromfile(f"{out}.ia.1").tobytes() == ia.tobytes()
    assert ia.shape == ref_ia.shape and np.all(np.abs(ia - ref_ia) <= 1e-12 * np.abs(ref_ia) + 1e-300)
    assert ia.min() >= 0 and ia.max() <= 1 and np.all(np.diff(ia.reshape(8, 40), axis=1) >= 0)
    bands, ref_bands = np.fromfile(f"{out}.bands.0"), np.loadtxt(unit / "lookup_reference_bands.txt")
    assert np.fromfile(f"{out}.bands.1").tobytes() == bands.tobytes() and len(bands) == 8 * 64 * 7 * 5
    i_bands = ref_bands[:, 0].astype(int)
    assert np.all(np.abs(bands[i_bands] - ref_bands[:, 1]) <= 1e-12 * np.abs(ref_bands[:, 1]) + 1e-300)
    bol, ref_bol = np.fromfile(f"{out}.bol.0"), np.loadtxt(unit / "lookup_reference_bol.txt")
    assert np.fromfile(f"{out}.bol.1").tobytes() == bol.tobytes() and len(bol) == 8 * 64 * 7 * 1
    i_bol = ref_bol[:, 0].astype(int)
    assert np.all(np.abs(bol[i_bol] - ref_bol[:, 1]) <= 1e-12 * np.abs(ref_bol[:, 1]) + 1e-300)
