"""single_star_record test: the compact record of IMF-sampled single stars (GALSF_SINGLE_STARS_FROM_IMF) follows its
star, and such stars are never merged. 64 stars of 0.1-50 Msun (some below the merge threshold of 0.49 Msun set by the
1 Msun gas cells) with drawn masses and birth metallicities from the initial conditions move at 20-60 km/s through a
5.3 pc periodic box (1.5-4.5 crossings by the end), so on 4 ranks they change rank many times.
  1. every snapshot (4 ranks) holds all 64 stars, each with its mass, drawn mass and birth metallicity bitwise as in the
     initial conditions (without the merge guard one star, ID 532, is merged in the first 0.1 time units);
  2. a run stopped halfway and continued from its restart files ends with the same records;
  3. 1 rank gives the same records;
  4. with GALSF_SINGLE_STARS_EVOLUTION_TABLES the record also holds the birth density: as 1 and 2, on 4 ranks.
"""
import os
import shutil
from pathlib import Path
import numpy as np
import h5py
import pytest
from gizmo.test import build_gizmo_for_test, run_test, mpi_launch_command, get_stellar_evolution_tables

TEST_NAME = "single_star_record"
TEST_DIR = Path(__file__).parent


@pytest.fixture(scope="module")
def built():
    build_gizmo_for_test(TEST_NAME)
    return True


def records(fname, birth_density=False):
    with h5py.File(fname, "r") as F:
        s = F["PartType4"]
        o = np.argsort(s["ParticleIDs"][()])
        out = (s["ParticleIDs"][()][o].astype(np.uint64), s["Masses"][()][o].astype(np.float64),
               s["SingleStarDrawnMass"][()][o].astype(np.float64), s["SingleStarBirthMetallicity"][()][o].astype(np.float64))
        return out + ((s["SingleStarBirthDensity"][()][o].astype(np.float64),) if birth_density else ())


def run(tag, ranks, time_max=None, restart_from=None, exe="GIZMO", extra_params=""):
    import sys
    sys.path.insert(0, str(TEST_DIR))
    from make_single_star_record_ics import make_single_star_record_ics
    d = TEST_DIR / f"run_{tag}"
    if restart_from is None and d.exists():
        shutil.rmtree(d)  # this run's own directory: no stale snapshots
    d.mkdir(exist_ok=True)
    os.chdir(d)
    try:
        make_single_star_record_ics(f"{TEST_NAME}_ics.hdf5")
        params = Path(TEST_DIR / f"{TEST_NAME}.params").read_text()
        if time_max:
            params = params.replace("TimeMax                            0.4", f"TimeMax                            {time_max}")
        Path(f"{TEST_NAME}.params").write_text(params + extra_params)
        if not Path("GIZMO").exists():
            os.symlink(TEST_DIR / exe, "GIZMO")
        if restart_from is None:
            run_test(TEST_NAME, num_mpi_ranks=ranks)
        else:  # continue the run in this directory from its restart files (start flag 1)
            import subprocess
            with open(f"test_{TEST_NAME}.out", "w") as out:
                subprocess.run(mpi_launch_command(ranks) + ["./GIZMO", f"{TEST_NAME}.params", "1"], stdout=out, stderr=subprocess.STDOUT, check=True)
        snaps = sorted(Path("output").resolve().glob("snapshot_*.hdf5"))
        ic = Path(f"{TEST_NAME}_ics.hdf5").resolve()
    finally:
        os.chdir("../../../")
    return ic, snaps


def test_records_follow_stars(built):
    ic, snaps = run("r4", 4)
    ref = records(ic)
    assert len(snaps) >= 4
    for s in snaps:
        got = records(s)
        assert len(got[0]) == len(ref[0]), s
        for a, b in zip(ref, got):
            assert a.tobytes() == b.tobytes(), s


def test_records_survive_restart(built):
    ic, _ = run("restart", 4, time_max=0.2)
    _, snaps = run("restart", 4, time_max=0.4, restart_from=True)
    ref = records(ic)
    for a, b in zip(ref, records(snaps[-1])):
        assert a.tobytes() == b.tobytes()


def test_one_rank(built):
    ic, snaps = run("r1", 1)
    for a, b in zip(records(ic), records(snaps[-1])):
        assert a.tobytes() == b.tobytes()


def test_split_threshold(built):
    """a cell splits above 2.01 x the largest gas cell of the initial conditions"""
    ic, snaps = run("split", 2, time_max=0.01)
    with h5py.File(ic, "r") as F:
        m_max = F["PartType0/Masses"][()].max()
    with h5py.File(snaps[-1], "r") as F:
        assert abs(F["Header"].attrs["Maximum_Mass_For_Cell_Split"] / (2.01 * m_max) - 1) < 1e-6


@pytest.fixture(scope="module")
def built_tables():
    tables = get_stellar_evolution_tables()
    build_gizmo_for_test(TEST_NAME, extra_config_flags=("GALSF_SINGLE_STARS_EVOLUTION_TABLES",))
    os.replace(TEST_DIR / "GIZMO", TEST_DIR / "GIZMO_tables")
    return f"StellarTablesFile                  {tables['StellarTablesFile']}\n"


def test_birth_density_follows_stars(built_tables):
    ic, snaps = run("tables", 4, exe="GIZMO_tables", extra_params=built_tables, time_max=0.2)
    _, more = run("tables", 4, time_max=0.4, restart_from=True, exe="GIZMO_tables", extra_params=built_tables)
    ref = records(ic, True)
    assert np.all(ref[4] > 0)
    for s in snaps + more:
        for a, b in zip(ref, records(s, True)):
            assert a.tobytes() == b.tobytes(), s
