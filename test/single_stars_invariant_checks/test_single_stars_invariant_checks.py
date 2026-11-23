"""single_stars_invariant_checks test: TEST_SINGLE_STARS_PRINT_INVARIANT_CHECKS prints a FAIL line when a single-star
invariant breaks and a RAN count per check site. Stella cloud production flags; lattice and stars of single_stars_metals plus
two births by the test hook. See README.md."""
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
import h5py
import pytest
from gizmo.test import build_gizmo_for_test, get_stellar_evolution_tables, get_cooling_tables, mpi_launch_command

TEST_NAME = "single_stars_invariant_checks"
TEST_DIR = Path(__file__).parent
sys.path.insert(0, str(TEST_DIR.parent / "single_stars_metals"))
from make_single_stars_metals_ics import make_single_stars_metals_ics  # noqa: E402

FIRE_ELEMENTS = [1, 2, 3, 4, 6, 8, 10, 12, 16, 22]
EXTRA_ELEMENTS = [5, 7, 9, 11, 13, 14, 15, 17, 18, 19, 20, 21, 23, 24, 25, 26]
TAGS = ("fire_slot_deposit", "fire_slot_invariant", "mass_sync", "fb_booked_deposited", "lifetime_consistent",
        "donor_mass_floor", "donor_cooldown", "tree_insert", "dgr_consistent")
BIRTHS = "1101:20,3277:2"  # gas cells in two octants without stars: lattice index ix*256 + iy*16 + iz, ID = index + 1


@pytest.fixture(scope="module")
def tables():
    if not os.environ.get("SUNDIALS_DIR"):
        pytest.skip("set SUNDIALS_DIR (SGCHEM builds against SUNDIALS 6.x or 7.x)")
    t = get_stellar_evolution_tables()
    get_cooling_tables(str(TEST_DIR))
    build_gizmo_for_test(TEST_NAME)
    return t


def solar27(tables_file):
    with h5py.File(tables_file, "r") as F:
        Zt = F["Z"][()]
        X = F["X_init"][()].reshape(len(Zt), -1)[int(np.argmin(np.abs(Zt - 0.014)))].astype(float)
    return np.concatenate([[X[2:].sum()], X[FIRE_ELEMENTS], X[EXTRA_ELEMENTS]])


def event_stars(tables_file):
    """as single_stars_metals: a 20 Msun core collapse, a 3 Msun AGB death, a 1 Gyr white dwarf (Type Ia), a 35 Msun star 1 Myr old"""
    with h5py.File(tables_file, "r") as F:
        Zt, Mt = F["Z"][()], F["M_init"][()]
        iz, n2 = int(np.argmin(np.abs(Zt - 0.014))), (len(Zt), len(Mt))
        life, pms, rem = (F[k][()].reshape(n2)[iz] for k in ("lifetime_yr", "t_PMS", "remnant_mass"))
    i = {m: int(np.argmin(np.abs(Mt - m))) for m in (20.0, 3.0)}
    return [(20.0, 20.0, 1.01 * (life[i[20.0]] + pms[i[20.0]]), (0.25, 0.25, 0.25), 0, np.log10(life[i[20.0]])),
            (3.0, 3.0, 1.001 * (life[i[3.0]] + pms[i[3.0]]), (0.75, 0.75, 0.25), 0),
            (3.0, float(rem[i[3.0]]), 1.0e9, (0.25, 0.75, 0.75), 2),
            (35.0, 35.0, 1.0e6, (0.75, 0.25, 0.75), 0)]


def run(tag, tables, metals, over):
    d = TEST_DIR / f"run_{tag}"
    if d.exists():
        shutil.rmtree(d)
    (d / "output").mkdir(parents=True)
    L = make_single_stars_metals_ics(str(d / "output" / "snapshot_000.hdf5"), metals, event_stars(tables["StellarTablesFile"]))
    for f in ("TREECOOL", "spcool_tables"):
        if (TEST_DIR / f).exists():
            os.symlink(TEST_DIR / f, d / f)
    T = tables["StellarTablesFile"]
    over = {"BoxSize": L, "TestSingleStarBirths": BIRTHS, **over}
    p = (TEST_DIR / f"{TEST_NAME}.params").read_text()
    p = p.replace("StellarTablesFile           SET_BY_TEST", f"StellarTablesFile           {T}")
    p = p.replace("SGChemTablesPath            SET_BY_TEST", f"SGChemTablesPath            {os.environ.get('SGCHEM_TABLES_DIR', os.path.dirname(T))}")
    p = "\n".join([l for l in p.splitlines() if not (l.split() and l.split()[0] in over)] + [f"{k:27s} {v}" for k, v in over.items()]) + "\n"
    (d / f"{TEST_NAME}.params").write_text(p)
    out = d / f"test_{TEST_NAME}.out"
    with open(out, "w") as f:
        subprocess.run(mpi_launch_command(2) + [str(TEST_DIR / "GIZMO"), f"{TEST_NAME}.params", "2", "0"], cwd=d, stdout=f,
                       stderr=subprocess.STDOUT, env={**os.environ, "OMP_NUM_THREADS": "1"}, check=True)
    return d, out.read_text()


def counts(log):
    """per tag: largest RAN count over ranks and the FAIL lines"""
    ran, fail = {}, {}
    for m in re.finditer(r"^SINGLE_STAR_CHECK (\S+) RAN Task=\d+ n=(\d+)$", log, re.M):
        ran[m.group(1)] = max(ran.get(m.group(1), 0), int(m.group(2)))
    for m in re.finditer(r"^SINGLE_STAR_CHECK (\S+) FAIL .*$", log, re.M):
        fail.setdefault(m.group(1), []).append(m.group(0))
    return ran, fail


def test_checks_pass_in_stella_run(tables):
    """bar (written before the run): births by the hook (2 stars collect their gas), a core collapse, an AGB death, two Type Ia
    and winds, 5e-3 time units: every check site ran (RAN count > 0 for all 9 tags), none failed, and every source wrote its
    yields trace"""
    d, log = run("stella", tables, solar27(tables["StellarTablesFile"]), {"TimeMax": 5e-3})
    ran, fail = counts(log)
    print("RAN:", ran)
    assert not fail, {k: v[:3] for k, v in fail.items()}
    assert all(ran.get(t, 0) > 0 for t in TAGS), ran
    for ch in ("SN", "AGB", "Ia", "wind"):
        assert re.search(rf"^SINGLE_STAR_YIELDS ch={ch} ID=\d+ t=\S+ table( \S+){{27}} slots", log, re.M), ch


def star_identities(snap):
    """per star (ID, child, generation, wrap count, lineage key); the last two read as 0 when the file has no such block"""
    with h5py.File(snap, "r") as F:
        g = F["PartType4"]
        n = len(g["ParticleIDs"])
        cols = [g[k][()].astype(np.uint64) if k in g else np.zeros(n, dtype=np.uint64)
                for k in ("ParticleIDs", "ParticleChildIDsNumber", "ParticleIDGenerationNumber", "ParticleIDWraps", "ParticleIDLineage")]
    return [tuple(int(c[k]) for c in cols) for k in range(n)]


def test_identities_unique_past_roll_over_and_restart(tables):
    """bar (written before the run): 40 stars from one cell in one call (the 32nd would repeat the 1st star's ID, child and
    generation, and so its random streams): every star identity in the snapshot distinct; a snapshot restart (start flag 2)
    from that snapshot brings every born star back with its full identity (an initial event star may die in the restart window)"""
    births = "1101:0.3*40"  # ID:M*count: the parameter holds 255 characters
    d, log = run("ids", tables, solar27(tables["StellarTablesFile"]), {"TimeMax": 1e-4, "TimeBetSnapshot": 1e-4, "TestSingleStarBirths": births})
    snap = sorted((d / "output").glob("snapshot_*.hdf5"))[-1]
    ids = star_identities(snap)
    born = [t for t in ids if t[0] == 1101]
    assert len(born) == 40, len(born)
    assert len(set(ids)) == len(ids), [t for t in ids if ids.count(t) > 1][:4]
    # restart from that snapshot: the identities come back from the file
    d2 = TEST_DIR / "run_ids_restart"
    if d2.exists():
        shutil.rmtree(d2)
    shutil.copytree(d, d2, symlinks=True, ignore=shutil.ignore_patterns("snapshot_*.hdf5", "*.out", "*.txt"))
    shutil.copy(snap, d2 / "output" / "snapshot_000.hdf5")
    with h5py.File(d2 / "output" / "snapshot_000.hdf5", "r+") as F:  # stars born together still share their position (GKS-TI18): set them
        x = F["PartType4"]["Coordinates"]                             # apart by distinct small offsets; identities are what is checked
        x[...] = x[()] + 1.0e-3 * np.arange(len(x))[:, None] * np.array([1.0, 0.7, 0.3])
    new = {"TimeMax": 1.1e-4, "TestSingleStarBirths": "none"}  # a little further, no births
    p = (d2 / f"{TEST_NAME}.params").read_text()
    p = "\n".join([l for l in p.splitlines() if l.split()[:1] not in ([k] for k in new)] + [f"{k:27s} {v}" for k, v in new.items()]) + "\n"
    (d2 / f"{TEST_NAME}.params").write_text(p)
    with open(d2 / "restart.out", "w") as f:
        subprocess.run(mpi_launch_command(2) + [str(TEST_DIR / "GIZMO"), f"{TEST_NAME}.params", "2", "0"], cwd=d2, stdout=f,
                       stderr=subprocess.STDOUT, env={**os.environ, "OMP_NUM_THREADS": "1"}, check=True)
    after = set(star_identities(sorted((d2 / "output").glob("snapshot_*.hdf5"))[-1]))
    missing = [t for t in born if t not in after]  # full identities: (ID, child, generation) alone repeats past the roll-over
    assert not missing, missing[:4]


def test_broken_invariant_is_reported(tables):
    """bar (written before the run): gas whose C..Fe slots exceed its total Z (slot 0 set to half the sum of the 25 metals) gives
    FAIL lines of the step check fire_slot_invariant on the first feedback step (the Type Ia at once); the per-cell check
    fire_slot_invariant_cell only tests cells that held the invariant before, so it stays silent; the run goes on to the end"""
    Z = solar27(tables["StellarTablesFile"])
    Z[0] = 0.5 * Z[0]
    d, log = run("broken", tables, Z, {"TimeMax": 2.5e-4, "TimeBetSnapshot": 2.5e-4})
    ran, fail = counts(log)
    assert "fire_slot_invariant" in fail and "fire_slot_invariant_cell" not in fail, list(fail)
    assert re.search(r"fire_slot_invariant FAIL Task=\d+ t=\S+ \d+ of \d+ cells", log)
    assert len(sorted((d / "output").glob("snapshot_*.hdf5"))) >= 2
