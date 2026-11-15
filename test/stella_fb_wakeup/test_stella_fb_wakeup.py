"""stella_fb_wakeup test: gas cells that Stella feedback heats or kicks between their own sync points are woken.

A 20 Msun star 1 Myr old (ID 1, winds) sits at the centre of a 16^3 lattice of 1 Msun cells and is forced to explode at
TestSNTime0 = 1e-3. Gas is held on long steps (MaxSizeTimestep 2.5e-4), the star on short ones (MaxStellarTimestep 10 yr),
so the winds and the supernova reach gas that is not active. process_wake_ups scans only cells marked in the dirty sidecar
(WakeupDirty) and only when a rank raised the wake-up flag; TEST_WAKEUP_REQUEST_AUDIT counts, before every scan, flagged
cells off their sync point that the scan would skip. See README.md."""
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

TEST_NAME = "stella_fb_wakeup"
TEST_DIR = Path(__file__).parent
sys.path.insert(0, str(TEST_DIR.parent / "single_stars_metals"))
sys.path.insert(0, str(TEST_DIR.parent / "single_stars_invariant_checks"))
from make_single_stars_metals_ics import make_single_stars_metals_ics  # noqa: E402
from test_single_stars_invariant_checks import solar27  # noqa: E402


@pytest.fixture(scope="module")
def tables():
    if not os.environ.get("SUNDIALS_DIR"):
        pytest.skip("set SUNDIALS_DIR (SGCHEM builds against SUNDIALS 6.x or 7.x)")
    t = get_stellar_evolution_tables()
    get_cooling_tables(str(TEST_DIR))
    build_gizmo_for_test(TEST_NAME)
    return t


def test_feedback_wakeups_are_processed(tables):
    """bar (written before the run): the forced supernova of star 1 is delivered, cells are woken, every flagged cell off its
    sync point is in the sidecar (missing 0) and no scan is skipped with requests pending (gate_shut 0)"""
    d = TEST_DIR / "run"
    if d.exists():
        shutil.rmtree(d)
    (d / "output").mkdir(parents=True)
    T = tables["StellarTablesFile"]
    ic = d / "output" / "snapshot_000.hdf5"
    L = make_single_stars_metals_ics(str(ic), solar27(T), [(20.0, 20.0, 1.0e6, (0.5, 0.5, 0.5), 0)])
    with h5py.File(ic, "r+") as F:  # the forced-SN hook acts on star ID 1
        F["PartType0/ParticleIDs"][...] = F["PartType0/ParticleIDs"][()] + 1
        F["PartType4/ParticleIDs"][...] = np.array([1], dtype=np.uint32)
    for f in ("TREECOOL", "spcool_tables"):
        if (TEST_DIR / f).exists():
            os.symlink(TEST_DIR / f, d / f)
    p = (TEST_DIR / f"{TEST_NAME}.params").read_text()
    p = p.replace("StellarTablesFile           SET_BY_TEST", f"StellarTablesFile           {T}")
    p = p.replace("SGChemTablesPath            SET_BY_TEST", f"SGChemTablesPath            {os.environ.get('SGCHEM_TABLES_DIR', os.path.dirname(T))}")
    p = re.sub(r"^BoxSize .*$", f"BoxSize                     {L}", p, flags=re.M)
    (d / f"{TEST_NAME}.params").write_text(p)
    out = d / f"test_{TEST_NAME}.out"
    with open(out, "w") as f:
        subprocess.run(mpi_launch_command(2) + [str(TEST_DIR / "GIZMO"), f"{TEST_NAME}.params", "2", "0"], cwd=d, stdout=f,
                       stderr=subprocess.STDOUT, env={**os.environ, "OMP_NUM_THREADS": "1"}, check=True)
    log = out.read_text()
    events = d / "output" / "single_star_events.txt"
    rows = np.atleast_2d(np.loadtxt(events)) if events.exists() else np.zeros((0, 12))
    woken = sum(int(n) for n in re.findall(r"^(\d+) particles activated \(in wakeup check\)", log, re.M))
    audit = re.findall(r"^WAKEUP_REQUEST_AUDIT pending (\d+) missing (\d+) gate_shut (\d+)$", log, re.M)
    assert audit, "no audit line"
    pending, missing, gate = (int(x) for x in audit[-1])
    print(f"woken {woken}; requests off their sync point {pending}, missing from the sidecar {missing}, behind a shut gate {gate}")
    assert any(int(r[1]) == 1 and int(r[2]) == 1 for r in rows), "the forced supernova of star 1 was not delivered"
    assert woken > 0 and pending > 0
    assert missing == 0 and gate == 0
