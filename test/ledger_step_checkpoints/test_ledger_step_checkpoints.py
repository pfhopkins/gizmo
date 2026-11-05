"""ledger_step_checkpoints: OUTPUT_MASS_BALANCE_STEP_CHECKPOINTS writes mass_balance_steps.txt, the baryon and metal mass at checkpoints
inside the step (before and after the hydro and feedback solve, before and after star formation), so a mass leak is attributed to
the operation that made it. The ledger_mass_metals box: 8^3 cells of 1 Msun with SGChem and single stars from the tables, two
births at the first step, a young 8 Msun star losing wind mass, a 3 Msun star dying on the AGB; 2 ranks, 0.29 Myr.

Bars: the file exists; the first line is the first call; no change of M_bary or of M_Z_ledger (gas metals + star metals - net yields,
which the ledger rows book) between two checkpoints exceeds 1e-12 of its value (this box closes its books: the births move mass between
gas and stars only, the stars' winds and the AGB death return it).
Tags: the first version asked for all four tags in the file, which the writer cannot give: it logs a checkpoint only when a number
changed, and nothing changes mass between post_star_formation and pre_hydro_fb in this box, so pre_hydro_fb stays silent. Moved, with
that reason: the box must show exactly the three tags that change something, and the call column, which counts silent calls too, must
show all four checkpoints ran: each step is four calls, in the order pre_star_formation, post_star_formation, pre_hydro_fb, post_hydro_fb.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
from gizmo.test import build_gizmo_for_test, get_stellar_evolution_tables, get_cooling_tables, mpi_launch_command

TEST_NAME = "ledger_step_checkpoints"
TEST_DIR = Path(__file__).parent
sys.path.insert(0, str(TEST_DIR.parent / "treecol_pixels"))
sys.path.insert(0, str(TEST_DIR.parent / "ledger_mass_metals"))
from make_treecol_pixels_ics import make_treecol_pixels_ics, lattice  # noqa: E402
from test_ledger_mass_metals import star_age, BIRTHS, L, N_SIDE  # noqa: E402
from stella_table_bands import Tables  # noqa: E402


@pytest.fixture(scope="module")
def steps():
    if not os.environ.get("SUNDIALS_DIR"):
        pytest.skip("set SUNDIALS_DIR (SGCHEM builds against SUNDIALS 6.x or 7.x)")
    tables = get_stellar_evolution_tables()
    get_cooling_tables(str(TEST_DIR))
    build_gizmo_for_test(TEST_NAME)
    tab = Tables(tables["StellarTablesFile"])
    lM, lZ = np.log10(3.0), np.log10(0.0142)
    death3 = tab._i2(tab.t_pms, lM, lZ) + tab._i2(tab.life, lM, lZ)
    stars = [(8.0, star_age(tab, 8.0, 0.4), np.full(3, 0.5 * L)), (3.0, death3 - 0.1e6, np.array([0.25, 0.25, 0.75]) * L)]
    d = TEST_DIR / "output"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir()
    make_treecol_pixels_ics(str(d / "snapshot_000.hdf5"), L, lattice(N_SIDE, L), stars, hsml=2.0)
    params = (TEST_DIR / f"{TEST_NAME}.params").read_text()
    for k, v in (("StellarTablesFile", tables["StellarTablesFile"]), ("TestSingleStarBirths", BIRTHS),
                 ("SGChemTablesPath", os.environ.get("SGCHEM_TABLES_DIR", os.path.dirname(tables["StellarTablesFile"])))):
        params = params.replace(f"{k:35s}SET_BY_TEST", f"{k:35s}{v}")
    (d / "run.params").write_text(params)
    with open(d / "run.out", "w") as out:
        subprocess.run(mpi_launch_command(2, 1) + [str(TEST_DIR / "GIZMO"), str(d / "run.params"), "2", "0"], cwd=TEST_DIR,
                       stdout=out, stderr=subprocess.STDOUT, check=True)
    return d


def test_ledger_step_checkpoints(steps):
    f = steps / "mass_balance_steps.txt"
    assert f.exists(), "mass_balance_steps.txt was not written"
    rows = [l.split() for l in f.read_text().splitlines() if l and not l.startswith("#")]
    tags = {r[1] for r in rows}
    print(f"{len(rows)} logged checkpoints, tags {sorted(tags)}")
    assert tags == {"post_hydro_fb", "pre_star_formation", "post_star_formation"}
    assert all(int(r[0]) % 4 == {"pre_star_formation": 0, "post_star_formation": 1, "post_hydro_fb": 3}[r[1]] for r in rows)
    assert int(rows[0][0]) == 0
    # columns: call_num tag t M_bary dM_bary M_Z_gas dM_Z_gas M_Z_stars M_Z_netyield M_Z_ledger dM_Z_ledger
    mb, dmb, zl, dzl = (np.array([float(r[k]) for r in rows]) for k in (3, 4, 9, 10))
    print(f"max |dM_bary|/M {np.max(np.abs(dmb)) / mb[0]:.2e}, max |dM_Z_ledger|/M_Z {np.max(np.abs(dzl)) / abs(zl[0]):.2e}")
    assert np.max(np.abs(dmb)) <= 1e-12 * mb[0]
    assert np.max(np.abs(dzl)) <= 1e-12 * abs(zl[0])
