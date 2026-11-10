"""ledger_step_breakdown: OUTPUT_MASS_BALANCE_BREAKDOWN on the box of test/ledger_step_checkpoints with metal diffusion: more
checkpoints (feedback, gas collection, diffusion apply), the metals of the diffusive exchange not yet applied in M_Z_ledger,
diff_unpack_breakdown.txt and MCBAL lines per element.

Bars: post_feedback and post_collect_gas lines appear (the file logs only checkpoints where a number changed; the first version asked
for post_diff_unpack too, which a conservative exchange can never write, so that tag is checked on the MCBAL lines, which print at
every call); no change of M_bary or M_Z_ledger between two checkpoints above 1e-12; the exchange is not empty and its pair fluxes sum
to below 1e-12 of the gas metals in a hydro step, the floor adds nothing; the mass per element of the MCBAL lines moves by less than
1e-5 (the print precision) across post_diff_unpack.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
from gizmo.test import build_gizmo_for_test, get_stellar_evolution_tables, get_cooling_tables, mpi_launch_command

TEST_NAME = "ledger_step_breakdown"
TEST_DIR = Path(__file__).parent
sys.path.insert(0, str(TEST_DIR.parent / "treecol_pixels"))
sys.path.insert(0, str(TEST_DIR.parent / "ledger_mass_metals"))
from make_treecol_pixels_ics import make_treecol_pixels_ics, lattice  # noqa: E402
from test_ledger_mass_metals import star_age, BIRTHS, L, N_SIDE  # noqa: E402
from stella_table_bands import Tables  # noqa: E402


@pytest.fixture(scope="module")
def out():
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
    with open(d / "run.out", "w") as o:
        subprocess.run(mpi_launch_command(2, 1) + [str(TEST_DIR / "GIZMO"), str(d / "run.params"), "2", "0"], cwd=TEST_DIR,
                       stdout=o, stderr=subprocess.STDOUT, check=True)
    return d


def test_checkpoints_close(out):
    rows = [l.split() for l in (out / "mass_balance_steps.txt").read_text().splitlines() if l and not l.startswith("#")]
    tags = {r[1] for r in rows}
    print(sorted(tags))
    assert {"post_feedback", "post_collect_gas"} <= tags
    mb, dmb, zl, dzl = (np.array([float(r[k]) for r in rows]) for k in (3, 4, 9, 10))
    print(f"max |dM_bary|/M {np.max(np.abs(dmb)) / mb[0]:.2e}, max |dM_Z_ledger|/M_Z {np.max(np.abs(dzl)) / abs(zl[0]):.2e}")
    assert np.max(np.abs(dmb)) <= 1e-12 * mb[0]
    assert np.max(np.abs(dzl)) <= 1e-12 * abs(zl[0])


def test_diffusion_breakdown(out):
    t, pair, clamp, gross = np.loadtxt(out / "diff_unpack_breakdown.txt", unpack=True)
    mz = float((out / "mass_balance_steps.txt").read_text().splitlines()[1].split()[5])
    print(f"{len(t)} hydro steps, max |pair sum|/M_Z {np.max(np.abs(pair)) / mz:.2e}, max clamp/M_Z {np.max(np.abs(clamp)) / mz:.2e}")
    print(f"max gross exchange {gross.max():.3e}")
    assert gross.max() > 1e4 * np.max(np.abs(pair)) and gross.max() > 0
    assert np.max(np.abs(pair)) <= 1e-12 * mz
    assert np.all(clamp == 0)


def test_mcbal_per_element(out):
    lines = [l.split() for l in (out / "run.out").read_text().splitlines() if l.startswith("MCBAL") and "per-element:" in l]
    assert lines, "no MCBAL per-element lines"
    pre = np.array([[float(x) for x in l[3:]] for l in lines if l[1] == "pre_diff_unpack"])
    post = np.array([[float(x) for x in l[3:]] for l in lines if l[1] == "post_diff_unpack"])
    assert len(pre) == len(post) and len(pre) > 0
    assert np.max(np.abs(post / pre - 1)) < 1e-5
