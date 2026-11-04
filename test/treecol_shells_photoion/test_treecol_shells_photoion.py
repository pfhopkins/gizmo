"""treecol_shells_photoion: StarBench D-type expansion (Bisbas et al. 2015) with the TreeRay photoionisation
(GALSF_RESOLVEDISM_TREECOL_PIXELS_RADIAL_SHELLS_PHOTOION). Photon ledger and R_IF of the coarsest rung against stella_dev
(83b7ed68 net 5 one band NSIDE 2: -25.0 % to 2.5 Myr, ledger 0.9-1.3 Q). See README.md."""
import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path
import pytest
from gizmo.test import build_gizmo_for_test, get_stellar_evolution_tables, mpi_launch_command

TEST_NAME = "treecol_shells_photoion"
TEST_DIR = Path(__file__).parent
sys.path.insert(0, str(TEST_DIR))
from make_treecol_shells_photoion_ics import make_ics, RUNGS  # noqa: E402
from starbench import grade, photon_ledger  # noqa: E402

STELLA_DEV_M10 = -0.250  # stella_dev, net 5, one band, NSIDE 2; D-type median R_IF / R_StarBench - 1, t < 2.5 Myr
T_MAX_MYR = 2.5


def chemistry_tables_dir(stellar_tables_file):
    return os.environ.get("SGCHEM_TABLES_DIR", os.path.dirname(stellar_tables_file))


def setup_run(d, rung, tables):
    if d.exists():
        shutil.rmtree(d)
    (d / "output").mkdir(parents=True)
    make_ics(str(d / "ics.hdf5"), RUNGS[rung])
    p = (TEST_DIR / f"{TEST_NAME}.params").read_text()
    p = p.replace("StellarTablesFile                  SET_BY_TEST", f"StellarTablesFile                  {tables['StellarTablesFile']}")
    p = p.replace("SGChemTablesPath                   SET_BY_TEST", f"SGChemTablesPath                   {chemistry_tables_dir(tables['StellarTablesFile'])}")
    (d / f"{TEST_NAME}.params").write_text(p)


@pytest.fixture(scope="module")
def run_m10():
    d = Path(os.environ.get("GIZMO_TREERAY_RUN", TEST_DIR / "run_m10"))
    if not os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        tables = get_stellar_evolution_tables()
        build_gizmo_for_test(TEST_NAME)
        setup_run(d, "m10", tables)
        with open(d / f"test_{TEST_NAME}.out", "w") as out:
            subprocess.run(mpi_launch_command(1, 4) + [str(TEST_DIR / "GIZMO"), f"{TEST_NAME}.params", "0"], cwd=d, stdout=out,
                           stderr=subprocess.STDOUT, env={**os.environ, "OMP_NUM_THREADS": "4"}, check=True)
    return d / "output"


def test_photon_ledger(run_m10):
    """every D-type snapshot pays for its recombinations with the star's photons: 0.85-1.7 Q (stella_dev net 5 0.9-1.3, net 17 1.0-1.7)"""
    snaps = sorted(glob.glob(str(run_m10 / "snapshot_*.hdf5")))
    rows = [photon_ledger(s) for s in snaps]
    rows = [(t, b) for t, b in rows if 0.2 * 3.15576e13 < t < T_MAX_MYR * 3.15576e13]
    assert rows and all(0.85 < b < 1.7 for _, b in rows), rows


def test_starbench_coarse_rung(run_m10):
    _, _, _, ref, med = grade(str(run_m10), T_MAX_MYR)
    assert abs(med - STELLA_DEV_M10) <= 0.04, (med, ref)
