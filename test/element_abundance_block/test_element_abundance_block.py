"""element_abundance_block test: with GALSF_SINGLE_STARS_ELEMENT_ABUNDANCE_BLOCK a snapshot restart reads the 27 element mass fractions
(H..Zn) of an IC into the metal slots (FIRE layout slots 1-10 and the 16 extra slots; total Z = sum of C..Zn), and the snapshot
writes them back. Gas lattice and star of single_stars_metals, 27-element build. See README.md."""
import os
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
import h5py
import pytest
from gizmo.test import build_gizmo_for_test, get_stellar_evolution_tables, get_cooling_tables, mpi_launch_command

TEST_NAME = "element_abundance_block"
TEST_DIR = Path(__file__).parent
sys.path.insert(0, str(TEST_DIR.parent / "single_stars_metals"))
from make_single_stars_metals_ics import make_single_stars_metals_ics  # noqa: E402

FIRE_ELEMENTS = [1, 2, 3, 4, 6, 8, 10, 12, 16, 22]
EXTRA_ELEMENTS = [5, 7, 9, 11, 13, 14, 15, 17, 18, 19, 20, 21, 23, 24, 25, 26]


@pytest.fixture(scope="module")
def tables():
    if not os.environ.get("SUNDIALS_DIR"):
        pytest.skip("set SUNDIALS_DIR (SGCHEM builds against SUNDIALS 6.x or 7.x)")
    t = get_stellar_evolution_tables()
    get_cooling_tables(str(TEST_DIR))
    build_gizmo_for_test(TEST_NAME)
    return t


def test_restart_reads_element_abundances(tables):
    """bar (written before the run): the IC carries zero Metallicity slots and a 27-element block (decreasing mass fractions, H the
    rest); the gas and star metal slots of the t = 0 snapshot-restart output equal the block through the slot map (1e-12), total Z
    = the sum of the 25 metals, and the ElementAbundance written back equals the block (1e-12, H = 1 - the rest)"""
    rng = np.random.default_rng(5)
    x = np.zeros(27); x[2:] = 10 ** rng.uniform(-7, -3, 25); x[1] = 0.27; x[0] = 1 - x[1:].sum()
    d = TEST_DIR / "run_restart"
    if d.exists():
        shutil.rmtree(d)
    (d / "output").mkdir(parents=True)
    ic = d / "output" / "snapshot_000.hdf5"
    L = make_single_stars_metals_ics(str(ic), np.zeros(27), [(3.0, 3.0, 1.0e6, (0.5, 0.5, 0.5), 0)], gas_fields={"ElementAbundance": x})
    with h5py.File(ic, "r+") as F:
        F["PartType4/ElementAbundance"] = x[None, :]
    for f in ("TREECOOL", "spcool_tables"):
        if (TEST_DIR / f).exists():
            os.symlink(TEST_DIR / f, d / f)
    T = tables["StellarTablesFile"]
    p = (TEST_DIR / f"{TEST_NAME}.params").read_text()
    p = p.replace("StellarTablesFile           SET_BY_TEST", f"StellarTablesFile           {T}")
    p = p.replace("SGChemTablesPath            SET_BY_TEST", f"SGChemTablesPath            {os.environ.get('SGCHEM_TABLES_DIR', os.path.dirname(T))}")
    p = "\n".join([l for l in p.splitlines() if not (l.split() and l.split()[0] in ("BoxSize", "TimeMax", "TimeBetSnapshot"))] + [f"BoxSize {L}", "TimeMax 1e-5", "TimeBetSnapshot 1e-5"]) + "\n"
    (d / f"{TEST_NAME}.params").write_text(p)
    with open(d / "run.out", "w") as out:
        subprocess.run(mpi_launch_command(2) + [str(TEST_DIR / "GIZMO"), f"{TEST_NAME}.params", "2", "0"], cwd=d, stdout=out,
                       stderr=subprocess.STDOUT, env={**os.environ, "OMP_NUM_THREADS": "1"}, check=True)
    snap = sorted((d / "output").glob("snapshot_*.hdf5"))[-1]
    want = np.concatenate([[x[2:].sum()], x[FIRE_ELEMENTS], x[EXTRA_ELEMENTS]])
    with h5py.File(snap, "r") as F:
        for t in ("PartType0", "PartType4"):
            Z = F[t]["Metallicity"][()].reshape(len(F[t]["Metallicity"]), -1)
            E = F[t]["ElementAbundance"][()]
            assert Z.shape[1] == 27 and np.allclose(Z, want, rtol=1e-12, atol=0), t
            assert E.shape[1] == 27 and np.allclose(E, x, rtol=1e-12, atol=0), t
