"""stella_rng_identity_keys test: Stella's per-particle draws (IMF masses, star formation, Type Ia, HII sources, radiation
pressure, split and ejecta-cell directions) are keyed on the particle's identity (ID, child number, generation, type) and
the integer time, not on its position, so runs on 1, 2 and 4 ranks draw the same numbers until their dynamics part.
Stella cloud production flags and parameters, cloud IC from STELLA_CLOUD_IC (a MakeCloud cloud, e.g. gmc_Z1_m1.0,
12377 cells), 2.7 Myr. See README.md."""
import os
import shutil
import subprocess
from pathlib import Path
import pytest
from gizmo.test import build_gizmo_for_test, get_stellar_evolution_tables, get_cooling_tables, mpi_launch_command

TEST_NAME = "stella_rng_identity_keys"
TEST_DIR = Path(__file__).parent
RANKS = (1, 2, 4)
N_SHARED = 10  # bar, set before the run: with position keys the rank counts shared 0 stars (113-run study, 2026-11-15)


@pytest.fixture(scope="module")
def tables():
    if not os.environ.get("SUNDIALS_DIR"):
        pytest.skip("set SUNDIALS_DIR (SGCHEM builds against SUNDIALS 6.x or 7.x)")
    if not os.environ.get("STELLA_CLOUD_IC"):
        pytest.skip("set STELLA_CLOUD_IC to a Stella cloud IC (HDF5)")
    t = get_stellar_evolution_tables()
    get_cooling_tables(str(TEST_DIR))
    build_gizmo_for_test(TEST_NAME)
    return t


def births(tables, n):
    d = TEST_DIR / f"run_{n}"
    if d.exists():
        shutil.rmtree(d)
    (d / "output").mkdir(parents=True)
    ic = Path(os.environ["STELLA_CLOUD_IC"])
    os.symlink(ic, d / ic.name)
    for f in ("TREECOOL", "spcool_tables"):
        if (TEST_DIR / f).exists():
            os.symlink(TEST_DIR / f, d / f)
    T = tables["StellarTablesFile"]
    sub = {"InitCondFile": ic.name[:-5] if ic.name.endswith(".hdf5") else ic.name, "StellarTablesFile": T,
           "SGChemTablesPath": os.environ.get("SGCHEM_TABLES_DIR", os.path.dirname(T))}
    p = [f"{l.split()[0]} {sub[l.split()[0]]}" if l.split() and l.split()[0] in sub else l
         for l in (TEST_DIR / f"{TEST_NAME}.params").read_text().splitlines()]
    (d / f"{TEST_NAME}.params").write_text("\n".join(p) + "\n")
    with open(d / f"test_{TEST_NAME}.out", "w") as f:
        subprocess.run(mpi_launch_command(n) + [str(TEST_DIR / "GIZMO"), f"{TEST_NAME}.params"], cwd=d, stdout=f,
                       stderr=subprocess.STDOUT, env={**os.environ, "OMP_NUM_THREADS": "1"}, check=True)
    rows = [l.split() for l in open(d / "output" / "single_star_births.txt") if not l.startswith("#")]
    # time, ID, child, generation, drawn mass (as printed); within one time in ID order (ranks write a step's rows in any order)
    return sorted((tuple(r[:5]) for r in rows), key=lambda r: (float(r[0]), int(r[1]), int(r[2]), int(r[3])))


def shared(a, b):
    k = 0
    while k < min(len(a), len(b)) and a[k] == b[k]:
        k += 1
    return k


def test_rank_counts_share_their_first_stars(tables):
    """bar (written before the run): the births logs of 1, 2 and 4 ranks agree row by row (time, ID, child, generation,
    drawn mass; rows of one time in ID order) for at least the first 10 stars"""
    b = {n: births(tables, n) for n in RANKS}
    k12, k24, k14 = shared(b[1], b[2]), shared(b[2], b[4]), shared(b[1], b[4])
    print(f"stars formed: {[len(b[n]) for n in RANKS]}; identical leading stars: 1-2 {k12}, 2-4 {k24}, 1-4 {k14}")
    assert min(k12, k24, k14) >= N_SHARED
