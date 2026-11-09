"""treecol_pixel_columns_output test: OUTPUT_TREECOL_PIXEL_COLUMNS writes the per-pixel columns of the gravity walk
(dust-weighted gas, H2, CO; 12 HEALPix pixels) to the snapshot as TreeColumnPixels [g cm^-2], shape (N, 3 x 12), order
[column][pixel]. Lattice of treecol_pixels with six young stars and shielding length 8 pc, one step. See README.md."""
import os
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
import h5py
import pytest
from gizmo.test import build_gizmo_for_test, get_stellar_evolution_tables, get_cooling_tables, mpi_launch_command

TEST_NAME = "treecol_pixel_columns_output"
TEST_DIR = Path(__file__).parent
SRC_DIR = TEST_DIR.parent / "treecol_pixels"
sys.path.insert(0, str(SRC_DIR))
from make_treecol_pixels_ics import make_treecol_pixels_ics, lattice  # noqa: E402
from stella_table_bands import Tables  # noqa: E402
from treecol_reference import direct_columns  # noqa: E402

L, N_SIDE, SHIELD = 20.0, 12, 8.0
UNIT_SURFDEN = 1.989e33 / 3.085678e18 ** 2  # g cm^-2 per code unit
STARS = [(8.0, 0.3), (15.0, 0.5), (25.0, 0.2), (40.0, 0.1), (60.0, 0.4), (100.0, 0.05)]


@pytest.fixture(scope="module")
def tables():
    if not os.environ.get("SUNDIALS_DIR"):
        pytest.skip("set SUNDIALS_DIR (SGCHEM builds against SUNDIALS 6.x or 7.x)")
    t = get_stellar_evolution_tables()
    get_cooling_tables(str(TEST_DIR))
    build_gizmo_for_test(TEST_NAME)
    return t


def run_one_step(tables, ranks):
    rng = np.random.default_rng(7)
    pos = lattice(N_SIDE, L) + rng.uniform(-0.2, 0.2, (N_SIDE ** 3, 3)) * (L / N_SIDE)
    z = 0.0142 * 10.0 ** rng.uniform(-1, np.log10(3), N_SIDE ** 3)
    ab = np.zeros((N_SIDE ** 3, 3)); ab[:, 0] = rng.uniform(0, 0.5, N_SIDE ** 3); ab[:, 1] = 1e-4; ab[:, 2] = rng.uniform(0, 1e-4, N_SIDE ** 3)
    tab = Tables(tables["StellarTablesFile"])
    rng = np.random.default_rng(11)
    stars = []
    for M, f in STARS:
        lM, lZ = np.log10(M), np.log10(0.0142)
        stars.append((M, tab._i2(tab.t_pms, lM, lZ) + f * tab._i2(tab.life, lM, lZ), rng.uniform(0.15, 0.85, 3) * L))
    d = TEST_DIR / f"run_np{ranks}"
    if d.exists():
        shutil.rmtree(d)
    (d / "output").mkdir(parents=True)
    make_treecol_pixels_ics(str(d / "output" / "snapshot_000.hdf5"), L, pos, stars, z_gas=z, tracabund=ab)
    for f in ("TREECOOL", "spcool_tables"):
        if (TEST_DIR / f).exists():
            os.symlink(TEST_DIR / f, d / f)
    params = (TEST_DIR / f"{TEST_NAME}.params").read_text()
    params = params.replace("StellarTablesFile           SET_BY_TEST", f"StellarTablesFile           {tables['StellarTablesFile']}")
    params = params.replace("SGChemTablesPath            SET_BY_TEST", f"SGChemTablesPath            {os.environ.get('SGCHEM_TABLES_DIR', os.path.dirname(tables['StellarTablesFile']))}")
    over = {"BoxSize": L, "TreeColShieldingLength": SHIELD}
    lines = [l for l in params.splitlines() if not (l.split() and l.split()[0] in over)]
    (d / f"{TEST_NAME}.params").write_text("\n".join(lines + [f"{k:27s} {v}" for k, v in over.items()]) + "\n")
    with open(d / "run.out", "w") as out:
        subprocess.run(mpi_launch_command(ranks, 1) + [str(TEST_DIR / "GIZMO"), f"{TEST_NAME}.params", "2", "0"], cwd=d, stdout=out,
                       stderr=subprocess.STDOUT, env={**os.environ, "OMP_NUM_THREADS": "1"}, check=True)
    snaps = sorted((d / "output").glob("snapshot_*.hdf5"))
    return snaps[-1], pos, z


@pytest.mark.parametrize("ranks", [1, 2])
def test_columns_in_snapshot(tables, ranks):
    """bar (written before the run): the field exists with 36 values per cell, finite and non-negative; the dust-weighted
    column summed over the pixels equals the direct sum over all cells (dust-weighted mass min(Z / 0.014, 10) per cell,
    within 8 pc) to the tree-error bars of the treecol_pixels test: median 1e-2, 99th percentile 5e-2"""
    snap, pos, z = run_one_step(tables, ranks)
    with h5py.File(snap, "r") as F:
        g = F["PartType0"]
        col = g["TreeColumnPixels"][()]
        ids, xyz = g["ParticleIDs"][()], g["Coordinates"][()]
    assert col.shape == (N_SIDE ** 3, 36) and np.all(np.isfinite(col)) and np.all(col >= 0)
    o = np.argsort(ids); col, xyz = col[o], xyz[o]
    assert np.all(col[:, :12].sum(1) > 0)
    mdust = np.minimum(z / 0.014, 10.0)
    r = []
    for i in range(len(xyz)):
        ref = direct_columns(xyz[i], xyz, mdust[(ids[o] - ids.min()).astype(int)], r_max=SHIELD).sum() * UNIT_SURFDEN
        r.append(abs(col[i, :12].sum() - ref) / ref)
    print(f"np{ranks}: dust column median {np.median(r):.1e} p99 {np.percentile(r, 99):.1e}")
    assert np.median(r) <= 1e-2 and np.percentile(r, 99) <= 5e-2
