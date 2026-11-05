"""treecol_tallbox: the per-direction columns and stellar band fluxes of the gravity walk in a tall box (x/y periodic
by image sums, open in z: BOX_PERIODIC, BOX_LONG_Z=3, BOX_OUTFLOW_Z, GRAVITY_PERIODIC_XY_IMAGES). Same set-up and
checks as treecol_pixels test_pixel_columns_and_fluxes (12 pixel columns of dust-weighted gas, H2 and CO; 6-13.6 eV
energy density per pixel; band totals), but the direct sums use the nearest image in x and y and none in z. A gas
lattice fills the lower third of the box (20 pc wide); a star sits 58 pc above it near the top face, where a z wrap
would put it 2 pc from the gas at the bottom face. Limits as treecol_pixels (columns 1e-2 / 5e-2, fluxes 2e-3 / 3e-2), plus the wrap gate: the median column error of cells within 8 pc
of an x/y face must be at most 1.5 times that of the others (without the wrap those cells are off by 0.2, ten times
the others). The column limit is NOT met: measured 1.9e-2 / 5.9e-2 (cells far from the faces: 2.3e-2, so it is the tree
error of this geometry and not the wrap). 1 and 2 ranks.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
from gizmo.test import build_gizmo_for_test, get_stellar_evolution_tables, get_cooling_tables, mpi_launch_command

TEST_NAME = "treecol_tallbox"
TEST_DIR = Path(__file__).parent
PIX_DIR = TEST_DIR.parent / "treecol_pixels"
sys.path.insert(0, str(PIX_DIR))
from make_treecol_pixels_ics import make_treecol_pixels_ics, lattice  # noqa: E402
from stella_table_bands import Tables  # noqa: E402
from treecol_reference import direct_columns, direct_energy_density  # noqa: E402

L, LZ, N, SHIELD = 20.0, 60.0, 12, 8.0
PIX_STARS = [(8.0, 0.3), (15.0, 0.5), (25.0, 0.2), (40.0, 0.1), (60.0, 0.4), (100.0, 0.05)]


@pytest.fixture(scope="module")
def tables():
    if not os.environ.get("SUNDIALS_DIR"):
        pytest.skip("set SUNDIALS_DIR (SGCHEM builds against SUNDIALS 6.x or 7.x)")
    t = get_stellar_evolution_tables()
    get_cooling_tables(str(TEST_DIR))
    return t


@pytest.fixture(scope="module")
def exe():
    build_gizmo_for_test(TEST_NAME, extra_config_flags=("GALSF_RESOLVEDISM_TREECOL_PIXELS=12",))
    shutil.move(TEST_DIR / "GIZMO", TEST_DIR / "GIZMO_pix12")
    return TEST_DIR / "GIZMO_pix12"


def wrap_xy(pos, target):
    """positions as seen from target: nearest image in x and y, none in z"""
    d = pos - target
    d[:, :2] -= L * np.round(d[:, :2] / L)
    return target + d


def run(exe, ranks, tables):
    rng = np.random.default_rng(7)
    gas = lattice(N, L) + rng.uniform(-0.2, 0.2, (N ** 3, 3)) * (L / N)
    z = 0.0142 * 10.0 ** rng.uniform(-1, np.log10(3), N ** 3)
    ab = np.zeros((N ** 3, 3)); ab[:, 0] = rng.uniform(0, 0.5, N ** 3); ab[:, 1] = 1e-4; ab[:, 2] = rng.uniform(0, 1e-4, N ** 3)
    tab = Tables(tables["StellarTablesFile"])
    rng = np.random.default_rng(11)
    stars = []
    for M, f in PIX_STARS:
        lM, lZ = np.log10(M), np.log10(0.0142)
        stars.append((M, tab._i2(tab.t_pms, lM, lZ) + f * tab._i2(tab.life, lM, lZ), rng.uniform(0.15, 0.85, 3) * L))
    lM, lZ = np.log10(40.0), np.log10(0.0142)
    stars.append((40.0, tab._i2(tab.t_pms, lM, lZ) + 0.1 * tab._i2(tab.life, lM, lZ), np.array([0.5 * L, 0.5 * L, LZ - 2.0])))
    d = TEST_DIR / f"run_np{ranks}"
    if d.exists():
        shutil.rmtree(d)
    (d / "output").mkdir(parents=True)
    make_treecol_pixels_ics(str(d / "output" / "snapshot_000.hdf5"), L, gas, stars, z_gas=z, tracabund=ab)
    for f in ("TREECOOL", "spcool_tables"):
        if (PIX_DIR / f).exists():
            os.symlink(PIX_DIR / f, d / f)
    params = (TEST_DIR / f"{TEST_NAME}.params").read_text()
    params = params.replace("StellarTablesFile           SET_BY_TEST", f"StellarTablesFile           {tables['StellarTablesFile']}")
    tdir = os.environ.get("SGCHEM_TABLES_DIR", os.path.dirname(tables["StellarTablesFile"]))
    params = params.replace("SGChemTablesPath            SET_BY_TEST", f"SGChemTablesPath            {tdir}")
    params = "\n".join(l for l in params.splitlines() if not l.startswith("BoxSize")) + f"\nBoxSize                     {L}\n"
    (d / f"{TEST_NAME}.params").write_text(params)
    with open(d / f"test_{TEST_NAME}.out", "w") as out:
        subprocess.run(mpi_launch_command(ranks, 1) + [str(exe), f"{TEST_NAME}.params", "2", "0"], cwd=d, stdout=out,
                       stderr=subprocess.STDOUT, env={**os.environ, "OMP_NUM_THREADS": "1"}, check=True)
    assert "TREECOL DUMP written" in (d / f"test_{TEST_NAME}.out").read_text()
    out = d / "output"
    def load(what):
        rows = [np.atleast_2d(np.loadtxt(f)) for f in sorted(out.glob(f"treecol_dump_{what}_*.txt")) if f.stat().st_size > 0]
        return np.vstack(rows)
    cells = load("cells"); cells = cells[np.argsort(cells[:, 0])]
    lum = load("stars"); lum = lum[np.argsort(lum[:, 0])]
    bins = np.loadtxt(out / "treecol_dump_bins_0.txt").astype(int)
    return cells, np.array([s[2] for s in stars]), lum[:, 2:][:, bins]


def rel(a, b):
    return np.abs(a - b) / np.maximum(np.abs(b), 1e-300)


@pytest.mark.parametrize("ranks", [1, 2])
def test_columns_and_fluxes_wrap_in_xy_only(tables, exe, ranks):
    cells, spos, bands = run(exe, ranks, tables)
    gpos, msp = cells[:, -3:], cells[:, 1:4]
    col, uv, tot = cells[:, 4:40].reshape(-1, 3, 12), cells[:, 40:52], cells[:, 52:56]
    L_uv = bands[:, 0] + bands[:, 2]
    stats, edge_err = {}, []
    edge = np.minimum.reduce([gpos[:, 0], L - gpos[:, 0], gpos[:, 1], L - gpos[:, 1]]) < SHIELD
    for i in range(len(cells)):
        g = wrap_xy(gpos, gpos[i]); s = wrap_xy(spos, gpos[i])
        for k in range(3):
            stats.setdefault(f"col{k}", []).append(rel(col[i, k].sum(), direct_columns(gpos[i], g, msp[:, k], r_max=SHIELD).sum()))
        edge_err.append(stats["col0"][-1])
        ref = direct_energy_density(gpos[i], s, L_uv)
        stats.setdefault("uv", []).append(rel(uv[i].sum(), ref.sum()))
        stats.setdefault("uv_pix", []).append(np.abs(uv[i] - ref).sum() / ref.sum())
        for k, b in enumerate((1, 3, 4, 2)):
            stats.setdefault(f"tot{k}", []).append(rel(tot[i, k], direct_energy_density(gpos[i], s, bands[:, b]).sum()))
    summary = {k: (np.median(v), np.percentile(v, 99)) for k, v in stats.items()}
    print(ranks, {k: f"{a:.1e}/{b:.1e}" for k, (a, b) in summary.items()})
    for k, (med, p99) in summary.items():
        bar = (1e-2, 5e-2) if k.startswith("col") else (2e-3, 3e-2) if k != "uv_pix" else (1e-2, 1e-1)
        assert med <= bar[0] and p99 <= bar[1], (k, med, p99)
    edge_err = np.array(edge_err)
    print("column error, cells near an x/y face / others:", np.median(edge_err[edge]), np.median(edge_err[~edge]))
    assert np.median(edge_err[edge]) <= 1.5 * np.median(edge_err[~edge])
