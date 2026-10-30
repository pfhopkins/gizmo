"""treecol_pixels: per-direction (HEALPix) columns and stellar band fluxes accumulated in the gravity walk
(GALSF_RESOLVEDISM_TREECOL_PIXELS...). Each test builds its configuration, runs from the initial conditions until the
first gravity computation is done (TEST_DUMP_TREECOL_PIXELS_AND_EXIT writes the tree radiation inputs and results and
stops) and checks them against exact values.
  test_star_band_luminosities: single stars of 1-100 Msun before, on and after the main sequence and dead shine in the
    disjoint table bands (PE 8-13.6, LW 11.2-13.6, 6-8, NUV 3.444-6, OPT eV) at their current age: each band equals the
    table interpolation to 1e-10, OPT + NUV + 6-8 + PE = OPT + NUV(3.444-8) + PE(8-13.6) of the tables, LW <= PE, and
    the root node of the tree holds their sum (1 and 2 ranks).
  test_node_species_masses: gas cells of 0.1-3 solar metallicity with H2 and CO carry dust-weighted gas, H2 and CO
    masses (dust-to-gas ratio as the chemistry input, H nuclei per mass 1/(1+4 ABHE)); the root node holds their sums
    (1e-12) after the first tree build and after three steps of chemistry and drift (1 and 2 ranks).
  test_pixel_columns_and_fluxes: a gas lattice of 0.1-3 solar metallicity with six young stars; every cell's 12 pixel
    columns (dust-weighted, H2, CO; within TreeColShieldingLength) and stellar band energy densities (6-13.6 eV per
    pixel; 11.2-13.6, 3.444-6, 0.4133-3.444, 6-8 eV totals) against direct sums over all particles: medians and 99th
    percentiles of the relative differences within the tree-error bars below (1 and 2 ranks; 2 ranks = 1 rank to the
    same bars).
  test_host_equals_device: the device walk (team-scratch sums, fixed fold order) gives the same cell outputs as the
    host walk, and repeats bitwise (2 ranks x 2 threads; 1 rank x 4 threads, where every target takes the device walk).
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
from gizmo.test import build_gizmo_for_test, get_stellar_evolution_tables, get_cooling_tables, mpi_launch_command

TEST_NAME = "treecol_pixels"
TEST_DIR = Path(__file__).parent
sys.path.insert(0, str(TEST_DIR))
from make_treecol_pixels_ics import make_treecol_pixels_ics, lattice  # noqa: E402
from stella_table_bands import Tables  # noqa: E402
from treecol_reference import direct_columns, direct_energy_density  # noqa: E402

BUILDS = {}


def chemistry_tables_dir(stellar_tables_file):
    return os.environ.get("SGCHEM_TABLES_DIR", os.path.dirname(stellar_tables_file))


@pytest.fixture(scope="module")
def tables():
    if not os.environ.get("SUNDIALS_DIR"):
        pytest.skip("set SUNDIALS_DIR (SGCHEM builds against SUNDIALS 6.x or 7.x)")
    t = get_stellar_evolution_tables()
    get_cooling_tables(str(TEST_DIR))
    return t


def build(tag, extra=()):
    """the test's Config.sh plus extra flags, as TEST_DIR/GIZMO_<tag>"""
    if tag not in BUILDS:
        build_gizmo_for_test(TEST_NAME, extra_config_flags=tuple(extra))
        shutil.move(TEST_DIR / "GIZMO", TEST_DIR / f"GIZMO_{tag}")
        BUILDS[tag] = TEST_DIR / f"GIZMO_{tag}"
    return BUILDS[tag]


def run(exe, tag, tables, L, gas_pos, stars, ranks=1, overrides=None, threads=1, **ic):
    d = TEST_DIR / f"run_{tag}"
    if d.exists():
        shutil.rmtree(d)  # this run's own directory
    (d / "output").mkdir(parents=True)
    make_treecol_pixels_ics(str(d / "output" / "snapshot_000.hdf5"), L, gas_pos, stars, **ic)
    for f in ("TREECOOL", "spcool_tables"):
        if (TEST_DIR / f).exists():
            os.symlink(TEST_DIR / f, d / f)
    params = (TEST_DIR / f"{TEST_NAME}.params").read_text()
    params = params.replace("StellarTablesFile           SET_BY_TEST", f"StellarTablesFile           {tables['StellarTablesFile']}")
    params = params.replace("SGChemTablesPath            SET_BY_TEST", f"SGChemTablesPath            {chemistry_tables_dir(tables['StellarTablesFile'])}")
    over = {"BoxSize": L, **(overrides or {})}
    lines = [l for l in params.splitlines() if not (l.split() and l.split()[0] in over)]
    params = "\n".join(lines + [f"{k:27s} {v}" for k, v in over.items()]) + "\n"
    (d / f"{TEST_NAME}.params").write_text(params)
    with open(d / f"test_{TEST_NAME}.out", "w") as out:
        subprocess.run(mpi_launch_command(ranks, threads) + [str(exe), f"{TEST_NAME}.params", "2", "0"], cwd=d, stdout=out,
                       stderr=subprocess.STDOUT, env={**os.environ, "OMP_NUM_THREADS": str(threads)}, check=True)
    assert "TREECOL DUMP written" in (d / f"test_{TEST_NAME}.out").read_text()
    return d / "output"


def load(out, what):
    rows = [np.atleast_2d(np.loadtxt(f)) for f in sorted(out.glob(f"treecol_dump_{what}_*.txt")) if f.stat().st_size > 0]
    return np.vstack(rows) if rows else np.zeros((0, 0))


STAR_SPECS = [(1.0, 0.5), (3.0, 0.3), (8.0, 0.01), (8.0, 0.97), (15.0, 0.6), (25.0, 1e-4), (40.0, 0.9), (60.0, 0.5),
              (100.0, 0.2), (20.0, -0.5), (30.0, 1.05)]  # (M, fraction of the main-sequence life; < 0: before it, > 1: dead)


def star_list(tab, L):
    rng = np.random.default_rng(4)
    stars = []
    for M, f in STAR_SPECS:
        lM, lZ = np.log10(M), np.log10(0.0142)
        t_pms, life = tab._i2(tab.t_pms, lM, lZ), tab._i2(tab.life, lM, lZ)
        age = t_pms * (1 + f) if f < 0 else t_pms + f * life
        stars.append((M, age, rng.uniform(0.2, 0.8, 3) * L))
    return stars


@pytest.mark.parametrize("ranks", [1, 2])
def test_star_band_luminosities(tables, ranks):
    tab = Tables(tables["StellarTablesFile"])
    L = 20.0
    stars = star_list(tab, L)
    out = run(build("bands"), f"bands_np{ranks}", tables, L, lattice(12, L), stars, ranks=ranks)
    bins = np.loadtxt(out / "treecol_dump_bins_0.txt").astype(int)
    rows = load(out, "stars")
    assert len(rows) == len(stars)
    N_gas = 12 ** 3
    shone = 0
    for row in rows:
        M, age, _ = stars[int(row[0]) - N_gas - 1]
        age += row[1] * 3.085678e13 / 3.155e7
        got, want = row[2:][bins], tab.bands(M, 0.0142, age)
        assert np.allclose(got, want, rtol=1e-10, atol=0), (M, age, got, want)
        if want.any():
            shone += 1
            assert got[1] <= got[0]
        np.testing.assert_allclose(got[4] + got[3] + got[2] + got[0], want.sum() - want[1], rtol=1e-12)
    assert shone == len(stars) - 2  # the pre-main-sequence and the dead star are dark
    root = np.loadtxt(out / "treecol_dump_root_lum_0.txt")
    np.testing.assert_allclose(root[bins], rows[:, 2:][:, bins].sum(0), rtol=1e-12)


def gas_with_species(n, L, seed=7):
    rng = np.random.default_rng(seed)
    pos = lattice(n, L) + rng.uniform(-0.2, 0.2, (n ** 3, 3)) * (L / n)
    z = 0.0142 * 10.0 ** rng.uniform(-1, np.log10(3), n ** 3)
    ab = np.zeros((n ** 3, 3)); ab[:, 0] = rng.uniform(0, 0.5, n ** 3); ab[:, 1] = 1e-4; ab[:, 2] = rng.uniform(0, 1e-4, n ** 3)
    return pos, z, ab


@pytest.mark.parametrize("ranks,step", [(1, 0), (2, 0), (2, 3)])
def test_node_species_masses(tables, ranks, step):
    L = 20.0
    pos, z, ab = gas_with_species(12, L)
    tag = "pix12" if step == 0 else f"pix12_step{step}"
    exe = build(tag, ("GALSF_RESOLVEDISM_TREECOL_PIXELS=12",) + (() if step == 0 else (f"TEST_DUMP_TREECOL_PIXELS_AT_STEP={step}",)))
    tab = Tables(tables["StellarTablesFile"])
    out = run(exe, f"species_np{ranks}_s{step}", tables, L, pos, star_list(tab, L)[:3], ranks=ranks, z_gas=z, tracabund=ab)
    cells = load(out, "cells")
    assert len(cells) == len(pos)
    o = np.argsort(cells[:, 0]); cells = cells[o]
    if step == 0:  # the initial conditions, exactly
        np.testing.assert_allclose(cells[:, 1], np.minimum(z / 0.014, 10.0), rtol=1e-12)
        np.testing.assert_allclose(cells[:, 2], 2 * ab[:, 0] / 1.4, rtol=1e-12)
        np.testing.assert_allclose(cells[:, 3], 28 * ab[:, 2] / 1.4, rtol=1e-12)
    root = np.loadtxt(out / "treecol_dump_root_species_0.txt")
    np.testing.assert_allclose(root, cells[:, 1:4].sum(0), rtol=1e-12)


PIX_STARS = [(8.0, 0.3), (15.0, 0.5), (25.0, 0.2), (40.0, 0.1), (60.0, 0.4), (100.0, 0.05)]
SHIELD = 8.0  # TreeColShieldingLength [pc]


def pixel_run(tables, ranks, threads=1, tag="", overrides=None):
    L, n = 20.0, 12
    pos, z, ab = gas_with_species(n, L)
    tab = Tables(tables["StellarTablesFile"])
    rng = np.random.default_rng(11)
    stars = []
    for M, f in PIX_STARS:
        lM, lZ = np.log10(M), np.log10(0.0142)
        stars.append((M, tab._i2(tab.t_pms, lM, lZ) + f * tab._i2(tab.life, lM, lZ), rng.uniform(0.15, 0.85, 3) * L))
    exe = build("pix12", ("GALSF_RESOLVEDISM_TREECOL_PIXELS=12",))
    out = run(exe, f"pixels_np{ranks}{tag}", tables, L, pos, stars, ranks=ranks, z_gas=z, tracabund=ab, threads=threads,
              overrides={"TreeColShieldingLength": SHIELD, **(overrides or {})})
    cells = load(out, "cells"); cells = cells[np.argsort(cells[:, 0])]
    lum = load(out, "stars"); lum = lum[np.argsort(lum[:, 0])]
    bins = np.loadtxt(out / "treecol_dump_bins_0.txt").astype(int)
    return cells, np.array([s[2] for s in stars]), lum[:, 2:][:, bins]


def rel(a, b):
    return np.abs(a - b) / np.maximum(np.abs(b), 1e-300)


@pytest.mark.parametrize("ranks", [1, 2])
def test_pixel_columns_and_fluxes(tables, ranks):
    cells, spos, bands = pixel_run(tables, ranks)
    gpos, msp = cells[:, -3:], cells[:, 1:4]
    col = cells[:, 4:40].reshape(-1, 3, 12)
    uv, tot = cells[:, 40:52], cells[:, 52:56]
    L_uv = bands[:, 0] + bands[:, 2]
    stats = {}
    for i in range(len(cells)):
        for s in range(3):
            ref = direct_columns(gpos[i], gpos, msp[:, s], r_max=SHIELD)
            stats.setdefault(f"col{s}", []).append(rel(col[i, s].sum(), ref.sum()))
        ref = direct_energy_density(gpos[i], spos, L_uv)
        stats.setdefault("uv", []).append(rel(uv[i].sum(), ref.sum()))
        stats.setdefault("uv_pix", []).append(np.abs(uv[i] - ref).sum() / ref.sum())
        for k, b in enumerate((1, 3, 4, 2)):  # LW, NUV, OPT, 6-8
            stats.setdefault(f"tot{k}", []).append(rel(tot[i, k], direct_energy_density(gpos[i], spos, bands[:, b]).sum()))
    summary = {k: (np.median(v), np.percentile(v, 99)) for k, v in stats.items()}
    print(ranks, {k: f"{a:.1e}/{b:.1e}" for k, (a, b) in summary.items()})
    for k, (med, p99) in summary.items():
        bar = (1e-2, 5e-2) if k.startswith("col") else (2e-3, 3e-2) if k != "uv_pix" else (1e-2, 1e-1)
        assert med <= bar[0] and p99 <= bar[1], (k, med, p99)


def test_columns_unsoftened(tables):
    """softening of 1 pc on a 1.7 pc lattice: the columns still follow the unsoftened direct sum, m / (4 pi / 12 r^2)"""
    soft = {f"Softening{t}": 1.0 for t in ("Gas", "Halo", "Disk", "Bulge", "Stars", "Bndry")}
    cells, _, _ = pixel_run(tables, 1, tag="_soft", overrides=soft)
    gpos, msp = cells[:, -3:], cells[:, 1:4]
    col = cells[:, 4:40].reshape(-1, 3, 12)
    for s in range(3):
        d = [rel(col[i, s].sum(), direct_columns(gpos[i], gpos, msp[:, s], r_max=SHIELD).sum()) for i in range(len(cells))]
        assert np.median(d) <= 1e-2 and np.percentile(d, 99) <= 5e-2, (s, np.median(d), np.percentile(d, 99))


@pytest.mark.parametrize("ranks,threads", [(2, 2), (1, 4)])
def test_host_equals_device(tables, ranks, threads):
    runs = {}
    for tag, below in (("dev_a", 0), ("dev_b", 0), ("host", 10 ** 9)):  # GravityHostWalkBelowActive: 0 = device for any work
        runs[tag] = pixel_run(tables, ranks, threads, f"_t{threads}_{tag}", {"GravityHostWalkBelowActive": below})[0]
    assert runs["dev_a"].tobytes() == runs["dev_b"].tobytes()
    np.testing.assert_allclose(runs["dev_a"], runs["host"], rtol=2.5e-7, atol=0)  # float storage: at most a rounding step
