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
  test_chemistry_inputs: the SGChem field of every cell from the tree energy densities (pixels floored at
    0.324e-2 SGChemBackgroundG0 / 12 Habing): G0, G0_LW and the band-resolved grain-heating G0_dust as stella_dev (1e-6).
  test_radial_shells_attenuate_bands: with GALSF_RESOLVEDISM_TREECOL_PIXELS_RADIAL_SHELLS the stored stellar bands are
    dust-attenuated per (pixel, shell) in the walk: 6-13.6, 11.2-13.6, 3.444-6 and 0.4133-3.444 eV against a direct
    sweep over all particles (tree-error bars, 1 and 2 ranks).
  test_radial_shells_without_dust: without dust the sweep returns the unattenuated pixels (3e-6).
  test_ionising_sources_and_absorbers: ionising energy and photon rate as the tables in two tree source bins, root node
    sums, H-ionising absorber masses per cell and in the root node.
  test_ionising_sweep: photon and energy flux per cell against a direct sweep through the HI column at NSIDE 2 (median 20 %, tree error at tau ~ 1).
  test_radial_shells_photon_cap: thick cells: implied absorption brought back to the stars' emission (2e-3).
  test_radial_shells_host_equals_device: device walk repeats bitwise and equals the host walk to float sums (1e-5).
  test_far_field_g0: after five steps with chemistry, during which one star dies, the tree 6-13.6 eV field of every cell
    equals the direct sum over the stars' current luminosities (median |ratio - 1| <= 1e-2, p99 <= 5e-2): node
    luminosities are refreshed, the dead star is dark (2 ranks).
"""
import os
import re
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
from treecol_reference import direct_columns, direct_energy_density, direct_shell_sweep  # noqa: E402

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
    out = run(build("pix12", ("GALSF_RESOLVEDISM_TREECOL_PIXELS=12",)), f"bands_np{ranks}", tables, L, lattice(12, L), stars, ranks=ranks)
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


def test_radiation_mass_cut(tables):
    """IMFSampleStellarMassCut 0.5 (stella_dev production): main-sequence stars at or below 0.5 Msun are dark in every
    band, the others shine as the tables; the root node holds only the shining stars"""
    tab = Tables(tables["StellarTablesFile"])
    L, cut = 20.0, 0.5
    rng = np.random.default_rng(5)
    stars = []
    for M in (0.1, 0.2, 0.3, 0.45, 0.5, 0.6, 1.0, 8.0):
        lM, lZ = np.log10(M), np.log10(0.0142)
        stars.append((M, tab._i2(tab.t_pms, lM, lZ) + 0.01 * tab._i2(tab.life, lM, lZ), rng.uniform(0.2, 0.8, 3) * L))
    out = run(build("pix12", ("GALSF_RESOLVEDISM_TREECOL_PIXELS=12",)), "masscut", tables, L, lattice(12, L), stars,
              overrides={"IMFSampleStellarMassCut": cut})
    bins = np.loadtxt(out / "treecol_dump_bins_0.txt").astype(int)
    rows = load(out, "stars")
    assert len(rows) == len(stars)
    N_gas = 12 ** 3
    for row in rows:
        M, age, _ = stars[int(row[0]) - N_gas - 1]
        age += row[1] * 3.085678e13 / 3.155e7
        want = tab.bands(M, 0.0142, age) if M > cut else np.zeros(5)
        assert M <= cut or want.all(), (M, want)
        np.testing.assert_allclose(row[2:][bins], want, rtol=1e-10, atol=0, err_msg=f"M={M}")
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


def pixel_run(tables, ranks, threads=1, tag="", overrides=None, build_tag="pix12", extra=(), **ic):
    L, n = 20.0, 12
    pos, z, ab = gas_with_species(n, L)
    tab = Tables(tables["StellarTablesFile"])
    rng = np.random.default_rng(11)
    stars = []
    for M, f in PIX_STARS:
        lM, lZ = np.log10(M), np.log10(0.0142)
        stars.append((M, tab._i2(tab.t_pms, lM, lZ) + f * tab._i2(tab.life, lM, lZ), rng.uniform(0.15, 0.85, 3) * L))
    exe = build(build_tag, ("GALSF_RESOLVEDISM_TREECOL_PIXELS=12",) + tuple(extra))
    out = run(exe, f"pixels_np{ranks}{tag}", tables, L, pos, stars, ranks=ranks, z_gas=z, tracabund=ab, threads=threads,
              overrides={"TreeColShieldingLength": SHIELD, **(overrides or {})}, **ic)
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


def slab_ics(L=8.0, z0=4.5, n_xy=20, n_z=9):
    """n_H ~ 1e3 slab z in [z0, L] (N_H ~ 1e22 through it), sparse fill below, one 40 Msun star 3 pc in front of the face"""
    g = (np.arange(n_xy) + 0.5) * L / n_xy; zs = z0 + (np.arange(n_z) + 0.5) * (L - z0) / n_z
    slab = np.array(np.meshgrid(g, g, zs, indexing="ij")).reshape(3, -1).T
    gf = (np.arange(6) + 0.5) * L / 6; zf = (np.arange(3) + 0.5) * z0 / 3
    fill = np.array(np.meshgrid(gf, gf, zf, indexing="ij")).reshape(3, -1).T
    return np.vstack([slab, fill])


def test_slab_shielding(tables):
    """cells deep in a slab lit by one star (N_H >~ 5e21 toward it) are shielded by the pixel columns: after 20 steps of
    chemistry (0.15 Myr) their H2 fraction exceeds 1e-2 and 30x that of the lit face (cells on the star's axis; off-axis
    face cells see the star through a 57 degree pixel that also holds slab gas)"""
    tab = Tables(tables["StellarTablesFile"])
    L, z0 = 8.0, 4.5
    pos = slab_ics(L, z0)
    lM, lZ = np.log10(40.0), np.log10(0.0142)
    star = (40.0, tab._i2(tab.t_pms, lM, lZ) + 0.3 * tab._i2(tab.life, lM, lZ), np.array([4.0, 4.0, 1.5]))
    exe = build("pix12_step20", ("GALSF_RESOLVEDISM_TREECOL_PIXELS=12", "TEST_DUMP_TREECOL_PIXELS_AT_STEP=20"))
    out = run(exe, "slab", tables, L, pos, [star], gas_mass=1.0, hsml=0.6,
              overrides={"TimeMax": 1.0, "MaxSizeTimestep": 0.01, "SGChemBackgroundG0": 1.7, "SGChemCosmicRayIonRate": 3.0e-17})
    cells = load(out, "cells"); z = cells[:, -1]
    axis = (np.abs(cells[:, -3] - 4.0) < 1.2) & (np.abs(cells[:, -2] - 4.0) < 1.2)
    x_h2 = cells[:, 2] * 1.4 / (2.0 * 1.0)  # H2 mass of a 1 Msun cell, per H nucleus
    t = np.loadtxt(out / "treecol_dump_stars_0.txt", ndmin=2)[0, 1]
    face, deep = x_h2[axis & (z > z0) & (z < z0 + 0.4)].mean(), x_h2[axis & (z > z0 + 2.0)].mean()
    print(f"t = {t * 0.978:.3f} Myr: x_H2 face {face:.2e}, deep {deep:.2e}")
    assert deep >= 30 * face and deep > 1e-2, (face, deep)


LOG16_STEP = 20.0 / 65535  # dex per step of the 16-bit storage
LOG16_MIN = {"col": -16.0, "egy": -24.0}  # range starts: columns [g/cm^2 at the code normalisation], energy densities [erg/cm^3]


FLOAT_STORAGE = ("GALSF_RESOLVEDISM_TREECOL_PIXELS_STORED_AS_FLOAT",)  # parity builds: per-pixel cell values as float


def test_pixels_stored_as_16bit_log(tables):
    """default storage: every stored pixel column and 6-13.6 eV energy density lies on the 16-bit log10 grid and equals the
    float build (GALSF_RESOLVEDISM_TREECOL_PIXELS_STORED_AS_FLOAT) to half a step (3.5e-4); zeros stay zero; species masses
    and the band totals are unchanged; G0, G0_LW, G0_dust to 3.5e-4"""
    a, _, _ = pixel_run(tables, 1, tag="_float", build_tag="pix12_float", extra=FLOAT_STORAGE)
    b, _, _ = pixel_run(tables, 1)
    assert (a[:, 0] == b[:, 0]).all()
    half = 10 ** (LOG16_STEP / 2) - 1 + 1e-7
    for sl, lo, scale in ((slice(4, 40), LOG16_MIN["col"], 1.989e33 / 3.085678e18 ** 2), (slice(40, 52), LOG16_MIN["egy"], 1.0)):
        x, y = a[:, sl], b[:, sl]
        assert ((x == 0) == (y == 0)).all()
        k = (np.log10(y[y > 0] * scale) - lo) / LOG16_STEP
        print(f"{sl}: {np.count_nonzero(y)} values, grid offset max {np.abs(k - np.round(k)).max():.1e}, rel. diff max {rel(y, x)[x > 0].max():.2e}")
        assert np.abs(k - np.round(k)).max() < 1e-3
        assert rel(y, x)[x > 0].max() <= half
    assert (a[:, 1:4] == b[:, 1:4]).all() and (a[:, 52:56] == b[:, 52:56]).all()
    assert rel(b[:, 56:59], a[:, 56:59]).max() <= half


@pytest.mark.parametrize("ranks,threads", [(2, 2), (1, 4)])
def test_host_equals_device(tables, ranks, threads):
    runs = {}
    for tag, below in (("dev_a", 0), ("dev_b", 0), ("host", 10 ** 9)):  # GravityHostWalkBelowActive: 0 = device for any work
        runs[tag] = pixel_run(tables, ranks, threads, f"_t{threads}_{tag}", {"GravityHostWalkBelowActive": below}, "pix12_float", FLOAT_STORAGE)[0]
    assert runs["dev_a"].tobytes() == runs["dev_b"].tobytes()
    np.testing.assert_allclose(runs["dev_a"], runs["host"], rtol=2.5e-7, atol=0)  # float storage: at most a rounding step


def test_chemistry_inputs(tables):
    cells = pixel_run(tables, 1, tag="_chem", overrides={"SGChemBackgroundG0": 1.7})[0]
    g0bg = 0.324e-2 * 1.7
    uv = np.maximum(cells[:, 40:52], g0bg / 12 * 5.257e-14).sum(1)
    G0 = uv / 5.257e-14
    G0_LW = cells[:, 52] / 7.799e-15
    G0_dust = 0.2362 * G0 + 0.1496 * (cells[:, 53] / 4.696e-14 + g0bg) + 0.6142 * (cells[:, 54] / 5.141e-13 + g0bg)
    for got, want in ((cells[:, 56], G0), (cells[:, 57], G0_LW), (cells[:, 58], G0_dust)):
        np.testing.assert_allclose(got, want, rtol=1e-6)
    assert np.median(G0) > 10 * g0bg  # the stars dominate the floor


SIGMA_UV, SIGMA_LW = 2.5 * 5.348e-22, 2.0e-21  # cm^2 per H at solar dust-to-gas ratio (chemistry df25, f_dust)
SIGMA_NUV, SIGMA_OPT = 1.3e-21, 0.5e-21
TO_NH = 1.989e33 / 3.085678e18 ** 2 * 0.7065 / 1.6726e-24  # Msun/pc^2 to H/cm^2, X_H of SGChem builds


@pytest.mark.parametrize("ranks", [1, 2])
def test_radial_shells_attenuate_bands(tables, ranks):
    cells, spos, bands = pixel_run(tables, ranks, tag="_shells", build_tag="shells", gas_mass=8.0,
                                   extra=("GALSF_RESOLVEDISM_TREECOL_PIXELS_RADIAL_SHELLS=12",))
    pos = cells[:, -3:]; m_dust = cells[:, 1]; floor = 0.01
    log = (TEST_DIR / f"run_pixels_np{ranks}_shells" / f"test_{TEST_NAME}.out").read_text()
    caps = re.findall(r"photon cap engaged, f=([0-9.e+-]+)", log)
    cells[:, 40:55] /= float(caps[-1]) if caps else 1.0  # the sweep before the global cap
    pick = np.random.default_rng(3).choice(len(pos), 150, replace=False)
    want = np.array([direct_shell_sweep(pos[i], pos, m_dust, spos, bands[:, 0] + bands[:, 2], SIGMA_UV * TO_NH, SHIELD, r_floor=floor) for i in pick])
    r = rel(cells[pick, 40:52].ravel(), want.ravel())[want.ravel() > 1e-6 * want.max()]
    assert np.median(r) < 0.05 and np.percentile(r, 90) < 0.3, (np.median(r), np.percentile(r, 90))
    for col, band, sig in ((52, 1, SIGMA_LW), (53, 3, SIGMA_NUV), (54, 4, SIGMA_OPT)):
        want = np.array([direct_shell_sweep(pos[i], pos, m_dust, spos, bands[:, band], sig * TO_NH, SHIELD, r_floor=floor).sum() for i in pick])
        assert np.median(rel(cells[pick, col], want)) < 0.05, (col, np.median(rel(cells[pick, col], want)))


def test_radial_shells_without_dust(tables):
    over = {"SGChemDustToGasRatio": 0.0}
    plain, _, _ = pixel_run(tables, 1, tag="_nodust", build_tag="pix12_float", extra=FLOAT_STORAGE, overrides=over)
    shells, _, _ = pixel_run(tables, 1, tag="_shells_nodust", build_tag="shellsf", overrides=over,
                             extra=FLOAT_STORAGE + ("GALSF_RESOLVEDISM_TREECOL_PIXELS_RADIAL_SHELLS=12",))
    assert plain[:, 40:52].max() > 0
    np.testing.assert_allclose(shells[:, 40:55], plain[:, 40:55], rtol=3e-6, atol=1e-30)


SHELLS = ("GALSF_RESOLVEDISM_TREECOL_PIXELS_RADIAL_SHELLS=12",)
ION = SHELLS + ("GALSF_RESOLVEDISM_TREECOL_PIXELS_RADIAL_SHELLS_PHOTOION=2",)


def test_ionising_sources_and_absorbers(tables):
    """stars carry the table's ionising energy and photon rate in two tree source bins, the root node their sums and the
    H-ionising absorbers of all cells (H atoms + 1.34 per H2); the pixel columns keep their three species"""
    tab = Tables(tables["StellarTablesFile"])
    L = 20.0
    pos, z, ab = gas_with_species(12, L)
    rng = np.random.default_rng(13)
    stars = []
    for M, f in PIX_STARS:
        lM, lZ = np.log10(M), np.log10(0.0142)
        stars.append((M, tab._i2(tab.t_pms, lM, lZ) + f * tab._i2(tab.life, lM, lZ), rng.uniform(0.15, 0.85, 3) * L))
    out = run(build("ion", ("GALSF_RESOLVEDISM_TREECOL_PIXELS=12",) + ION), "ion", tables, L, pos, stars, z_gas=z, tracabund=ab,
              overrides={"TreeColShieldingLength": SHIELD})
    bins = np.loadtxt(out / "treecol_dump_bins_0.txt").astype(int)
    assert len(bins) == 7
    rows = load(out, "stars"); rows = rows[np.argsort(rows[:, 0])]
    want = np.array([tab.ionising(M, 0.0142, age) for M, age, _ in stars])
    EV = 1.60217733e-12
    np.testing.assert_allclose(rows[:, 2:][:, bins[5]], want[:, 0], rtol=1e-10)
    np.testing.assert_allclose(rows[:, 2:][:, bins[6]] / (13.6 * EV), want[:, 1], rtol=1e-10)
    root = np.loadtxt(out / "treecol_dump_root_lum_0.txt")
    np.testing.assert_allclose(root[bins[5:]], rows[:, 2:][:, bins[5:]].sum(0), rtol=1e-12)
    cells = load(out, "cells"); cells = cells[np.argsort(cells[:, 0])]
    hi = (1 - ab[:, 1] - 2 * ab[:, 0] + 1.34 * ab[:, 0]) * 0.7065
    np.testing.assert_allclose(cells[:, 4], hi, rtol=1e-12)
    np.testing.assert_allclose(np.loadtxt(out / "treecol_dump_root_species_0.txt")[3], hi.sum(), rtol=1e-12)


def test_ionising_sweep(tables):
    """photon and energy flux of each cell against a direct sweep through the HI column at NSIDE 2 (light cells: tau ~ 1;
    the tree lumps mass per pixel, flux biased high by ~10 % there; a wrong pixel area in the column gives > 50 %)"""
    tab = Tables(tables["StellarTablesFile"])
    L, m_gas = 20.0, 4e-4
    pos, z, ab = gas_with_species(12, L)
    ab[:, 0] = 0  # atomic
    rng = np.random.default_rng(13)
    stars = []
    for M, f in PIX_STARS:
        lM, lZ = np.log10(M), np.log10(0.0142)
        stars.append((M, tab._i2(tab.t_pms, lM, lZ) + f * tab._i2(tab.life, lM, lZ), rng.uniform(0.15, 0.85, 3) * L))
    out = run(build("ion", ("GALSF_RESOLVEDISM_TREECOL_PIXELS=12",) + ION), "ion_sweep", tables, L, pos, stars, z_gas=z, tracabund=ab,
              gas_mass=m_gas, overrides={"TreeColShieldingLength": SHIELD})
    cells = load(out, "cells"); cells = cells[np.argsort(cells[:, 0])]
    rows = load(out, "stars"); rows = rows[np.argsort(rows[:, 0])]
    bins = np.loadtxt(out / "treecol_dump_bins_0.txt").astype(int)
    Q_e, L_e = rows[:, 2:][:, bins[6]], rows[:, 2:][:, bins[5]]
    gas, m_hi, spos = cells[:, -3:], cells[:, 4], rows[:, -3:]
    sigma = 6.3e-18 * 1.989e33 / 3.085678e18 ** 2 / 1.6726e-24
    pick = np.random.default_rng(3).choice(len(gas), 150, replace=False)
    u_to_f = 2.9979e10
    want_ph = np.array([direct_shell_sweep(gas[i], gas, m_hi, spos, Q_e, sigma, SHIELD, npix=48, r_floor=0.01).sum() for i in pick]) * u_to_f / (13.6 * 1.60217733e-12)
    want_en = np.array([direct_shell_sweep(gas[i], gas, m_hi, spos, L_e, sigma, SHIELD, npix=48, r_floor=0.01).sum() for i in pick]) * u_to_f
    assert np.median(rel(cells[pick, -5], want_ph)) < 0.2, np.median(rel(cells[pick, -5], want_ph))
    assert np.median(rel(cells[pick, -4], want_en)) < 0.2, np.median(rel(cells[pick, -4], want_en))


def test_radial_shells_photon_cap(tables):
    """500 Msun cells: each is thick in itself, so the implied absorption exceeds the emission and the cap brings it back"""
    tag = "_shells_cap"
    cells, _, _ = pixel_run(tables, 2, tag=tag, build_tag="shells", gas_mass=500.0, extra=("GALSF_RESOLVEDISM_TREECOL_PIXELS_RADIAL_SHELLS=12",))
    out = TEST_DIR / f"run_pixels_np2{tag}" / "output"
    bins = np.loadtxt(out / "treecol_dump_bins_0.txt").astype(int)
    root = np.loadtxt(out / "treecol_dump_root_lum_0.txt")
    L_emit = root[bins[0]] + root[bins[2]]
    absorbed = (cells[:, 40:52].sum(1) * 2.9979e10 * SIGMA_UV * cells[:, 1] * 1.989e33 * 0.7065 / 1.6726e-24).sum()
    assert abs(absorbed / L_emit - 1) < 2e-3, absorbed / L_emit


def test_radial_shells_host_equals_device(tables):
    sh = FLOAT_STORAGE + ("GALSF_RESOLVEDISM_TREECOL_PIXELS_RADIAL_SHELLS=12",)
    runs = {}
    for tag, below in (("dev_a", 0), ("dev_b", 0), ("host", 10 ** 9)):
        runs[tag] = pixel_run(tables, 1, 4, f"_shells_{tag}", {"GravityHostWalkBelowActive": below}, "shellsf", sh, gas_mass=8.0)[0]
    assert runs["dev_a"].tobytes() == runs["dev_b"].tobytes()
    np.testing.assert_allclose(runs["dev_a"], runs["host"], rtol=1e-5, atol=1e-40)  # float shell sums, other order


def test_far_field_g0(tables):
    L, n = 20.0, 12
    pos, z, ab = gas_with_species(n, L)
    tab = Tables(tables["StellarTablesFile"])
    rng = np.random.default_rng(11)
    stars = []
    for M, f in PIX_STARS:
        lM, lZ = np.log10(M), np.log10(0.0142)
        stars.append((M, tab._i2(tab.t_pms, lM, lZ) + f * tab._i2(tab.life, lM, lZ), rng.uniform(0.15, 0.85, 3) * L))
    lM, lZ = np.log10(60.0), np.log10(0.0142)
    stars.append((60.0, tab._i2(tab.t_pms, lM, lZ) + tab._i2(tab.life, lM, lZ) - 150.0, np.array([0.5, 0.5, 0.5]) * L))  # dies at ~150 yr
    exe = build("pix12_step5", ("GALSF_RESOLVEDISM_TREECOL_PIXELS=12", "TEST_DUMP_TREECOL_PIXELS_AT_STEP=5"))
    out = run(exe, "far_field", tables, L, pos, stars, ranks=2, z_gas=z, tracabund=ab, overrides={"MaxSizeTimestep": 1e-4})
    cells = load(out, "cells"); cells = cells[np.argsort(cells[:, 0])]
    st = load(out, "stars"); st = st[np.argsort(st[:, 0])]
    bins = np.loadtxt(out / "treecol_dump_bins_0.txt").astype(int)
    assert st[0, 1] * 3.085678e13 / 3.155e7 > 150.0  # the run went past the death
    lum = st[:, 2:][:, bins]
    assert lum[-1].sum() == 0  # the dead star is dark
    L_uv = lum[:, 0] + lum[:, 2]
    ref = np.array([direct_energy_density(p, st[:, -3:], L_uv).sum() for p in cells[:, -3:]])
    r = np.abs(cells[:, 40:52].sum(1) / ref - 1)
    print(f"far field: median {np.median(r):.1e} p99 {np.percentile(r, 99):.1e}")
    assert np.median(r) <= 1e-2 and np.percentile(r, 99) <= 5e-2
