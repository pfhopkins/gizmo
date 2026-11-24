"""treecol_shells_helium: helium in the TreeRay photoionisation (net 17). Sources and absorbers on the tree, from the first
gravity computation. See README.md."""
import os
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
from gizmo.test import build_gizmo_for_test, get_stellar_evolution_tables, get_cooling_tables, mpi_launch_command

TEST_NAME = "treecol_shells_helium"
TEST_DIR = Path(__file__).parent
PIX_DIR = TEST_DIR.parent / "treecol_pixels"
sys.path.insert(0, str(PIX_DIR))
from make_treecol_pixels_ics import make_treecol_pixels_ics, lattice  # noqa: E402
from stella_table_bands import Tables  # noqa: E402
from treecol_reference import direct_shell_sweep  # noqa: E402

EV = 1.60217733e-12
EDGES = (13.6, 24.6, 54.4, 70.0, 500.0)  # eV
SIGMA_HI = (3.658e-18, 8.482e-19, 1.0315e-19, 4.972e-20)  # cm^2, the walk's band means (gravtree_treecol_pixels.h)
SIGMA_HEI = (0.0, 5.460e-18, 7.330e-19, 3.6306e-19)
SIGMA_HEII = (0.0, 0.0, 1.3488e-18, 7.002e-19)
E_HE1, E_HE2 = 58.006, 73.785  # eV, mean photon energies of the two bands above 54.4 eV at 4e4 K
L_BOX, SHIELD = 20.0, 3.0
STARS = [(8.0, 0.5), (15.0, 0.6), (30.0, 0.2), (60.0, 0.5), (100.0, 0.2)]  # (M, fraction of the main sequence)
PIXELS = ("GALSF_RESOLVEDISM_TREECOL_PIXELS=12", "TEST_DUMP_TREECOL_PIXELS_AND_EXIT", "BOX_PERIODIC", "GRAVITY_NOT_PERIODIC")


EXE = {}


def run_dump(tag, gas_mass=1.0):
    if not os.environ.get("SUNDIALS_DIR"):
        pytest.skip("set SUNDIALS_DIR (SGCHEM builds against SUNDIALS 6.x or 7.x)")
    tables = get_stellar_evolution_tables()
    get_cooling_tables(str(TEST_DIR))
    exe = TEST_DIR / "GIZMO_dump"
    if "dump" not in EXE:
        build_gizmo_for_test(TEST_NAME, extra_config_flags=PIXELS)
        shutil.move(TEST_DIR / "GIZMO", exe)
        EXE["dump"] = exe
    tab = Tables(tables["StellarTablesFile"])
    rng = np.random.default_rng(13)
    n = 12; pos = lattice(n, L_BOX) + rng.uniform(-0.2, 0.2, (n ** 3, 3)) * (L_BOX / n)
    stars = []
    for M, f in STARS:
        lM, lZ = np.log10(M), np.log10(0.0142)
        stars.append((M, tab._i2(tab.t_pms, lM, lZ) + f * tab._i2(tab.life, lM, lZ), rng.uniform(0.15, 0.85, 3) * L_BOX))
    ab = np.zeros((n ** 3, 7)); ab[:, 0] = rng.uniform(0, 0.1, n ** 3); ab[:, 1] = rng.uniform(0, 0.9, n ** 3)
    ab[:, 3] = rng.uniform(0, 0.08, n ** 3); ab[:, 4] = rng.uniform(0, 0.02, n ** 3)
    d = TEST_DIR / f"run_{tag}"
    if d.exists():
        shutil.rmtree(d)
    (d / "output").mkdir(parents=True)
    make_treecol_pixels_ics(str(d / "output" / "snapshot_000.hdf5"), L_BOX, pos, stars, tracabund=ab, gas_mass=gas_mass)
    for f in ("TREECOOL", "spcool_tables"):
        if (TEST_DIR / f).exists():
            os.symlink(TEST_DIR / f, d / f)
    p = (PIX_DIR / "treecol_pixels.params").read_text()
    p = p.replace("StellarTablesFile           SET_BY_TEST", f"StellarTablesFile           {tables['StellarTablesFile']}")
    tabdir = os.environ.get("SGCHEM_TABLES_DIR", os.path.dirname(tables["StellarTablesFile"]))
    p = p.replace("SGChemTablesPath            SET_BY_TEST", f"SGChemTablesPath            {tabdir}")
    over = {"BoxSize": L_BOX, "TreeColShieldingLength": SHIELD}
    p = "\n".join([l for l in p.splitlines() if not (l.split() and l.split()[0] in over)] + [f"{k} {v}" for k, v in over.items()]) + "\n"
    (d / "treecol_pixels.params").write_text(p)
    with open(d / "run.out", "w") as out:
        subprocess.run(mpi_launch_command(1, 1) + [str(exe), "treecol_pixels.params", "2", "0"], cwd=d, stdout=out,
                       stderr=subprocess.STDOUT, env={**os.environ, "OMP_NUM_THREADS": "1"}, check=True)
    assert "TREECOL DUMP written" in (d / "run.out").read_text()
    return d / "output", tab, stars, ab


@pytest.fixture(scope="module")
def dump():
    return run_dump("dump")


def load(out, what):
    r = np.vstack([np.atleast_2d(np.loadtxt(f)) for f in sorted(out.glob(f"treecol_dump_{what}_*.txt")) if f.stat().st_size > 0])
    return r[np.argsort(r[:, 0])]


def test_four_source_bands(dump):
    """four bands from the tables: photons and energy add up to the star's, the band energies are the tables'"""
    out, tab, stars, _ = dump
    bins = np.loadtxt(out / "treecol_dump_bins_0.txt").astype(int)
    assert len(bins) == 13
    rows = load(out, "stars")[:, 2:]
    L, Q = rows[:, bins[5::2]], rows[:, bins[6::2]] / (13.6 * EV)
    want = np.array([tab.ionising(M, 0.0142, age) for M, age, _ in stars])
    np.testing.assert_allclose(L.sum(1), want[:, 0], rtol=1e-10)
    np.testing.assert_allclose(Q.sum(1), want[:, 1], rtol=1e-10)
    for b, key in ((1, "log_L_ion_He0"), (2, "log_L_ion_He1"), (3, "log_L_ion_He2")):
        np.testing.assert_allclose(L[:, b], [tab.band(key, M, 0.0142, age) for M, age, _ in stars], rtol=1e-10)
    np.testing.assert_allclose(Q[:, 2], L[:, 2] / (E_HE1 * EV), rtol=1e-10)
    np.testing.assert_allclose(Q[:, 3], L[:, 3] / (E_HE2 * EV), rtol=1e-10)
    above = np.array([tab.band("log_Q_ion", M, 0.0142, age) - tab.band("log_Q_ion_H0", M, 0.0142, age) for M, age, _ in stars]) - Q[:, 2] - Q[:, 3]
    np.testing.assert_allclose(Q[:, 1], np.clip(above, L[:, 1] / (54.4 * EV), L[:, 1] / (24.6 * EV)), rtol=1e-10)
    root = np.loadtxt(out / "treecol_dump_root_lum_0.txt")
    np.testing.assert_allclose(root[bins[5:]], rows[:, bins[5:]].sum(0), rtol=1e-12)


def test_helium_absorbers(dump):
    """node species: HI = (1 - x_H+) per H, HeI = ABHE - x_He+ - x_He++ per H, He+ = x_He+ per H, cell by cell and in the root node"""
    out, _, _, ab = dump
    cells = load(out, "cells")
    hi = (1 - ab[:, 1]) * 0.7065
    hei = (0.1 - ab[:, 3] - ab[:, 4]) * 0.7065
    np.testing.assert_allclose(cells[:, 4], hi, rtol=1e-12)
    np.testing.assert_allclose(cells[:, 5], hei, rtol=1e-12, atol=1e-12)
    heii = ab[:, 3] * 0.7065
    np.testing.assert_allclose(cells[:, 6], heii, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.loadtxt(out / "treecol_dump_root_species_0.txt")[3:6], [hi.sum(), hei.sum(), heii.sum()], rtol=1e-12)


def test_band_sweeps():
    """photon flux of each band against a direct sweep through HI, HeI and He+ (light cells, tau ~ 1, tree-error bar as
    test_ionising_sweep); energy flux = photon flux x the stars' band mean energy, one hardness per band"""
    out, tab, stars, ab = run_dump("sweep", gas_mass=4e-4)
    cells, rows = load(out, "cells"), load(out, "stars")
    bins = np.loadtxt(out / "treecol_dump_bins_0.txt").astype(int)
    gas, spos = cells[:, -3:], rows[:, -3:]
    m_hi, m_hei, m_heii = cells[:, 4], cells[:, 5], cells[:, 6]
    to_tau = 1.989e33 / 3.085678e18 ** 2 / 1.6726e-24
    pick = np.random.default_rng(3).choice(len(gas), 150, replace=False)
    F = cells[pick, -11:-3]  # photons 0..3, energy 0..3
    for b in range(4):
        Qb = rows[:, 2:][:, bins[6 + 2 * b]]
        if Qb.max() <= 0:
            continue
        m = m_hi + (SIGMA_HEI[b] * m_hei + SIGMA_HEII[b] * m_heii) / SIGMA_HI[b]
        want = np.array([direct_shell_sweep(gas[i], gas, m, spos, Qb, SIGMA_HI[b] * to_tau, SHIELD, npix=48, r_floor=0.01).sum() for i in pick]) * 2.9979e10 / (13.6 * EV)
        assert np.median(abs(F[:, b] / want - 1)) < 0.2, (b, np.median(abs(F[:, b] / want - 1)))
        q = F[:, b] > 0
        assert np.all(F[q, 4 + b] / F[q, b] / EV > EDGES[b] * (1 - 1e-6)) and np.all(F[q, 4 + b] / F[q, b] / EV < EDGES[b + 1])


def test_cross_sections_as_blackbody_integrals():
    """the band means of the integrals (photon-weighted, 4e4 K, Osterbrock fits), the band mean energies: the constants in the code"""
    def sig(e, s0, th):
        f = np.sqrt(np.maximum(e / th - 1, 1e-10))
        return np.where(e < th, 0, s0 * (th / e) ** 4 * np.exp(4 - 4 * np.arctan(f) / f) / (1 - np.exp(-2 * np.pi / f)))
    for b in range(4):
        e = np.linspace(EDGES[b], EDGES[b + 1], 200000); x = e * EV / (1.380658e-16 * 4e4); w = e ** 2 / np.expm1(x)
        for s0, th, want in ((6.3e-18, 13.6, SIGMA_HI[b]), (7.83e-18, 24.6, SIGMA_HEI[b]), (1.58e-18, 54.4, SIGMA_HEII[b])):
            if want:
                np.testing.assert_allclose(np.trapezoid(sig(e, s0, th) * w, e) / np.trapezoid(w, e), want, rtol=1e-3)
        if b > 1:
            np.testing.assert_allclose(np.trapezoid(e * w, e) / np.trapezoid(w, e), (E_HE1, E_HE2)[b - 2], rtol=1e-3)
