"""sgchem_recent_sfr: SGChem background field and cosmic-ray rate scaled with the recent star formation rate
(ISRF_AND_CR_RATE_SCALE_WITH_RECENT_SFR, as stella_dev GHOSDT_G0_CR with CR_SCALE_WITH_G0, Hu+21):
F = [M(stars younger than 30 Myr) / 30 Myr] / [M_gas,0 / 4 Gyr], G0 = max(F SGChemG0, 0.324e-2), zeta = F SGChemCosmicRayIonRate.
An 8^3 box of 1 Msun cells with SGChem and single stars from the tables; 2 ranks.
  test_recent_sfr_scale: three 1 Msun stars pass 30 Myr during the run, a young 8 Msun star loses wind mass, two stars
    are born at the first step (drawn mass collected from the gas); 0.59 Myr. At every star-formation call F equals a
    full scan over all particles (stella_dev's sum, 1e-12); F rises with the births, drops as each old star passes
    30 Myr, falls with the wind mass loss; G0 and zeta the chemistry gets follow F.
  test_quiet_calls_reduce_nothing: no young star loses mass (an 8 Msun star of 31.5 Myr does); 0.29 Myr. The
    young-star mass is reduced over ranks only at the births and when a star passes 30 Myr; the other calls add no
    collective, and F still equals the full scan.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path
import h5py
import numpy as np
import pytest
from gizmo.test import build_gizmo_for_test, get_stellar_evolution_tables, get_cooling_tables, mpi_launch_command

TEST_NAME = "sgchem_recent_sfr"
TEST_DIR = Path(__file__).parent
sys.path.insert(0, str(TEST_DIR.parent / "treecol_pixels"))
from make_treecol_pixels_ics import make_treecol_pixels_ics, lattice, YR_PER_TIME_UNIT  # noqa: E402
from stella_table_bands import Tables  # noqa: E402

L, N_SIDE = 8.0, 8
BIRTHS = "100:0.5,300:3.0"  # cell ID: drawn mass [Msun]
OLD_AGES_MYR = (29.85, 29.7, 29.55)  # 1 Msun stars, pass 30 Myr at about 0.15, 0.3, 0.45 Myr


@pytest.fixture(scope="module")
def setup():
    if not os.environ.get("SUNDIALS_DIR"):
        pytest.skip("set SUNDIALS_DIR (SGCHEM builds against SUNDIALS 6.x or 7.x)")
    tables = get_stellar_evolution_tables()
    get_cooling_tables(str(TEST_DIR))
    build_gizmo_for_test(TEST_NAME)
    return tables, Tables(tables["StellarTablesFile"])


def star_age(tab, M, f):
    lM, lZ = np.log10(M), np.log10(0.0142)
    return tab._i2(tab.t_pms, lM, lZ) + f * tab._i2(tab.life, lM, lZ)


def run(tables, tag, stars, t_max):
    d = TEST_DIR / f"output_{tag}"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir()
    make_treecol_pixels_ics(str(d / "snapshot_000.hdf5"), L, lattice(N_SIDE, L), stars, hsml=2.0)
    params = (TEST_DIR / f"{TEST_NAME}.params").read_text()
    for k, v in (("StellarTablesFile", tables["StellarTablesFile"]), ("TestSingleStarBirths", BIRTHS),
                 ("SGChemTablesPath", os.environ.get("SGCHEM_TABLES_DIR", os.path.dirname(tables["StellarTablesFile"])))):
        params = params.replace(f"{k:35s}SET_BY_TEST", f"{k:35s}{v}")
    for k, v in (("OutputDir", f"output_{tag}"), ("TimeMax", t_max), ("TimeBetSnapshot", t_max), ("TimeBetStatistics", t_max)):
        params = params.replace(f"{k:35s}", f"{k:35s}{v} %", 1)
    (d / "run.params").write_text(params)
    log = d / "run.out"
    with open(log, "w") as out:
        subprocess.run(mpi_launch_command(2, 1) + [str(TEST_DIR / "GIZMO"), str(d / "run.params"), "2", "0"], cwd=TEST_DIR,
                       stdout=out, stderr=subprocess.STDOUT, check=True)
    rows = [l.split()[1:] for l in log.read_text().splitlines() if l.startswith("RECENT_SFR:")]
    assert rows, "no RECENT_SFR lines"
    return np.array(rows, dtype=float), d  # time F F_full_scan reduced G0 zeta


def check_scale(r):
    F, F_scan, G0, zeta = r[:, 1], r[:, 2], r[:, 4], r[:, 5]
    np.testing.assert_allclose(F, F_scan, rtol=1e-12, atol=0)
    np.testing.assert_allclose(G0, np.maximum(F * 1.7, 0.324e-2), rtol=1e-14, atol=0)
    np.testing.assert_allclose(zeta, F * 2.0e-16, rtol=1e-14, atol=0)


def check_expiries(r, ages):
    t_myr, F = r[:, 0] * YR_PER_TIME_UNIT / 1e6, r[:, 1]
    for a in ages:  # each old star leaves the sum when it passes 30 Myr
        k = np.searchsorted(t_myr, 30.0 - a)
        assert 0 < k < len(F) and F[k] < F[k - 1], (a, t_myr[k - 1:k + 1], F[k - 1:k + 1])


def test_recent_sfr_scale(setup):
    tables, tab = setup
    rng = np.random.default_rng(3)
    stars = [(1.0, a * 1e6, rng.uniform(0.1, 0.9, 3) * L) for a in OLD_AGES_MYR] + [(8.0, star_age(tab, 8.0, 0.4), np.full(3, 0.5 * L))]
    assert stars[-1][1] < 29e6
    r, _ = run(tables, "scale", stars, 0.6)
    check_scale(r)
    check_expiries(r, OLD_AGES_MYR)
    jumps = np.diff(r[:, 1])
    print(f"{len(r)} calls, F {r[:, 1].min():.6g}-{r[:, 1].max():.6g}, {np.sum(jumps < 0)} drops")
    assert r[0, 1] > 0 and np.sum(jumps < 0) > len(OLD_AGES_MYR)  # births at the first call; wind mass loss between expiries


def test_quiet_calls_reduce_nothing(setup):
    tables, tab = setup
    rng = np.random.default_rng(3)
    assert star_age(tab, 8.0, 1.0) > 32e6  # an 8 Msun star of 31.5 Myr is alive and loses wind mass, but is not young
    stars = [(1.0, OLD_AGES_MYR[0] * 1e6, rng.uniform(0.1, 0.9, 3) * L), (8.0, 31.5e6, np.full(3, 0.5 * L))]
    r, d = run(tables, "quiet", stars, 0.3)
    check_scale(r)
    check_expiries(r, OLD_AGES_MYR[:1])
    reduced, changed = r[:, 3].astype(int), np.r_[True, np.diff(r[:, 1]) != 0]
    print(f"{reduced.sum()} reductions in {len(r)} calls; F changed on {changed.sum()} calls")
    assert np.array_equal(reduced, changed.astype(int)) and reduced.sum() == 2  # the births and the expiry
    with h5py.File(d / "snapshot_000.hdf5") as s0, h5py.File(d / "snapshot_001.hdf5") as s1:
        m0 = dict(zip(s0["PartType4/ParticleIDs"][:], s0["PartType4/Masses"][:]))
        m1 = dict(zip(s1["PartType4/ParticleIDs"][:], s1["PartType4/Masses"][:]))
    old = max(m0, key=m0.get)
    assert m1[old] < m0[old]  # the old star did lose wind mass
