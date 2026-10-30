"""single_stars_radiation_pressure: photon momentum of single stars in the local radiation-pressure loop
(GALSF_FB_RADPRESSURE_SINGLE_STAR_BANDS; design_feedback_coupling.md 4.6). One 40 Msun main-sequence star in a uniform
lattice at n_H ~ 100 cm^-3 (dust optical depths 0.2-0.8 over the kernel column), 1 Msun cells, chemistry on, no gravity.
  test_momentum_budget: the momentum offered per step (MomWinds.txt) equals the stella_dev estimator from the tables:
    L_abs = f_UV (L_PE + L_6-8) + f_NUV L_NUV(3.444-6) + f_OPT L_OPT, f_b = 1 - exp(-sigma_b DGR N_H) with
    sigma = 2.0, 1.3, 0.5 x 1e-21 cm^2, N_H = (rho / m_p) h from the uniform density and the star's kernel radius (64 cells),
    times (1 + tau_IR), tau_IR = 5 DGR Sigma, over dt / c (2 %); every neighbour gets its share, so the momentum coupled equals it (5 %).
  test_count_6_8ev_twice: GALSF_FB_RADPRESSURE_COUNT_6_8EV_TWICE (stella_dev production) adds f_NUV L_6-8: the ratio of
    the two budgets equals the table ratio (1e-3, density estimate cancels to first order).
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
from gizmo.test import build_gizmo_for_test, get_stellar_evolution_tables, get_cooling_tables, mpi_launch_command

TEST_NAME = "single_stars_radiation_pressure"
TEST_DIR = Path(__file__).parent
sys.path.insert(0, str(TEST_DIR.parent / "treecol_pixels"))
from make_treecol_pixels_ics import make_treecol_pixels_ics, lattice, YR_PER_TIME_UNIT  # noqa: E402
from stella_table_bands import Tables  # noqa: E402

BUILDS = {}
N_H, NGB, M_STAR, F_LIFE = 100.0, 32, 40.0, 0.3
MSUN, PC, MP, C = 1.989e33, 3.085678e18, 1.6726e-24, 2.9979e10


@pytest.fixture(scope="module")
def tables():
    if not os.environ.get("SUNDIALS_DIR"):
        pytest.skip("set SUNDIALS_DIR (SGCHEM builds against SUNDIALS 6.x or 7.x)")
    t = get_stellar_evolution_tables()
    get_cooling_tables(str(TEST_DIR))
    return t


def build(tag, extra=()):
    if tag not in BUILDS:
        build_gizmo_for_test(TEST_NAME, extra_config_flags=tuple(extra))
        shutil.move(TEST_DIR / "GIZMO", TEST_DIR / f"GIZMO_{tag}")
        BUILDS[tag] = TEST_DIR / f"GIZMO_{tag}"
    return BUILDS[tag]


def run(exe, tag, tables, n=16):
    rho = N_H * 1.4 * MP * PC ** 3 / MSUN  # Msun / pc^3 (H nuclei per mass as the chemistry)
    dx = (1.0 / rho) ** (1.0 / 3.0); L = n * dx
    tab = Tables(tables["StellarTablesFile"])
    lM, lZ = np.log10(M_STAR), np.log10(0.0142)
    age = tab._i2(tab.t_pms, lM, lZ) + F_LIFE * tab._i2(tab.life, lM, lZ)
    d = TEST_DIR / f"run_{tag}"
    if d.exists():
        shutil.rmtree(d)
    (d / "output").mkdir(parents=True)
    make_treecol_pixels_ics(str(d / "output" / "snapshot_000.hdf5"), L, lattice(n, L), [(M_STAR, age, np.full(3, 0.5 * L + 0.31 * dx))], hsml=2 * dx)
    for f in ("TREECOOL", "spcool_tables"):
        if (TEST_DIR / f).exists():
            os.symlink(TEST_DIR / f, d / f)
    T = tables["StellarTablesFile"]
    p = (TEST_DIR / f"{TEST_NAME}.params").read_text()
    p = p.replace("StellarTablesFile           SET_BY_TEST", f"StellarTablesFile           {T}")
    p = p.replace("SGChemTablesPath            SET_BY_TEST", f"SGChemTablesPath            {os.environ.get('SGCHEM_TABLES_DIR', os.path.dirname(T))}")
    over = {"BoxSize": L, "TimeMax": 2e-3, "MaxSizeTimestep": 1e-4, "DesNumNgb": NGB}
    p = "\n".join([l for l in p.splitlines() if not (l.split() and l.split()[0] in over)] + [f"{k:27s} {v}" for k, v in over.items()]) + "\n"
    (d / f"{TEST_NAME}.params").write_text(p)
    with open(d / f"test_{TEST_NAME}.out", "w") as out:
        subprocess.run(mpi_launch_command(2) + [str(exe), f"{TEST_NAME}.params", "2", "0"], cwd=d, stdout=out,
                       stderr=subprocess.STDOUT, env={**os.environ, "OMP_NUM_THREADS": "1"}, check=True)
    rows = np.atleast_2d(np.loadtxt(d / "output" / "MomWinds.txt"))
    return rows, tab, age, rho


def expected_per_time(tab, age_yr, rho, twice=False):
    """stella_dev estimator, L_abs (1 + tau_IR) / c [Msun km/s per code time]: N_H = (rho / m_p) h, Sigma = rho h"""
    L = tab.bands(M_STAR, 0.0142, age_yr)  # PE, LW, 6-8, NUV 3.444-6, OPT
    h = (3 * max(NGB, 64) / (4 * np.pi * rho)) ** (1.0 / 3.0) * PC  # star kernel: max(DesNumNgb, 64) cells of 1 Msun (density_loop.h) [cm]
    rho_cgs = rho * MSUN / PC ** 3
    NH, Sigma, dgr = rho_cgs / MP * h, rho_cgs * h, 0.0142 / 0.014
    f = [1 - np.exp(-s * 1e-21 * dgr * NH) for s in (2.0, 1.3, 0.5)]
    L_abs = f[0] * (L[0] + L[2]) + f[1] * (L[3] + (L[2] if twice else 0)) + f[2] * L[4]
    return L_abs * (1 + 5.0 * dgr * Sigma) / C * (PC / 1e5) / (MSUN * 1e5)


def supply_per_time(rows):
    """MomWinds.txt: time, n_checked, n_firing, n_touched, <R>, momentum offered, momentum coupled, ..."""
    dt = np.diff(rows[:, 0]); ok = dt > 0
    return rows[:-1][ok, 5] / dt[ok], rows[:-1][ok, 6] / dt[ok], rows[:-1][ok, 0]


def test_momentum_budget(tables):
    rows, tab, age, rho = run(build("default"), "default", tables)
    offered, coupled, t = supply_per_time(rows)
    ok = offered > 0; offered, coupled, t = offered[ok], coupled[ok], t[ok]
    assert len(offered) >= 5 and (rows[1:, 2] == 1).all()  # the star fires every step (no feedback step at t = 0)
    want = np.array([expected_per_time(tab, age + ti * YR_PER_TIME_UNIT, rho) for ti in t])
    np.testing.assert_allclose(offered, want, rtol=2e-2)
    assert abs(coupled.sum() / offered.sum() - 1) < 5e-2, (coupled.sum(), offered.sum())


def test_count_6_8ev_twice(tables):
    rows_d, tab, age, rho = run(build("default"), "default_ratio", tables)
    rows_t, _, _, _ = run(build("twice", ("GALSF_FB_RADPRESSURE_COUNT_6_8EV_TWICE",)), "twice", tables)
    got = rows_d[1, 5] / rows_t[1, 5]  # the first feedback step, same state in both runs
    want = expected_per_time(tab, age, rho) / expected_per_time(tab, age, rho, twice=True)
    print(f"disjoint / counted twice: {got:.4f} (tables {want:.4f})")
    assert abs(got / want - 1) < 1e-3
