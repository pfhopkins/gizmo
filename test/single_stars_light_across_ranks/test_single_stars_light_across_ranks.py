"""single_stars_light_across_ranks: a star that starts to shine or dies between tree builds is seen the same on every rank.

A 16^3 lattice of 1 Msun cells at n_H = 1 (optically thin), chemistry on, band fluxes from the gravity walk, 2 kyr, no tree or
domain rebuild after the start (both active fractions 1e4), so the light switches are handled by the node refresh only.
Stars (40 Msun, solar Z): 'switch' has one star ending its pre-main sequence (starts to shine) and one reaching the end of its
life (goes dark), both near 1 kyr, at (0.3, 0.3, 0.3) L and (0.7, 0.7, 0.7) L; 'control' has one star shining throughout at
(0.3, 0.3, 0.3) L. Each runs on 1, 2 and 4 ranks; G0_FUV (final snapshot, the last gravity call) and the radiation-pressure
momentum per step (MomWinds.txt) of N ranks are compared with 1 rank.
Bar (set before the first run): in 'switch', for every cell with G0_FUV(1 rank) above 1e-3 of its maximum and for every step,
|X_N / X_1 - 1| <= max(2 x the control's largest deviation, 1 %) (1 % = the node refresh tolerance), for X = G0_FUV and the momentum.
Mass constant to 1e-12; gas energy change minus the ledger's chemistry, floor and radiation-pressure kick rows below 1e-3 of the initial thermal energy.
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

NAME = "single_stars_light_across_ranks"
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "treecol_pixels"))
from make_treecol_pixels_ics import make_treecol_pixels_ics, lattice  # noqa: E402
from stella_table_bands import Tables  # noqa: E402

M_STAR, NGRID, Z, T_EVENT = 40.0, 16, 0.0142, 1000.0
MSUN, PC, MP = 1.989e33, 3.085678e18, 1.6726e-24
RANKS = (1, 2, 4)
RES = {}


@pytest.fixture(scope="module")
def built():
    if not os.environ.get("SUNDIALS_DIR"):
        pytest.skip("set SUNDIALS_DIR (SGCHEM builds against SUNDIALS 6.x or 7.x)")
    t = get_stellar_evolution_tables()
    get_cooling_tables(str(HERE))
    if not os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        build_gizmo_for_test(NAME)
    return t


def faint_star(tab):
    """(mass, age) of a star whose 6-13.6 eV luminosity changes most over the run (at least 4 %), staying below 0.1 of the
    bright star's, so its change stays under 1 % of the root's"""
    lZ = np.log10(Z)
    L_b = sum(tab.bands(M_STAR, Z, tab._i2(tab.t_pms, np.log10(M_STAR), lZ) + 0.3 * tab._i2(tab.life, np.log10(M_STAR), lZ))[[0, 2]])
    best = (0.0, None)
    for M in (8.0, 9.0, 10.0, 12.0, 15.0, 20.0):
        t_pms, life = tab._i2(tab.t_pms, np.log10(M), lZ), tab._i2(tab.life, np.log10(M), lZ)
        for x in np.concatenate((np.geomspace(100.0, 0.99 * life, 300), life - np.geomspace(3.0e3, 0.01 * life, 100))):
            a, b = sum(tab.bands(M, Z, t_pms + x)[[0, 2]]), sum(tab.bands(M, Z, t_pms + x + 2000.0)[[0, 2]])
            if a > 0 and b > 0 and max(a, b) < 0.1 * L_b and abs(b - a) < 0.005 * L_b and abs(b / a - 1) > best[0]:
                best = (abs(b / a - 1), (M, t_pms + x))
    print(f"faint star: M {best[1][0]} Msun, age {best[1][1]:.6g} yr, 6-13.6 eV change over the run {best[0]:.3f}")
    assert best[0] > 0.04
    return best[1]


def run(tables, case, ranks):
    key = (case, ranks)
    if key in RES:
        return RES[key]
    rho = 1.0 * MP / 0.7065 * PC ** 3 / MSUN
    dx = (1.0 / rho) ** (1.0 / 3.0); L = NGRID * dx
    tab = Tables(tables["StellarTablesFile"])
    lM, lZ = np.log10(M_STAR), np.log10(Z)
    t_pms, life = tab._i2(tab.t_pms, lM, lZ), tab._i2(tab.life, lM, lZ)
    assert t_pms > T_EVENT
    if case == "switch":
        stars = [(M_STAR, t_pms - T_EVENT, np.full(3, 0.3 * L)), (M_STAR, t_pms + life - T_EVENT, np.full(3, 0.7 * L))]
    elif case.startswith("faint"):
        M_f, age_f = faint_star(tab)  # a group of 8 faint stars, seen through its tree node from a few cells away
        stars = [(M_STAR, t_pms + 0.3 * life, np.full(3, 0.15 * L))] + [(M_f, age_f, np.full(3, 0.85 * L) + 0.25 * dx * np.array(o))
                                                                      for o in np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).reshape(3, -1).T]
    else:
        stars = [(M_STAR, t_pms + 0.3 * life, np.full(3, 0.3 * L))]
    d = HERE / f"run_{case}_r{ranks}"
    if d.exists():
        shutil.rmtree(d)
    (d / "output").mkdir(parents=True)
    make_treecol_pixels_ics(str(d / "output" / "snapshot_000.hdf5"), L, lattice(NGRID, L), stars, hsml=2 * dx)
    for f in ("TREECOOL", "spcool_tables"):
        if (HERE / f).exists():
            os.symlink(HERE / f, d / f)
    T = tables["StellarTablesFile"]
    p = (HERE / f"{NAME}.params").read_text()
    p = p.replace("StellarTablesFile           SET_BY_TEST", f"StellarTablesFile           {T}")
    p = p.replace("SGChemTablesPath            SET_BY_TEST", f"SGChemTablesPath            {os.environ.get('SGCHEM_TABLES_DIR', os.path.dirname(T))}")
    over = {"BoxSize": L, "TimeMax": 2e-3, "TimeBetSnapshot": 2e-3, "MaxSizeTimestep": 1e-4, "DesNumNgb": 32,
            "DomainBuild_ActiveFraction": 1e4, "TreeRebuild_ActiveFraction": 0 if case == "faint_ref" else 1e4}
    p = "\n".join([l for l in p.splitlines() if not (l.split() and l.split()[0] in over)] + [f"{k:27s} {v}" for k, v in over.items()]) + "\n"
    (d / f"{NAME}.params").write_text(p)
    with open(d / f"test_{NAME}.out", "w") as out:
        subprocess.run(mpi_launch_command(ranks) + [str(HERE / "GIZMO"), f"{NAME}.params", "2", "0"], cwd=d, stdout=out,
                       stderr=subprocess.STDOUT, env={**os.environ, "OMP_NUM_THREADS": "1"}, check=True)
    snaps = sorted((d / "output").glob("snapshot_*.hdf5"))
    with h5py.File(snaps[-1], "r") as F:
        g = F["PartType0"]; o = np.argsort(g["ParticleIDs"][()])
        g0 = g["G0_FUV"][()][o]
    rows = np.atleast_2d(np.loadtxt(d / "output" / "MomWinds.txt"))
    builds = (d / f"test_{NAME}.out").read_text().count("Tree construction done")
    RES[key] = dict(d=d, g0=g0, rows=rows, snaps=snaps, builds=builds)
    return RES[key]


def deviations(tables, case, ranks):
    a, b = run(tables, case, 1), run(tables, case, ranks)
    lit = a["g0"] > 1e-3 * a["g0"].max()
    dg = np.abs(b["g0"][lit] / a["g0"][lit] - 1)
    ta, tb = a["rows"][:, 0], b["rows"][:, 0]
    assert np.array_equal(ta, tb), (case, ranks)
    ok = a["rows"][:, 5] > 0
    dp = np.abs(b["rows"][ok, 5] / a["rows"][ok, 5] - 1)
    return dg, dp, (b["rows"][:, 5] > 0) != ok


@pytest.mark.parametrize("ranks", RANKS[1:])
def test_light_switch_seen_across_ranks(built, ranks):
    cg, cp, _ = deviations(built, "control", ranks)
    bar = max(2 * max(cg.max(), cp.max() if cp.size else 0), 0.01)
    dg, dp, onoff = deviations(built, "switch", ranks)
    s1, sn = run(built, "switch", 1), run(built, "switch", ranks)
    print(f"{ranks} ranks: tree builds 1 rank {s1['builds']}, {ranks} ranks {sn['builds']}; control: max |dG0| {cg.max():.2e}, "
          f"max |dp| {cp.max() if cp.size else 0:.2e}; bar {bar:.2e}")
    print(f"  switch: G0_FUV: cells off by > bar {np.sum(dg > bar)} of {dg.size}, max |dG0| {dg.max():.3e}, median {np.median(dg):.2e}; "
          f"momentum per step: max |dp| {dp.max() if dp.size else 0:.3e}, steps with momentum on one side only {int(onoff.sum())}")
    p1, pn = s1["rows"][:, 5].sum(), sn["rows"][:, 5].sum()
    print(f"  total radiation-pressure momentum 1 rank {p1:.6e}, {ranks} ranks {pn:.6e} (ratio {pn / p1:.4f})")
    assert dg.max() <= bar and (dp.max() if dp.size else 0) <= bar and not onoff.any()


@pytest.mark.parametrize("case", ["switch", "control"])
def test_mass_and_energy(built, case):
    for ranks in RANKS:
        r = run(built, case, ranks)
        with h5py.File(r["snaps"][0], "r") as a, h5py.File(r["snaps"][-1], "r") as b:
            m = [f["PartType0/Masses"][()].sum() + f["PartType4/Masses"][()].sum() for f in (a, b)]
            e = [np.sum(f["PartType0/Masses"][()] * (f["PartType0/InternalEnergy"][()] + 0.5 * np.sum(f["PartType0/Velocities"][()] ** 2, axis=1)))
                 * MSUN * 1e10 for f in (a, b)]
            eth0 = np.sum(a["PartType0/Masses"][()] * a["PartType0/InternalEnergy"][()]) * MSUN * 1e10
        led = np.atleast_2d(np.loadtxt(r["d"] / "output" / "ledger.txt"))
        names = [c for c in open(r["d"] / "output" / "ledger.txt").readline().split()[1:] if c != "(cumulative;"]
        booked = sum(led[-1, names.index(n)] * s for n, s in (("E_chem_heating_erg", 1), ("E_chem_cooling_erg", -1), ("E_kick_floor_erg", 1), ("E_radiation_pressure_kick_erg", 1)))
        res = (e[1] - e[0]) - booked
        print(f"{case} r{ranks}: mass change {m[1] / m[0] - 1:.1e}, gas energy change {e[1] - e[0]:.4e} erg, booked {booked:.4e}, "
              f"residual / E_th(0) {res / eth0:.2e}")
        assert abs(m[1] / m[0] - 1) < 1e-12
        assert abs(res) < 1e-3 * eth0


@pytest.mark.parametrize("ranks", RANKS)
def test_faint_star_drift_seen(built, ranks):
    """faint stars drifting by more than the refresh tolerance of their own light are seen: a 40 Msun star and a group of 8
    faint ones (each changing by at least 4 % over the run, below 1 % of the root's luminosity) at opposite corners, 1, 2 and
    4 ranks, no rebuild, against a run that rebuilds the tree every step; G0_FUV within 2 % in the cells more than 2.5 cells from
    the group where the group gives more 6-13.6 eV flux than the bright star (there the walk uses the group's node; the old
    rule, 1 % of the root's luminosity, never refreshed it)"""
    a, ref = run(built, "faint", ranks), run(built, "faint_ref", 1)
    with h5py.File(ref["snaps"][-1], "r") as F:
        g = F["PartType0"]; o = np.argsort(g["ParticleIDs"][()]); x = g["Coordinates"][()][o]
        s = F["PartType4"]; xs, ms = s["Coordinates"][()], s["Masses"][()]
        L = F["Header"].attrs["BoxSize"]
    faint, bright = ms < M_STAR, np.argmax(ms)
    assert faint.sum() == 8
    tab = Tables(built["StellarTablesFile"])
    lZ = np.log10(Z)
    M_f, age_f = faint_star(tab)
    t_end = 2e-3 * 3.085678e13 / 3.155e7
    L_f = sum(tab.bands(M_f, Z, age_f + t_end)[[0, 2]])
    L_b = sum(tab.bands(M_STAR, Z, tab._i2(tab.t_pms, np.log10(M_STAR), lZ) + 0.3 * tab._i2(tab.life, np.log10(M_STAR), lZ) + t_end)[[0, 2]])
    c = xs[faint].mean(0); r_c = np.sqrt(((x - c) ** 2).sum(1))
    f_f = 8 * L_f / r_c ** 2; f_b = L_b / ((x - xs[bright]) ** 2).sum(1)
    sel = np.logical_and(f_f > f_b, r_c > 2.5 * L / NGRID)
    dev = np.abs(a["g0"][sel] / ref["g0"][sel] - 1)
    print(f"{ranks} ranks: {sel.sum()} cells lit mainly by the faint group, beyond 2.5 cells: max |G0/G0_ref - 1| {dev.max():.3e}, median {np.median(dev):.2e}; "
          f"tree builds {a['builds']} vs reference {ref['builds']}")
    assert sel.sum() > 0 and dev.max() < 0.02
