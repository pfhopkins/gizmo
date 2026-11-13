"""ism_dust_five_element_model test: the five-element dust model (GALSF_ISMDUSTCHEM_FIVE_ELEMENT_MODEL) on the fields of the
ISM dust model with SGChem: sputtering and growth against the stella_dev formulas, the seed, the ledger, swept-mass SN
destruction. See README.md."""
import os
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
import h5py
import pytest
from gizmo.test import build_gizmo_for_test, get_cooling_tables, get_stellar_evolution_tables, mpi_launch_command

TEST_NAME = "ism_dust_five_element_model"
TEST_DIR = Path(__file__).parent
sys.path.insert(0, str(TEST_DIR.parent / "single_stars_metals"))
sys.path.insert(0, str(TEST_DIR.parent / "ism_dust_five_element_kernels"))
from make_single_stars_metals_ics import make_single_stars_metals_ics  # noqa: E402
from test_ism_dust_five_element_kernels import ref_step, yth, YC, YS  # noqa: E402  the stella_dev formulas

MP = 1.67262192e-24
T_UNIT_YR = 3.085678e13 / 3.155e7  # time unit [yr], with the code's year
ATOMS, ATOMIC = np.array([4.0, 1.0, 1.0, 1.0]), np.array([15.99, 24.305, 28.086, 55.845])  # FeMgSiO4: O, Mg, Si, Fe
FRAC = ATOMS * ATOMIC / (ATOMS * ATOMIC).sum()
SOLAR = np.array([0.014, 0.2485, 2.47653512e-3, 7.25454958e-4, 6.00285929e-3, 1.31578971e-3, 7.41243589e-4, 6.96209192e-4, 3.23764213e-4, 6.7169384e-5, 1.35284992e-3])  # FIRE slots, the tables' Z = 0.014
SLOTS = [2, 4, 6, 7, 10]  # C, O, Mg, Si, Fe
STARS = ("GALSF_FB_MECHANICAL", "GALSF_SINGLE_STARS_FROM_IMF", "GALSF_SINGLE_STARS_EVOLUTION_TABLES")


@pytest.fixture(scope="module")
def exe():
    if not os.environ.get("SUNDIALS_DIR"):
        pytest.skip("set SUNDIALS_DIR (SGCHEM builds against SUNDIALS 6.x or 7.x)")
    get_cooling_tables(str(TEST_DIR))
    if not os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        build_gizmo_for_test(TEST_NAME)
    return TEST_DIR / "GIZMO"


def seed(Z, fc=0.3, fsi=0.3):
    """fc of the carbon and fsi of the silicon in dust (stoichiometric silicate): element dust (slot 0 total), four sources
    (fractions, all SNe), species silicate and carbon"""
    sil = fsi * Z[7] / FRAC[2]
    d = np.zeros(11)
    d[2] = fc * Z[2]
    d[[4, 6, 7, 10]] = sil * FRAC
    d[0] = d[1:].sum()
    return np.concatenate([d, [0, 0, 1, 0]]), np.array([sil, d[2]])


def launch(d, exe, over, flag, ranks=2):
    for f in ("TREECOOL", "spcool_tables"):
        if (TEST_DIR / f).exists():
            os.symlink(TEST_DIR / f, d / f)
    p = (TEST_DIR / f"{TEST_NAME}.params").read_text().replace("SGChemTablesPath            SET_BY_TEST", f"SGChemTablesPath            {os.environ['SGCHEM_TABLES_DIR']}")
    p = "\n".join([l for l in p.splitlines() if not (l.split() and l.split()[0] in over)] + [f"{k:27s} {v}" for k, v in over.items()]) + "\n"
    (d / "run.params").write_text(p)
    with open(d / "run.out", "w") as out:
        subprocess.run(mpi_launch_command(ranks) + [str(exe), "run.params", flag, "0"], cwd=d, stdout=out, stderr=subprocess.STDOUT,
                       env={**os.environ, "OMP_NUM_THREADS": "1"}, check=True)


def run(tag, exe, n_H, T, over, ranks=2, n=8):
    d = TEST_DIR / f"run_{tag}"
    if not os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        if d.exists():
            shutil.rmtree(d)
        (d / "output").mkdir(parents=True)
        dust, species = seed(SOLAR)
        L = make_single_stars_metals_ics(str(d / "output" / "snapshot_000.hdf5"), SOLAR, n=n, n_H=n_H, T=T, gas_fields={"DustMetallicity": dust, "DustSpeciesAbundance": species})
        launch(d, exe, {"BoxSize": L, **over}, "2", ranks)
    return sorted((d / "output").glob("snapshot_*.hdf5"))


def snap(f):
    with h5py.File(f, "r") as F:
        g = F["PartType0"]
        i = np.argsort(g["ParticleIDs"][()])
        o = dict(t=float(F["Header"].attrs["Time"]) * T_UNIT_YR, m=g["Masses"][()][i], Z=g["Metallicity"][()][i][:, :11],
                 D=g["DustMetallicity"][()][i], S=g["DustSpeciesAbundance"][()][i], pos=g["Coordinates"][()][i])
        for k, n in (("rho", "Density"), ("T", "Temperature"), ("Td", "SGChemDustTemp")):
            o[k] = g[n][()][i] if n in g else None
    return o


def nH(s):
    return s["rho"] * 1.989e33 / 3.085678e18 ** 3 / MP  # as stella_dev: total density over m_p


def ledger(snaps):
    """the metals do not change, slot by slot; dust inside its element; sources, species and elements add up to the dust"""
    a = [snap(f) for f in snaps]
    mz0 = (a[0]["m"][:, None] * a[0]["Z"]).sum(0)
    for s in a[1:]:
        mz = (s["m"][:, None] * s["Z"]).sum(0)
        assert np.all(np.abs(mz / mz0 - 1) < 1e-12), mz / mz0 - 1
        assert np.all(s["D"][:, 1:11] <= s["Z"][:, 1:11] * (1 + 1e-12))
        assert np.all(s["D"][:, [1, 3, 5, 8, 9]] == 0)  # only C, O, Mg, Si, Fe hold dust
        assert np.all(np.abs(s["D"][:, 11:15].sum(1) - 1) < 1e-12)  # the snapshots hold the sources as fractions
        assert np.all(np.abs(s["S"].sum(1) / s["D"][:, 0] - 1) < 1e-12)
        assert np.all(np.abs(s["D"][:, 1:11].sum(1) / s["D"][:, 0] - 1) < 1e-12)


def test_thermal_sputtering(exe):
    """hot gas at rest: carbon and silicate fall as exp(-3 n Y(T) t / 0.1 micron) with the Nozawa+06 fits"""
    snaps = run("hot", exe, 1.0, 1e7, {"TimeMax": 0.02, "TimeBetSnapshot": 0.01})
    ledger(snaps)
    for a, b in zip(snaps[1:-1], snaps[2:]):
        s0, s1 = snap(a), snap(b)
        dt = s1["t"] - s0["t"]
        T = 0.5 * (s0["T"] + s1["T"])
        for slot, c, name in ((2, YC, "C"), (7, YS, "Si")):
            got = np.log(s1["D"][:, slot].sum() / s0["D"][:, slot].sum())
            want = -(3 * nH(s0) * np.array([yth(t, c) for t in T]) / 0.1 * dt).mean()
            print(f"{name}: T = {T.mean():.3g} K, ln(D1/D0) = {got:.5f}, stella_dev formula {want:.5f}, ratio {got / want:.4f}")
            assert abs(got / want - 1) < 0.01


def test_growth(exe):
    """cold dense gas: the growth of the stella_dev step (Hirashita sticking with the dust temperature, Dwek rates), integrated
    in small steps cell by cell at the mean gas and dust temperature; booked as the gas-dust accretion source"""
    snaps = run("cold", exe, 1.0e3, 50.0, {"TimeMax": 0.15, "TimeBetSnapshot": 0.05, "MaxSizeTimestep": 2.5e-3})
    ledger(snaps)
    s0, s1 = snap(snaps[1]), snap(snaps[-1])
    dt, n = s1["t"] - s0["t"], 200
    T, Td, nh = 0.5 * (s0["T"] + s1["T"]), 0.5 * (s0["Td"] + s1["Td"]), nH(s0)
    want = np.zeros(5)
    for c in range(len(T)):
        d = list(s0["D"][c, SLOTS]); z = list(s0["Z"][c, SLOTS])
        for _ in range(n):
            d = ref_step(nh[c], T[c], Td[c], dt / n, d, z, frac=tuple(FRAC))
        want += s0["m"][c] * (np.array(d) - s0["D"][c, SLOTS])
    got = ((s1["m"][:, None] * s1["D"][:, SLOTS]) - (s0["m"][:, None] * s0["D"][:, SLOTS])).sum(0)
    print(f"T {T.mean():.1f} K, T_dust {Td.mean():.1f} K; growth C {got[0]:.5g} (stella_dev step {want[0]:.5g}), Si {got[3]:.5g} ({want[3]:.5g})")
    assert got[0] > 0 and got[3] > 0
    assert abs(got[0] / want[0] - 1) < 0.03 and abs(got[3] / want[3] - 1) < 0.03
    assert np.allclose(got[[1, 2, 4]] / got[3], FRAC[[0, 1, 3]] / FRAC[2], rtol=1e-6)  # stoichiometric silicate
    assert (s1["m"] * s1["D"][:, 0] * s1["D"][:, 11]).sum() > 0  # booked as gas-dust accretion


def test_two_and_four_ranks(exe):
    ref = snap(run("hot", exe, 1.0, 1e7, {"TimeMax": 0.02, "TimeBetSnapshot": 0.01})[-1])
    a = snap(run("hot_r4", exe, 1.0, 1e7, {"TimeMax": 0.02, "TimeBetSnapshot": 0.01}, ranks=4)[-1])
    assert abs(a["D"][:, 0].sum() / ref["D"][:, 0].sum() - 1) < 1e-8


def test_seed_on_a_fresh_start(exe):
    """a fresh start seeds dust-to-gas 0.01 Z/0.014: 30% carbon (at most half the C), 70% FeMgSiO4 (at most 90% of O, Mg, Si,
    Fe), from supernovae (stella_dev resolvedism_dust_seed_cell)"""
    d = TEST_DIR / "run_seed"
    if not os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        if d.exists():
            shutil.rmtree(d)
        (d / "output").mkdir(parents=True)
        L = make_single_stars_metals_ics(str(d / "snapshot.hdf5"), SOLAR, n=8, n_H=1e3, T=50.0)
        launch(d, exe, {"BoxSize": L, "TimeMax": 1e-3, "TimeBetSnapshot": 1e-3}, "0")
    s = snap(sorted((d / "output").glob("snapshot_*.hdf5"))[0])
    Z = s["Z"]
    dgr = 0.01 * Z[:, 0] / 0.014
    C = np.minimum(0.3 * dgr, 0.5 * Z[:, 2])
    sil = np.minimum(0.7 * dgr, 0.9 * np.min(Z[:, SLOTS[1:]] / FRAC, axis=1))
    print(f"seed: dust-to-gas {s['D'][:, 0].mean():.5g} (stella_dev {(C + sil).mean():.5g}), carbon {s['D'][:, 2].mean():.5g}, silicate {s['S'][:, 0].mean():.5g}")
    assert Z[:, 0].min() > 0
    assert np.allclose(s["D"][:, 2], C, rtol=1e-6) and np.allclose(s["S"][:, 0], sil, rtol=1e-6)
    assert np.allclose(s["D"][:, SLOTS[1:]], sil[:, None] * FRAC, rtol=1e-6)
    assert np.allclose(s["D"][:, 13], 1.0)  # sources as fractions: all supernovae


def star_run(tag, flags, tmax=5e-5, every=None):
    """a 20 Msun supernova and a 3 Msun AGB death in seeded gas (16^3 cells of 1 Msun)"""
    from test_single_stars_metals import event_stars, event_life
    T = get_stellar_evolution_tables()["StellarTablesFile"]
    exe = TEST_DIR / f"GIZMO_{tag}"
    build_gizmo_for_test(TEST_NAME, extra_config_flags=flags)
    shutil.move(TEST_DIR / "GIZMO", exe)
    build_gizmo_for_test(TEST_NAME)  # the other tests' binary
    d = TEST_DIR / f"run_{tag}"
    if d.exists():
        shutil.rmtree(d)
    (d / "output").mkdir(parents=True)
    stars = event_stars(T)[:2]
    stars[1] = stars[1] + (np.log10(event_life(T, 3.0)),)
    dust, species = seed(SOLAR, 0.5, 0.5)
    L = make_single_stars_metals_ics(str(d / "output" / "snapshot_000.hdf5"), SOLAR, stars, gas_fields={"DustMetallicity": dust, "DustSpeciesAbundance": species})
    launch(d, exe, {"BoxSize": L, "TimeMax": tmax, "TimeBetSnapshot": every or tmax, "StellarTablesFile": T}, "2")
    rows = np.loadtxt(d / "output" / "single_star_events.txt", ndmin=2)
    assert sorted(rows[:, 2].astype(int).tolist()) == [1, 2]
    snaps = [snap(f) for f in sorted((d / "output").glob("snapshot_*.hdf5"))]
    return (snaps[0], snaps[-1], d) if every is None else (snaps, d)


def test_supernova_destroys_the_dust_of_its_kernel():
    """50 yr after a supernova in 1 Msun cells: McKee 1989 sweeps 0.4 x 2460 Msun per 1e51 erg, far above the cell mass, so every
    cell that takes a real share of the blast loses all its old dust (stella_dev: fraction min(0.4 M_swept / M_cell, 1)); the shared
    loop's own rule (efficiency 0.4 after the cap) would leave 60% of it; cells at the kernel edge lose part"""
    get_stellar_evolution_tables()
    s0, s1, _ = star_run("stars", STARS)
    left = s1["S"][:, 0] / s0["S"][0, 0]  # silicate mass fraction per cell over the seeded one (the ejecta bring almost none)
    core = s1["Z"][:, 0] > 1.5 * s0["Z"][0, 0]  # cells the supernova enriched by over half
    print(f"cells: {len(left)}, strongly enriched {core.sum()}, their silicate left: max {left[core].max():.3g}; "
          f"cells that lost any {(left < 0.99).sum()}, all of it {(left < 0.05).sum()}")
    assert core.sum() >= 8
    assert left[core].max() < 0.05


def ejecta_dust(d, species=False):
    """dust the events of a star run bring (table ejecta, Choban+22 condensation, FeMgSiO4; no metallic iron species tracked);
    species=True: the silicate alone"""
    import test_single_stars_metals as ssm
    ssm.SIL_ATOMS = {4: 4.0, 6: 1.0, 7: 1.0, 10: 1.0}  # FeMgSiO4 (GALSF_ISMDUSTCHEM_SILICATE_COMPOSITION=16)
    T = get_stellar_evolution_tables()["StellarTablesFile"]
    rows = np.atleast_2d(np.loadtxt(d / "output" / "single_star_events.txt"))
    with h5py.File(T, "r") as F:
        Zt, Mt = F["Z"][()], F["M_init"][()]
        iz, n3 = int(np.argmin(np.abs(Zt - 0.014))), (len(Zt), len(Mt), -1)
        ej = {"sn": F["elem_ej_SN_mass"][()].reshape(n3)[iz, int(np.argmin(np.abs(Mt - 20.0)))],
              "agb": F["elem_ej_AGB_mass"][()].reshape(n3)[iz, int(np.argmin(np.abs(Mt - 3.0)))]}
    total = 0.0
    for r in rows:
        kind = {1: "sn", 2: "agb"}[int(r[2])]
        y = np.asarray(ej[kind], dtype=float); y = y * r[6] / y.sum()
        dd, ss = ssm.stella_dust(np.concatenate([[y[2:].sum()], y[ssm.FIRE_ELEMENTS]]), kind)
        total += ss[0] if species else dd[0] - ss[2]
    return total


def lost(s0, s1, d):
    return (s0["m"] * s0["D"][:, 0]).sum() + ejecta_dust(d) - (s1["m"] * s1["D"][:, 0]).sum()


def test_dust_budget_through_feedback_merge_and_split():
    """a 3 Msun AGB death alone in cold seeded gas (100 K: no sputtering): the cells within two
    spacings of the star start at 1.45 Msun with the split threshold at 1.5 Msun, eight 0.3 Msun cells elsewhere with the
    merge threshold at 0.35 Msun; after the deposit splits and the mergers the gas holds the seeded dust plus the condensation of
    the table ejecta plus the growth booked as gas-dust accretion, to the yields' precision (1e-10 of the ejecta dust) and the
    double roundoff, element by element inside its metals"""
    from test_single_stars_metals import event_stars, event_life
    T = get_stellar_evolution_tables()["StellarTablesFile"]
    exe = TEST_DIR / "GIZMO_stars"
    if not exe.exists():
        build_gizmo_for_test(TEST_NAME, extra_config_flags=STARS)
        shutil.move(TEST_DIR / "GIZMO", exe)
    d = TEST_DIR / "run_budget"
    if d.exists():
        shutil.rmtree(d)
    (d / "output").mkdir(parents=True)
    star = event_stars(T)[1] + (np.log10(event_life(T, 3.0)),)
    dust, species = seed(SOLAR, 0.5, 0.5)
    ic = d / "output" / "snapshot_000.hdf5"
    L = make_single_stars_metals_ics(str(ic), SOLAR, [star], gas_fields={"DustMetallicity": dust, "DustSpeciesAbundance": species})
    with h5py.File(ic, "r+") as F:
        g, x_s = F["PartType0"], F["PartType4/Coordinates"][0]
        m, x = g["Masses"][()], g["Coordinates"][()]
        dx = L / round(len(m) ** (1 / 3))
        r = np.linalg.norm((x - x_s + L / 2) % L - L / 2, axis=1)
        m[r < 2 * dx] = 1.45
        m[np.argsort(-r)[::97][:8]] = 0.3  # far from the star, spread over the box
        g["Masses"][...] = m
        F["Header"].attrs["Maximum_Mass_For_Cell_Split"] = 1.5
        F["Header"].attrs["Minimum_Mass_For_Cell_Merge"] = 0.35
    launch(d, exe, {"BoxSize": L, "TimeMax": 2e-4, "TimeBetSnapshot": 2e-4, "StellarTablesFile": T}, "2")
    rows = np.loadtxt(d / "output" / "single_star_events.txt", ndmin=2)
    assert rows[:, 2].astype(int).tolist() == [2]
    out = (d / "run.out").read_text()
    import re
    merged = sum(int(a) for a in re.findall(r"check: (\d+) particles merged", out))
    split = sum(int(a) for a in re.findall(r"merged, (\d+) particles split", out))
    s0, s1 = snap(sorted((d / "output").glob("snapshot_*.hdf5"))[0]), snap(sorted((d / "output").glob("snapshot_*.hdf5"))[-1])
    M_ej, gap = ejecta_dust(d), lost(s0, s1, d)
    grown = (s1["m"] * s1["D"][:, 0] * s1["D"][:, 11]).sum() - (s0["m"] * s0["D"][:, 0] * s0["D"][:, 11]).sum()  # booked gas-dust accretion
    steps = out.count("Sync-Point")
    M_dust = (s1["m"] * s1["D"][:, 0]).sum()
    bar = 1e-10 * M_ej + (10 * steps + len(s1["m"])) * 2.2e-16 * M_dust  # yields precision + double roundoff per step and in the sum
    el = [((s1["m"][:, None] * s1["D"][:, SLOTS]).sum(0) - (s0["m"][:, None] * s0["D"][:, SLOTS]).sum(0))]
    print(f"cells {len(s0['m'])} -> {len(s1['m'])}, merged {merged}, split {split}, steps {steps}; T max {s1['T'].max():.3g} K; "
          f"ejecta dust {M_ej:.6g} Msun, booked growth {grown:.4g} Msun, dust missing net of growth {gap + grown:.4g} Msun, bar {bar:.3g}; element dust change {el[0]}")
    assert merged > 0 and split > 0
    assert s1["T"].max() < 3e4  # no sputtering below 3e4 K (Nozawa fits)
    assert grown >= 0 and abs(gap + grown) < bar
    assert np.all(s1["D"][:, 1:11] <= s1["Z"][:, 1:11] * (1 + 1e-12))


def test_no_swept_mass_destruction():
    """GALSF_ISMDUSTCHEM_FIVE_ELEMENT_NO_SWEPT_MASS_DESTRUCTION: in 50 yr after the supernova the box loses at most the silicate
    thermal sputtering takes, M1 (exp(3 n Y(T) dt / a) - 1) per cell at its final n, T, with the rate x1.5 for the density and
    temperature change in 50 yr (cells move < 0.1 pc, spacing 1.4 pc); the swept-mass rule would clear the kernel"""
    s0, s1, d = star_run("noswept", STARS + ("GALSF_ISMDUSTCHEM_FIVE_ELEMENT_NO_SWEPT_MASS_DESTRUCTION",))
    dt = s1["t"] - s0["t"]
    lost_sil = (s0["m"] * s0["S"][:, 0]).sum() + ejecta_dust(d, species=True) - (s1["m"] * s1["S"][:, 0]).sum()
    rate = 3 * nH(s1) * np.array([yth(t, YS) for t in s1["T"]]) / 0.1
    bound = (s1["m"] * s1["S"][:, 0] * np.expm1(1.5 * rate * dt)).sum()
    print(f"silicate lost in {dt:.3g} yr: {lost_sil:.4g} Msun, sputtering bound {bound:.4g} Msun")
    assert lost_sil <= bound


def test_shock_tracked_destruction():
    """GALSF_ISMDUSTCHEM_FIVE_ELEMENT_SHOCK_DESTRUCTION in place of the swept-mass rule, 200 yr: compressed cells with a shock above
    50 km/s lose dust at 0.4 v^2/(v^2 + 100^2) per crossing time (carbon half), on top of the same sputtering: more lost than with
    no SN destruction (above the double roundoff); per cell at most 1 - exp(-0.4 dt v/h) <~ 20% (v < 3000 km/s, h > 1.4 pc), so
    less than the swept-mass rule, which clears the kernel"""
    s, d = star_run("shock", STARS + ("GALSF_ISMDUSTCHEM_FIVE_ELEMENT_SHOCK_DESTRUCTION",), 2e-4, 2.5e-5)
    n, dn = star_run("noswept", STARS + ("GALSF_ISMDUSTCHEM_FIVE_ELEMENT_NO_SWEPT_MASS_DESTRUCTION",), 2e-4, 2.5e-5)  # sputtering only
    m, dm = star_run("stars", STARS, 2e-4, 2.5e-5)
    shock, none, mckee = lost(s[0], s[-1], d), lost(n[0], n[-1], dn), lost(m[0], m[-1], dm)
    floor = 1e-12 * (s[-1]["m"] * s[-1]["D"][:, 0]).sum()
    print(f"dust lost in 200 yr [Msun]: shock-tracked {shock:.4g}, none (sputtering only) {none:.4g}, swept mass {mckee:.4g}")
    assert shock > none + floor and shock < mckee
