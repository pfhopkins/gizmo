"""single_stars_supernovae test: IMF-sampled single stars as supernova sources of the shared mechanical feedback loop,
from the stellar-evolution tables (GALSF_SINGLE_STARS_EVOLUTION_TABLES + GALSF_SINGLE_STARS_FROM_IMF +
GALSF_FB_MECHANICAL, pure-thermal coupling). See README.md."""
import os
import re
import shutil
import subprocess
from pathlib import Path
import numpy as np
import h5py
import pytest
from gizmo.test import build_gizmo_for_test, get_stellar_evolution_tables, get_cooling_tables, mpi_launch_command

TEST_NAME = "single_stars_supernovae"
TEST_DIR = Path(__file__).parent
E_UNIT = 1.989e33 * 1e10  # erg per code energy unit (Msun (km/s)^2)
ELEMENTS = ("COOLING", "COOL_METAL_LINES_BY_SPECIES")  # the FIRE metal layout: total Z, He, C, N, O, Ne, Mg, Si, S, Ca, Fe
FIRE_ELEMENTS = [1, 2, 3, 4, 6, 8, 10, 12, 16, 22]  # table element index of slots 1..10
EXPECTED_TYPES = {20.0: 2, 40.0: 3, 150.0: 4, 320.0: 5}  # CCSN, failed SN, PPISN, PISN at Z = 0.014


@pytest.fixture(scope="module")
def built():
    tables = get_stellar_evolution_tables()
    if not os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        build_gizmo_for_test(TEST_NAME, extra_config_flags=ELEMENTS)
        os.replace(TEST_DIR / "GIZMO", TEST_DIR / "GIZMO_elements")
        get_cooling_tables(str(TEST_DIR))
        build_gizmo_for_test(TEST_NAME)
    return tables


def run(tag, tables, ranks=2, threads=1, before_death_yr=None, overrides=None, exe="GIZMO", n_metals=1, ic_args=None):
    import sys
    sys.path.insert(0, str(TEST_DIR))
    from make_single_stars_supernovae_ics import make_single_stars_supernovae_ics
    d = TEST_DIR / f"run_{tag}"
    if not os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN") and d.exists():
        shutil.rmtree(d)  # this run's own directory: no stale logs or snapshots
    (d / "output").mkdir(parents=True, exist_ok=True)
    ic = d / "output" / "snapshot_000.hdf5"
    if not os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        L, _ = make_single_stars_supernovae_ics(str(ic), tables["StellarTablesFile"], before_death_yr=before_death_yr, n_metals=n_metals, **(ic_args or {}))
        if exe != "GIZMO":
            for f in ("TREECOOL", "spcool_tables"):
                os.symlink(TEST_DIR / f, d / f)
        params = Path(TEST_DIR / f"{TEST_NAME}.params").read_text().replace("SET_BY_TEST", tables["StellarTablesFile"])
        params = params.replace("BoxSize                            45.0", f"BoxSize                            {L}")
        for k, v in (overrides or {}).items():
            params = "\n".join(f"{k:34s} {v}" if l.split() and l.split()[0] == k else l for l in params.splitlines()) + "\n"
        (d / f"{TEST_NAME}.params").write_text(params)
        env = {**os.environ, "OMP_NUM_THREADS": str(max(threads, 1))}
        with open(d / f"test_{TEST_NAME}.out", "w") as out:
            subprocess.run(mpi_launch_command(ranks, threads) + [str(TEST_DIR / exe), f"{TEST_NAME}.params", "2", "0"],
                           cwd=d, stdout=out, stderr=subprocess.STDOUT, env=env, check=True)
    snaps = sorted((d / "output").glob("snapshot_*.hdf5"))
    rows = np.atleast_2d(np.loadtxt(d / "output" / "single_star_events.txt"))
    return ic, snaps[-1], rows


def totals(fname):
    """gas mass, metal mass per slot, energy (thermal + kinetic), momentum; star masses by ID"""
    with h5py.File(fname, "r") as F:
        g = F["PartType0"]
        m, v, u = g["Masses"][()], g["Velocities"][()], g["InternalEnergy"][()]
        Z = g["Metallicity"][()].reshape(len(m), -1)
        out = dict(M=m.sum(), MZ=(m[:, None] * Z).sum(0), E=(m * u).sum() + 0.5 * (m * (v ** 2).sum(1)).sum(), p=(m[:, None] * v).sum(0))
        stars = {}
        if "PartType4" in F:
            s = F["PartType4"]
            ms, vs = s["Masses"][()], s["Velocities"][()]
            out["p"] = out["p"] + (ms[:, None] * vs).sum(0)
            out["E"] += 0.5 * (ms * (vs ** 2).sum(1)).sum()
            stars = dict(zip(s["ParticleIDs"][()].tolist(), ms.tolist()))
        out["stars"] = stars
    return out


def test_supernovae(built):
    ic, snap, rows = run("r2", built)
    a, b = totals(ic), totals(snap)
    # one event per star, of the class the table gives its drawn mass
    assert len(rows) == 4 and len(set(rows[:, 1])) == 4
    assert np.all(rows[:, 2] == 1)  # kind: supernova
    rows = rows[:, [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11]]  # the columns as before winds were returned (drop the kind)
    for r in rows:
        assert EXPECTED_TYPES[r[3]] == int(r[2])
        assert r[4] == r[3]  # mass at the trigger = drawn mass (the initial conditions count the winds as returned)
    # each star keeps exactly its remnant (the logged one: a fraction of its mass at death, test_death_ejecta_fractions); the pair-instability SN leaves none
    for r in rows:
        sid = int(r[1])
        if r[6] > 0:
            assert b["stars"][sid] == r[6]
        else:
            assert sid not in b["stars"]
    assert np.all(np.abs(rows[:, 9]) <= 1e-12 * rows[:, 4])  # what the loop removed = trigger mass - remnant, to rounding
    # gas gains the ejecta and the ejected metals
    assert abs(b["M"] - a["M"] - rows[:, 5].sum()) <= 1e-12 * b["M"]
    assert abs(b["MZ"][0] - a["MZ"][0] - rows[:, 8].sum()) <= 1e-10 * rows[:, 8].sum()
    assert np.all(b["MZ"][1:] >= a["MZ"][1:])
    # energy: the explosion energies (failed SN: none) as heat; the hydro step conserves energy
    dE = (b["E"] - a["E"]) * E_UNIT
    assert abs(dE / rows[:, 7].sum() - 1) < 1e-3, dE / rows[:, 7].sum()
    # momentum: ejecta carry the stars' velocity, nothing radial is added
    assert np.all(np.abs(b["p"] - a["p"]) <= 1e-10 * np.abs(a["p"]).max())


class EnergyDependsOnRanks(AssertionError):
    """the gas energy differs between rank or thread counts by more than the bar"""


def compare_ranks_and_threads(built, exe, prefix, energy_bar=True):
    """the events (class, ejecta, remnant, metals, energy) are the same on 1, 2 and 4 ranks and with 2 threads, bitwise; the gas
    mass agrees to rounding (the loop adds deposits with atomics) and, with energy_bar, the gas energy to 1e-5 (EnergyDependsOnRanks if not)"""
    key = lambda r: r[np.argsort(r[:, 1])][:, [1, 2, 3, 4, 5, 6, 7, 8, 9]]
    _, s2, r2 = run(prefix + "r2", built, exe=exe)
    t2 = totals(s2)
    for tag, kw in [("r1", dict(ranks=1)), ("r4", dict(ranks=4)), ("r2t2", dict(ranks=2, threads=2))]:
        _, s, r = run(prefix + tag, built, exe=exe, **kw)
        assert key(r).tobytes() == key(r2).tobytes(), tag
        t = totals(s)
        assert abs(t["M"] / t2["M"] - 1) < 1e-13, tag
        dE = t["E"] / t2["E"] - 1
        print(f"{prefix}{tag} against {prefix}r2: dM/M {t['M'] / t2['M'] - 1:.3e}, dE/E {dE:.3e}")
        if energy_bar and not abs(dE) < 1e-5:
            raise EnergyDependsOnRanks(f"{prefix}{tag}: dE/E = {dE:.3e}")


def test_ranks_and_threads(built):
    """events and gas mass on 1, 2 and 4 ranks and with 2 threads, with the default cell splitting and merging. The gas energy is
    printed, not asserted: merge partners and split neighbours are searched on the calling task by design, so cells near a task
    boundary split differently on another number of tasks (a different, equally valid realisation); the energy bar is asserted
    in test_ranks_and_threads_without_merge_split"""
    compare_ranks_and_threads(built, "GIZMO", "", energy_bar=False)


def test_ranks_and_threads_without_merge_split(built):
    """the same comparison with PREVENT_PARTICLE_MERGE_SPLIT: no cell splits, and events, mass and energy are rank-independent"""
    if not os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        build_gizmo_for_test(TEST_NAME, extra_config_flags=("PREVENT_PARTICLE_MERGE_SPLIT",))
        os.replace(TEST_DIR / "GIZMO", TEST_DIR / "GIZMO_nosplit")
        build_gizmo_for_test(TEST_NAME)
    compare_ranks_and_threads(built, "GIZMO_nosplit", "nosplit_")
    for tag in ("r1", "r2", "r4", "r2t2"):
        log = (TEST_DIR / f"run_{tag}" / f"test_{TEST_NAME}.out")
        if log.exists():
            assert "Particle split/merge check" in log.read_text(errors="replace"), f"run_{tag}: the default build made no split/merge pass, so this control controls nothing"
        assert "Particle split/merge check" not in (TEST_DIR / f"run_nosplit_{tag}" / f"test_{TEST_NAME}.out").read_text(errors="replace"), tag


@pytest.mark.parametrize("cap_kyr", [10.0, 50.0])
def test_death_step_first_step_past_lifetime(built, cap_kyr):
    """stars 37 kyr from death and a long largest step: no step before the first death falls below the bin of
    MaxStellarTimestep, and each star dies at its first step past the table death time, at most one cap later"""
    kyr = 1e3 / (3.085678e13 / 3.155e7)  # code time per kyr, with the code's year
    _, _, rows = run(f"death_step_{cap_kyr:.0f}", built, before_death_yr=37e3,
                     overrides=dict(MaxStellarTimestep=cap_kyr * 1e3, MaxSizeTimestep=120 * kyr, TimeMax=120 * kyr, TimeBetSnapshot=120 * kyr, TimeBetStatistics=120 * kyr))
    assert len(rows) == 4
    t_kyr = rows[:, 0] / kyr
    assert np.all((t_kyr > 37.0) & (t_kyr <= 37.0 + cap_kyr)), t_kyr
    steps = [(float(m.group(1)), float(m.group(2))) for m in re.finditer(r"Sync-Point \d+, Time: (\S+), Systemstep: (\S+)", (TEST_DIR / f"run_death_step_{cap_kyr:.0f}" / f"test_{TEST_NAME}.out").read_text())]
    before = [dt / kyr for t, dt in steps if 0 < t < rows[:, 0].min()]
    assert len(before) <= 37.0 / (0.5 * cap_kyr) + 3, len(before)
    assert min(before) >= 0.5 * cap_kyr, min(before)


def test_death_with_winds_owed(built):
    """stars past death with the winds of the last 5 % of their life not returned: the owed winds leave in the first step as a wind
    delivery, at the table wind speed, and the death fires on the next step, with the table's fraction of the mass left"""
    ic, snap, rows = run("winds_owed", built, ic_args={"winds_owed_dex": np.log10(1 / 0.95)})
    a, b = totals(ic), totals(snap)
    assert len(rows) == 4 and np.all(rows[:, 2] == 1)
    assert np.ptp(rows[:, 0]) == 0 and rows[0, 0] > 0
    import sys
    sys.path.insert(0, str(TEST_DIR.parent / "treecol_pixels"))
    from stella_table_bands import Tables
    T = Tables(built["StellarTablesFile"])
    for r in rows:
        life = np.interp(np.log10(r[4]), T.lM, T.life.reshape(len(T.lZ), len(T.lM))[int(np.argmin(np.abs(T.lZ - np.log10(0.014))))])
        owed = T.wind_cumulative(r[4], 0.014, np.log10(life)) - T.wind_cumulative(r[4], 0.014, np.log10(0.95 * life))  # the table at the star's own end of life
        assert owed > 0 and abs((r[4] - r[5]) / owed - 1) < 1e-9, (r[4], r[5], owed)  # what left before the death
    assert abs(b["M"] - a["M"] - (rows[:, 6].sum() + (rows[:, 4] - rows[:, 5]).sum())) <= 1e-12 * b["M"]
    failed = rows[rows[:, 4] == 40.0][0]
    assert failed[6] == 0 and failed[9] == 0  # no table ejecta: its winds were all it gave


def test_element_yields(built):
    """FIRE metal layout (cooling on; metals do not depend on it): every slot of the gas gains exactly the table SN element
    ejecta of the three stars, scaled to their ejecta mass (the reference implementation's rule; the A9 comparison found
    the same element masses); slot 0 gains all table metals C..Zn"""
    _, snap, rows = run("elements", built, exe="GIZMO_elements", n_metals=11)
    ic = TEST_DIR / "run_elements" / "output" / "snapshot_000.hdf5"
    with h5py.File(ic) as F, h5py.File(snap) as G:
        dZ = (G["PartType0/Masses"][()][:, None] * G["PartType0/Metallicity"][()]).sum(0) - (F["PartType0/Masses"][()][:, None] * F["PartType0/Metallicity"][()]).sum(0)
    rows = rows[:, [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
    with h5py.File(built["StellarTablesFile"], "r") as T:
        Zt, Mt = T["Z"][()], T["M_init"][()]
        ej = T["elem_ej_SN_mass"][()].reshape(len(Zt), len(Mt), -1)[int(np.argmin(np.abs(Zt - 0.014)))]
    expect = np.zeros(11)
    for r in rows:
        y = ej[int(np.argmin(np.abs(Mt - r[3])))].astype(float)
        if r[5] <= 0 or y.sum() <= 0:
            continue
        y *= r[5] / y.sum()
        expect[0] += y[2:].sum()
        expect[1:] += y[FIRE_ELEMENTS]
    assert np.all(np.abs(dZ - expect) <= 1e-9 * np.maximum(expect, 1e-3)), (dZ, expect)


def test_no_mass_gain_at_death(built):
    """stars already below their table remnant mass (winds took more than the table's pre-SN mass): the remnant is the star, nothing is created"""
    ic, snap, rows = run("light", built, ic_args={"star_mass_factor": 0.3})
    a, b = totals(ic), totals(snap)
    m0 = {i: m for i, m in a["stars"].items()}
    for i, m in b["stars"].items():
        assert m <= m0[i] * (1 + 1e-12), f"star {i} grew from {m0[i]} to {m} at death"
    M0, M1 = a["M"] + sum(m0.values()), b["M"] + sum(b["stars"].values())
    assert abs(M1 - M0) <= 1e-12 * M0


@pytest.mark.parametrize("factor", [1.0, 0.3])
def test_death_ejecta_fractions(built, factor):
    """death ejecta as fractions of the mass at death (stella_dev F1, 8837629e8): with no winds owed, a star of the table grid
    holding M at death ejects M sum e / (rem + sum e) with the table's element ejecta e (SN table; PISN rem 0), and the
    metals in the same proportion (1e-10; checked for the stars on the table grid, at least two); every star: 0 <= rem <= M;
    a failed SN (table ejecta 0) ejects nothing.
    factor 0.3: stars at 0.3 of their drawn mass at death, below the table remnant of some (the old rule clipped the remnant
    to the star and ejected nothing there; a 1.0 star ejected M - rem_tab, which differs from the fraction whenever the
    star holds more than the table's pre-SN mass)"""
    _, _, rows = run(f"fractions_{factor}", built, ic_args={"star_mass_factor": factor})
    with h5py.File(built["StellarTablesFile"], "r") as T:
        Zt, Mt = T["Z"][()], T["M_init"][()]
        iz = int(np.argmin(np.abs(np.log10(Zt) - np.log10(0.014))))
        ej = T["elem_ej_SN_mass"][()].reshape(len(Zt), len(Mt), -1)[iz]
        rem = T["remnant_mass"][()].reshape(len(Zt), len(Mt))[iz]
    assert len(rows) == 4
    n_grid = 0
    for r in rows:
        M_drawn, M_trig, M_ej, M_rem, M_Z = r[4], r[5], r[6], r[7], r[9]
        assert -1e-12 * M_trig <= M_rem <= M_trig * (1 + 1e-12) and abs(M_rem + M_ej - M_trig) <= 1e-12 * M_trig
        im = int(np.argmin(np.abs(Mt - M_drawn)))
        if abs(Mt[im] / M_drawn - 1) > 1e-6:
            print(f"M_drawn {M_drawn:g} off the table grid: remnant within the star only")
            continue
        n_grid += 1
        e = np.maximum(ej[im].astype(float), 0)
        rt = 0 if int(r[3]) == EXPECTED_TYPES[320.0] else max(float(rem[im]), 0)
        f = e.sum() / (e.sum() + rt) if e.sum() + rt > 0 else 0
        fz = e[2:].sum() / (e.sum() + rt) if e.sum() + rt > 0 else 0
        print(f"M_drawn {M_drawn:g}, type {int(r[3])}, M_trigger {M_trig:.6g}: M_ej {M_ej:.10g} expected {f * M_trig:.10g}; "
              f"M_Z {M_Z:.6g} expected {fz * M_trig:.6g}; M_rem {M_rem:.6g}")
        assert abs(M_ej - f * M_trig) <= 1e-10 * M_trig and abs(M_Z - fz * M_trig) <= 1e-10 * M_trig
        if e.sum() == 0:
            assert M_ej == 0
    assert n_grid >= 2


def test_cold_ejecta(built):
    """1e7 K gas, where the ejecta mass x ambient u is 2 % of the explosion energy: the ejecta carry no ambient u, so the gas
    gains the three explosions (20, 150 and 320 Msun; the failed SN gives none) and no more"""
    ic, snap, _ = run("hot", built, ic_args={"T": 1.0e7})
    dE = (totals(snap)["E"] - totals(ic)["E"]) * E_UNIT
    assert abs(dE / 3.0e51 - 1) < 1e-3, dE / 3.0e51


def ecsn_stars(tables):
    """the IC stars plus the first table mass of an electron-capture SN (remnant type 1) at the IC metallicity, in a free corner"""
    import sys
    sys.path.insert(0, str(TEST_DIR))
    from make_single_stars_supernovae_ics import STARS
    with h5py.File(tables["StellarTablesFile"], "r") as T:
        Zt, Mt = T["Z"][()], T["M_init"][()]
        rt = T["remnant_type"][()].reshape(len(Zt), len(Mt))[int(np.argmin(np.abs(np.log10(Zt) - np.log10(0.014))))]
    ec = [float(m) for m, t in zip(Mt, rt) if int(t) == 1]
    assert ec, "no ECSN in the table row"
    return [(m, x) for m, x in STARS] + [(ec[0], (0.25, 0.75, 0.25))]


@pytest.mark.parametrize("variable", [False, True], ids=["default", "variable"])
def test_sn_energies(built, variable):
    """SN energies (Uli 2026-11-19, stella_dev fix/fb-audit 97b488f26): by default every explosion deposits 1e51 erg (failed SN 0);
    with RESOLVED_ISM_VARIABLE_SN_ENERGY a pair-instability SN 1e52 erg and an electron-capture SN 5e50 erg, the others 1e51.
    Stars: the four IC stars and the first ECSN mass of the table row; energies from single_star_events.txt to 1e-6"""
    exe = "GIZMO"
    if variable:
        if not os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
            build_gizmo_for_test(TEST_NAME, extra_config_flags=("RESOLVED_ISM_VARIABLE_SN_ENERGY",))
            os.replace(TEST_DIR / "GIZMO", TEST_DIR / "GIZMO_variable")
            build_gizmo_for_test(TEST_NAME)
        exe = "GIZMO_variable"
    _, _, rows = run(f"energies_{'variable' if variable else 'default'}", built, exe=exe, ic_args={"stars": ecsn_stars(built)})
    rows = rows[rows[:, 2] == 1]  # supernovae (time ID kind remnant_type M_drawn M_trigger M_ejecta M_remnant E ...)
    types = sorted(int(r[3]) for r in rows)
    print("remnant types", types, "energies", [f"{r[8]:.3e}" for r in rows])
    assert 1 in types and 5 in types and len(rows) == 5
    for r in rows:
        t = int(r[3])
        want = 0.0 if t == 3 else (1e52 if (variable and t == 5) else (5e50 if (variable and t == 1) else 1e51))
        assert abs(r[8] - want) <= 1e-6 * max(want, 1), (r[4], t, r[8], want)
