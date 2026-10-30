"""single_stars_supernovae test: IMF-sampled single stars as supernova sources of the shared mechanical feedback loop,
from the stellar-evolution tables (GALSF_SINGLE_STARS_EVOLUTION_TABLES + GALSF_SINGLE_STARS_FROM_IMF +
GALSF_FB_MECHANICAL, pure-thermal coupling). See README.md."""
import os
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
    rows = rows[:, [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11]]  # drop the kind
    for r in rows:
        assert EXPECTED_TYPES[r[3]] == int(r[2])
        assert r[4] == r[3]  # mass at the trigger = drawn mass (no winds are returned)
    # each star keeps exactly its table remnant; the pair-instability SN leaves none
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


def test_ranks_and_threads(built):
    """the events (class, ejecta, remnant, metals, energy) are the same on 1 and 4 ranks and with 2 threads; the gas totals
    agree to rounding (the loop adds deposits with atomics)"""
    _, s2, r2 = run("r2", built) if not (TEST_DIR / "run_r2" / "output" / "single_star_events.txt").exists() else (None, sorted((TEST_DIR / "run_r2" / "output").glob("snapshot_*.hdf5"))[-1], np.atleast_2d(np.loadtxt(TEST_DIR / "run_r2" / "output" / "single_star_events.txt")))
    key = lambda r: r[np.argsort(r[:, 1])][:, [1, 2, 3, 4, 5, 6, 7, 8, 9]]
    t2 = totals(s2)
    for tag, kw in [("r1", dict(ranks=1)), ("r4", dict(ranks=4)), ("r2t2", dict(ranks=2, threads=2))]:
        _, s, r = run(tag, built, **kw)
        assert key(r).tobytes() == key(r2).tobytes(), tag
        t = totals(s)
        # energy: a few hydro steps of hot gas after the deposit, which are not bitwise rank-independent
        assert abs(t["M"] / t2["M"] - 1) < 1e-13 and abs(t["E"] / t2["E"] - 1) < 1e-5, tag


def test_element_yields(built):
    """FIRE metal layout (cooling on; metals do not depend on it): every slot of the gas gains exactly the table SN element
    ejecta of the three stars, scaled to their ejecta mass (the reference implementation's rule; the A9 comparison found
    the same element masses in stella_dev); slot 0 gains all table metals C..Zn"""
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
