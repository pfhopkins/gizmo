"""single_stars_metal_origin test: GALSF_SINGLE_STARS_METAL_ORIGIN_TRACERS keeps the metal mass fraction each single-star source
(SN, AGB, winds, Type Ia) brought into a cell in four extra Metallicity slots. Lattice, chemistry and stars of single_stars_metals,
27-element build. See README.md."""
import os
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
import h5py
import pytest
from gizmo.test import build_gizmo_for_test, get_stellar_evolution_tables, get_cooling_tables, mpi_launch_command

TEST_NAME = "single_stars_metal_origin"
TEST_DIR = Path(__file__).parent
sys.path.insert(0, str(TEST_DIR.parent / "single_stars_metals"))
from make_single_stars_metals_ics import make_single_stars_metals_ics  # noqa: E402

FIRE_ELEMENTS = [1, 2, 3, 4, 6, 8, 10, 12, 16, 22]
EXTRA_ELEMENTS = [5, 7, 9, 11, 13, 14, 15, 17, 18, 19, 20, 21, 23, 24, 25, 26]
ORIGIN = 27  # first origin slot: 11 FIRE slots + 16 extra elements
SN, AGB, WIND, IA = range(4)
KIND_OF_SLOT = {SN: 1, AGB: 2, IA: 4}  # event-log kind of each logged source (winds are not logged)


@pytest.fixture(scope="module")
def tables():
    if not os.environ.get("SUNDIALS_DIR"):
        pytest.skip("set SUNDIALS_DIR (SGCHEM builds against SUNDIALS 6.x or 7.x)")
    t = get_stellar_evolution_tables()
    get_cooling_tables(str(TEST_DIR))
    build_gizmo_for_test(TEST_NAME)
    return t


def solar27(tables_file):
    """the 27-element layout at the tables' solar composition (slot 0 = the sum of the 25 metals), origin slots zero"""
    with h5py.File(tables_file, "r") as F:
        Zt = F["Z"][()]
        X = F["X_init"][()].reshape(len(Zt), -1)[int(np.argmin(np.abs(Zt - 0.014)))].astype(float)
    return np.concatenate([[X[2:].sum()], X[FIRE_ELEMENTS], X[EXTRA_ELEMENTS], np.zeros(4)])


def event_stars(tables_file):
    """as single_stars_metals: a 20 Msun core collapse, a 3 Msun AGB death, a 1 Gyr white dwarf (Type Ia) and a 35 Msun star
    1 Myr old (winds), each in its own octant; the AGB white dwarf explodes too"""
    with h5py.File(tables_file, "r") as F:
        Zt, Mt = F["Z"][()], F["M_init"][()]
        iz, n2 = int(np.argmin(np.abs(Zt - 0.014))), (len(Zt), len(Mt))
        life, pms, rem = (F[k][()].reshape(n2)[iz] for k in ("lifetime_yr", "t_PMS", "remnant_mass"))
    i = {m: int(np.argmin(np.abs(Mt - m))) for m in (20.0, 3.0)}
    return [(20.0, 20.0, 1.01 * (life[i[20.0]] + pms[i[20.0]]), (0.25, 0.25, 0.25), 0, np.log10(life[i[20.0]])),
            (3.0, 3.0, 1.001 * (life[i[3.0]] + pms[i[3.0]]), (0.75, 0.75, 0.25), 0),
            (3.0, float(rem[i[3.0]]), 1.0e9, (0.25, 0.75, 0.75), 2),
            (35.0, 35.0, 1.0e6, (0.75, 0.25, 0.75), 0)]


ALLOWED = {(0, 0, 0): {SN}, (1, 1, 0): {AGB, IA}, (0, 1, 1): {IA}, (1, 0, 1): {WIND}}  # octant of each star: its sources


def run(tag, tables, over, ic=None):
    d = TEST_DIR / f"run_{tag}"
    if d.exists():
        shutil.rmtree(d)
    (d / "output").mkdir(parents=True)
    L = make_single_stars_metals_ics(str(d / "output" / "snapshot_000.hdf5"), solar27(tables["StellarTablesFile"]), event_stars(tables["StellarTablesFile"]))
    if ic is not None:
        shutil.copy(ic, d / "output" / "snapshot_000.hdf5")
    for f in ("TREECOOL", "spcool_tables"):
        if (TEST_DIR / f).exists():
            os.symlink(TEST_DIR / f, d / f)
    T = tables["StellarTablesFile"]
    over = {"BoxSize": L, **over}
    p = (TEST_DIR / f"{TEST_NAME}.params").read_text()
    p = p.replace("StellarTablesFile           SET_BY_TEST", f"StellarTablesFile           {T}")
    p = p.replace("SGChemTablesPath            SET_BY_TEST", f"SGChemTablesPath            {os.environ.get('SGCHEM_TABLES_DIR', os.path.dirname(T))}")
    p = "\n".join([l for l in p.splitlines() if not (l.split() and l.split()[0] in over)] + [f"{k:27s} {v}" for k, v in over.items()]) + "\n"
    (d / f"{TEST_NAME}.params").write_text(p)
    with open(d / f"test_{TEST_NAME}.out", "w") as out:
        subprocess.run(mpi_launch_command(2) + [str(TEST_DIR / "GIZMO"), f"{TEST_NAME}.params", "2", "0"], cwd=d, stdout=out,
                       stderr=subprocess.STDOUT, env={**os.environ, "OMP_NUM_THREADS": "1"}, check=True)
    return d


def gas(snap):
    with h5py.File(snap, "r") as F:
        g = F["PartType0"]
        m = g["Masses"][()]
        key = g["ParticleIDs"][()].astype(np.int64) * 2 ** 32 + (g["ParticleChildIDsNumber"][()] if "ParticleChildIDsNumber" in g else 0)
        return dict(t=float(F["Header"].attrs["Time"]), L=float(F["Header"].attrs["BoxSize"]), m=m, Z=g["Metallicity"][()].reshape(len(m), -1),
                    x=g["Coordinates"][()], key=key, zkey=F["Header"].attrs.get("Metals_Atomic_Number_Or_Key"))


EVENTS = {}


def events(tables):
    if not EVENTS:
        d = run("events", tables, {"TimeMax": 5e-3, "TimeBetSnapshot": 5e-3})
        EVENTS["snaps"] = sorted((d / "output").glob("snapshot_*.hdf5"))
        EVENTS["rows"] = np.atleast_2d(np.loadtxt(d / "output" / "single_star_events.txt"))
    return EVENTS["snaps"], EVENTS["rows"]


def test_metal_origin_by_source(tables):
    """bar (written before the run): four origin slots after the 27 elements, labelled -4..-7 in the snapshot header; the gas metal
    mass of each logged source (SN, AGB, Type Ia) equals the metals of its events in the event log (1e-10), the winds' is positive;
    the metal mass not from stars (total minus the four sources) stays the initial gas metal mass (1e-12); each star's octant
    carries only its own sources, every other origin slot is exactly 0"""
    snaps, rows = events(tables)
    a, b = gas(snaps[0]), gas(snaps[-1])
    assert b["Z"].shape[1] == ORIGIN + 4 and list(b["zkey"][ORIGIN:]) == [-4, -5, -6, -7]
    assert sorted(rows[:, 2].astype(int).tolist()) == [1, 2, 4, 4] and np.all(rows[:, 0] < b["t"])
    M_src = (b["m"][:, None] * b["Z"][:, ORIGIN:]).sum(0)
    for s, kind in KIND_OF_SLOT.items():
        z_ev = rows[rows[:, 2] == kind, 9].sum()
        print(f"source {s}: gas {M_src[s]:.12g} Msun, events {z_ev:.12g} Msun, rel {M_src[s] / z_ev - 1:.3g}")
        assert z_ev > 0 and abs(M_src[s] / z_ev - 1) <= 1e-10
    assert M_src[WIND] > 0
    M_init, M_rest = (a["m"] * a["Z"][:, 0]).sum(), (b["m"] * (b["Z"][:, 0] - b["Z"][:, ORIGIN:].sum(1))).sum()
    print(f"metals not from stars: start {M_init:.15g}, end {M_rest:.15g}, rel {M_rest / M_init - 1:.3g}; winds {M_src[WIND]:.6g} Msun")
    assert abs(M_rest / M_init - 1) <= 1e-12
    octant = tuple((b["x"] // (0.5 * b["L"])).astype(int).T)
    for c in range(len(b["m"])):
        allowed = ALLOWED.get(tuple(int(o[c]) for o in octant), set())
        for s in range(4):
            if s not in allowed:
                assert b["Z"][c, ORIGIN + s] == 0, (c, s)
    for s in range(4):
        assert np.any(b["Z"][:, ORIGIN + s] > 0)


def test_metal_origin_snapshot_restart(tables):
    """bar (written before the run): a snapshot restart from the last snapshot reads the origin slots back: after 1e-5 time
    units every gas cell outside the wind star's octant (the only source still active) has the origin slots it was written
    with (1e-14)"""
    snaps, _ = events(tables)
    a = gas(snaps[-1])
    d = run("restart", tables, {"TimeMax": a["t"] + 1e-5, "TimeBetSnapshot": 1e-5, "TimeBegin": a["t"]}, ic=snaps[-1])
    b = gas(sorted((d / "output").glob("snapshot_*.hdf5"))[-1])
    assert b["t"] > a["t"]
    ia, ib = {k: i for i, k in enumerate(a["key"])}, {k: i for i, k in enumerate(b["key"])}
    n = 0
    for k, j in ib.items():
        i = ia.get(k)
        if i is None or tuple((b["x"][j] // (0.5 * b["L"])).astype(int)) == (1, 0, 1):
            continue
        assert np.all(np.abs(b["Z"][j, ORIGIN:] - a["Z"][i, ORIGIN:]) <= 1e-14 * max(a["Z"][i, 0], 1e-30)), k
        n += 1
    assert n > 0.5 * len(b["m"]) and np.any(a["Z"][:, ORIGIN:] > 0)
