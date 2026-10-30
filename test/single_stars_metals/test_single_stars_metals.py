"""single_stars_metals test: the metals of single-star builds in the FIRE-3 layout (slot 0 total Z, 1 He, 2-10 C, N, O,
Ne, Mg, Si, S, Ca, Fe; H implicit), with SGChem network 5 (production chemistry). See README.md."""
import os
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
import h5py
import pytest
from gizmo.test import build_gizmo_for_test, get_stellar_evolution_tables, get_cooling_tables, mpi_launch_command

TEST_NAME = "single_stars_metals"
TEST_DIR = Path(__file__).parent
sys.path.insert(0, str(TEST_DIR))
from make_single_stars_metals_ics import make_single_stars_metals_ics  # noqa: E402

FIRE_ELEMENTS = [1, 2, 3, 4, 6, 8, 10, 12, 16, 22]  # table element (H=0 .. Zn=26) of slots 1..10
X_H_PRIMORDIAL = 0.7065  # HYDROGEN_MASSFRAC of SGChem builds
EXTRA_ELEMENTS = [5, 7, 9, 11, 13, 14, 15, 17, 18, 19, 20, 21, 23, 24, 25, 26]  # F Na Al P Cl Ar K Sc Ti V Cr Mn Co Ni Cu Zn
ELEMENTS27 = ("GALSF_SINGLE_STARS_27_ELEMENT_YIELDS", "METALS_EXTRA_PASSIVE_SPECIES=16")
BUILDS = {}


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


def solar_table(tables_file):
    """the tables' initial composition at their Z = 0.014 node (Asplund+09 present day), elements H..Zn"""
    with h5py.File(tables_file, "r") as F:
        Zt = F["Z"][()]
        iz = int(np.argmin(np.abs(Zt - 0.014)))
        assert Zt[iz] == 0.014
        return F["X_init"][()].reshape(len(Zt), -1)[iz].astype(float)


def run(exe, tag, tables, metals, stars=(), flag=2, over=None, ranks=2, **ic_args):
    d = TEST_DIR / f"run_{tag}"
    if d.exists():
        shutil.rmtree(d)
    (d / "output").mkdir(parents=True)
    ic = d / ("snapshot.hdf5" if flag == 0 else "output/snapshot_000.hdf5")
    L = make_single_stars_metals_ics(str(ic), metals, stars, **ic_args)
    for f in ("TREECOOL", "spcool_tables"):
        if (TEST_DIR / f).exists():
            os.symlink(TEST_DIR / f, d / f)
    T = tables["StellarTablesFile"]
    over = {"BoxSize": L, **(over or {})}
    p = (TEST_DIR / f"{TEST_NAME}.params").read_text()
    p = p.replace("StellarTablesFile           SET_BY_TEST", f"StellarTablesFile           {T}")
    p = p.replace("SGChemTablesPath            SET_BY_TEST", f"SGChemTablesPath            {os.environ.get('SGCHEM_TABLES_DIR', os.path.dirname(T))}")
    p = "\n".join([l for l in p.splitlines() if not (l.split() and l.split()[0] in over)] + [f"{k:27s} {v}" for k, v in over.items()]) + "\n"
    (d / f"{TEST_NAME}.params").write_text(p)
    with open(d / f"test_{TEST_NAME}.out", "w") as out:
        subprocess.run(mpi_launch_command(ranks) + [str(exe), f"{TEST_NAME}.params", str(flag), "0"], cwd=d, stdout=out,
                       stderr=subprocess.STDOUT, env={**os.environ, "OMP_NUM_THREADS": "1"}, check=True)
    return d


def gas_metals(snap):
    with h5py.File(snap, "r") as F:
        m = F["PartType0/Masses"][()]
        return m, F["PartType0/Metallicity"][()].reshape(len(m), -1)


@pytest.mark.parametrize("z_solar,layout", [(1.0, "fire"), (0.3, "fire"), (0.3, "27")])
def test_solar_abundances(tables, z_solar, layout):
    """a fresh start with InitMetallicity: total Z = 0.014 x InitMetallicity and the tables' solar element pattern, as
    stella_dev (Asplund+09 present day; the tables' initial composition at Z = 0.014); He primordial + scaled excess;
    with the 27 elements the other 16 elements too"""
    n = 27 if layout == "27" else 11
    exe = build("elements27", ELEMENTS27) if layout == "27" else build("default")
    d = run(exe, f"solar_{z_solar}_{layout}", tables, np.zeros(n), flag=0, over={"InitMetallicity": z_solar})
    _, Z = gas_metals(d / "output" / "snapshot_000.hdf5")
    X = solar_table(tables["StellarTablesFile"])
    expect = np.concatenate([[0.014], X[FIRE_ELEMENTS], X[EXTRA_ELEMENTS]])[:n] * z_solar
    expect[1] = (1 - X_H_PRIMORDIAL) + (X[1] - (1 - X_H_PRIMORDIAL)) * z_solar
    assert Z.shape[1] == n and np.all(np.abs(Z / expect - 1) <= 1e-10), (Z[0], expect)


def solar27(tables_file):
    """the 27-element layout at the tables' solar composition; slot 0 = the sum of the 25 metals"""
    X = solar_table(tables_file)
    return np.concatenate([[X[2:].sum()], X[FIRE_ELEMENTS], X[EXTRA_ELEMENTS]])


def event_stars(tables_file):
    """a 20 Msun core collapse (winds returned), a 3 Msun AGB death, a 1 Gyr white dwarf (Type Ia) and a 35 Msun star
    1 Myr old (winds), each alone in its region, Z = 0.014"""
    with h5py.File(tables_file, "r") as F:
        Zt, Mt = F["Z"][()], F["M_init"][()]
        iz, n2 = int(np.argmin(np.abs(Zt - 0.014))), (len(Zt), len(Mt))
        life, pms, rem = (F[k][()].reshape(n2)[iz] for k in ("lifetime_yr", "t_PMS", "remnant_mass"))
    i = {m: int(np.argmin(np.abs(Mt - m))) for m in (20.0, 3.0)}
    return [(20.0, 20.0, 1.01 * (life[i[20.0]] + pms[i[20.0]]), (0.25, 0.25, 0.25), 0, np.log10(life[i[20.0]])),
            (3.0, 3.0, 1.001 * (life[i[3.0]] + pms[i[3.0]]), (0.75, 0.75, 0.25), 0),
            (3.0, float(rem[i[3.0]]), 1.0e9, (0.25, 0.75, 0.75), 2),
            (35.0, 35.0, 1.0e6, (0.75, 0.25, 0.75), 0)]


EVENT_RUNS = {}


def event_life(tables_file, m):
    """table main-sequence lifetime [yr] of a star of drawn mass m at Z = 0.014"""
    with h5py.File(tables_file, "r") as F:
        Zt, Mt = F["Z"][()], F["M_init"][()]
        return float(F["lifetime_yr"][()].reshape(len(Zt), len(Mt))[int(np.argmin(np.abs(Zt - 0.014))), int(np.argmin(np.abs(Mt - m)))])


def run_events(tables, exe, tag, n_metals):
    if tag in EVENT_RUNS:
        return EVENT_RUNS[tag]
    over = {"TimeMax": 5e-3, "TimeBetSnapshot": 5e-3}
    d = run(exe, tag, tables, solar27(tables["StellarTablesFile"])[:n_metals], event_stars(tables["StellarTablesFile"]), over=over)
    EVENT_RUNS[tag] = sorted((d / "output").glob("snapshot_*.hdf5")), np.atleast_2d(np.loadtxt(d / "output" / "single_star_events.txt"))
    return EVENT_RUNS[tag]


def test_27_element_closure(tables):
    """27 elements, winds, a core collapse, an AGB death and two Type Ia: in every cell (split ones too) total Z equals
    the sum of the 25 metal slots (the element yields sum to the ejecta metals; 1e-12, the start holds it exactly); the
    gas metal gain equals the events' metals plus the winds (positive in every slot)"""
    snaps, rows = run_events(tables, build("elements27", ELEMENTS27), "events27", 27)
    m0, Z0 = gas_metals(snaps[0]); m1, Z1 = gas_metals(snaps[-1])
    assert Z1.shape[1] == 27 and sorted(rows[:, 2].astype(int).tolist()) == [1, 2, 4, 4]
    closure = np.abs(Z1[:, 0] - Z1[:, 2:].sum(1)) / Z1[:, 0]
    dZ = (m1[:, None] * Z1).sum(0) - (m0[:, None] * Z0).sum(0)
    print(f"closure: max |Z0 - sum Z_k| / Z0 per cell {closure.max():.3g} (start {np.max(np.abs(Z0[:, 0] - Z0[:, 2:].sum(1)) / Z0[:, 0]):.3g}); "
          f"box {abs(dZ[0] - dZ[2:].sum()) / (m1 * Z1[:, 0]).sum():.3g}; metals gained {dZ[0]:.6g} Msun, events {rows[:, 9].sum():.6g}")
    assert np.all(closure <= 1e-12)
    assert np.all(dZ[2:] > 0) and dZ[0] > rows[:, 9].sum()


def test_27_element_shared_slots_identical(tables):
    """the same events with and without the 27 elements: slots 0-10, masses, energies, velocities and the star masses
    are bitwise identical"""
    a, ra = run_events(tables, build("default"), "events11", 11)
    b, rb = run_events(tables, build("elements27", ELEMENTS27), "events27", 27)
    assert ra.tobytes() == rb.tobytes()
    with h5py.File(a[-1], "r") as A, h5py.File(b[-1], "r") as B:
        for pt in ("PartType0", "PartType4"):
            key = lambda F: np.lexsort((F[pt]["ParticleChildIDsNumber"][()], F[pt]["ParticleIDs"][()])) if "ParticleChildIDsNumber" in F[pt] else np.argsort(F[pt]["ParticleIDs"][()])
            ia, ib = key(A), key(B)
            for k in A[pt]:
                if k == "Metallicity":
                    za, zb = A[pt][k][()][ia], B[pt][k][()][ib]
                    assert za.tobytes() == np.ascontiguousarray(zb[:, :11]).tobytes(), pt
                elif k in B[pt]:
                    assert A[pt][k][()][ia].tobytes() == B[pt][k][()][ib].tobytes(), (pt, k)


def test_27_element_supernova_yields(tables):
    """one 20 Msun core collapse: every element slot of the gas gains the table SN element ejecta scaled to the ejecta
    mass (1e-11); total Z gains the event's metals (1e-12 of the gas metal mass)"""
    d = run(build("elements27", ELEMENTS27), "sn27", tables, solar27(tables["StellarTablesFile"]), event_stars(tables["StellarTablesFile"])[:1],
            over={"TimeMax": 5e-5, "TimeBetSnapshot": 5e-5})
    snaps = sorted((d / "output").glob("snapshot_*.hdf5"))
    rows = np.atleast_2d(np.loadtxt(d / "output" / "single_star_events.txt"))
    m0, Z0 = gas_metals(snaps[0]); m1, Z1 = gas_metals(snaps[-1])
    dZ = (m1[:, None] * Z1).sum(0) - (m0[:, None] * Z0).sum(0)
    with h5py.File(tables["StellarTablesFile"], "r") as T:
        Zt, Mt = T["Z"][()], T["M_init"][()]
        y = T["elem_ej_SN_mass"][()].reshape(len(Zt), len(Mt), -1)[int(np.argmin(np.abs(Zt - 0.014))), int(np.argmin(np.abs(Mt - 20.0)))].astype(float)
    y *= rows[0, 6] / y.sum()
    expect = np.concatenate([[y[2:].sum()], y[FIRE_ELEMENTS], y[EXTRA_ELEMENTS]])
    rel = np.where(expect > 0, np.abs(dZ / np.where(expect > 0, expect, 1) - 1), np.abs(dZ))  # Ni: none from this star
    ledger = abs(dZ[0] - rows[0, 9]) / (m1 * Z1[:, 0]).sum()
    print(f"element gains vs table, max relative deviation {rel.max():.3g}; total Z gain - event metals, relative to the gas metals {ledger:.3g}")
    assert np.all(rel <= 1e-11) and ledger <= 1e-12


def test_dust_above_metals_stops(tables):
    """SGCHEM_DGR_INDEPENDENT_OF_METALLICITY on a fresh start: a constant dust-to-gas ratio above the metal mass
    (SGChemDustToGasRatio 1 = 0.01, InitMetallicity 0.3 = Z 0.0042) stops at start-up with a message, as stella_dev;
    at InitMetallicity 1 it runs"""
    exe = build("dgr_const", ("SGCHEM_DGR_INDEPENDENT_OF_METALLICITY",))
    with pytest.raises(subprocess.CalledProcessError):
        run(exe, "dgr_const_low_z", tables, np.zeros(11), flag=0, over={"InitMetallicity": 0.3})
    assert "dust mass above the metal mass" in (TEST_DIR / "run_dgr_const_low_z" / f"test_{TEST_NAME}.out").read_text()
    run(exe, "dgr_const_solar", tables, np.zeros(11), flag=0, over={"InitMetallicity": 1.0})


DUST = ("GALSF_ISMDUSTCHEM_MODEL=(1+2)",)  # silicate, carbon, metallic iron
DUST_PARAMS = {"Initial_ISMDustChem_Depletion": 0, "Initial_ISMDustChem_SiltoCarbRatio": 2.0, "SNeIIDustScaling": 1, "SNeIaDustScaling": 1,
               "AGBDustScaling": 1, "DustAccretionScaling": 0, "ThermalSputteringScaling": 0, "SNeGasClearedOfDustScaling": 0, "AccretionTcutoffScaling": 1}
SIL_ATOMS = {4: 3.63, 6: 1.06, 7: 1.0, 10: 0.571}  # O, Mg, Si, Fe per Si in the model's default silicate
ATOMIC = {2: 12.01, 4: 15.99, 6: 24.305, 7: 28.086, 10: 55.845}


def stella_dust(y, kind):
    """dust from an event's ejecta y (FIRE slots, Msun) by stella_dev's condensation (Choban+22 efficiencies), in the
    species of GALSF_ISMDUSTCHEM_MODEL: element dust (FIRE slots, slot 0 total) and species (silicate, carbon, iron)"""
    y = np.array(y, dtype=float)
    f_sil = f_c = f_fe = 0.0; carb = None
    if kind == "sn":
        f_c, f_sil, f_fe = 0.15, 3.5e-4, 1e-3
    elif kind == "ia":
        f_fe = 5e-3
    elif y[2] / 12.0 > y[4] / 16.0:
        carb = max(y[2] - 0.75 * y[4], 0.0)  # carbon-rich: C not in CO
    else:
        f_sil = 0.8; y[4] = max(y[4] - 4.0 / 3.0 * y[2], 0.0) / 0.8  # oxygen-rich: O not in CO
    W = sum(n * ATOMIC[k] for k, n in SIL_ATOMS.items())
    key = min(SIL_ATOMS, key=lambda k: y[k] / ATOMIC[k] / SIL_ATOMS[k])
    sil = f_sil * y[key] * W / (SIL_ATOMS[key] * ATOMIC[key]) if all(y[k] > 0 for k in SIL_ATOMS) else 0.0
    species = np.array([sil, f_c * y[2] if carb is None else carb, f_fe * y[10]])
    dust = np.zeros(11)
    for k, n in SIL_ATOMS.items():
        dust[k] += sil * n * ATOMIC[k] / W
    dust[2] += species[1]; dust[10] += species[2]; dust[0] = dust[1:].sum()
    return dust, species


def test_single_star_dust_yields(tables):
    """GALSF_ISMDUSTCHEM_MODEL with single stars: a 20 Msun core collapse, a 3 Msun AGB death and a white dwarf's Type Ia
    give the gas the dust of stella_dev's condensation of their table ejecta, in the model's element
    dust, species and sources (1e-10); the model's accretion, sputtering and SN shock destruction off for the test"""
    T = tables["StellarTablesFile"]
    stars = event_stars(T)[:3]
    stars[1] = stars[1] + (np.log10(event_life(T, 3.0)),)  # AGB: winds counted as returned
    d = run(build("dust", DUST), "dust", tables, np.concatenate([solar27(T)[:11], np.zeros(18)]), stars,
            over={"TimeMax": 5e-4, "TimeBetSnapshot": 5e-4, **DUST_PARAMS})
    snaps = sorted((d / "output").glob("snapshot_*.hdf5"))
    rows = np.atleast_2d(np.loadtxt(d / "output" / "single_star_events.txt"))
    rows = rows[rows[:, 0] < 5e-4 * (1 - 1e-9)]  # an event at TimeMax deposits after the last snapshot
    assert sorted(rows[:, 2].astype(int).tolist()) == [1, 2, 4]
    with h5py.File(T, "r") as F:
        Zt, Mt = F["Z"][()], F["M_init"][()]
        iz, n3 = int(np.argmin(np.abs(Zt - 0.014))), (len(Zt), len(Mt), -1)
        ej = {"sn": F["elem_ej_SN_mass"][()].reshape(n3)[iz, int(np.argmin(np.abs(Mt - 20.0)))],
              "agb": F["elem_ej_AGB_mass"][()].reshape(n3)[iz, int(np.argmin(np.abs(Mt - 3.0)))], "ia": F["type_ia_yields"][()]}
    want_d, want_s, want_src = np.zeros(11), np.zeros(3), np.zeros(4)
    for r in rows:
        kind, src = {1: ("sn", 2), 2: ("agb", 3), 4: ("ia", 1)}[int(r[2])]
        y = np.asarray(ej[kind], dtype=float); y = y * r[6] / y.sum()
        dd, ss = stella_dust(np.concatenate([[y[2:].sum()], y[FIRE_ELEMENTS]]), kind)
        want_d += dd; want_s += ss; want_src[src] += dd[0]
    with h5py.File(snaps[-1], "r") as F:
        m = F["PartType0/Masses"][()]
        dz = F["PartType0/DustMetallicity"][()].reshape(len(m), -1)  # element dust, then the sources as fractions of it
        got_d = np.concatenate([(m[:, None] * dz[:, :11]).sum(0), ((m * dz[:, 0])[:, None] * dz[:, 11:15]).sum(0)])
        got_s = (m[:, None] * F["PartType0/DustSpeciesAbundance"][()].reshape(len(m), -1)).sum(0)
    print(f"dust from the stars [Msun]: total {got_d[0]:.10g} (expected {want_d[0]:.10g}), species {got_s} (expected {want_s}), "
          f"max relative deviation {np.max(np.abs(got_d[:11] - want_d)) / want_d.max():.3g}")
    for got, want in ((got_d[:11], want_d), (got_s, want_s)):
        assert np.all(np.abs(got - want) <= 1e-10 * want.max()), (got, want)
    if len(got_d) >= 15:
        assert np.all(np.abs(got_d[11:15] - want_src) <= 1e-10 * want_src.max())


def test_chemistry_gas_phase_from_live_dust(tables):
    """GALSF_ISMDUSTCHEM_MODEL with SGChem: the chemistry's gas-phase C, O, Si are the cell's metals minus its live dust
    (as stella_dev with its dust model): with all carbon in dust, no CO forms (n_H = 1e3, 100 K, 20 kyr); the
    default 50 % carbon depletion would leave half of it in the gas"""
    X = solar_table(tables["StellarTablesFile"])
    Z = np.concatenate([solar27(tables["StellarTablesFile"])[:11], np.zeros(18)])
    dust = np.zeros(15); dust[2] = Z[2]; dust[0] = Z[2]; dust[11 + 2] = 1.0  # all C as carbonaceous dust, from SNe
    d = run(build("dust", DUST), "dust_all_carbon", tables, Z, n_H=1e3, over={"TimeMax": 2e-2, "TimeBetSnapshot": 2e-2, **DUST_PARAMS},
            gas_fields={"DustMetallicity": dust, "DustSpeciesAbundance": np.array([0.0, Z[2], 0.0])})
    with h5py.File(sorted((d / "output").glob("snapshot_*.hdf5"))[-1], "r") as F:
        g = F["PartType0"]
        x_co, x_h2 = g["SGChemTracAbund"][()][:, 2], g["SGChemTracAbund"][()][:, 0]
        c_dust = g["DustMetallicity"][()][:, 2]
    print(f"all C in dust: x(CO) max {x_co.max():.3g}, x(H2) {x_h2.mean():.3g}, C dust / C {np.mean(c_dust) / Z[2]:.6f}")
    assert x_h2.mean() > 1e-4 and np.all(c_dust > 0.99 * Z[2])  # the chemistry ran; the carbon stayed in dust
    assert np.all(x_co == 0)
