"""single_stars_winds_agb_ia test: stellar winds, AGB deaths and Type Ia SNe of IMF-sampled single stars, from the
stellar-evolution tables, through the mass-return (winds, AGB) and supernova (Type Ia) modes of the shared mechanical
feedback loop. See README.md."""
import os
import shutil
import subprocess
from pathlib import Path
import numpy as np
import h5py
import pytest
from gizmo.test import build_gizmo_for_test, get_stellar_evolution_tables, get_cooling_tables, mpi_launch_command

TEST_NAME = "single_stars_winds_agb_ia"
TEST_DIR = Path(__file__).parent
PC_KMS_IN_YR = 3.085678e13 / 3.155e7
FIRE_ELEMENTS = [1, 2, 3, 4, 6, 8, 10, 12, 16, 22]  # table element of metal slots 1..10 (slot 0: all metals C..Zn)


@pytest.fixture(scope="module")
def built():
    tables = get_stellar_evolution_tables()
    if not os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        build_gizmo_for_test(TEST_NAME)
    return tables


@pytest.fixture(scope="module")
def built_elements(built):
    """the FIRE metal layout (11 slots: total, He, C, N, O, Ne, Mg, Si, S, Ca, Fe); cooling on, metals do not depend on it"""
    if not os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        os.replace(TEST_DIR / "GIZMO", TEST_DIR / "GIZMO_base")
        build_gizmo_for_test(TEST_NAME, extra_config_flags=("COOLING", "COOL_METAL_LINES_BY_SPECIES"))
        os.replace(TEST_DIR / "GIZMO", TEST_DIR / "GIZMO_elements")
        os.replace(TEST_DIR / "GIZMO_base", TEST_DIR / "GIZMO")
        get_cooling_tables(str(TEST_DIR))
    return built


def run(tag, tables, ranks=2, which=("wind", "agb", "wd"), ic_args={}, overrides=None, exe="GIZMO"):
    import sys
    sys.path.insert(0, str(TEST_DIR))
    from make_single_stars_winds_agb_ia_ics import make_single_stars_winds_agb_ia_ics
    d = TEST_DIR / f"run_{tag}"
    if not os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN") and d.exists():
        shutil.rmtree(d)  # this run's own directory: no stale logs or snapshots
    (d / "output").mkdir(parents=True, exist_ok=True)
    ic = d / "output" / "snapshot_000.hdf5"
    if not os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        L = make_single_stars_winds_agb_ia_ics(str(ic), tables["StellarTablesFile"], which=which, **ic_args)
        params = Path(TEST_DIR / f"{TEST_NAME}.params").read_text().replace("SET_BY_TEST", tables["StellarTablesFile"])
        params = params.replace("BoxSize                            45.0", f"BoxSize                            {L}")
        for k, v in (overrides or {}).items():
            lines = [l for l in params.splitlines() if not (l.split() and l.split()[0] == k)]
            params = "\n".join(lines + [f"{k:34s} {v}"]) + "\n"
        (d / f"{TEST_NAME}.params").write_text(params)
        if exe != "GIZMO":
            for f in ("TREECOOL", "spcool_tables"):
                os.symlink(TEST_DIR / f, d / f)
        with open(d / f"test_{TEST_NAME}.out", "w") as out:
            subprocess.run(mpi_launch_command(ranks) + [str(TEST_DIR / exe), f"{TEST_NAME}.params", "2", "0"],
                           cwd=d, stdout=out, stderr=subprocess.STDOUT, env={**os.environ, "OMP_NUM_THREADS": "1"}, check=True)
    events = d / "output" / "single_star_events.txt"  # deaths only: winds are not logged
    return ic, sorted((d / "output").glob("snapshot_*.hdf5")), np.atleast_2d(np.loadtxt(events)) if events.exists() else np.zeros((0, 12))


def state(fname):
    with h5py.File(fname, "r") as F:
        g = F["PartType0"]
        m = g["Masses"][()]
        out = dict(t=float(F["Header"].attrs["Time"]), M=m.sum(), E=(0.5 * m * (g["Velocities"][()] ** 2).sum(1) + m * g["InternalEnergy"][()]).sum(), MZ=(m * g["Metallicity"][()].reshape(len(m), -1)[:, 0]).sum(),
                   p=(m[:, None] * g["Velocities"][()]).sum(0), stars={})
        if "PartType4" in F:
            s = F["PartType4"]
            out["p"] = out["p"] + (s["Masses"][()][:, None] * s["Velocities"][()]).sum(0)
            out["E"] += 0.5 * (s["Masses"][()] * (s["Velocities"][()] ** 2).sum(1)).sum()
            for k, i in enumerate(s["ParticleIDs"][()]):
                get = lambda name: float(s[name][k]) if name in s else 0.0
                out["stars"][int(i)] = dict(m=float(s["Masses"][k]), md=get("SingleStarDrawnMass"), bits=int(get("SingleStarEventBits")),
                                            wa=get("SingleStarLastWindLogAge"))
    return out


def wind_cumulative(tables_file, M, Z, log_age):
    """total cumulative wind ejecta [Msun] of a star at a (Z, M) grid node: linear in log age, as the code"""
    with h5py.File(tables_file, "r") as F:
        Zt, Mt, la = F["Z"][()], F["M_init"][()], F["log_age_yr"][()]
        iz, im = int(np.argmin(np.abs(Zt - Z))), int(np.argmin(np.abs(Mt - M)))
        assert Zt[iz] == Z and Mt[im] == M
    with h5py.File(str(Path(tables_file).parent / "stellar_wind_cumulative.hdf5"), "r") as W:
        cum = W["elem_ej_wind_cumulative"][iz, im, :, :].astype(np.float64)
    return sum(np.interp(log_age, la, cum[:, k]) for k in range(cum.shape[1]))


def wind_speed(tables_file, M, Z, log_age):
    """table wind speed [km/s] at a (Z, M) grid node: linear in log age, as the code"""
    with h5py.File(tables_file, "r") as F:
        Zt, Mt, la = F["Z"][()], F["M_init"][()], F["log_age_yr"][()]
        iz, im = int(np.argmin(np.abs(Zt - Z))), int(np.argmin(np.abs(Mt - M)))
        return float(np.interp(log_age, la, F["v_wind"][iz, im, :].astype(np.float64)))


def test_wind_energy(built):
    """the 35 Msun star alone: the energy the gas and the star gained (kinetic + thermal) against what the loop books, 1/2 M v^2
    of the mass returned at the table speed (the speed changes by 0.1% over the run, so the speed at the last delivery stands
    for all). The supernovae of the other test are left out: their 1e51 erg would hide the wind's 4e49."""
    ic, snaps, _ = run("wind", built, which=("wind",))
    s0, s1 = state(ic), state(snaps[-1])
    (w,) = s1["stars"].values()
    Msun_kms2_in_erg = 1.989e33 * 1.0e10
    booked = 0.5 * (35.0 - w["m"]) * wind_speed(built["StellarTablesFile"], 35.0, 0.014, w["wa"]) ** 2 * Msun_kms2_in_erg
    coupled = (s1["E"] - s0["E"]) * Msun_kms2_in_erg
    assert abs(coupled / booked - 1) < 1e-3, (coupled, booked)


def test_winds_agb_ia(built):
    ic, snaps, rows = run("r2", built)
    s0, s1 = state(ic), state(snaps[-1])
    ids = sorted(s0["stars"])
    wind, agb, wd = ids
    # the 35 Msun star has returned exactly the table's cumulative wind mass up to its last delivery
    w = s1["stars"][wind]
    assert w["wa"] > 0
    delivered = 35.0 - w["m"]
    expected = wind_cumulative(built["StellarTablesFile"], 35.0, 0.014, w["wa"])
    assert delivered > 0 and abs(delivered / expected - 1) < 1e-9, (delivered, expected)
    # it is still on the main sequence: its last delivery age is the age at the last step (to one step)
    age_yr = 1.0e6 + s1["t"] * PC_KMS_IN_YR
    with h5py.File(built["StellarTablesFile"], "r") as F:
        Zt, Mt = F["Z"][()], F["M_init"][()]
        tpms = F["t_PMS"][()].reshape(len(Zt), len(Mt))[int(np.argmin(np.abs(Zt - 0.014))), int(np.argmin(np.abs(Mt - 35.0)))]
    assert abs(10 ** w["wa"] - (age_yr - tpms)) <= 3e-4 * PC_KMS_IN_YR
    # the AGB death: one event, white-dwarf remnant of the table mass, ejecta = mass - remnant
    r_agb = rows[rows[:, 2] == 2]
    assert len(r_agb) == 1 and int(r_agb[0, 1]) == agb and int(r_agb[0, 3]) == 0
    assert abs(r_agb[0, 6] - (3.0 - r_agb[0, 7])) < 1e-12
    # Type Ia (hazard multiplied so that every eligible white dwarf explodes at once): the white dwarf of the initial
    # conditions at the first step, the new one from the AGB death at a later step; the whole mass at 1e51 erg, no remnant
    r_ia = rows[rows[:, 2] == 4]
    assert sorted(r_ia[:, 1].astype(int).tolist()) == sorted([wd, agb])
    assert wd not in s1["stars"] and agb not in s1["stars"]
    t_ia = dict(zip(r_ia[:, 1].astype(int), r_ia[:, 0]))
    assert t_ia[agb] > r_agb[0, 0]
    for r in r_ia:
        m_wd = s0["stars"][wd]["m"] if int(r[1]) == wd else r_agb[0, 7]
        assert r[5] == m_wd and r[6] == m_wd and r[8] == 1e51 and r[7] == 0
        assert r[9] > 0.9 * r[6]  # Type Ia ejecta are almost all metals
    # conservation: gas + stars mass and metal mass, total momentum
    M0 = s0["M"] + sum(v["m"] for v in s0["stars"].values())
    M1 = s1["M"] + sum(v["m"] for v in s1["stars"].values())
    assert abs(M1 - M0) <= 1e-12 * M0
    assert s1["MZ"] - s0["MZ"] >= r_ia[:, 9].sum() + r_agb[0, 9] - 1e-9  # plus the wind metals
    # momentum: the supernovae add none; the momentum-conserving wind coupling of the loop balances its radial kicks only
    # approximately on this lattice (measured: 3% of the first step's wind momentum, 0.35 Msun at ~2000 km/s)
    v_max = 3000.0  # km/s, above the table wind speed
    assert np.all(np.abs(s1["p"] - s0["p"]) <= 0.05 * delivered * v_max)


def test_wind_energy_in_outflowing_gas(built):
    """a heavy star's steady wind in hot gas moving out of it at 0.75 v_w: the gas gains what the wind offers. The sums of the
    weighting pass must use the wind's mass, not the star's: with the star's the momentum solve loses the work on the moving gas"""
    tables = built["StellarTablesFile"]
    M = 150.0
    with h5py.File(tables, "r") as F:
        Zt, Mt = F["Z"][()], F["M_init"][()]
        tpms = F["t_PMS"][()].reshape(len(Zt), len(Mt))[int(np.argmin(np.abs(Zt - 0.014))), int(np.argmin(np.abs(Mt - M)))]
    la = np.log10(1.0e6 - tpms)
    args = dict(M=M, n_H=1.0, T=1.0e7, v_radial=0.75 * wind_speed(tables, M, 0.014, la))
    _, snaps, _ = run("outflow", built, which=("wind",), ic_args=dict(args, last_wind_log_age=la))
    _, twin, _ = run("outflow_twin", built, which=("wind",), ic_args=dict(args, last_wind_log_age=9.9))  # same flow, no wind owed: its hydro drift cancels
    s0, s1 = state(twin[-1]), state(snaps[-1])
    (w,) = s1["stars"].values()
    assert s0["t"] == s1["t"]
    Msun_kms2_in_erg = 1.989e33 * 1.0e10
    offered = 0.5 * (M - w["m"]) * wind_speed(tables, M, 0.014, w["wa"]) ** 2 * Msun_kms2_in_erg
    gained = (s1["E"] - s0["E"]) * Msun_kms2_in_erg
    print(f"wind offered {offered:.5e} erg, gas and star gained {gained:.5e} erg, ratio {gained / offered:.4f}")
    assert offered > 0 and abs(gained / offered - 1) < 0.02, (gained, offered)


def test_one_and_four_ranks(built):
    """the same wind mass, AGB death and Type Ia on 1 and 4 ranks (wind mass to rounding: the loop removes a sum of
    deposits)"""
    _, s_ref, rows_ref = run("r2", built)
    for r in (1, 4):
        _, snaps, rows = run(f"r{r}", built, ranks=r)
        a, b = state(s_ref[-1]), state(snaps[-1])
        assert sorted(a["stars"]) == sorted(b["stars"])
        for k in a["stars"]:
            assert abs(a["stars"][k]["m"] - b["stars"][k]["m"]) <= 1e-12 * a["stars"][k]["m"], (r, k)
        key = lambda x: x[np.lexsort((x[:, 1], x[:, 2]))][:, [1, 2, 3, 4, 6, 7, 8]]
        assert np.allclose(key(rows), key(rows_ref), rtol=1e-12, atol=0), r


def test_wind_elements_independent_of_cadence(built_elements):
    """a 40 Msun star from 4.40 to 4.84 Myr, in metal-free gas, delivering every 1 kyr or every 20 kyr; from 4.59 Myr one element's
    table cumulative is flat while the total still rises. In both runs each metal slot of the gas gains exactly the table's
    cumulative wind ejecta of its elements between the start and the age of the last delivery (1e-12), and the star loses what the
    gas gains (1e-12): the winds telescope, so how often a star delivers does not change what it returns"""
    tf = built_elements["StellarTablesFile"]
    with h5py.File(tf, "r") as F, h5py.File(str(Path(tf).parent / "stellar_wind_cumulative.hdf5"), "r") as W:
        Zt, Mt, la = F["Z"][()], F["M_init"][()], F["log_age_yr"][()]
        iz, im = int(np.argmin(np.abs(Zt - 0.014))), int(np.argmin(np.abs(Mt - 40.0)))
        assert Mt[im] == 40.0
        cum = W["elem_ej_wind_cumulative"][iz, im, :, :].astype(np.float64)
        t_pms = F["t_PMS"][()].reshape(len(Zt), len(Mt))[iz, im]
    def table(log_age):  # as the code interpolates, at a grid node
        ia = int(np.searchsorted(la, log_age) - 1); f = (log_age - la[ia]) / (la[ia + 1] - la[ia])
        y = cum[ia] * (1 - f) + cum[ia + 1] * f
        return np.r_[y[2:].sum(), y[FIRE_ELEMENTS]], y.sum()
    def slots(fname):
        with h5py.File(fname, "r") as G:
            g = G["PartType0"]
            return (g["Masses"][()][:, None] * g["Metallicity"][()]).sum(0), g["Masses"][()].sum(), float(G["PartType4/Masses"][0]), float(G["PartType4/SingleStarLastWindLogAge"][0])
    la0 = np.log10(4.40e6 - t_pms)
    for cap in (1.0e3, 2.0e4):
        ic, snaps, _ = run(f"wind_cadence_{cap:.0f}", built_elements, which=("wind",), exe="GIZMO_elements",
                           ic_args=dict(n_metals=11, M=40.0, wind_age_yr=4.40e6, gas_metals=False, last_wind_log_age=la0),
                           overrides=dict(MaxStellarTimestep=cap, TimeMax=0.45, MaxSizeTimestep=0.05, TimeBetSnapshot=0.45, TimeBetStatistics=0.45))
        (z0, m0, _, _), (z1, m1, ms, wa) = slots(ic), slots(snaps[-1])
        assert 10 ** wa > 4.80e6 - t_pms, wa  # the star delivered through the flat interval
        (y1, M1), (y0, M0) = table(wa), table(la0)
        assert np.all(np.abs((z1 - z0) / (y1 - y0) - 1) < 1e-12), (cap, (z1 - z0) / (y1 - y0) - 1)
        assert abs((m1 - m0) / (40.0 - ms) - 1) < 1e-12 and abs((40.0 - ms) / (M1 - M0) - 1) < 1e-12
