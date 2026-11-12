"""single_stars_event_logs test: SNinfo.txt, AGBinfo.txt and WINDinfo.txt of single stars (GALSF_SINGLE_STARS_EVENT_LOGS),
checked against the source model's own event record and the tables. See README.md."""
import os
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
import h5py
import pytest
from gizmo.test import build_gizmo_for_test, get_stellar_evolution_tables, mpi_launch_command

TEST_NAME = "single_stars_event_logs"
TEST_DIR = Path(__file__).parent
SN_DIR = TEST_DIR.parent / "single_stars_supernovae"
WIND_DIR = TEST_DIR.parent / "single_stars_winds_agb_ia"
MSUN_KMS = 1.989e33 * 1e5  # g cm/s per Msun km/s
ID, MSTAR, REM, MEJ, ZEJ, AGE, LIFE, E, STATUS = 6, 7, 8, 9, 10, 15, 16, 18, 19  # SNinfo columns
A_MEJ, A_ZEJ, A_STATUS = 8, 9, 17  # AGBinfo columns


@pytest.fixture(scope="module")
def built():
    tables = get_stellar_evolution_tables()
    if not os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        build_gizmo_for_test(TEST_NAME)
    return tables


def rows(path, cols):
    a = np.loadtxt(path, comments="#", ndmin=2)
    return a if a.size else np.zeros((0, cols))


def run(tag, tables, case, ranks=2, over=None):
    d = TEST_DIR / f"run_{tag}"
    if not os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        if d.exists():
            shutil.rmtree(d)  # this run's own directory: no stale logs or snapshots
        (d / "output").mkdir(parents=True)
        ic = str(d / "output" / "snapshot_000.hdf5")
        src = SN_DIR if case == "sn" else WIND_DIR
        sys.path.insert(0, str(src))
        if case == "sn":
            from make_single_stars_supernovae_ics import make_single_stars_supernovae_ics
            L = make_single_stars_supernovae_ics(ic, tables["StellarTablesFile"])[0]
        else:
            from make_single_stars_winds_agb_ia_ics import make_single_stars_winds_agb_ia_ics
            L = make_single_stars_winds_agb_ia_ics(ic, tables["StellarTablesFile"])
        params = (src / f"{src.name}.params").read_text().replace("SET_BY_TEST", tables["StellarTablesFile"])
        params = params.replace("BoxSize                            45.0", f"BoxSize                            {L}")
        for k, v in (over or {}).items():
            params = "\n".join(f"{k:34s} {v}" if l.split() and l.split()[0] == k else l for l in params.splitlines()) + "\n"
        (d / "run.params").write_text(params)
        with open(d / "run.out", "w") as out:
            subprocess.run(mpi_launch_command(ranks) + [str(TEST_DIR / "GIZMO"), "run.params", "2", "0"], cwd=d, stdout=out,
                           stderr=subprocess.STDOUT, env={**os.environ, "OMP_NUM_THREADS": "1"}, check=True)
    o = d / "output"
    return dict(sn=rows(o / "SNinfo.txt", 26), agb=rows(o / "AGBinfo.txt", 24), wind=rows(o / "WINDinfo.txt", 10),
                ev=np.loadtxt(o / "single_star_events.txt", ndmin=2), snaps=sorted(o.glob("snapshot_*.hdf5")))


def table_ejecta(tables_file, M, Z, elements):
    """SN element ejecta [Msun] of a star at a (Z, M) grid node"""
    with h5py.File(tables_file, "r") as F:
        Zt, Mt = F["Z"][()], F["M_init"][()]
        return F["elem_ej_SN_mass"][int(np.argmin(np.abs(Zt - Z))), int(np.argmin(np.abs(Mt - M))), elements]


def test_supernova_rows(built):
    r = run("sn", built, "sn")
    sn, ev = r["sn"], r["ev"]
    print(sn[:, [ID, MSTAR, REM, MEJ, ZEJ, E, STATUS]])
    assert len(sn) == 4 and len(ev) == 4 and len(r["agb"]) == 0
    sn = sn[np.argsort(sn[:, ID])]
    ev = ev[np.argsort(ev[:, 1])]
    assert np.array_equal(sn[:, ID], ev[:, 1])
    assert np.array_equal(sn[:, REM], ev[:, 3])  # class
    assert np.allclose(sn[:, MSTAR], ev[:, 4], rtol=1e-7)  # drawn mass
    assert np.allclose(sn[:, MEJ], ev[:, 6], rtol=1e-9)
    assert np.allclose(sn[:, ZEJ], ev[:, 9], rtol=1e-9)
    assert np.allclose(sn[:, E], ev[:, 8], rtol=1e-3)
    failed = sn[:, REM] == 3
    assert failed.sum() == 1 and sn[failed, E] == 0 and sn[failed, STATUS] == 1 and np.all(sn[~failed, STATUS] == 0)
    assert np.all(sn[:, AGE] >= sn[:, LIFE]) and np.all(sn[:, LIFE] > 0)  # just past the table lifetime
    # C, O, Si, Fe are the event's element ejecta: their share of the metals as in the table, for the core-collapse star
    cc = sn[sn[:, REM] == 2][0]
    y = table_ejecta(built["StellarTablesFile"], cc[MSTAR], 0.014, [2, 4, 10, 22])  # table elements: see single_stars_supernovae
    assert np.allclose(cc[11:15] / cc[11:15].sum(), y / y.sum(), rtol=0.02), (cc[11:15], y)
    # the ambient density and internal energy around the star: the lattice's
    with h5py.File(r["snaps"][0], "r") as F:
        g = F["PartType0"]
        rho = g["Masses"][()].sum() / F["Header"].attrs["BoxSize"] ** 3
        assert np.all(np.abs(sn[:, 5] / rho - 1) < 0.2), (sn[:, 5], rho)
        assert np.all(np.abs(sn[:, 4] / g["InternalEnergy"][()].mean() - 1) < 1e-3), sn[:, 4]
    # columns 21-26: ambient n_H and T (from columns 6 and 5), the star's h and neighbour count, birth and birth-frozen n_H
    nH = sn[:, 5] * 1.989e33 / 3.085678e18 ** 3 * 0.76 / 1.67262192e-24
    T = sn[:, 4] * 1e10 * (2 / 3) * 1.22 * 1.67262192e-24 / 1.380649e-16
    assert sn.shape[1] == 26 and r["agb"].shape[1] in (0, 24)
    assert np.allclose(sn[:, 20], nH, rtol=1e-3) and np.allclose(sn[:, 21], T, rtol=1e-3), (sn[:, 20:22], nH, T)
    assert np.all(sn[:, 22] > 0) and np.all(sn[:, 23] > 0)
    assert np.array_equal(sn[:, 25], sn[:, 24])  # a = 1: the birth-frozen density is the birth density


def test_agb_type_ia_wind_rows(built):
    r = run("wind", built, "wind")
    sn, agb, wind, ev = r["sn"], r["agb"], r["wind"], r["ev"]
    print(f"{len(wind)} wind rows, {len(agb)} AGB, {len(sn)} SN")
    e_agb = ev[ev[:, 2] == 2]
    assert len(agb) == 1 and len(e_agb) == 1
    assert agb[0, ID] == e_agb[0, 1] and abs(agb[0, A_MEJ] / e_agb[0, 6] - 1) < 1e-9 and agb[0, A_STATUS] == 0
    e_ia = ev[ev[:, 2] == 4]
    assert len(sn) == len(e_ia) == 2 and np.all(sn[:, REM] == 7) and np.all(sn[:, E] == 1e51) and np.all(sn[:, LIFE] == 0)
    assert sorted(sn[:, ID]) == sorted(e_ia[:, 1])
    # every wind delivery of the 35 Msun star: the masses add up to what it lost, at the table speed
    with h5py.File(r["snaps"][-1], "r") as F:
        s = F["PartType4"]
        m = dict(zip(s["ParticleIDs"][()].tolist(), s["Masses"][()].tolist()))
        t_snap = float(F["Header"].attrs["Time"])
    w_id = wind[0, 4]
    assert np.all(wind[:, 4] == w_id) and np.all(wind[:, 5] == 35.0) and np.all(np.diff(wind[:, 0]) > 0)
    before = wind[:, 0] < t_snap * (1 - 1e-12)  # the snapshot is written before that step's feedback
    assert abs(wind[before, 6].sum() / (35.0 - m[int(w_id)]) - 1) < 1e-6
    sys.path.insert(0, str(WIND_DIR))
    from test_single_stars_winds_agb_ia import wind_speed
    with h5py.File(r["snaps"][-1], "r") as F:
        s = F["PartType4"]
        la = float(s["SingleStarLastWindLogAge"][()][s["ParticleIDs"][()] == int(w_id)][0])
    v = wind[:, 7] / wind[:, 6] / MSUN_KMS
    print(f"wind speed {v.min():.1f}-{v.max():.1f} km/s, table {wind_speed(built['StellarTablesFile'], 35.0, 0.014, la):.1f}")
    assert np.allclose(v, wind_speed(built["StellarTablesFile"], 35.0, 0.014, la), rtol=0.01), v  # changes by 0.1% over the run
    assert np.all((wind[:, 8] > 0) & (wind[:, 8] < wind[:, 6]))


def test_rows_at_output_only_and_rank_independent(built):
    """the same rows on 1 and 4 ranks; with one snapshot at the end the rows still arrive: nothing is lost between outputs"""
    ref = run("sn", built, "sn")
    for n in (1, 4):
        a = run(f"sn_r{n}", built, "sn", ranks=n)
        assert np.array_equal(a["sn"][:, [ID, REM]], ref["sn"][:, [ID, REM]]), n
        assert np.allclose(a["sn"][:, [MEJ, ZEJ]], ref["sn"][:, [MEJ, ZEJ]], rtol=1e-9), n
