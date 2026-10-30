"""single_stars_winds_agb_ia test: stellar winds of IMF-sampled single stars, from the stellar-evolution tables, through
the mass-return mode of the shared mechanical feedback loop. See README.md."""
import os
import shutil
import subprocess
from pathlib import Path
import numpy as np
import h5py
import pytest
from gizmo.test import build_gizmo_for_test, get_stellar_evolution_tables, mpi_launch_command

TEST_NAME = "single_stars_winds_agb_ia"
TEST_DIR = Path(__file__).parent
PC_KMS_IN_YR = 3.085678e13 / 3.155e7


@pytest.fixture(scope="module")
def built():
    tables = get_stellar_evolution_tables()
    if not os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        build_gizmo_for_test(TEST_NAME)
    return tables


def run(tag, tables, ranks=2):
    import sys
    sys.path.insert(0, str(TEST_DIR))
    from make_single_stars_winds_agb_ia_ics import make_single_stars_winds_agb_ia_ics
    d = TEST_DIR / f"run_{tag}"
    if not os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN") and d.exists():
        shutil.rmtree(d)  # this run's own directory: no stale logs or snapshots
    (d / "output").mkdir(parents=True, exist_ok=True)
    ic = d / "output" / "snapshot_000.hdf5"
    if not os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        L = make_single_stars_winds_agb_ia_ics(str(ic), tables["StellarTablesFile"])
        params = Path(TEST_DIR / f"{TEST_NAME}.params").read_text().replace("SET_BY_TEST", tables["StellarTablesFile"])
        params = params.replace("BoxSize                            45.0", f"BoxSize                            {L}")
        (d / f"{TEST_NAME}.params").write_text(params)
        with open(d / f"test_{TEST_NAME}.out", "w") as out:
            subprocess.run(mpi_launch_command(ranks) + [str(TEST_DIR / "GIZMO"), f"{TEST_NAME}.params", "2", "0"],
                           cwd=d, stdout=out, stderr=subprocess.STDOUT, env={**os.environ, "OMP_NUM_THREADS": "1"}, check=True)
    events = d / "output" / "single_star_events.txt"  # deaths only: winds are not logged
    return ic, sorted((d / "output").glob("snapshot_*.hdf5")), np.atleast_2d(np.loadtxt(events)) if events.exists() else np.zeros((0, 12))


def state(fname):
    with h5py.File(fname, "r") as F:
        g = F["PartType0"]
        m = g["Masses"][()]
        out = dict(t=float(F["Header"].attrs["Time"]), M=m.sum(), MZ=(m * g["Metallicity"][()].reshape(len(m), -1)[:, 0]).sum(),
                   p=(m[:, None] * g["Velocities"][()]).sum(0), stars={})
        if "PartType4" in F:
            s = F["PartType4"]
            out["p"] = out["p"] + (s["Masses"][()][:, None] * s["Velocities"][()]).sum(0)
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


def test_winds(built):
    ic, snaps, rows = run("r2", built)
    s0, s1 = state(ic), state(snaps[-1])
    ids = sorted(s0["stars"])
    (wind,) = ids
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
    # conservation: gas + stars mass and metal mass, total momentum
    M0 = s0["M"] + sum(v["m"] for v in s0["stars"].values())
    M1 = s1["M"] + sum(v["m"] for v in s1["stars"].values())
    assert abs(M1 - M0) <= 1e-12 * M0
    assert s1["MZ"] > s0["MZ"]  # the wind metals
    assert len(rows) == 0  # winds are not logged per step
    # momentum: the momentum-conserving wind coupling of the loop balances its radial kicks only
    # approximately on this lattice (measured: 3% of the first step's wind momentum, 0.35 Msun at ~2000 km/s)
    v_max = 3000.0  # km/s, above the table wind speed
    assert np.all(np.abs(s1["p"] - s0["p"]) <= 0.05 * delivered * v_max)


def test_one_and_four_ranks(built):
    """the same wind mass on 1 and 4 ranks (wind mass to rounding: the loop removes a sum of
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
