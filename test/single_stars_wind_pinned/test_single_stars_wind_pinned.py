"""single_stars_wind_pinned test: TEST_SINGLE_STARS_WIND_PINNED_RATE_AND_SPEED pins the wind of a single star to a constant
rate TestFixedWindMdot [Msun/yr] at TestFixedWindVel [km/s] (no table, no pre-main-sequence delay), so that the wind power
1/2 Mdot v^2 is known exactly (a Weaver bubble needs it). A 35 Msun star 1 Myr old with winds returned up to that age,
in the lattice of single_stars_winds_agb_ia, Mdot = 1e-4 Msun/yr, v = 3000 km/s, 0.01 code time (9.8 kyr).
Bars, written before the runs: (1) the mass the star lost equals Mdot times the time up to the last delivery, 10^wa - 1e6 yr
(wa = SingleStarLastWindLogAge), to 1e-9; (2) the last delivery is at the star's age to within 3e-4 code time; (3) the energy
gained by gas and star (kinetic + thermal) equals 1/2 M_lost v^2 to 1 %. The first versions had Mdot = 1e-5 and v = 2000 and
bars of 1e-3, then 2 % against a twin without wind: both logically wrong. The hydro does not conserve the total energy of the
box to better than about 4e3 code units (3e-4 of its thermal energy), a number that changes from run to run, so a wind of
2e5 code units cannot be closed to 1e-3 or 2 %; the wind here has 4e6, and the drift is 0.1 % of it."""
import os
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
import h5py
import pytest
from gizmo.test import build_gizmo_for_test, get_stellar_evolution_tables, mpi_launch_command

TEST_NAME = "single_stars_wind_pinned"
TEST_DIR = Path(__file__).parent
sys.path.insert(0, str(TEST_DIR.parent / "single_stars_winds_agb_ia"))
from make_single_stars_winds_agb_ia_ics import make_single_stars_winds_agb_ia_ics  # noqa: E402

PC_KMS_IN_YR = 3.085678e13 / 3.155e7
MDOT, VW, AGE0 = 1.0e-4, 3000.0, 1.0e6


@pytest.fixture(scope="module")
def built():
    tables = get_stellar_evolution_tables()
    if not os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        build_gizmo_for_test(TEST_NAME)
    return tables


def run(tables):
    d = TEST_DIR / "run_pinned"
    if not os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN") and d.exists():
        shutil.rmtree(d)
    (d / "output").mkdir(parents=True, exist_ok=True)
    ic = d / "output" / "snapshot_000.hdf5"
    if not os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        L = make_single_stars_winds_agb_ia_ics(str(ic), tables["StellarTablesFile"], which=("wind",), last_wind_log_age=np.log10(AGE0))
        p = (TEST_DIR / f"{TEST_NAME}.params").read_text().replace("SET_BY_TEST", tables["StellarTablesFile"])
        p = p.replace("BoxSize                            45.0", f"BoxSize                            {L}")
        (d / f"{TEST_NAME}.params").write_text(p)
        with open(d / f"test_{TEST_NAME}.out", "w") as out:
            subprocess.run(mpi_launch_command(2) + [str(TEST_DIR / "GIZMO"), f"{TEST_NAME}.params", "2", "0"], cwd=d, stdout=out,
                           stderr=subprocess.STDOUT, env={**os.environ, "OMP_NUM_THREADS": "1"}, check=True)
    return ic, sorted((d / "output").glob("snapshot_*.hdf5"))[-1]


def state(fname):
    with h5py.File(fname, "r") as F:
        g, s = F["PartType0"], F["PartType4"]
        m = g["Masses"][()]
        E = (0.5 * m * (g["Velocities"][()] ** 2).sum(1) + m * g["InternalEnergy"][()]).sum() + 0.5 * (s["Masses"][()] * (s["Velocities"][()] ** 2).sum(1)).sum()
        return dict(t=float(F["Header"].attrs["Time"]), E=E, m=float(s["Masses"][0]), wa=float(s["SingleStarLastWindLogAge"][0]))


def test_pinned_wind(built):
    ic, snap = run(built)
    s0, s1 = state(ic), state(snap)
    lost, t_wind = 35.0 - s1["m"], 10 ** s1["wa"] - AGE0
    assert lost > 0.5 and abs(lost / (MDOT * t_wind) - 1) < 1e-9, (lost, MDOT * t_wind)
    assert abs(10 ** s1["wa"] - (AGE0 + s1["t"] * PC_KMS_IN_YR)) <= 3e-4 * PC_KMS_IN_YR, (s1["wa"], s1["t"])
    coupled = (s1["E"] - s0["E"]) * 1.989e33 * 1.0e10
    booked = 0.5 * lost * VW ** 2 * 1.989e33 * 1.0e10
    assert abs(coupled / booked - 1) < 1e-2, (coupled, booked)
