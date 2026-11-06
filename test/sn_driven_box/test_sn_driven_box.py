"""sn_driven_box: energy, mass and metal closure over many SNe in driven turbulence, Stella cloud model. See README.md.
Bars: reports of the SN energy check; mass and each element to 1e-12, each deposit within 1e-3 of what the star offers,
ledger residual within 0.01 E_SN at every full sync, whatever the event count."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
from gizmo.test import build_gizmo_for_test, get_stellar_evolution_tables, get_cooling_tables, mpi_launch_command

TEST_NAME = "sn_driven_box"
TEST_DIR = Path(__file__).parent
sys.path.insert(0, str(TEST_DIR))
from make_sn_driven_box_ics import make_box, events, add_stars  # noqa: E402

SUITE = dict(L_pc=30.0, n_H=10.0, t_drive=0.004, n_sn=12, window_yr=6.0e5, t_after=1.5e-4)
RANKS = int(os.environ.get("GIZMO_TEST_SN_BOX_RANKS", 32))


def params(d, tables, cooling, L_pc, t_max, t_out):
    text = (TEST_DIR / f"{TEST_NAME}.params").read_text().splitlines()
    set_ = {"TimeMax": t_max, "BoxSize": L_pc * 1e-3, "TimeBetSnapshot": t_out, "TimeBetStatistics": t_out, "StellarTablesFile": tables,
            "SGChemTablesPath": cooling, "TurbDrive_MaxWavelength": L_pc * 1e-3, "TurbDrive_MinWavelength": L_pc * 1e-3 / 3}
    out = [f"{l.split()[0]:<34} {set_[l.split()[0]]}" if "SET_BY_TEST" in l else l for l in text]
    (d / f"{TEST_NAME}.params").write_text("\n".join(out) + "\n")


def run(d, env=None):
    with open(d / "gizmo.out", "w") as f:
        rc = subprocess.run(mpi_launch_command(RANKS) + [str(TEST_DIR / "GIZMO"), f"{TEST_NAME}.params", "2", "0"], cwd=d, stdout=f,
                            stderr=subprocess.STDOUT, env={**os.environ, "OMP_NUM_THREADS": "1", **(env or {})}, check=False).returncode
    assert rc == 0, f"GIZMO exited with {rc}, see {d}/gizmo.out"


@pytest.fixture(scope="module")
def driven():
    tables = get_stellar_evolution_tables()["StellarTablesFile"]
    get_cooling_tables(str(TEST_DIR))
    cooling = os.environ.get("SGCHEM_TABLES_DIR", os.path.dirname(tables))
    build_gizmo_for_test(TEST_NAME)
    d = TEST_DIR / "run_drive"
    shutil.rmtree(d, ignore_errors=True)
    (d / "output").mkdir(parents=True)
    make_box(d / "output" / "snapshot_000.hdf5", SUITE["L_pc"], SUITE["n_H"])
    params(d, tables, cooling, SUITE["L_pc"], SUITE["t_drive"], SUITE["t_drive"])
    run(d)
    return sorted((d / "output").glob("snapshot_*.hdf5"))[-1], tables, cooling


def sn_stage(driven, tag, env=None):
    snap, tables, cooling = driven
    d = TEST_DIR / f"run_{tag}"
    shutil.rmtree(d, ignore_errors=True)
    (d / "output").mkdir(parents=True)
    shutil.copy(snap, d / "output" / "snapshot_000.hdf5")
    M, age, pos = events(tables, SUITE["n_sn"], SUITE["window_yr"])
    add_stars(d / "output" / "snapshot_000.hdf5", M, age, pos)
    t_end = SUITE["t_drive"] + SUITE["window_yr"] / 9.78e8 + SUITE["t_after"]
    params(d, tables, cooling, SUITE["L_pc"], t_end, t_end)
    run(d, env)
    (d / "events.json").write_text(json.dumps(dict(M=M.tolist(), age=age.tolist())))
    return read(d / "output")


def read(out):
    names = open(out / "energy_ledger.txt").readline().split()[1:]
    A = np.loadtxt(out / "energy_ledger.txt", ndmin=2)
    return {k: A[:, i] for i, k in enumerate(names)}, np.loadtxt(out / "ledger_mass.txt", ndmin=2), np.loadtxt(out / "ledger_sn_events.txt", ndmin=2)


def G_cum(c, n):
    d = lambda k: c[k][-1] - c[k][0]
    return (d("dE") + d("dE_B") - d("cool") - d("grav_W") - d("turb_W") - d("fb_other") - d("photoion")) / n


@pytest.fixture(scope="module")
def wake_on(driven):
    return sn_stage(driven, "wake_on")


def test_all_sne_fired(wake_on):
    assert int(wake_on[2][:, 1].sum()) == SUITE["n_sn"]


def test_mass_closes(wake_on):
    M = wake_on[1][:, 3] + wake_on[1][:, 4]
    assert np.abs(M - M[0]).max() / M[0] < 1e-12


def test_each_element_closes(wake_on):
    m = wake_on[1]
    gas, offered = m[:, 5::2], m[:, 6::2]
    for k in range(27):
        if gas[0, k] > 0:
            assert np.abs((gas[:, k] - gas[0, k]) - (offered[:, k] - offered[0, k])).max() / gas[0, k] < 1e-12, f"table element {k}"


def test_deposit_matches_offer(wake_on):
    e = wake_on[2]
    assert np.abs(e[:, 2] / e[:, 33] - 1).max() < 1e-3  # energy actually in the gas, from the sum over cells
    assert np.abs(e[:, 3] / e[:, 32] - 1).max() < 1e-3


def test_ledger_residual_bounded(wake_on):
    r = wake_on[0]["residual"]
    assert np.abs(r - r[0]).max() < 0.01


def test_wakeup_off_shows_the_hydro_gain(driven, wake_on):
    off = sn_stage(driven, "wake_off", {"LEDGER_WAKEUP_OFF": "1"})
    assert G_cum(off[0], SUITE["n_sn"]) > G_cum(wake_on[0], SUITE["n_sn"]) + 0.1
