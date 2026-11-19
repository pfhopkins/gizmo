"""fb_after_closing_kick_radp: with RESOLVED_ISM_FB_AFTER_CLOSING_KICK (stella_dev c68d0ced8) radiation pressure is deposited after
the closing kick with the other feedback, and with RESOLVED_ISM_RECOMPUTE_HYDRO_POST_FB (stella_dev 90786d02f, 667879eb8) a step in
which it pushed any cell repeats the hydro pass; with the first flag alone only SN and Ia deposits do (test/sn_hydro_recompute).

Problem: Stella cloud production flags; one 5 Msun main-sequence star (below the 8 Msun wind mass, so no wind; no death in the run)
in gas of n_H = 10 at 100 K, 8^3 cells of 0.68 Msun in a 10 pc periodic box, 20 kyr, 2 ranks.

Bars (set before the runs):
- first flag alone: no repeat;
- both flags: one repeat on every step whose radiation pressure coupled momentum (MomWinds.txt column 7 > 0), at least one;
- both builds reach TimeMax; gas plus star mass constant to 1e-12 (double-precision snapshots).
"""
import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from gizmo.test import build_gizmo_for_test, get_stellar_evolution_tables, get_cooling_tables, mpi_launch_command

NAME = "fb_after_closing_kick_radp"
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "treecol_pixels"))
from make_treecol_pixels_ics import make_treecol_pixels_ics, lattice  # noqa: E402

FLAG, FLAG_ALL = "RESOLVED_ISM_FB_AFTER_CLOSING_KICK", "RESOLVED_ISM_RECOMPUTE_HYDRO_POST_FB"
L, NG, M_STAR, AGE, N_H = 10.0, 8, 5.0, 1.0e7, 10.0
REPEAT = "repeating the hydro pass"
RUN_TIMEOUT = 1800  # s


def run(tag, exe, tables):
    d = HERE / f"run_{tag}"
    shutil.rmtree(d, ignore_errors=True)
    (d / "output").mkdir(parents=True)
    m_gas = N_H * 1.6726e-24 / 0.7065 * (3.085678e18 ** 3) / 1.989e33 * L ** 3 / NG ** 3
    make_treecol_pixels_ics(str(d / "output" / "snapshot_000.hdf5"), L, lattice(NG, L), [(M_STAR, AGE, (0.5 * L + 0.11, 0.5 * L + 0.07, 0.5 * L))],
                            gas_mass=m_gas, T=100.0, hsml=2 * L / NG, tracabund=np.tile([0.0, 1.0e-4, 0.0], (NG ** 3, 1)))
    for f in ("TREECOOL", "spcool_tables"):
        if (HERE / f).exists():
            os.symlink(HERE / f, d / f)
    T = tables["StellarTablesFile"]
    p = (HERE / f"{NAME}.params").read_text().replace("StellarTablesFile           SET_BY_TEST", f"StellarTablesFile           {T}")
    p = p.replace("SGChemTablesPath            SET_BY_TEST", f"SGChemTablesPath            {os.environ.get('SGCHEM_TABLES_DIR', os.path.dirname(T))}")
    (d / f"{NAME}.params").write_text(p)
    with open(d / "run.out", "w") as out:
        subprocess.run(mpi_launch_command(2) + [str(HERE / exe), f"{NAME}.params", "2", "0"], cwd=d, stdout=out,
                       stderr=subprocess.STDOUT, env={**os.environ, "OMP_NUM_THREADS": "1"}, check=True, timeout=RUN_TIMEOUT)
    text = (d / "run.out").read_text()
    assert "Final time=" in text, f"{tag}: run did not reach TimeMax"
    mw = np.loadtxt(d / "output" / "MomWinds.txt", ndmin=2)
    snaps = sorted(glob.glob(str(d / "output" / "snapshot_*.hdf5")))
    mass = []
    for s in (snaps[0], snaps[-1]):
        with h5py.File(s, "r") as f:
            mass.append(f["PartType0/Masses"][()].sum() + f["PartType4/Masses"][()].sum())
    return {"repeats": text.count(REPEAT), "pushed_steps": int((mw[:, 6] > 0).sum()), "rows": len(mw), "mass": mass}


@pytest.fixture(scope="module")
def runs():
    if not os.environ.get("SUNDIALS_DIR"):
        pytest.skip("set SUNDIALS_DIR (SGCHEM builds against SUNDIALS 6.x or 7.x)")
    tables = get_stellar_evolution_tables()
    get_cooling_tables(str(HERE))
    build_gizmo_for_test(NAME, extra_config_flags=(FLAG, FLAG_ALL))
    os.replace(HERE / "GIZMO", HERE / "GIZMO_all")
    build_gizmo_for_test(NAME, extra_config_flags=(FLAG,))
    out = {"closing": run("closing", "GIZMO", tables), "all": run("all", "GIZMO_all", tables)}
    for k, v in out.items():
        print(f"{k}: repeats {v['repeats']}, steps with radiation-pressure momentum {v['pushed_steps']} of {v['rows']}, "
              f"mass change {v['mass'][1] / v['mass'][0] - 1:.1e}")
    return out


def test_radiation_pressure_repeats_only_with_the_second_flag(runs):
    assert runs["closing"]["repeats"] == 0
    assert runs["all"]["pushed_steps"] >= 1 and runs["all"]["repeats"] == runs["all"]["pushed_steps"]


def test_mass(runs):
    for v in runs.values():
        assert abs(v["mass"][1] / v["mass"][0] - 1) < 1e-12
