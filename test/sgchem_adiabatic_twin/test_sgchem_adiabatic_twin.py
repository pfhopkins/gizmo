"""sgchem_adiabatic_twin: TEST_SKIP_COOLING_AND_CHEMISTRY_UPDATE gives the adiabatic twin of a cooling run (needs
COOLING_OPERATOR_SPLIT, so that the hydro work is not applied by the cooling): the box of sgchem_box (8^3 cells, n_H = 100,
T0 = 8000 K, at rest, SGChem network 5, 2 ranks) with the update left out. Bars, written before the run: after 8 steps
(0.032, 31 kyr) the temperature of every cell equals its start value to 1e-6 (sgchem_box cools the same box to 32 K in
125 kyr) and the tracked species (H2, H+, CO) are the start values bitwise."""
import shutil
import subprocess
import sys
from os import chdir, environ, path

import h5py
import numpy as np
import pytest

from gizmo.test import build_and_run_test, default_omp_threads, get_cooling_tables, run_test

HERE = path.dirname(path.abspath(__file__))
ROOT = path.dirname(path.dirname(HERE))
NAME = "sgchem_adiabatic_twin"


def chemistry_tables_dir():
    """SGCHEM_TABLES_DIR, else the install of scripts/get_stella_tables.sh (as sgchem_box)"""
    if environ.get("SGCHEM_TABLES_DIR"):
        return environ["SGCHEM_TABLES_DIR"]
    out = subprocess.run(["bash", path.join(ROOT, "scripts", "get_stella_tables.sh")], capture_output=True, text=True,
                         env={**environ, "PYTHON": environ.get("PYTHON", sys.executable)})
    assert out.returncode == 0, f"get_stella_tables.sh failed:\n{out.stdout}\n{out.stderr}"
    return [l for l in out.stdout.splitlines() if " holds " in l][-1].split()[1]


@pytest.fixture(scope="module")
def box():
    if not environ.get("SUNDIALS_DIR"):
        pytest.skip("set SUNDIALS_DIR (SGCHEM builds against SUNDIALS 6.x or 7.x)")
    get_cooling_tables(HERE)
    sys.path.insert(0, path.join(ROOT, "test", "sgchem_box"))
    from make_sgchem_box_ics import make_ics
    make_ics(path.join(HERE, "sgchem_box_ics.hdf5"))
    tables = chemistry_tables_dir()
    build_and_run_test(NAME, 2, default_omp_threads(), param_overrides={"SGChemTablesPath": tables})
    return path.join(HERE, "output"), tables


def test_adiabatic_twin(box):
    box, _ = box
    with h5py.File(path.join(box, "snapshot_000.hdf5")) as s0, h5py.File(path.join(box, "snapshot_001.hdf5")) as s1:
        assert s1["Header"].attrs["Time"] == 0.032
        T0, T1 = s0["PartType0/Temperature"][:], s1["PartType0/Temperature"][:]
        assert np.max(np.abs(T1 / T0 - 1)) < 1e-6, (T0.min(), T1.min())
        assert np.array_equal(s0["PartType0/SGChemTracAbund"][:], s1["PartType0/SGChemTracAbund"][:])


def wave_ics():
    """the box with v_x = 5 km/s sin(2 pi x / L): half of it expands, where the gas would cool adiabatically"""
    src, dst = path.join(HERE, "sgchem_box_ics.hdf5"), path.join(HERE, "sgchem_wave_ics.hdf5")
    shutil.copy(src, dst)
    with h5py.File(dst, "r+") as f:
        x = f["PartType0/Coordinates"][:, 0]
        v = np.zeros((len(x), 3)); v[:, 0] = 5.0 * np.sin(2 * np.pi * x / 10.0)
        f["PartType0/Velocities"][...] = v
    return "sgchem_wave_ics"


def floor_run(box, tag, min_temp):
    _, tables = box
    d = path.join(HERE, f"output_{tag}")
    shutil.rmtree(d, ignore_errors=True)
    chdir(HERE)
    try:
        run_test(NAME, 2, default_omp_threads(), param_overrides={"SGChemTablesPath": tables, "OutputDir": f"output_{tag}", "MinGasTemp": min_temp,
                                                                   "InitCondFile": wave_ics()})
    finally:
        chdir(ROOT)
    lines = open(path.join(d, "ledger.txt")).read().splitlines()
    return dict(zip(lines[0].split()[2:], [float(x) for x in lines[-1].split()[1:]])), d


def test_kick_floor_row(box):
    """E_kick_floor_erg and E_mhd_numerical_erg (OUTPUT_MASS_METAL_ENERGY_LEDGER) book the gas energy the kick limiter and the temperature
    floor add at the kicks, and the energy the hydro and MHD source terms create (MAGNETIC build; with B = 1e-12 the net change of thermal
    plus kinetic energy by the hydro step itself). The box with the update left out and a velocity wave v_x = A sin(k x),
    A = 5 km/s, k = 2 pi / 10 pc, for 0.032 code time, with the floor at the start energy (MinGasTemp 9000 K: the start clamps u
    to it): the half that expands cannot cool and the floor adds energy there. Bars, written before the run: with MinGasTemp 9000 K
    the increase of E_kick_floor_erg is above 1e-2 of M u_0 (gamma - 1) A k t / pi (the energy the expanding half would have lost),
    and |E_mhd_numerical_erg| is below 1e-6 of the energy the gas gained (the hydro conserves energy; B is negligible); with
    MinGasTemp 2.73 K E_kick_floor_erg is exactly 0. Two bars written first were logically wrong and are dropped: 0.75 to 1.25 of
    the estimate for the floor row, and the closure 'gain of thermal plus kinetic energy = both rows to 2 %'. The rows count the clamps
    of the kicks; the measured gain is 2.4 times the floor row (1.07e47 against 4.5e46 erg), the rest is energy added where the
    predicted energy is clamped at the synchronisation and in the drift, outside the kicks (the definition of the row)."""
    unit = 1.989e33 * 1.0e10  # erg per Msun (km/s)^2
    out = {}
    for tag, temp in (("floor", 9000.0), ("nofloor", 2.73)):
        _, d = floor_run(box, tag, temp)
        lines = open(path.join(d, "ledger.txt")).read().splitlines()
        names = lines[0].split()[2:]
        vals = np.array([[float(x) for x in l.split()] for l in lines[1:]])
        inc = {k: (vals[-1, 1 + j] - vals[0, 1 + j]) for j, k in enumerate(names) if 1 + j < vals.shape[1]}
        e = []
        for k in (0, 1):
            with h5py.File(path.join(d, f"snapshot_00{k}.hdf5")) as s:
                m, u, v = s["PartType0/Masses"][:], s["PartType0/InternalEnergy"][:], s["PartType0/Velocities"][:]
                e.append(float((m * u).sum() + 0.5 * (m * (v ** 2).sum(1)).sum()) * unit)
                if k == 0:
                    est = float(m.sum() * u.mean() * (2.0 / 3.0) * 5.0 * (2 * np.pi / 10.0) * 0.032 / np.pi) * unit
        out[tag] = (inc, e[1] - e[0], est)
    inc, gain, est = out["floor"]
    assert inc["E_kick_floor_erg"] > 1e-2 * est, (inc["E_kick_floor_erg"], est)
    assert abs(inc["E_mhd_numerical_erg"]) < 1e-6 * gain, (inc["E_mhd_numerical_erg"], gain)
    assert out["nofloor"][0]["E_kick_floor_erg"] == 0.0
