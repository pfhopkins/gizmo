"""single_stars_gas_timestep: with the stars' band fluxes in the gravity walk, gas takes its gravitational step from the tidal
criterion alone (stella_dev timestep.cc). Cold gas around a heavy star, every cell many kernels away, so the acceleration
criterion would be well below the tidal one. The shortest gas step must lie above the acceleration criterion sqrt(2 eta eps / a)
of the nearest cell by a factor 1.5 (steps are powers of two)."""
import os
import shutil
import subprocess
import sys
from pathlib import Path
import h5py
import numpy as np
import pytest
from gizmo.test import build_gizmo_for_test, get_stellar_evolution_tables, get_cooling_tables, mpi_launch_command

NAME = "single_stars_gas_timestep"
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from make_ic import make, M_STAR  # noqa: E402
G = 4.30091e-3  # pc (km/s)^2 / Msun


@pytest.fixture(scope="module")
def run():
    if not os.environ.get("SUNDIALS_DIR"):
        pytest.skip("set SUNDIALS_DIR (SGCHEM builds against SUNDIALS 6.x or 7.x)")
    t = get_stellar_evolution_tables()
    get_cooling_tables(str(HERE))
    build_gizmo_for_test(NAME)
    d = HERE / "run"
    if d.exists():
        shutil.rmtree(d)
    (d / "output").mkdir(parents=True)
    L, c, dx = make(str(d / "output" / "snapshot_000.hdf5"))
    for f in ("TREECOOL", "spcool_tables"):
        if (HERE / f).exists():
            os.symlink(HERE / f, d / f)
    T = t["StellarTablesFile"]
    p = (HERE / f"{NAME}.params").read_text()
    p = p.replace("StellarTablesFile           SET_BY_TEST", f"StellarTablesFile           {T}")
    p = p.replace("SGChemTablesPath            SET_BY_TEST", f"SGChemTablesPath            {os.environ.get('SGCHEM_TABLES_DIR', os.path.dirname(T))}")
    p = "\n".join(l for l in p.splitlines() if not (l.split() and l.split()[0] == "BoxSize")) + f"\nBoxSize {L}\n"
    (d / f"{NAME}.params").write_text(p)
    with open(d / f"test_{NAME}.out", "w") as out:
        subprocess.run(mpi_launch_command(2) + [str(HERE / "GIZMO"), f"{NAME}.params", "2", "0"], cwd=d, stdout=out,
                       stderr=subprocess.STDOUT, env={**os.environ, "OMP_NUM_THREADS": "1"}, check=True)
    return d, c


def test_gas_step_is_tidal(run):
    d, c = run
    with h5py.File(sorted((d / "output").glob("snapshot_*.hdf5"))[-1]) as F:
        g = F["PartType0"]; r = np.sqrt(((g["Coordinates"][:] - c) ** 2).sum(1)); dt = g["TimeStep"][:]
    eta, eps = 0.02, 0.028  # ErrTolIntAccuracy (begrun.cc), softening of the run
    accel_step = np.sqrt(2 * eta * eps / (G * M_STAR / r.min() ** 2))
    print(f"min gas step {dt.min():.4g}, acceleration criterion with gravity {accel_step:.4g}, r_min {r.min():.3g}")
    assert dt.min() > 1.5 * accel_step
