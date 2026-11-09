"""TIDAL_TIMESTEP_PREFAC (gpu_kokkos_unified c84215640) as the tidal prefactor of stella_dev (TidalTimestepPrefacGas, default 1). Narrow.
Cold gas around a heavy star, factor f: the tidal step of a point mass is f sqrt(eta r^3 / (G M)), and the shortest gas step must lie
in (1/2, 1] of it at the nearest cell (power-of-two bins); without the factor the step is well above that."""
import os
import shutil
import subprocess
import sys
from pathlib import Path
import h5py
import numpy as np
import pytest
from gizmo.test import build_gizmo_for_test, get_stellar_evolution_tables, get_cooling_tables, mpi_launch_command

NAME = "tidal_timestep_prefac"
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "single_stars_gas_timestep"))
from make_ic import make, M_STAR  # noqa: E402
G = 4.30091e-3  # pc (km/s)^2 / Msun
ETA, F_GAS = 0.02, 1.0e-3  # ErrTolIntAccuracy (begrun.cc), gas prefactor of the test


def run_one(tag, tables, extra, exe="GIZMO"):
    d = HERE / f"run_{tag}"
    if d.exists():
        shutil.rmtree(d)
    (d / "output").mkdir(parents=True)
    L, c, _ = make(str(d / "output" / "snapshot_000.hdf5"))
    for f in ("TREECOOL", "spcool_tables"):
        if (HERE / f).exists():
            os.symlink(HERE / f, d / f)
    T = tables["StellarTablesFile"]
    p = (HERE / f"{NAME}.params").read_text()
    p = p.replace("StellarTablesFile           SET_BY_TEST", f"StellarTablesFile           {T}")
    p = p.replace("SGChemTablesPath            SET_BY_TEST", f"SGChemTablesPath            {os.environ.get('SGCHEM_TABLES_DIR', os.path.dirname(T))}")
    p = "\n".join(l for l in p.splitlines() if not (l.split() and l.split()[0] == "BoxSize")) + f"\nBoxSize {L}\n" + extra
    (d / f"{NAME}.params").write_text(p)
    with open(d / f"test_{NAME}.out", "w") as out:
        subprocess.run(mpi_launch_command(2) + [str(HERE / exe), f"{NAME}.params", "2", "0"], cwd=d, stdout=out,
                       stderr=subprocess.STDOUT, env={**os.environ, "OMP_NUM_THREADS": "1"}, check=True)
    with h5py.File(sorted((d / "output").glob("snapshot_*.hdf5"))[-1]) as F:
        g = F["PartType0"]
        return g["TimeStep"][:].min(), np.sqrt(((g["Coordinates"][:] - c) ** 2).sum(1)).min()


@pytest.fixture(scope="module")
def steps():
    if not os.environ.get("SUNDIALS_DIR"):
        pytest.skip("set SUNDIALS_DIR (SGCHEM builds against SUNDIALS 6.x or 7.x)")
    t = get_stellar_evolution_tables()
    get_cooling_tables(str(HERE))
    build_gizmo_for_test(NAME)
    os.replace(HERE / "GIZMO", HERE / "GIZMO_default")
    build_gizmo_for_test(NAME, extra_config_flags=(f"TIDAL_TIMESTEP_PREFAC=({F_GAS})",))
    return {"default": run_one("default", t, "", "GIZMO_default"), "gas": run_one("gas", t, "")}


def test_factor_sets_the_gas_step(steps):
    print(steps)
    dt, r = steps["gas"]
    pred = F_GAS * np.sqrt(ETA * r ** 3 / (G * M_STAR))
    assert steps["default"][0] > 2 * pred, "the default step is well above the factor's"
    assert 0.5 * pred < dt <= pred * (1 + 1e-2)
