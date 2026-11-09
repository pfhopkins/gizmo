"""sgchem_restart_temperature test: SGCHEM_TEMPERATURE_FROM_CHEMISTRY_AT_SNAPSHOT_RESTART sets the chemistry temperature of every gas cell
from its energy and abundances when a snapshot is restarted (start flag 2); a snapshot carries no Temp and the cell would otherwise
start at 100 K. The box of test/sgchem_box as a snapshot, atomic gas (x(H2) = 0) of 940 K to 7500 K. See README.md."""
import re
import subprocess
import os
import shutil
import sys
from os import environ, path
from pathlib import Path

import h5py
import numpy as np
import pytest

from gizmo.test import build_gizmo_for_test, get_cooling_tables, mpi_launch_command

HERE = Path(path.dirname(path.abspath(__file__)))
ROOT = HERE.parent.parent
NAME = "sgchem_restart_temperature"
sys.path.insert(0, str(ROOT / "test" / "sgchem_box"))
ABHE, X_HP = 0.1, 0.1  # the chemistry's helium abundance; x(H+) of the IC
K_B, M_P, UNIT_U = 1.380649e-16, 1.6726e-24, 1.0e10  # (km/s)^2 -> cm^2/s^2


def test_temperature_at_restart():
    """bar (written before the run): every cell prints its start temperature; it equals the atomic-gas value
    T = (2/3) u m_p (1 + 4 ABHE) / (k_B (1 + ABHE + x(H+))) to 1e-3 (the energy is the IC's) and differs from 100 K by more than 10%"""
    from test_sgchem_box import chemistry_tables_dir
    from make_sgchem_box_ics import make_ics
    if not environ.get("SUNDIALS_DIR"):
        pytest.skip("set SUNDIALS_DIR (SGCHEM builds against SUNDIALS 6.x or 7.x)")
    get_cooling_tables(str(HERE))
    build_gizmo_for_test(NAME)
    d = HERE / "run_restart"
    if d.exists():
        shutil.rmtree(d)
    (d / "output").mkdir(parents=True)
    ic = d / "output" / "snapshot_000.hdf5"
    make_ics(str(ic))
    with h5py.File(ic, "r+") as f:
        ids = f["PartType0/ParticleIDs"][:]
        u = 10.0 * (1 + ids % 8)
        f["PartType0/InternalEnergy"][...] = u
        f["PartType0/SGChemTracAbund"] = np.column_stack([np.zeros(len(ids)), np.full(len(ids), X_HP), np.full(len(ids), 1.0e-5)])
        f["Header"].attrs["Flag_Metals"] = 0
    for fn in ("TREECOOL", "spcool_tables"):
        if (HERE / fn).exists():
            os.symlink(HERE / fn, d / fn)
    p = (HERE / f"{NAME}.params").read_text().replace("sgchem_tables          % overridden by the test", f"{chemistry_tables_dir()}")
    (d / f"{NAME}.params").write_text(p)
    with open(d / "run.out", "w") as out:
        subprocess.run(mpi_launch_command(2) + [str(HERE / "GIZMO"), f"{NAME}.params", "2", "0"], cwd=d, stdout=out,
                       stderr=subprocess.STDOUT, env={**os.environ, "OMP_NUM_THREADS": "1"}, check=True)
    rows = re.findall(r"^RESTARTTEMP (\d+) (\S+)$", (d / "run.out").read_text(), re.M)
    T = {int(i): float(t) for i, t in rows}
    assert len(T) == len(ids)
    want = {int(i): (2.0 / 3.0) * (10.0 * (1 + int(i) % 8)) * UNIT_U * M_P * (1 + 4 * ABHE) / (K_B * (1 + ABHE + X_HP)) for i in ids}
    r = np.array([T[i] / want[i] - 1 for i in T])
    print("max |T/T_atomic - 1| =", np.abs(r).max())
    assert np.abs(r).max() < 1e-3 and all(abs(T[i] / 100.0 - 1) > 0.1 for i in T)
