"""hot_gas_timestep_sync: hydro fluxes are one-sided, so a pair conserves energy only if both partners step together. Around a young blast one bin of
difference makes or destroys energy. HOT_GAS_TIMESTEP_SYNC keeps everything near hot gas (T > 1e6 K) on a step 4 times finer than the hot gas's own
Courant step, while hot gas exists. Adiabatic blast (no cooling, so total energy is the bar): a 20 Msun star dies in a 24^3 lattice of 1 Msun cells at
n_H = 10 as a new ejecta cell (GALSF_SINGLE_STARS_SN_EJECTA_CELL=1000), 3 kyr, 2 ranks. stella_dev at 1 Msun: worst error -5.07 % without, +0.47 % with.

Bars, written before the first run:
  test_energy_conserved: total energy (heat + kinetic, gas and star) at every snapshot after the explosion = IC + 1e51 erg to 1 %.
  test_quiet_before_the_blast_ends: at the end the hot gas is still there (the sync ran on a live blast, the test means something).
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
from gizmo.test import build_gizmo_for_test, get_stellar_evolution_tables, mpi_launch_command

TEST_NAME = "hot_gas_timestep_sync"
TEST_DIR = Path(__file__).parent
sys.path.insert(0, str(TEST_DIR.parent / "single_stars_supernovae"))
sys.path.insert(0, str(TEST_DIR.parent / "single_stars_sn_ejecta_cell"))
from make_single_stars_supernovae_ics import make_single_stars_supernovae_ics  # noqa: E402
from test_single_stars_sn_ejecta_cell import gas, energy  # noqa: E402

E_UNIT = 1.989e33 * 1e10
N = 24


@pytest.fixture(scope="module")
def run():
    t = get_stellar_evolution_tables()
    build_gizmo_for_test(TEST_NAME)
    d = TEST_DIR / "run"
    if d.exists():
        shutil.rmtree(d)
    (d / "output").mkdir(parents=True)
    ic = d / "output" / "snapshot_000.hdf5"
    L, _ = make_single_stars_supernovae_ics(str(ic), t["StellarTablesFile"], n=N, stars=[(20.0, (0.5, 0.5, 0.5))])
    p = (TEST_DIR / f"{TEST_NAME}.params").read_text().replace("SET_BY_TEST", t["StellarTablesFile"])
    p = "\n".join(l for l in p.splitlines() if not (l.split() and l.split()[0] == "BoxSize")) + f"\nBoxSize {L}\n"
    (d / f"{TEST_NAME}.params").write_text(p)
    with open(d / f"test_{TEST_NAME}.out", "w") as out:
        subprocess.run(mpi_launch_command(2) + [str(TEST_DIR / "GIZMO"), f"{TEST_NAME}.params", "2", "0"], cwd=d, stdout=out,
                       stderr=subprocess.STDOUT, env={**os.environ, "OMP_NUM_THREADS": "1"}, check=True)
    return sorted((d / "output").glob("snapshot_*.hdf5"))


def test_energy_conserved(run):
    e0 = energy(gas(run[0]))
    err = [(energy(gas(s)) - e0) * E_UNIT / 1e51 - 1 for s in run[1:]]
    print("energy error per snapshot:", ["%+.4f" % e for e in err])
    assert len(err) == 6
    assert np.max(np.abs(err)) < 0.01, err


def test_quiet_before_the_blast_ends(run):
    a = gas(run[-1])
    assert a["u"].max() * 1e10 * 0.6 * 1.6726e-24 / (1.5 * 1.380649e-16) > 1e6
