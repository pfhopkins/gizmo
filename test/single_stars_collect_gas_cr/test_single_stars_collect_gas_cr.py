"""single_stars_collect_gas_cr: a new single star that collects its drawn mass from neighbouring gas takes the cosmic-ray energy of
the removed gas with it, so the donors keep their specific CR energy (COSMIC_RAY_FLUID).

The single_stars_collect_gas lattice (1 Msun cells, two stars of 8 and 40 Msun drawn at the first step, merging of light cells off)
with 2 CR energy units per cell and a 2 microG field, start flag 2, one step.
Bar (set before the first run): the total CR energy in the gas after the step equals the initial one times the gas mass ratio
(CR transport conserves it), |E/E0 - M/M0| < 1e-6; the stars carry mass 48 of 13824 Msun, so a missing scaling shows as 3.5e-3.
The ledger row E_cr_collected holds what left with the gas, to the same 1e-6.

The first version had no field and missed the bar by 5.5e-6: the scaling was exact (ledger: collected = E0 x 48/13824), but without a field
the streaming loss takes the sound speed as Alfven speed and every cell lost the same 5e-6 in the step. Same bar, field added.
"""
import shutil
import subprocess
import sys
from pathlib import Path
import h5py
import numpy as np
from gizmo.test import build_gizmo_for_test, mpi_launch_command

TEST_NAME = "single_stars_collect_gas_cr"
TEST_DIR = Path(__file__).parent
sys.path.insert(0, str(TEST_DIR.parent / "single_stars_collect_gas"))
from make_single_stars_collect_gas_ics import make_single_stars_collect_gas_ics, cell_id, N_SIDE  # noqa: E402

STARS = [((12, 12, 12), 8.0), ((12, 12, 13), 40.0)]
E_CR_CELL = 2.0


def test_collect_gas_scales_cr_energy():
    build_gizmo_for_test(TEST_NAME)
    d = TEST_DIR / "run"
    shutil.rmtree(d, ignore_errors=True)
    (d / "output").mkdir(parents=True)
    ic = d / "output" / "snapshot_000.hdf5"
    L = make_single_stars_collect_gas_ics(str(ic))
    with h5py.File(ic, "a") as F:
        n = len(F["PartType0/Masses"])
        F["PartType0/CosmicRayEnergy"] = np.full(n, E_CR_CELL)
        b = np.zeros((n, 3))
        b[:, 2] = 2e-6
        F["PartType0/MagneticField"] = b
    births = ",".join(f"{cell_id(*c)}:{m}" for c, m in STARS)
    params = (TEST_DIR / f"{TEST_NAME}.params").read_text().replace("SET_BY_TEST", births)
    params = params.replace("BoxSize                            15.84", f"BoxSize                            {L}")
    (d / f"{TEST_NAME}.params").write_text(params)
    with open(d / f"test_{TEST_NAME}.out", "w") as out:
        subprocess.run(mpi_launch_command(2, 1) + [str(TEST_DIR / "GIZMO"), f"{TEST_NAME}.params", "2", "0"], cwd=d, stdout=out,
                       stderr=subprocess.STDOUT, check=True)
    snap = sorted((d / "output").glob("snapshot_*.hdf5"))[-1]

    def gas(fname):
        with h5py.File(fname, "r") as F:
            return F["PartType0/Masses"][()].sum(), F["PartType0/CosmicRayEnergy"][()].sum()

    (m0, e0), (m1, e1) = gas(ic), gas(snap)
    print(f"gas mass {m1 / m0:.6f} of the initial, CR energy {e1 / e0:.6f} of the initial")
    assert m1 / m0 < 1 - 3e-3, "the stars did not collect their mass"
    assert abs(e1 / e0 - m1 / m0) < 1e-6
    head = (d / "output" / "ledger.txt").read_text().splitlines()[0].split()[1:]
    row = dict(zip(head, np.atleast_2d(np.loadtxt(d / "output" / "ledger.txt"))[-1]))
    collected = row["E_cr_collected_erg"] / (1.989e33 * 1e10)  # code units
    assert abs(collected / (e0 * (1 - m1 / m0)) - 1) < 1e-6
