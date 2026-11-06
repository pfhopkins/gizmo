"""Local radiation pressure from a young star must push gas away from it, on the FIRE-2 and FIRE-3 paths.

One star in a uniform periodic lattice, gas at rest, no self-gravity. Radiation pressure couples discrete kicks to single cells.
Each kick is read as the change of a cell's velocity between two snapshots (0.001 apart). The single-scattering (UV) kick is exactly
along the line to the star; the infrared kick follows the density gradient and does not depend on the pair vector's sign, but some
of those kicks also land in the radial bin. So the fraction below is not 1 even when the UV kick is right; the test works because
UV flips. Of the kicks that are radial (|cos| > COS_RADIAL), more than half must point away from the star.
Bars fixed before the first run.
"""
import sys
from glob import glob
from pathlib import Path

import h5py
import numpy as np
import pytest

from gizmo.test import build_and_run_test, get_cooling_tables, variant_output_dir, default_omp_threads, default_mpi_ranks

TEST_NAME = "fire_rp_direction"
TEST_DIR = Path(__file__).parent
DV_KICK = 1.0          # km/s: a cell kicked between two snapshots
COS_RADIAL = 0.9       # a kick along the line to the star within 26 degrees
N_MIN = 10             # radial kicks needed for the fraction to mean anything
FRAC_OUT_MIN = 0.5
# star mass (Msun) and photon-momentum factor per FIRE level. FIRE-3 couples in quanta of 2 max(v_grav, 15 km/s) times the star's mass,
# so it needs a light star (small kicks) and a larger factor (events within 20 kyr).
SETUP = {"2": (1.0e5, "1"), "3": (1.0e3, "300")}


def radial_kicks(snaps):
    n_out = n_in = n_other = 0
    prev = None
    for f in snaps:
        with h5py.File(f, "r") as F:
            g = F["PartType0"]
            ids, v, x = g["ParticleIDs"][()], g["Velocities"][()], g["Coordinates"][()]
            star, box = F["PartType4/Coordinates"][0], float(F["Header"].attrs["BoxSize"])
        order = np.argsort(ids)
        v, x = v[order], x[order]
        if prev is not None:
            dv = v - prev
            sp = np.linalg.norm(dv, axis=1)
            k = sp > DV_KICK
            d = x - star
            d -= box * np.round(d / box)
            cos = np.einsum("ij,ij->i", dv, d) / np.linalg.norm(d, axis=1) / np.where(sp > 0, sp, 1)
            n_out += int(np.sum(k & (cos > COS_RADIAL)))
            n_in += int(np.sum(k & (cos < -COS_RADIAL)))
            n_other += int(np.sum(k & (np.abs(cos) <= COS_RADIAL)))
        prev = v
    return n_out, n_in, n_other


@pytest.mark.parametrize("fire", ["2", "3"])
def test_fire_rp_direction(fire):
    m_star, boost = SETUP[fire]
    ic = TEST_DIR / f"{TEST_NAME}_ics.hdf5"
    sys.path.insert(0, str(TEST_DIR))
    from make_fire_rp_direction_ics import make_ics
    make_ics(str(ic), m_star)
    get_cooling_tables(str(TEST_DIR))
    flags = (f"FIRE_PHYSICS_DEFAULTS={fire}",)
    build_and_run_test(TEST_NAME, default_mpi_ranks(), default_omp_threads(), flags,
                       param_overrides={"RP_Local_Momentum_Renormalization": boost})
    out = variant_output_dir(TEST_NAME, flags)
    # radiation pressure only: no mechanical (wind or supernova) event may have fired
    events = Path(out) / "MechFeedbackEvents.txt"
    n_events = sum(1 for l in events.read_text().splitlines() if l.strip() and not l.startswith("#")) if events.exists() else 0
    assert n_events == 0, f"{n_events} mechanical feedback events: the run is not radiation pressure only"
    n_out, n_in, n_other = radial_kicks(sorted(glob(f"{out}/snapshot_*.hdf5")))
    print(f"FIRE-{fire}: radial kicks away from the star {n_out}, toward the star {n_in}, other directions {n_other}")
    assert n_out + n_in >= N_MIN, f"only {n_out + n_in} radial kicks reached the gas"
    assert n_out / (n_out + n_in) > FRAC_OUT_MIN, f"radiation pressure pushes gas toward the star: {n_in} of {n_out + n_in} radial kicks point at it"
