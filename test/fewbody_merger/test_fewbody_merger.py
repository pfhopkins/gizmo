"""A sink-sink merger in a gas-free run, on one rank and across two.

The IC: a pair (1 and 0.5 Msun) at rest 2e-5 pc apart, inside each other's sink radius and on
either side of the box centre, so that at 2 ranks each rank holds one; they merge in the first
step. Two 0.1 Msun sinks on near-circular orbits 7e-4 pc out never merge. The run must end with
the merged sink and the two outer sinks, mass and momentum conserved, and sink_events/merger.npy
must hold exactly that one merger.

Gas-free IC sinks have Sink_Formation_Mass 0, so the default merger mass ceiling (10x that) refuses
every merger; fewbody_merger.params sets SinkMergerMaxMass to admit the pair. The default_ceiling
variant sets it back to 0 and checks that the pair is refused.
"""

import numpy as np
import h5py
import pytest

from gizmo.units import G_CODE
from gizmo.test import build_and_run_test, get_final_snapshot, variant_output_dir

TEST_NAME = "fewbody_merger"
TEST_DIR = f"test/{TEST_NAME}"
IC_FILE = f"{TEST_DIR}/{TEST_NAME}_ics.hdf5"

M_A, M_B, SEP = 1.0, 0.5, 2e-5        # the merging pair: Msun, Msun, pc
M_OUT, S_OUT = 0.1, 5e-4              # outer sinks: Msun, offset per axis in pc
MASSES = np.array([M_A, M_B, M_OUT, M_OUT])
MTOT = MASSES.sum()
V_SCALE = np.sqrt(G_CODE * (M_A + M_B) / SEP)   # the pair's infall speed scale


def _write_ic():
    c = np.array([0.5, 0.5, 0.5])
    mp = M_A + M_B
    v_out = np.sqrt(G_CODE * mp / (np.sqrt(2.0) * S_OUT)) / np.sqrt(2.0)   # circular, per axis
    pos = c + np.array([[-SEP * M_B / mp, 0, 0], [SEP * M_A / mp, 0, 0],
                        [-S_OUT, S_OUT, 0], [S_OUT, -S_OUT, 0]])
    vel = np.array([[0, 0, 0], [0, 0, 0], [v_out, v_out, 0], [-v_out, -v_out, 0]], dtype=float)
    with h5py.File(IC_FILE, "w") as F:
        h = F.create_group("Header").attrs
        h["NumPart_ThisFile"] = [0, 0, 0, 0, 0, 4]
        h["NumPart_Total"] = [0, 0, 0, 0, 0, 4]
        h["NumPart_Total_HighWord"] = [0, 0, 0, 0, 0, 0]
        h["MassTable"] = [0, 0, 0, 0, 0, 0]
        h["BoxSize"] = 1.0
        h["Time"] = 0.0
        h["Redshift"] = 0.0
        h["NumFilesPerSnapshot"] = 1
        g = F.create_group("PartType5")
        g.create_dataset("Coordinates", data=pos)
        g.create_dataset("Velocities", data=vel)
        g.create_dataset("Masses", data=MASSES)
        g.create_dataset("ParticleIDs", data=np.arange(1, 5, dtype=np.uint32))
        # mature stars: eligible_for_hermite() KDK-integrates a star for ~2 dt after its formation
        g.create_dataset("StellarFormationTime", data=-np.ones(4))


def _final_sinks():
    with h5py.File(get_final_snapshot(TEST_NAME), "r") as f:
        g = f["PartType5"]
        ids, mass, vel = np.array(g["ParticleIDs"]), np.array(g["Masses"]), np.array(g["Velocities"])
    live = mass > 0
    o = np.argsort(ids[live])
    return ids[live][o], mass[live][o], vel[live][o]


@pytest.mark.parametrize("num_mpi_ranks", (1, 2))
def test_merger(num_mpi_ranks):
    _write_ic()
    build_and_run_test(TEST_NAME, num_mpi_ranks, 1)

    ids, mass, vel = _final_sinks()
    p = np.linalg.norm((mass[:, None] * vel).sum(0)) / (MTOT * V_SCALE)
    print(f"\n  {num_mpi_ranks} rank(s): IDs {ids}, masses {mass}, |P|/(M v_scale) {p:.3e}")
    assert list(ids) == [1, 3, 4], f"expected sink 2 merged into sink 1 and the outer two kept, got IDs {ids}"
    assert mass == pytest.approx([M_A + M_B, M_OUT, M_OUT], rel=1e-6, abs=0), "mass not conserved"
    assert p < 1e-3, f"total momentum |P| / (M v_scale) = {p:.3e}; the IC has P = 0"

    events = np.load(f"{variant_output_dir(TEST_NAME)}/sink_events/merger.npy")
    assert len(events) == 1, f"expected one merger record, got {len(events)}"
    e = events[0]
    assert (e["id1"], e["id2"]) == (1, 2), f"merger record has sink {e['id2']} absorbed by {e['id1']}"
    assert (e["mass_dyn1"], e["mass_dyn2"]) == pytest.approx((M_A, M_B), rel=1e-6, abs=0)
    assert e["mass_dyn1"] + e["mass_dyn2"] == pytest.approx(mass[0], rel=1e-6, abs=0), \
        "merger record masses do not sum to the survivor's"


def test_default_ceiling_refuses():
    """SinkMergerMaxMass = 0 restores the default ceiling, which is 0 for gas-free IC sinks."""
    _write_ic()
    build_and_run_test(TEST_NAME, 1, 1, param_overrides={"SinkMergerMaxMass": 0})

    ids, mass, _ = _final_sinks()
    assert list(ids) == [1, 2, 3, 4], f"expected no merger under the default ceiling, got IDs {ids}"
    assert len(np.load(f"{variant_output_dir(TEST_NAME)}/sink_events/merger.npy")) == 0
