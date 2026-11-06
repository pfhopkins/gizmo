"""A sink merger across ranks, followed by a Hermite pass with no tree rebuild.

A merger in calculate_non_standard_physics zeroes the swallowed sink's Mass. The Hermite pass
that follows does not rebuild the tree, so on another rank the swallowed sink is still a
foreign leaf with mass > 0, while the owner-side table that pass reads (hermite_ghost_entry)
skips particles with Mass <= 0. The lookup misses, and the walk must keep the tree-build
snapshot for that leaf, as the device walk does, instead of stopping.

The IC: a pair (1 and 0.5 Msun) at rest 2e-5 pc apart, inside each other's sink radius and on
either side of the box centre, so that at 2 ranks each rank holds one; they merge in the first
step. Two 0.1 Msun sinks on near-circular orbits 7e-4 pc out, one on each side of the centre,
so that whichever rank the swallowed sink is on, a sink on the other rank walks the tree in
that Hermite pass (the merged sink accreted this step and sits that pass out). The params force
the host gravity walk. One rank has no foreign leaves, so the 1-rank run is the control: both
must end with the merged sink and the two outer sinks, mass and momentum conserved.
"""

import glob
import re

import h5py
import numpy as np
import pytest

from gizmo.units import G_CODE
from gizmo.test import build_and_run_test, problem_output_dir

TEST_NAME = "fewbody_merger"
VARIANT = "merger"
TEST_DIR = f"test/{TEST_NAME}"
IC_FILE = f"{TEST_DIR}/{VARIANT}_ics.hdf5"

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


@pytest.mark.parametrize("num_mpi_ranks", (1, 2))
def test_merger(num_mpi_ranks):
    _write_ic()
    build_and_run_test(TEST_NAME, num_mpi_ranks, 1, (), params_name=VARIANT)

    outdir = problem_output_dir(TEST_NAME, VARIANT)
    snaps = sorted(glob.glob(outdir + "/snapshot_*.hdf5"),
                   key=lambda f: int(re.search(r"snapshot_(\d+)", f).group(1)))
    with h5py.File(snaps[-1], "r") as f:
        g = f["PartType5"]
        ids = np.array(g["ParticleIDs"])
        mass = np.array(g["Masses"])
        vel = np.array(g["Velocities"])
    live = mass > 0
    ids, mass, vel = ids[live], mass[live], vel[live]
    o = np.argsort(ids)
    ids, mass, vel = ids[o], mass[o], vel[o]
    p = np.linalg.norm((mass[:, None] * vel).sum(0)) / (MTOT * V_SCALE)
    print(f"\n  {num_mpi_ranks} rank(s), {snaps[-1]}: IDs {ids}, masses {mass}, |P|/(M v_scale) {p:.3e}")
    assert list(ids) == [1, 3, 4], f"expected sink 2 merged into sink 1 and the outer two kept, got IDs {ids}"
    assert mass == pytest.approx([M_A + M_B, M_OUT, M_OUT], rel=1e-6, abs=0), "mass not conserved"
    assert p < 1e-3, f"total momentum |P| / (M v_scale) = {p:.3e}; the IC has P = 0"
