"""Uniform elastic tension: a solid held in isotropic tension must stay uniform (the tensile instability), and material
that has stopped flowing plastically must get the instability's correction back.

A periodic 2D box of the ring material (see make_ics.py) is set at an areal density 5% below the reference
(about 2.6% linear stretch in each direction) and left at rest:
pressure -0.05, ten times the yield strength, and no shear. The exact solution is that nothing moves. Particle
methods are unstable here: the negative pressure pulls neighbouring elements into pairs and clumps (Swegle et al.
1995), and the elastic solver's tensile-instability correction exists to prevent it (with it switched off this box
clumps, nearest-neighbour distances falling below 0.2 of the lattice spacing by t = 20). The correction is switched
off in material that has recently flowed plastically, where it is not needed and its repulsion drives a runaway, and
comes back as a per-element memory of recent yielding decays (over ten sound-crossing times of the kernel radius).

Two problems share this directory:
  elastic_tension_box_fresh       the box as described: nothing yields, the correction acts throughout;
  elastic_tension_box_preyielded  the same box with every element's RecentYieldMemory set to 0.88 in the IC, so the
                                  correction starts off everywhere and must re-engage as the memory decays.

Validates, for each (t <= 20, about 400 cell sound-crossing times):
  1. The run reaches its end time with mass conserved.
  2. The lattice stays ordered: the smallest nearest-neighbour distance stays above 0.9 of the lattice spacing in
     every snapshot (measured: 0.98).
  3. The box stays in tension: the mean pressure at the end is within 10% of the initial value.
  4. (pre-yielded) the recent-yield memory has decayed by at least a factor 1000 by the end (measured: about 1e6).
"""

import glob

import h5py
import numpy as np
import pytest
from scipy.spatial import cKDTree

from gizmo.test import (
    build_and_run_test,
    default_mpi_ranks,
    default_omp_threads,
    assert_final_time,
    problem_output_dir,
    parse_params,
)

TEST_NAME = "elastic_tension_box"
N_SIDE, L, MEMORY_PRE = 40, 2.0, 0.88   # must match make_ics.py
DP = L / N_SIDE


def _make_ics():
    import subprocess, sys
    subprocess.run([sys.executable, "make_ics.py"], check=True, cwd=f"test/{TEST_NAME}")


@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(),))
@pytest.mark.parametrize("num_omp_threads", (default_omp_threads(),))
@pytest.mark.parametrize("problem", ["elastic_tension_box_fresh", "elastic_tension_box_preyielded"], ids=["fresh", "preyielded"])
def test_elastic_tension_box(num_mpi_ranks, num_omp_threads, problem):
    _make_ics()
    build_and_run_test(TEST_NAME, num_mpi_ranks, num_omp_threads, params_name=problem)
    snaps = sorted(glob.glob(f"{problem_output_dir(TEST_NAME, problem)}/snapshot_*.hdf5"))
    assert snaps, "no snapshots: GIZMO did not run"
    assert_final_time(snaps[-1], TEST_NAME, time_max=float(parse_params(f"test/{TEST_NAME}/{problem}.params")["TimeMax"]))

    nn_min, p_mean, mass, memory = [], [], [], []
    for snap in snaps:
        with h5py.File(snap, "r") as f:
            g = f["PartType0"]
            x = g["Coordinates"][:, :2] % L
            x[x >= L] = 0.0          # a coordinate just below 0 can round to exactly L
            p_mean.append(np.mean(g["Pressure"][:])); mass.append(g["Masses"][:].sum())
            memory.append(g["RecentYieldMemory"][:].max())
        nn_min.append(cKDTree(x, boxsize=L).query(x, k=2)[0][:, 1].min() / DP)
    print(f"{problem}: min nearest-neighbour distance / spacing {min(nn_min):.3f}; mean pressure {p_mean[0]:+.4f} -> {p_mean[-1]:+.4f}; "
          f"memory max {memory[0]:.3g} -> {memory[-1]:.3g}")

    assert abs(mass[-1] / mass[0] - 1) < 1e-12, "mass not conserved"
    assert min(nn_min) > 0.9, f"the lattice clumped: nearest-neighbour distance fell to {min(nn_min):.3f} of the spacing"
    assert p_mean[0] < 0 and abs(p_mean[-1] / p_mean[0] - 1) < 0.1, f"tension not held: mean pressure {p_mean[0]:+.4f} -> {p_mean[-1]:+.4f}"
    if problem.endswith("preyielded"):
        assert memory[0] > 0.5 * MEMORY_PRE, f"the IC's RecentYieldMemory was not read (first snapshot: {memory[0]:.3g})"
        assert memory[-1] < 1e-3 * MEMORY_PRE, f"recent-yield memory did not decay: {MEMORY_PRE} -> {memory[-1]:.3g}"
