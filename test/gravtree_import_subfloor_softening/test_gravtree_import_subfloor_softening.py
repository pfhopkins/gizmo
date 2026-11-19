"""gravtree_import_subfloor_softening: the imported tree carries what the walk opens when gas kernels lie below the fixed softening.

With ADAPTIVE_GRAVSOFT_FORALL the walk takes, for each target, the larger of its kernel radius and the fixed softening of its
type (grav_target_select_soft_and_zeta), and with adaptive softening it opens every node within max(that softening, the node's)
+ len sqrt(3)/2 of the target (NEIGHBORS_MUST_BE_COMPUTED_EXPLICITLY_IN_FORCETREE). The rank that packs
the subtrees other ranks import must assume the same softening, or the walk opens imported nodes that arrived without
children and the run stops (gravtree.cc: 'the imported tree did not carry the structure the walk resolved'). Production
Stella runs have no gas kernel floor (MinGasKernelRadiusFractional 0), so kernels below SofteningGas occur.

Problem: a cold 16^3 gas lattice (unit masses and spacing, open boundaries), kernels about 2 cells, fixed gas softening 5
cells, ErrTolTheta 0.7, 4 steps; 1 and 8 ranks.
test_import_carries_what_the_walk_opens: the 8-rank run completes without an import shortfall message; its start-up
  gravitational accelerations (snapshot 0) agree with 1 rank: 99th percentile of |a_8 - a_1| below 1 % and maximum below 3 %
  of the median |a_1|.
test_mass_and_energy: on both runs the mass is constant to 1e-12 and E_kin + E_th + E_pot (EVALPOTENTIAL) changes by less
  than 1e-3 of |E_pot| over the run.
"""
import glob
from os import chdir, path

import h5py
import numpy as np
import pytest

from gizmo.test import build_gizmo_for_test, run_test

HERE = path.dirname(path.abspath(__file__))
ROOT = path.dirname(path.dirname(HERE))
NAME = "gravtree_import_subfloor_softening"
N = 16


def make_ics(fname):
    x = np.arange(N) + 0.5
    pos = np.array(np.meshgrid(x, x, x, indexing="ij")).reshape(3, -1).T
    n = len(pos)
    with h5py.File(fname, "w") as f:
        h = f.create_group("Header")
        h.attrs["NumPart_ThisFile"] = h.attrs["NumPart_Total"] = [n, 0, 0, 0, 0, 0]
        h.attrs["NumPart_Total_HighWord"] = [0] * 6
        h.attrs["MassTable"] = [0.0] * 6
        h.attrs["Time"] = 0.0
        h.attrs["Redshift"] = 0.0
        h.attrs["BoxSize"] = float(N)
        h.attrs["NumFilesPerSnapshot"] = 1
        g = f.create_group("PartType0")
        g["Coordinates"] = pos
        g["Velocities"] = np.zeros((n, 3))
        g["ParticleIDs"] = np.arange(1, n + 1, dtype=np.uint32)
        g["Masses"] = np.ones(n)
        g["InternalEnergy"] = np.full(n, 0.01)


@pytest.fixture(scope="module")
def runs():
    chdir(ROOT)
    build_gizmo_for_test(NAME)
    make_ics(path.join(HERE, f"{NAME}_ics.hdf5"))
    out = {}
    for ranks in (1, 8):
        d = f"output_r{ranks}"
        chdir(HERE)
        try:
            run_test(NAME, ranks, 1, param_overrides={"OutputDir": d}, allow_nonzero_exit=True)
            out[ranks] = (path.join(HERE, d), open(path.join(HERE, f"test_{NAME}.out")).read())
        finally:
            chdir(ROOT)
    return out


def snap(d, k):
    with h5py.File(sorted(glob.glob(path.join(d, "snapshot_*.hdf5")))[k], "r") as f:
        g = f["PartType0"]; o = np.argsort(g["ParticleIDs"][()])
        return {q: g[q][()][o] for q in ("Masses", "Velocities", "InternalEnergy", "Potential", "Acceleration")}, f["Header"].attrs["Time"]


def test_import_carries_what_the_walk_opens(runs):
    d8, log8 = runs[8]
    short = [l for l in log8.splitlines() if "arrived without children" in l]
    print(f"8 ranks: {len(short)} import shortfall message(s); {short[0][:300] if short else ''}")
    assert "Final time=" in log8, log8[-1500:]
    assert not short
    a1, a8 = snap(runs[1][0], 0)[0]["Acceleration"], snap(d8, 0)[0]["Acceleration"]
    dev = np.linalg.norm(a8 - a1, axis=1) / np.median(np.linalg.norm(a1, axis=1))
    print(f"8 vs 1 rank, start-up gravity: 99th percentile {np.percentile(dev, 99):.2e}, max {dev.max():.2e} of the median |a|")
    assert np.percentile(dev, 99) < 0.01 and dev.max() < 0.03


@pytest.mark.parametrize("ranks", [1, 8])
def test_mass_and_energy(runs, ranks):
    (a, ta), (b, tb) = snap(runs[ranks][0], 0), snap(runs[ranks][0], -1)
    e = [np.sum(s["Masses"] * (0.5 * np.sum(s["Velocities"] ** 2, axis=1) + s["InternalEnergy"] + 0.5 * s["Potential"])) for s in (a, b)]
    epot = abs(np.sum(0.5 * a["Masses"] * a["Potential"]))
    print(f"{ranks} ranks, t {ta:g} -> {tb:g}: mass change {b['Masses'].sum() / a['Masses'].sum() - 1:.1e}, energy change {(e[1] - e[0]) / epot:.2e} of |E_pot|")
    assert tb > ta
    assert abs(b["Masses"].sum() / a["Masses"].sum() - 1) < 1e-12 and abs(e[1] - e[0]) < 1e-3 * epot
