"""FOF sink seeding from stars (SINK_SEED_FROM_FOF=1).

Two FOF groups of dark matter (Type 1) and stars (Type 4) linked together, no gas, from the clump layout of the
fof_subfind test: a bridged pair of clumps and a detached compact halo, each holding a compact star cluster. The
on-the-fly FOF pass at TimeBegin turns the minimum-potential star of each group with more than MinFoFMassForNewSeed of
stars into a sink. A star has no gas-cell data, so the seed must keep its own mass; and with no gas anywhere, nothing
can change the sinks' masses afterwards."""

import glob
import h5py
import numpy as np
from os import path
from gizmo.test import build_and_run_test

TEST_NAME = "fof_sink_seed"
TEST_DIR = path.join("test", TEST_NAME)
BOX = 10.0
STAR_MASS = 0.25  # 32 per cluster: 8 mass units, above MinFoFMassForNewSeed = 1


def make_ics(seed=12345):
    rng = np.random.default_rng(seed)
    c1, c2, c3 = np.array([4.65, 5.0, 5.0]), np.array([5.35, 5.0, 5.0]), np.array([8.0, 8.0, 8.0])
    bridge = np.column_stack([np.linspace(c1[0] + 0.10, c2[0] - 0.10, 24), np.full(24, 5.0), np.full(24, 5.0)])
    dm = np.vstack([c1 + rng.normal(0, 0.045, (96, 3)), bridge + rng.normal(0, 0.012, (24, 3)),
                    c2 + rng.normal(0, 0.045, (96, 3)), c3 + rng.normal(0, 0.055, (48, 3))]) % BOX
    stars = np.vstack([c1 + rng.normal(0, 0.02, (32, 3)), c3 + rng.normal(0, 0.02, (32, 3))]) % BOX
    npart = np.array([0, len(dm), 0, 0, len(stars), 0], dtype=np.int32)
    with h5py.File(path.join(TEST_DIR, f"{TEST_NAME}_ics.hdf5"), "w") as f:
        h = f.create_group("Header")
        h.attrs["NumPart_ThisFile"] = npart
        h.attrs["NumPart_Total"] = npart.astype(np.uint32)
        h.attrs["NumPart_Total_HighWord"] = np.zeros(6, dtype=np.uint32)
        h.attrs["MassTable"] = np.zeros(6)
        h.attrs["Time"] = 0.0
        h.attrs["BoxSize"] = BOX
        h.attrs["NumFilesPerSnapshot"] = 1
        h.attrs["Flag_DoublePrecision"] = 1
        for ptype, x, m, id0 in ((1, dm, 1.0, 1), (4, stars, STAR_MASS, len(dm) + 1)):
            g = f.create_group(f"PartType{ptype}")
            g.create_dataset("Coordinates", data=x)
            g.create_dataset("Velocities", data=np.zeros_like(x))
            g.create_dataset("Masses", data=np.full(len(x), m))
            g.create_dataset("ParticleIDs", data=np.arange(id0, id0 + len(x), dtype=np.uint32))


def test_fof_sink_seed():
    make_ics()
    build_and_run_test(TEST_NAME, 2, 0)
    with h5py.File(sorted(glob.glob(f"{TEST_DIR}/output/snapshot_*.hdf5"))[-1], "r") as F:
        sinks = F["PartType5"] if "PartType5" in F else None
        m = sinks["Masses"][:] if sinks is not None else np.array([])
        n_star = F["Header"].attrs["NumPart_Total"][4]
    print(f"\n{len(m)} FOF seeds, masses {m}; {n_star} stars left of 64")
    assert len(m) == 2, f"{len(m)} sink seeds, expected one per group"
    assert n_star == 62
    assert np.all(m == STAR_MASS), f"seed masses {m}, expected the star mass {STAR_MASS}"
