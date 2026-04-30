#!/usr/bin/env python3
"""Generate a tiny collisionless IC for FOF/SUBFIND restart-flag tests."""

import os
import shutil
from pathlib import Path

import h5py
import numpy as np


def _gaussian_clump(rng, center, n, sigma):
    return center + rng.normal(0.0, sigma, size=(n, 3))


def make_fof_subfind_ics(base_dir=None, box_size=10.0, seed=12345):
    base = Path(base_dir or Path(__file__).resolve().parent)
    out_dir = base / "output"
    out_dir.mkdir(exist_ok=True)

    rng = np.random.default_rng(seed)

    # One bridged FOF halo with two density peaks, plus one detached compact halo.
    c1 = np.array([4.65, 5.0, 5.0])
    c2 = np.array([5.35, 5.0, 5.0])
    c3 = np.array([8.0, 8.0, 8.0])
    clump_a = _gaussian_clump(rng, c1, 96, 0.045)
    clump_b = _gaussian_clump(rng, c2, 96, 0.045)
    bridge_x = np.linspace(c1[0] + 0.10, c2[0] - 0.10, 24)
    bridge = np.column_stack(
        [
            bridge_x,
            np.full_like(bridge_x, 5.0),
            np.full_like(bridge_x, 5.0),
        ]
    )
    bridge += rng.normal(0.0, 0.012, size=bridge.shape)
    halo_b = _gaussian_clump(rng, c3, 48, 0.055)

    coords = np.vstack([clump_a, bridge, clump_b, halo_b]).astype(np.float64)
    coords %= box_size
    n_halo = len(coords)

    velocities = np.zeros((n_halo, 3), dtype=np.float64)
    masses = np.ones(n_halo, dtype=np.float64)
    ids = np.arange(1, n_halo + 1, dtype=np.uint32)

    npart = np.array([0, n_halo, 0, 0, 0, 0], dtype=np.int32)

    ic_path = base / "fof_subfind_ics.hdf5"
    with h5py.File(ic_path, "w") as f:
        header = f.create_group("Header")
        header.attrs["NumPart_ThisFile"] = npart
        header.attrs["NumPart_Total"] = npart.astype(np.uint32)
        header.attrs["NumPart_Total_HighWord"] = np.zeros(6, dtype=np.uint32)
        header.attrs["MassTable"] = np.zeros(6, dtype=np.float64)
        header.attrs["Time"] = 0.0
        header.attrs["Redshift"] = 0.0
        header.attrs["BoxSize"] = box_size
        header.attrs["NumFilesPerSnapshot"] = 1
        header.attrs["Omega0"] = 1.0
        header.attrs["OmegaLambda"] = 0.0
        header.attrs["HubbleParam"] = 1.0
        header.attrs["Flag_Sfr"] = 0
        header.attrs["Flag_Cooling"] = 0
        header.attrs["Flag_StellarAge"] = 0
        header.attrs["Flag_Metals"] = 0
        header.attrs["Flag_Feedback"] = 0
        header.attrs["Flag_DoublePrecision"] = 1

        p1 = f.create_group("PartType1")
        p1.create_dataset("Coordinates", data=coords)
        p1.create_dataset("Velocities", data=velocities)
        p1.create_dataset("Masses", data=masses)
        p1.create_dataset("ParticleIDs", data=ids)

    # RestartFlag=3 reads from OutputDir/SnapshotFileBase_000.
    shutil.copyfile(ic_path, out_dir / "snapshot_000.hdf5")
    return ic_path


if __name__ == "__main__":
    path = make_fof_subfind_ics()
    print(f"Wrote {path}")
