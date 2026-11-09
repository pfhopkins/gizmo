"""readsnap_ids test: analysis/readsnap.py returns particle IDs as stored (the file's own dtype). It used to add 2^31 to IDs above 1e9
or below 0 (the bit-flip of 32-bit IDs of new stars), which corrupts 64-bit IDs. No GIZMO build. See README.md."""
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis"))
import readsnap  # noqa: E402

IDS = np.array([1, 7, 2 ** 31 + 5, 2 ** 40 + 3, 2 ** 62 + 11], dtype=np.uint64)


def write_snapshot(path, dtype):
    n = len(IDS)
    with h5py.File(path, "w") as F:
        h = F.create_group("Header")
        h.attrs["NumPart_ThisFile"] = np.array([0, 0, 0, 0, n, 0], dtype=np.uint32); h.attrs["NumPart_Total"] = np.array([0, 0, 0, 0, n, 0], dtype=np.uint32)
        h.attrs["MassTable"] = np.zeros(6); h.attrs["Time"] = 1.0; h.attrs["Redshift"] = 0.0; h.attrs["BoxSize"] = 1.0
        h.attrs["NumFilesPerSnapshot"] = 1; h.attrs["HubbleParam"] = 1.0
        for k, v in (("Flag_Sfr", 1), ("Flag_Cooling", 0), ("Flag_StellarAge", 1), ("Flag_Metals", 1), ("Flag_Feedback", 0)): h.attrs[k] = v
        g = F.create_group("PartType4")
        g["Coordinates"] = np.zeros((n, 3)); g["Velocities"] = np.zeros((n, 3)); g["Masses"] = np.ones(n)
        g["ParticleIDs"] = IDS.astype(dtype)
        g["Metallicity"] = np.zeros(n); g["StellarFormationTime"] = np.zeros(n)


@pytest.mark.parametrize("dtype", [np.uint64, np.int64])
def test_ids_as_stored(tmp_path, dtype):
    """bar (written before the run): every ID, among them 2^31 + 5, 2^40 + 3 and 2^62 + 11, comes back equal to the stored one"""
    write_snapshot(tmp_path / "snapshot_000.hdf5", dtype)
    out = readsnap.readsnap(str(tmp_path), 0, 4)
    assert out["k"] == 1
    assert np.array_equal(np.asarray(out["id"]).astype(np.uint64), IDS)
