#!/usr/bin/env python3
"""Compare GIZMO snapshot pairs dataset by dataset after sorting each particle type by (ID, child ID).
Prints IDENTICAL when every dataset and the header (Time, NumPart_Total) are bitwise equal, else the differences.
usage: compare_snaps.py A1.hdf5 B1.hdf5 [A2.hdf5 B2.hdf5 ...]; exit status 0 only if every pair is identical."""
import sys

import h5py
import numpy as np


def order(g):
    ids = g["ParticleIDs"][:]
    if "ParticleChildIDsNumber" in g:
        return np.lexsort((g["ParticleChildIDsNumber"][:], ids))
    return np.argsort(ids, kind="stable")


def compare(fa, fb):
    same = True
    with h5py.File(fa, "r") as A, h5py.File(fb, "r") as B:
        for key in ("Time", "NumPart_Total"):
            if not np.array_equal(A["Header"].attrs[key], B["Header"].attrs[key]):
                print(f"  Header/{key}: {A['Header'].attrs[key]!r} vs {B['Header'].attrs[key]!r}"); same = False
        types = sorted(k for k in set(A) | set(B) if k.startswith("PartType"))
        for pt in types:
            if pt not in A or pt not in B:
                print(f"  {pt} only in {'A' if pt in A else 'B'}"); same = False; continue
            ga, gb = A[pt], B[pt]
            if ga["ParticleIDs"].shape != gb["ParticleIDs"].shape:
                print(f"  {pt} count {ga['ParticleIDs'].shape} vs {gb['ParticleIDs'].shape}"); same = False; continue
            oa, ob = order(ga), order(gb)
            for name in sorted(set(ga) | set(gb)):
                if name not in ga or name not in gb:
                    print(f"  {pt}/{name} only in {'A' if name in ga else 'B'}"); same = False; continue
                a, b = ga[name][:][oa], gb[name][:][ob]
                if a.dtype.kind in "biu":
                    if not np.array_equal(a, b):
                        print(f"  {pt}/{name}: integer mismatch ({np.count_nonzero(a != b)} entries)"); same = False
                    continue
                if np.array_equal(a, b, equal_nan=True):
                    continue
                d = np.abs(a.astype(np.float64) - b.astype(np.float64))
                s = np.maximum(np.abs(a.astype(np.float64)), 1e-300)
                print(f"  {pt}/{name}: max|d|={np.nanmax(d):.3e} max rel={np.nanmax(d / s):.3e} n_diff={np.count_nonzero(d > 0)}")
                same = False
    return same


ok = True
args = sys.argv[1:]
for i in range(0, len(args), 2):
    s = compare(args[i], args[i + 1])
    print(("IDENTICAL " if s else "DIFFERENT ") + args[i] + " " + args[i + 1])
    ok = ok and s
sys.exit(0 if ok else 1)
