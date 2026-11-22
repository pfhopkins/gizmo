#!/usr/bin/env python3
"""Compare two GIZMO snapshots dataset by dataset after sorting each particle type by (ID, child ID).
Prints 'IDENTICAL' when every dataset is bitwise equal, else the max abs / rel difference per dataset.
usage: compare_snaps.py A.hdf5 B.hdf5 [more pairs ...]"""
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
        ta, tb = A["Header"].attrs["Time"], B["Header"].attrs["Time"]
        if ta != tb:
            print(f"  Time differs {ta!r} {tb!r}"); same = False
        for pt in sorted(k for k in A if k.startswith("PartType")):
            if pt not in B:
                print(f"  {pt} missing in B"); same = False; continue
            ga, gb = A[pt], B[pt]
            if ga["ParticleIDs"].shape != gb["ParticleIDs"].shape:
                print(f"  {pt} count {ga['ParticleIDs'].shape} vs {gb['ParticleIDs'].shape}"); same = False; continue
            oa, ob = order(ga), order(gb)
            for name in sorted(ga):
                if name not in gb:
                    continue
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
