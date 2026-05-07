#!/usr/bin/env python3
"""Per-field max-abs-diff between two GIZMO HDF5 snapshots.

Usage:
  snapshot_diff.py A.hdf5 B.hdf5 [tolerance]

Default tolerance = 1e-5 (relative).  Reports any field exceeding that as
a potential physics-affecting drift; smaller diffs are tagged as roundoff-
consistent.

Labeled WEAKER than strict NGL pair-set equivalence: matches do NOT prove
neighbor-set equivalence (FP rounding could cancel in early outputs).
Mismatches DO prove physics divergence (modulo MPI nondeterminism baseline).
"""

import sys
import h5py
import numpy as np

def diff_h5(a_path, b_path, rel_tol=1e-5):
    fa = h5py.File(a_path, 'r')
    fb = h5py.File(b_path, 'r')

    n_fields = 0
    n_match  = 0
    n_close  = 0
    n_diff   = 0
    worst = []

    def visit(name, obj):
        nonlocal n_fields, n_match, n_close, n_diff
        if not isinstance(obj, h5py.Dataset): return
        if name not in fb:
            print(f"MISSING in B: {name}")
            n_diff += 1
            return
        a = obj[()]
        b = fb[name][()]
        if a.shape != b.shape:
            print(f"SHAPE DIFF {name}: A={a.shape} B={b.shape}")
            n_diff += 1
            return
        if a.dtype.kind not in 'fiu':
            return  # skip non-numeric
        n_fields += 1
        if np.array_equal(a, b):
            n_match += 1
            return
        # Numeric diff
        a64 = a.astype(np.float64)
        b64 = b.astype(np.float64)
        absdiff = np.abs(a64 - b64)
        max_abs = float(absdiff.max())
        scale = max(float(np.abs(a64).max()), float(np.abs(b64).max()), 1e-30)
        rel = max_abs / scale
        worst.append((rel, name, max_abs, scale))
        if rel <= rel_tol:
            n_close += 1
        else:
            n_diff += 1
            print(f"DIFF {name}: max_abs={max_abs:.3e} rel={rel:.3e} (scale={scale:.3e})")

    fa.visititems(visit)

    print(f"\n=== summary ===")
    print(f"fields compared:  {n_fields}")
    print(f"bit-identical:    {n_match}")
    print(f"close (rel<={rel_tol}): {n_close}")
    print(f"divergent:        {n_diff}")
    if worst:
        worst.sort(reverse=True)
        print(f"\ntop 10 by relative drift:")
        for rel, name, mabs, scale in worst[:10]:
            print(f"  rel={rel:.3e}  max_abs={mabs:.3e}  scale={scale:.3e}  {name}")

    return n_diff

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} A.hdf5 B.hdf5 [rel_tol]", file=sys.stderr)
        sys.exit(2)
    rel_tol = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-5
    n_diff = diff_h5(sys.argv[1], sys.argv[2], rel_tol)
    sys.exit(0 if n_diff == 0 else 1)
