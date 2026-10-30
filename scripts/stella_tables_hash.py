#!/usr/bin/env python3
# Copyright (c) 2026 Ulrich Steinwandel
# Author: Ulrich Steinwandel (uli@mpa-garching.mpg.de)
"""Content hash of a Stella table file, as checked by GIZMO at startup (galaxy_sf/stella_tables_check.h).

HDF5: sha256 over every dataset, sorted by path: path, a zero byte, then the data in native byte order
(object timestamps differ between rebuilds of the same table; the data do not). Other files: sha256 of the bytes.
Usage: stella_tables_hash.py FILE [FILE ...]  -> "<hash>  <basename>" per file. Needs numpy and h5py for HDF5.
"""
import hashlib, os, sys


def table_hash(path):
    h = hashlib.sha256()
    if path.endswith(".hdf5"):
        import h5py, numpy as np
        with h5py.File(path, "r") as f:
            names = []
            f.visititems(lambda n, o: names.append(n) if isinstance(o, h5py.Dataset) else None)
            for n in sorted(names, key=lambda s: s.encode()):
                a = np.asarray(f[n][()])
                if a.dtype.kind in "iufc":
                    a = a.astype(a.dtype.newbyteorder("="), copy=False)
                h.update(n.encode() + b"\0" + np.ascontiguousarray(a).tobytes())
    else:
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
    return h.hexdigest()


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(f"{table_hash(p)}  {os.path.basename(p)}")
