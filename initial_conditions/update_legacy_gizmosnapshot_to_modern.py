#!/usr/bin/env python3
"""Update a legacy GIZMO HDF5 snapshot to modern dataset-name conventions.

Several IO fields were renamed in the modern GIZMO code. A legacy snapshot that still
carries the pre-rename dataset names will restart (restartflag=2) with those fields
silently zero-filled -- e.g. the gas kernel length reads back as 0, so the density
iteration cannot bracket and the run aborts (endrun 90001009).

This copies the snapshot and renames the affected datasets in place, so the modern
code finds them under the names it now expects. Unchanged fields are left alone.

The modern reader (read_ic.cc) also carries a legacy-name fallback for these same
fields, so this script is only needed if you prefer a converted file on disk (or are
running a build without the fallback).

Usage:
    python3 update_legacy_gizmosnapshot_to_modern.py <legacy_snapshot.hdf5> [output.hdf5]

If no output path is given, writes '<input>_modern.hdf5' next to the input.
The input file is never modified.

Rename table (legacy name -> modern name), applied per PartType group where present:
    SmoothingLength        -> KernelMaxRadius        (gas kernel length; PartType0)
    StellarSmoothingLength -> StellarKernelMaxRadius (non-gas kernel length)
    BH_Mass                -> Sink_Mass
    BH_Mass_AlphaDisk      -> Sink_Mass_Reservoir
    BH_Specific_AngMom     -> Sink_Specific_AngMom
    BH_AccretionLength     -> Sink_AccretionLength
    SinkRadius             -> Sink_Radius
    SinkInitialMass        -> Sink_InitialMass
    BH_Mdot                -> Sink_Mdot
    BH_NProgs              -> Sink_NProgenitors
    BH_Dist                -> Sink_Distance
    BH_Dust_Mass           -> Sink_Dust_Mass
"""

import sys
import os
import shutil

import h5py

# legacy name -> modern name (see io.cc get_dataset_name_legacy_alias for the SSOT).
# Derived by diffing legacy gizmo/io.c get_dataset_name against the modern list; the
# full set of renames whose field still exists in the modern code.
RENAMES = {
    "SmoothingLength": "KernelMaxRadius",           # gas kernel length
    "StellarSmoothingLength": "StellarKernelMaxRadius",  # non-gas kernel length
    "BH_Mass": "Sink_Mass",
    "BH_Mass_AlphaDisk": "Sink_Mass_Reservoir",
    "BH_Specific_AngMom": "Sink_Specific_AngMom",
    "BH_AccretionLength": "Sink_AccretionLength",
    "SinkRadius": "Sink_Radius",
    "SinkInitialMass": "Sink_InitialMass",
    "BH_Mdot": "Sink_Mdot",
    "BH_NProgs": "Sink_NProgenitors",
    "BH_Dist": "Sink_Distance",
    "BH_Dust_Mass": "Sink_Dust_Mass",
}


def convert(src, dst):
    if os.path.abspath(src) == os.path.abspath(dst):
        raise SystemExit("refusing to write output over the input file; choose a different output path")
    shutil.copy(src, dst)
    nrenamed = 0
    with h5py.File(dst, "r+") as f:
        groups = [k for k in f.keys() if k.startswith("PartType")]
        for g in groups:
            grp = f[g]
            for legacy, modern in RENAMES.items():
                if legacy in grp:
                    if modern in grp:
                        # modern name already exists -- do not clobber; leave the legacy copy as-is
                        print(f"  {g}: '{modern}' already present; leaving '{legacy}' untouched")
                        continue
                    grp[modern] = grp[legacy]        # HDF5 hard-link the new name to the same data
                    del grp[legacy]                  # drop the old name
                    print(f"  {g}: {legacy} -> {modern}")
                    nrenamed += 1
    print(f"wrote {dst}  ({nrenamed} dataset(s) renamed)")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)
    src = sys.argv[1]
    if len(sys.argv) >= 3:
        dst = sys.argv[2]
    else:
        base, ext = os.path.splitext(src)
        dst = base + "_modern" + (ext or ".hdf5")
    convert(src, dst)


if __name__ == "__main__":
    main()
