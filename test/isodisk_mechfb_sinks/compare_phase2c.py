#!/usr/bin/env python3
"""Compare GPU vs CPU outputs for Phase 2-C (SINK_PHOTONMOMENTUM + SINK_COMPTON_HEATING
+ SINK_DYNFRICTION_FROMTREE) in the isodisk_mechfb_sinks test.

Validates at snapshot_000 (first gravity call, identical positions):
  - Coordinates : bitwise identical (< 1e-12 reldiff)
  - Velocities  : FP-order reldiff < 1e-4
  - Rad_Flux_AGN: reldiff < 1e-4 (SINK_COMPTON_HEATING scatter)
"""

import h5py, numpy as np, sys, os

CPU_DIR = "output_cpu_phase2c"
GPU_DIR = "output_gpu_phase2c"
SNAP    = 0


def load_snap(outdir, snap_idx):
    path = os.path.join(outdir, f"snapshot_{snap_idx:03d}.hdf5")
    f = h5py.File(path, "r")
    data = {}
    for ptype in ("PartType0", "PartType1", "PartType2", "PartType3", "PartType5"):
        if ptype not in f:
            continue
        grp = f[ptype]
        entry = {
            "ids":         grp["ParticleIDs"][:],
            "Coordinates": grp["Coordinates"][:] if "Coordinates" in grp else None,
            "Velocities":  grp["Velocities"][:]  if "Velocities"  in grp else None,
        }
        for field in ("Rad_Flux_AGN",):
            if field in grp:
                entry[field] = grp[field][:]
        data[ptype] = entry
    f.close()
    return data


def compare_ptype(cpu_d, gpu_d, label, tols):
    ids_c = cpu_d["ids"];  ids_g = gpu_d["ids"]
    common = np.intersect1d(ids_c, ids_g)
    n = len(common)
    print(f"\n{label}: CPU={len(ids_c)} GPU={len(ids_g)} common={n}")
    if n == 0:
        print("  ERROR: no common particle IDs"); return False

    sort_c = np.argsort(ids_c); sort_g = np.argsort(ids_g)
    idx_c  = np.searchsorted(ids_c[sort_c], common)
    idx_g  = np.searchsorted(ids_g[sort_g], common)
    sel_c  = sort_c[idx_c];  sel_g = sort_g[idx_g]

    all_pass = True
    for field, tol in tols.items():
        if field not in cpu_d or cpu_d[field] is None:
            print(f"  {field}: absent in CPU snap (field not written — check compile flags)")
            continue
        if field not in gpu_d or gpu_d[field] is None:
            print(f"  {field}: absent in GPU snap"); all_pass = False; continue
        c = np.array(cpu_d[field], dtype=np.float64)[sel_c].ravel()
        g = np.array(gpu_d[field], dtype=np.float64)[sel_g].ravel()
        diff = np.abs(c - g)
        max_cpu  = np.max(np.abs(c))
        max_diff = np.max(diff)
        reldiff  = (max_diff / max_cpu) if max_cpu > 0 else (0.0 if max_diff == 0 else float("inf"))
        ok = reldiff < tol
        if not ok: all_pass = False
        print(f"  {field}: max|cpu|={max_cpu:.3e} max|diff|={max_diff:.3e} "
              f"reldiff={reldiff:.2e} [{'PASS' if ok else 'FAIL'} tol={tol:.0e}]")
    return all_pass


GAS_TOLS = {
    "Coordinates": 1e-12,
    "Velocities":  1e-4,
    "Rad_Flux_AGN": 1e-4,
}
OTHER_TOLS = {
    "Coordinates": 1e-12,
    "Velocities":  1e-4,
}
SINK_TOLS = {
    "Coordinates": 1e-12,
    "Velocities":  1e-4,
}

cpu = load_snap(CPU_DIR, SNAP)
gpu = load_snap(GPU_DIR, SNAP)

all_pass = True
print(f"=== isodisk_mechfb_sinks Phase 2-C: snapshot_{SNAP:03d} CPU vs GPU ===")
for ptype, tols, label in [
    ("PartType0", GAS_TOLS,   "PartType0 (gas)"),
    ("PartType1", OTHER_TOLS, "PartType1 (DM)"),
    ("PartType2", OTHER_TOLS, "PartType2 (disk)"),
    ("PartType3", OTHER_TOLS, "PartType3 (bulge)"),
    ("PartType5", SINK_TOLS,  "PartType5 (sink)"),
]:
    if ptype in cpu and ptype in gpu:
        all_pass &= compare_ptype(cpu[ptype], gpu[ptype], label, tols)
    else:
        print(f"WARN: {ptype} missing from one or both runs")

print()
if all_pass:
    print("RESULT: PASS — GPU Phase 2-C walk matches CPU within tolerances")
else:
    print("RESULT: FAIL — see above"); sys.exit(1)
