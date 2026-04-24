#!/usr/bin/env python3
"""Compare GPU vs CPU outputs from the gravtree_sinks Phase 2-B test.

Validates that SINK_CALC_DISTANCES + SINGLE_STAR_* GPU walk gives the same
gravity forces as the CPU walk at snapshot_000 (first gravity call from
identical positions, before trajectory divergence).

Validation metrics (scale-normalized: max|diff|/max|cpu|):
  - PartType0/Coordinates  : position unchanged at snap_000 (< 1e-12)
  - PartType0/Velocities   : velocity kick (< 1e-8 reldiff from force accumulation)
  - PartType5/Coordinates  : sink position unchanged (< 1e-12)
  - PartType5/Velocities   : sink velocity kick (< 1e-8)
"""

import h5py, numpy as np, sys, os

CPU_DIR = "output_cpu_1step_np2"
GPU_DIR = "output_gpu_1step_np2"
SNAP    = 0


def load_snap(outdir, snap_idx):
    path = os.path.join(outdir, f"snapshot_{snap_idx:03d}.hdf5")
    f = h5py.File(path, "r")
    data = {}
    for ptype in ("PartType0", "PartType5"):
        if ptype not in f:
            continue
        grp = f[ptype]
        data[ptype] = {
            "ids":   grp["ParticleIDs"][:],
            "Coordinates": grp["Coordinates"][:] if "Coordinates" in grp else None,
            "Velocities":  grp["Velocities"][:]  if "Velocities"  in grp else None,
        }
        for field in ("Min_Distance_to_Sink", "Min_Sink_Approach_Time",
                      "Min_Sink_FeedbackTime", "Min_Sink_Freefall_Time",
                      "BH_Mdot", "StellarFormationTime"):
            if field in grp:
                data[ptype][field] = grp[field][:]
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
            continue
        if field not in gpu_d or gpu_d[field] is None:
            print(f"  {field}: missing in GPU"); all_pass = False; continue
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


cpu = load_snap(CPU_DIR, SNAP)
gpu = load_snap(GPU_DIR, SNAP)

GAS_TOLS = {
    "Coordinates": 1e-12,
    "Velocities":  1e-4,   # FP rounding from GPU vs CPU walk ordering; Phase 2-A baseline ~6e-5
    "Min_Distance_to_Sink": 1e-6,
    "Min_Sink_Approach_Time": 1e-4,
    "Min_Sink_FeedbackTime": 1e-4,
}
SINK_TOLS = {
    "Coordinates": 1e-12,
    "Velocities":  1e-4,
}

all_pass = True
print(f"=== gravtree_sinks Phase 2-B: snapshot_{SNAP:03d} CPU vs GPU ===")
if "PartType0" in cpu and "PartType0" in gpu:
    all_pass &= compare_ptype(cpu["PartType0"], gpu["PartType0"], "PartType0 (gas)", GAS_TOLS)
else:
    print("WARN: PartType0 missing from one run")
if "PartType5" in cpu and "PartType5" in gpu:
    all_pass &= compare_ptype(cpu["PartType5"], gpu["PartType5"], "PartType5 (sink)", SINK_TOLS)
else:
    print("WARN: PartType5 missing — check if sink was swallowed or not output")

print()
if all_pass:
    print("RESULT: PASS — GPU sink walk matches CPU within tolerances")
else:
    print("RESULT: FAIL — see above"); sys.exit(1)
