#!/usr/bin/env python3
"""Compare RT outputs from two fire_gravtree_rt runs of the same params.

Usage: compare_rt.py [reference_dir] [test_dir]   (defaults: output_reference output_kokkos)

The two runs use the same Config.sh and the same fire_gravtree_rt.params; they
differ only in which build produced them (see README.md), so run one, rename
output/, then run the other.

PhotonEnergy (= Rad_E_gamma) is the snapshot-level output of RT_USE_GRAVTREE_SAVE_RAD_ENERGY.
Matches particles by ParticleID. Uses scale-normalized metric: max|diff|/max|reference|.
"""
import h5py, numpy as np, sys, os

SNAP = 7

def load_snap(outdir, snap_idx):
    path = os.path.join(outdir, f"snapshot_{snap_idx:03d}.hdf5")
    f = h5py.File(path, 'r')
    data = {}
    if 'PartType0' in f:
        gas = f['PartType0']
        data['ids'] = gas['ParticleIDs'][:]
        for field in ['PhotonEnergy', 'Rad_E_gamma', 'Rad_Flux_UV', 'Rad_Flux_EUV']:
            if field in gas:
                data[field] = gas[field][:]
    f.close()
    return data

REF_DIR = sys.argv[1] if len(sys.argv) > 1 else "output_reference"
TEST_DIR = sys.argv[2] if len(sys.argv) > 2 else "output_kokkos"

cpu = load_snap(REF_DIR, SNAP)
gpu = load_snap(TEST_DIR, SNAP)

if 'ids' not in cpu or 'ids' not in gpu:
    print("ERROR: missing ParticleIDs"); sys.exit(1)

# Match on common particle IDs
cpu_ids = cpu['ids']; gpu_ids = gpu['ids']
common = np.intersect1d(cpu_ids, gpu_ids)
n_common = len(common)
print(f"Comparing snapshot_{SNAP:03d}: CPU={len(cpu_ids)} GPU={len(gpu_ids)} common={n_common}\n")

cpu_idx = np.searchsorted(np.sort(cpu_ids), common)
gpu_idx = np.searchsorted(np.sort(gpu_ids), common)
cpu_order = np.argsort(cpu_ids); gpu_order = np.argsort(gpu_ids)
cpu_sel = cpu_order[cpu_idx]; gpu_sel = gpu_order[gpu_idx]

all_pass = True
fields_checked = set(cpu.keys()) | set(gpu.keys())
fields_checked.discard('ids')

for field in sorted(fields_checked):
    if field not in cpu:
        print(f"  {field}: missing in CPU"); continue
    if field not in gpu:
        print(f"  {field}: missing in GPU"); all_pass = False; continue

    c = np.array(cpu[field], dtype=np.float64)
    g = np.array(gpu[field], dtype=np.float64)
    c = c[cpu_sel].ravel(); g = g[gpu_sel].ravel()

    diff = np.abs(c - g)
    max_cpu = np.max(np.abs(c))
    max_diff = np.max(diff)
    n_nonzero_cpu = int(np.sum(c != 0))
    n_nonzero_gpu = int(np.sum(g != 0))

    if max_cpu > 0:
        reldiff = max_diff / max_cpu
    else:
        reldiff = 0.0 if max_diff == 0 else float('inf')

    ok = reldiff < 1e-4
    status = "PASS" if ok else "FAIL"
    if not ok: all_pass = False

    print(f"  {field}:")
    print(f"    max|cpu|={max_cpu:.4e}  max|diff|={max_diff:.4e}  reldiff={reldiff:.2e}  [{status}]")
    print(f"    non-zero: CPU={n_nonzero_cpu}  GPU={n_nonzero_gpu}")
    if max_cpu == 0 and n_nonzero_cpu == 0:
        print(f"    WARNING: all zero — stellar sources not emitting?"); all_pass = False

print()
if all_pass:
    print("RESULT: PASS — RT matches the reference within 1e-4")
else:
    print("RESULT: FAIL — see above"); sys.exit(1)
