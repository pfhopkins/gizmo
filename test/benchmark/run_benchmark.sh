#!/bin/bash
# run_benchmark.sh — Compare GPU neighbor-list path vs CPU tree-walk path.
#
# Runs the same problem twice: once with GIZMO_USE_NEIGHBOR_LIST_FOR_DENSITY
# (GPU/Kokkos path) and once without (CPU tree-walk). Extracts per-phase
# timing from [BENCH] lines and cpu.txt.
#
# Usage:
#   ./test/benchmark/run_benchmark.sh [nranks] [params_file]
#
# Defaults: nranks=1, params_file=test/gmc_cooling/gmc_cooling_quick.params

set -u

GIZMO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$GIZMO_ROOT"

NRANKS="${1:-1}"
PARAMS="${2:-test/gmc_cooling/gmc_cooling_quick.params}"
NJOBS=8

echo "============================================================"
echo "GIZMO Performance Benchmark — $(date)"
echo "Ranks: $NRANKS  Problem: $PARAMS"
echo "============================================================"

# Save originals
cp -f Config.sh Config.sh.bench_backup 2>/dev/null || true

# --- Base config from the test's Config.sh ---
TEST_DIR="$(dirname "$PARAMS")"
if [[ -f "$TEST_DIR/Config.sh" ]]; then
    BASE_CONFIG=$(cat "$TEST_DIR/Config.sh")
else
    BASE_CONFIG="SINGLE_STAR_STARFORGE_DEFAULTS
MAGNETIC
COOLING
METALS
COOL_METAL_LINES_BY_SPECIES
BOX_PERIODIC
GRAVITY_NOT_PERIODIC
ADAPTIVE_TREEFORCE_UPDATE=0.0625"
fi

# ---- RUN 1: GPU/Kokkos neighbor-list path ----
echo ""
echo ">>> Building GPU path (GIZMO_USE_NEIGHBOR_LIST_FOR_DENSITY)..."
{
    echo "GIZMO_USE_NEIGHBOR_LIST_FOR_DENSITY"
    echo "$BASE_CONFIG"
} > Config.sh
make clean > /dev/null 2>&1 && make -j${NJOBS} > /dev/null 2>&1
if [[ $? -ne 0 ]]; then echo "GPU build FAILED"; exit 1; fi

echo ">>> Running GPU path ($NRANKS ranks)..."
rm -f output/cpu.txt
mpirun -np "$NRANKS" ./GIZMO "$PARAMS" > bench_gpu.log 2>&1
echo ">>> GPU run complete."

# Extract timing
echo ""
echo "=== GPU PATH TIMING ==="
grep "\[BENCH\]" bench_gpu.log
echo ""
echo "cpu.txt summary (last step):"
tail -30 output/cpu.txt 2>/dev/null | head -25
cp output/cpu.txt bench_gpu_cpu.txt 2>/dev/null

# ---- RUN 2: CPU tree-walk path ----
echo ""
echo ">>> Building tree-walk path (no GIZMO_USE_NEIGHBOR_LIST_FOR_DENSITY)..."
echo "$BASE_CONFIG" > Config.sh
make clean > /dev/null 2>&1 && make -j${NJOBS} > /dev/null 2>&1
if [[ $? -ne 0 ]]; then echo "Tree-walk build FAILED"; exit 1; fi

echo ">>> Running tree-walk path ($NRANKS ranks)..."
rm -f output/cpu.txt
mpirun -np "$NRANKS" ./GIZMO "$PARAMS" > bench_tree.log 2>&1
echo ">>> Tree-walk run complete."

# Extract timing
echo ""
echo "=== TREE-WALK PATH TIMING ==="
grep "\[BENCH\]" bench_tree.log
echo ""
echo "cpu.txt summary (last step):"
tail -30 output/cpu.txt 2>/dev/null | head -25
cp output/cpu.txt bench_tree_cpu.txt 2>/dev/null

# ---- COMPARISON ----
echo ""
echo "============================================================"
echo "SIDE-BY-SIDE COMPARISON (last [BENCH] line from each)"
echo "============================================================"
echo "GPU:  $(grep '\[BENCH\]' bench_gpu.log | tail -1)"
echo "Tree: $(grep '\[BENCH\]' bench_tree.log | tail -1)"
echo ""

# Restore
cp -f Config.sh.bench_backup Config.sh 2>/dev/null || true

echo "Full logs: bench_gpu.log, bench_tree.log"
echo "cpu.txt:   bench_gpu_cpu.txt, bench_tree_cpu.txt"
echo "============================================================"
