#!/bin/bash
# benchmark_analyze.sh — Analyze benchmark results after SLURM jobs complete.
#
# Reads gizmo.out from bench_gpu/ and bench_tree/ directories,
# extracts [BENCH] timing lines, and prints side-by-side comparison.
#
# Usage:
#   ./test/benchmark/benchmark_analyze.sh

BENCH_DIR="$(cd "$(dirname "$0")" && pwd)"
GPU_DIR="$BENCH_DIR/bench_gpu"
TREE_DIR="$BENCH_DIR/bench_tree"

echo "============================================================"
echo "GIZMO Benchmark Analysis — $(date)"
echo "============================================================"

for DIR in "$GPU_DIR" "$TREE_DIR"; do
    NAME=$(basename "$DIR")
    OUT="$DIR/gizmo.out"
    if [[ ! -f "$OUT" ]]; then
        echo "WARNING: $OUT not found — job may not have completed"
        continue
    fi
    echo ""
    echo "=== $NAME ==="
    echo "--- Per-step timing ---"
    grep "\[BENCH\]" "$OUT" || echo "(no [BENCH] lines found)"
    echo ""
    echo "--- HYDRO DIAG ---"
    grep "\[HYDRO DIAG\]" "$OUT" | head -5
    echo ""
    echo "--- cpu.txt (last step) ---"
    if [[ -f "$DIR/output/cpu.txt" ]]; then
        # Find the last step block
        tail -30 "$DIR/output/cpu.txt" | head -25
    else
        echo "(cpu.txt not found)"
    fi
done

# Side-by-side comparison
echo ""
echo "============================================================"
echo "SIDE-BY-SIDE (last [BENCH] from each)"
echo "============================================================"
GPU_LAST=$(grep '\[BENCH\]' "$GPU_DIR/gizmo.out" 2>/dev/null | tail -1)
TREE_LAST=$(grep '\[BENCH\]' "$TREE_DIR/gizmo.out" 2>/dev/null | tail -1)
echo "GPU:  $GPU_LAST"
echo "Tree: $TREE_LAST"

# Extract totals for ratio
GPU_TOTAL=$(echo "$GPU_LAST" | grep -oP 'total=\K[0-9.]+' 2>/dev/null || echo "")
TREE_TOTAL=$(echo "$TREE_LAST" | grep -oP 'total=\K[0-9.]+' 2>/dev/null || echo "")

if [[ -n "$GPU_TOTAL" && -n "$TREE_TOTAL" ]]; then
    RATIO=$(python3 -c "print(f'{float(\"$TREE_TOTAL\")/float(\"$GPU_TOTAL\"):.2f}')" 2>/dev/null || echo "?")
    echo ""
    echo "Speedup (tree/GPU): ${RATIO}x"
    echo "(>1 = GPU faster, <1 = tree faster)"
fi

# Per-phase comparison if both have data
if [[ -n "$GPU_LAST" && -n "$TREE_LAST" ]]; then
    echo ""
    echo "--- Per-phase breakdown ---"
    python3 -c "
import re, sys

def parse_bench(line):
    d = {}
    for m in re.finditer(r'(\w+)=([\d.]+)', line):
        d[m.group(1)] = float(m.group(2))
    return d

gpu = parse_bench('''$GPU_LAST''')
tree = parse_bench('''$TREE_LAST''')

phases = ['density', 'grad', 'hydro', 'ghost', 'symlist', 'total']
print(f'{\"Phase\":>10s}  {\"GPU (s)\":>10s}  {\"Tree (s)\":>10s}  {\"Speedup\":>10s}')
print('-' * 46)
for p in phases:
    g = gpu.get(p, 0)
    t = tree.get(p, 0)
    if g > 0 and t > 0:
        ratio = t / g
        print(f'{p:>10s}  {g:>10.4f}  {t:>10.4f}  {ratio:>9.2f}x')
    elif g > 0:
        print(f'{p:>10s}  {g:>10.4f}  {\"---\":>10s}  {\"---\":>10s}')
    elif t > 0:
        print(f'{p:>10s}  {\"---\":>10s}  {t:>10.4f}  {\"---\":>10s}')
" 2>/dev/null || echo "(python3 not available for detailed breakdown)"
fi

echo ""
echo "============================================================"
