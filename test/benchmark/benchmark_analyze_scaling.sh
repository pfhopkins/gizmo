#!/bin/bash
# benchmark_analyze_scaling.sh — Analyze scaling benchmark results.
#
# Reads gizmo.out from bench_gpu_NNN/ and bench_tree_NNN/ directories,
# extracts [BENCH] timing, and prints scaling table.
#
# Usage:
#   ./test/benchmark/benchmark_analyze_scaling.sh

BENCH_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "============================================================"
echo "GIZMO Scaling Benchmark Analysis — $(date)"
echo "============================================================"

# Collect all resolution directories
SIZES=""
for d in "$BENCH_DIR"/bench_gpu_*; do
    if [[ -d "$d" ]]; then
        n=$(basename "$d" | sed 's/bench_gpu_//')
        SIZES="$SIZES $n"
    fi
done

if [[ -z "$SIZES" ]]; then
    echo "No benchmark directories found. Run benchmark_scaling.sh first."
    exit 1
fi

# Header
printf "\n%-8s  %10s  %10s  %10s  %10s  %10s  %10s  %10s  %10s\n" \
    "N^(1/3)" "Npart" "GPU_dens" "GPU_grad" "GPU_hydro" "GPU_total" "Tree_total" "Speedup" "Hydro_spd"
printf "%-8s  %10s  %10s  %10s  %10s  %10s  %10s  %10s  %10s\n" \
    "--------" "----------" "----------" "----------" "----------" "----------" "----------" "----------" "----------"

for N in $SIZES; do
    NPART=$(python3 -c "print(${N}**3)" 2>/dev/null || echo "?")
    GPU_OUT="$BENCH_DIR/bench_gpu_${N}/gizmo.out"
    TREE_OUT="$BENCH_DIR/bench_tree_${N}/gizmo.out"

    # Get LAST [BENCH] line from each (most representative: not first-step warmup)
    GPU_BENCH=$(grep '\[BENCH\]' "$GPU_OUT" 2>/dev/null | tail -1)
    TREE_BENCH=$(grep '\[BENCH\]' "$TREE_OUT" 2>/dev/null | tail -1)

    if [[ -z "$GPU_BENCH" || -z "$TREE_BENCH" ]]; then
        printf "%-8s  %10s  (data missing — check if jobs completed)\n" "$N" "$NPART"
        continue
    fi

    python3 -c "
import re
def parse(line):
    d = {}
    for m in re.finditer(r'(\w+)=([\d.]+)', line):
        d[m.group(1)] = float(m.group(2))
    return d

g = parse('''$GPU_BENCH''')
t = parse('''$TREE_BENCH''')

gt = g.get('total', 0)
tt = t.get('total', 0)
spd = tt / gt if gt > 0 else 0
gh = g.get('hydro', 0)
th = t.get('hydro', 0)
hspd = th / gh if gh > 0 else 0

print(f'$N        {$NPART:>10}  {g.get(\"density\",0):>10.4f}  {g.get(\"grad\",0):>10.4f}  {gh:>10.4f}  {gt:>10.4f}  {tt:>10.4f}  {spd:>9.2f}x  {hspd:>9.1f}x')
" 2>/dev/null || printf "%-8s  %10s  (parse error)\n" "$N" "$NPART"
done

echo ""

# Also print per-run details
for N in $SIZES; do
    for MODE in gpu tree; do
        OUT="$BENCH_DIR/bench_${MODE}_${N}/gizmo.out"
        if [[ -f "$OUT" ]]; then
            echo "--- ${MODE} N=${N} ---"
            grep '\[BENCH\]' "$OUT"
            echo ""
        fi
    done
done

echo "============================================================"
