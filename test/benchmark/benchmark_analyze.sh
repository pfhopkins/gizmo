#!/bin/bash
# benchmark_analyze.sh — Analyze benchmark results after SLURM jobs complete.
#
# Reads gizmo.out from bench_gpu/ and bench_tree/ directories,
# extracts [BENCH] timing lines, and prints side-by-side comparison.
#
# Usage:
#   ./test/benchmark/benchmark_analyze.sh [bench_dir]
#   Default: looks for bench_gpu/ and bench_tree/ next to this script.
#   If a single directory is given, just analyzes that directory.

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
    echo "--- Per-step [BENCH] timing ---"
    grep "\[BENCH\]" "$OUT" || echo "(no [BENCH] lines found)"
    echo ""
    echo "--- Phase breakdowns ---"
    grep -E "(density breakdown|ghost breakdown|symlist breakdown)" "$OUT" | head -15
    echo ""
    echo "--- cpu.txt (last step) ---"
    if [[ -f "$DIR/output/cpu.txt" ]]; then
        tail -30 "$DIR/output/cpu.txt" | head -25
    else
        echo "(cpu.txt not found)"
    fi
done

# Side-by-side comparison
echo ""
echo "============================================================"
echo "SIDE-BY-SIDE COMPARISON"
echo "============================================================"

# Collect all [BENCH] lines (main hydro line has density= on it)
GPU_LAST=$(grep '\[BENCH\].*density=' "$GPU_DIR/gizmo.out" 2>/dev/null | tail -1)
TREE_LAST=$(grep '\[BENCH\].*density=' "$TREE_DIR/gizmo.out" 2>/dev/null | tail -1)
GPU_COOL=$(grep '\[BENCH\].*cooling=' "$GPU_DIR/gizmo.out" 2>/dev/null | tail -1)
TREE_COOL=$(grep '\[BENCH\].*cooling=' "$TREE_DIR/gizmo.out" 2>/dev/null | tail -1)

echo "GPU hydro:    $GPU_LAST"
echo "Tree hydro:   $TREE_LAST"
echo "GPU cooling:  $GPU_COOL"
echo "Tree cooling: $TREE_COOL"

# Detailed per-phase comparison
if [[ -n "$GPU_LAST" || -n "$TREE_LAST" ]]; then
    echo ""
    echo "--- Per-phase breakdown ---"
    python3 -c "
import re, sys

def parse_bench(line):
    d = {}
    for m in re.finditer(r'(\w+)=([\d.]+)', line):
        d[m.group(1)] = float(m.group(2))
    return d

gpu_hydro = parse_bench('''$GPU_LAST''')
tree_hydro = parse_bench('''$TREE_LAST''')
gpu_cool = parse_bench('''$GPU_COOL''')
tree_cool = parse_bench('''$TREE_COOL''')

# Merge cooling into the main dict
for k, v in gpu_cool.items():
    gpu_hydro[k] = v
for k, v in tree_cool.items():
    tree_hydro[k] = v

# Primary phases (order matters for readability)
phases = [
    ('drift',         'drift (move_particles)'),
    ('ghost',         'ghost exchange'),
    ('density_only',  'density kernel'),
    ('density',       'density (total incl hmax)'),
    ('hmax',          'force_update_hmax'),
    ('ghost_redo',    'ghost re-exchange'),
    ('symlist',       'symmetric NL build'),
    ('grad',          'gradients'),
    ('hydro',         'hydro force'),
    ('cooling',       'cooling'),
    ('nuclear',       'nuclear network'),
    ('starformation', 'star formation'),
    ('total',         'TOTAL (dens+grad+hydro)'),
]

print(f'{\"Phase\":>25s}  {\"GPU (s)\":>10s}  {\"Tree (s)\":>10s}  {\"Speedup\":>10s}')
print('-' * 60)
for key, label in phases:
    g = gpu_hydro.get(key, 0)
    t = tree_hydro.get(key, 0)
    if g > 0 or t > 0:
        gs = f'{g:>10.4f}' if g > 0 else f'{\"---\":>10s}'
        ts = f'{t:>10.4f}' if t > 0 else f'{\"---\":>10s}'
        if g > 0 and t > 0:
            ratio = t / g
            rs = f'{ratio:>9.2f}x'
        else:
            rs = f'{\"---\":>10s}'
        print(f'{label:>25s}  {gs}  {ts}  {rs}')

# Compute grand total including cooling
gpu_grand = gpu_hydro.get('total', 0) + gpu_hydro.get('ghost', 0) + gpu_hydro.get('symlist', 0) + gpu_hydro.get('drift', 0) + gpu_hydro.get('cooling', 0)
tree_grand = tree_hydro.get('total', 0) + tree_hydro.get('cooling', 0)
if gpu_grand > 0 and tree_grand > 0:
    print('-' * 60)
    print(f'{\"grand total (w/ cooling)\":>25s}  {gpu_grand:>10.4f}  {tree_grand:>10.4f}  {tree_grand/gpu_grand:>9.2f}x')
" 2>/dev/null || echo "(python3 not available for detailed breakdown)"
fi

# Also show the sub-phase breakdowns for GPU path
echo ""
echo "--- GPU sub-phase breakdowns (from last timestep) ---"
if [[ -f "$GPU_DIR/gizmo.out" ]]; then
    echo "Density:"
    grep "density breakdown" "$GPU_DIR/gizmo.out" | tail -1
    echo "Ghost exchange:"
    grep "ghost breakdown" "$GPU_DIR/gizmo.out" | tail -1
    echo "Symmetric neighbor list:"
    grep "symlist breakdown" "$GPU_DIR/gizmo.out" | tail -1
fi

echo ""
echo "============================================================"
