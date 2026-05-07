#!/bin/bash
# Generate three concrete Phase 0 sbatch files from phase0_np.sbatch.template
# Usage: ./gen_phase0_sbatch.sh [partition]   (default: gh)
set -eu
PARTITION="${1:-gh}"
DIR="$(dirname "$0")"
TPL="$DIR/phase0_np.sbatch.template"
for NR in 2 4 8; do
    case $NR in
        2) OMPT=36 ;;
        4) OMPT=18 ;;
        8) OMPT=9  ;;
    esac
    OUT="$DIR/phase0_np${NR}.sbatch"
    sed -e "s|__NRANKS__|$NR|g" -e "s|__OMPT__|$OMPT|g" -e "s|__PARTITION__|$PARTITION|g" "$TPL" > "$OUT"
    chmod +x "$OUT"
    echo "wrote $OUT  (np=$NR omp=$OMPT partition=$PARTITION)"
done
