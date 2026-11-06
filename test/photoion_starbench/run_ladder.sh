#!/bin/bash
# run_ladder.sh [DIR]: set up and submit the finer rungs (m1, m0.1) as Slurm jobs, one node each, with the GIZMO built
# for this test (pytest test/photoion_starbench builds it). Grade with GIZMO_STARBENCH_DIR=DIR pytest test/photoion_starbench.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; DIR="${1:-$HERE}"; mkdir -p "$DIR"; DIR="$(cd "$DIR" && pwd)"
[ -x "$HERE/GIZMO" ] || { echo "run_ladder: build first (pytest test/photoion_starbench)"; exit 1; }
for rung in m1 m0.1; do
    case $rung in m1) c=32; t=16:00:00;; m0.1) c=64; t=24:00:00;; esac
    d="$DIR/run_$rung"
    (cd "$HERE" && PYTHONPATH="$HERE/../harness" python3 -c "
import sys; sys.path.insert(0, '.')
from pathlib import Path
from test_photoion_starbench import setup_run
from gizmo.test import get_stellar_evolution_tables
setup_run(Path('$d'), '$rung', get_stellar_evolution_tables())")
    cp "$HERE/GIZMO" "$d/GIZMO"
    cat > "$d/job.sbatch" <<JOB
#!/bin/bash -l
#SBATCH -o ./run.out.%j
#SBATCH -e ./run.err.%j
#SBATCH -D $d
#SBATCH -J starbench_$rung
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=$c
#SBATCH --time=$t
cd "$d" || exit 1
export OMP_NUM_THREADS=$c OMP_PROC_BIND=spread OMP_PLACES=cores
srun --cpu-bind=cores ./GIZMO photoion_starbench.params 0
JOB
    (cd "$d" && sbatch ${SBATCH_EXTRA:-} job.sbatch)
done
