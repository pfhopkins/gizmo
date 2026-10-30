#!/bin/bash
# run_performance.sh DIR: cost of the per-direction photoionisation at 1e6 and 8e6 cells, one source, flag on and off.
# Needs GIZMO_on and GIZMO_off in DIR (this test's Config.sh with and without GALSF_FB_PHOTOIONIZATION_PER_DIRECTION).
# Submits 4 Slurm jobs (8 ranks x 8 threads, 10 steps); per-solve wall time is column 13 of output/HIIheating.txt.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; DIR="$(cd "$1" && pwd)"
TABLES="$(cd "$HERE" && PYTHONPATH="$HERE/../harness" python3 -c 'from gizmo.test import get_stellar_evolution_tables as g; print(g()["StellarTablesFile"])')"
for n in 100 200; do
    L=$(cd "$HERE" && python3 -c "from make_photoion_per_direction_ics import make_photoion_per_direction_ics as m; print(m('$DIR/ic_$n.hdf5', [(35.0, 1.0e6, (0.5, 0.5, 0.5))], n=$n)[0])")
    for f in on off; do
        d="$DIR/run_${n}_$f"; mkdir -p "$d/output"; ln -sf "$DIR/ic_$n.hdf5" "$d/output/snapshot_000.hdf5"
        sed -e "s#SET_BY_TEST#$TABLES#" -e "s/^BoxSize .*/BoxSize $L/" -e "s/^TimeMax .*/TimeMax 5e-3/" -e "s/^TimeBetSnapshot .*/TimeBetSnapshot 5e-3/" \
            -e "s/^TimeBetStatistics .*/TimeBetStatistics 5e-3/" "$HERE/photoion_per_direction.params" > "$d/run.params"
        (cd "$d" && sbatch -J pf_${n}_$f -N 1 -n 8 -c 8 -t 04:00:00 ${SBATCH_EXTRA:-} \
            --wrap "export OMP_NUM_THREADS=8 OMP_PROC_BIND=spread OMP_PLACES=cores; srun --cpu-bind=cores $DIR/GIZMO_$f run.params 2 0")
    done
done
