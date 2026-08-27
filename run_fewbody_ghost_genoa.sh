#!/bin/bash
# fewbody acceptance run for the Hermite ghost-source fix.
#
#   sbatch run_fewbody_ghost_genoa.sh
#
# The bar (user-set): all 48 problems under ENERGY_TOL=0.10 on the TREE variants, matching
# starforge_dev. Mechanism already confirmed on one problem: with the ghost table, np=2 is
# identical to np=1 through the pre-encounter window (flat ~2.4e-8; previously secular from
# step 2). This runs the statistics.
#
# Runs the two Hermite tree variants plus direct_gravity as the control row. Skips the
# _kdk/equaldt/freshtree diagnostics -- they answered their questions already.
#
# GENOA: 96 cores runs all 48 two-rank problems in one wave. The icelake run of this same
# script (job 6948624) finished everything in 33 min, so 4h is generous headroom.
#
# Post-detachment-fix expectations: the suite's three worst problems dropped 2.17/2.00/1.81
# -> 0.008/0.0014/0.0006 at np=2 locally, so tree should now be at direct-gravity levels.
#
#SBATCH --job-name=fewbody_ghost
#SBATCH --partition=cca
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --constraint=genoa
#SBATCH --time=4:00:00
#SBATCH --output=fewbody_ghost-%j.out
#SBATCH --error=fewbody_ghost-%j.err
set -uo pipefail

# Runs against the tree it was SUBMITTED from (sbatch copies the script into spool, so $0
# is useless); GIZMO_TREE overrides. Enables one-job-per-worktree concurrency.
TREE="${GIZMO_TREE:-${SLURM_SUBMIT_DIR:-$PWD}}"
PY=/mnt/home/mgrudic/python_work/bin/python
export GIZMO_TEST_TIMEOUT="${GIZMO_TEST_TIMEOUT:-3600}"
export FEWBODY_PROBLEM_TIMEOUT="${FEWBODY_PROBLEM_TIMEOUT:-3600}"
export OMP_PROC_BIND=false
unset OMP_PLACES

module --force purge
module load modules/2.4-20250724 gcc openmpi gsl hdf5/mpi-1.12.3 fftw texlive
export SYSTYPE=RUSTY_GENOA   # the only Rusty block in this tree; carries no -march, portable to Intel
export KOKKOS_PATH="${KOKKOS_PATH:-$HOME/opt/kokkos-openmp}"

cd "$TREE" || exit 1
echo "=== node $(hostname)  cores=$(nproc)  HEAD $(git rev-parse --short HEAD)  (uncommitted changes INCLUDED)"
if [ ! -f "$KOKKOS_PATH/include/Kokkos_Core.hpp" ]; then
    ./scripts/build_kokkos_rusty.sh "$KOKKOS_PATH" || exit 1
fi

rc=0
# The upstream 2x2: {tree, direct} x {individual, equal timesteps}. Selecting by exclusion
# keeps the diagnostic-only variants (freshtree, *_kdk) out. equaldt matters: it multiplies
# the step count, which is exactly what amplified the detachment defect (pre-fix it was the
# worst row, median 1.91), so it is the sharpest regression guard for that fix.
"$PY" -u -m pytest test/fewbody -k "not kdk and not freshtree" -ra -s -v --tb=short || rc=$?

# The upstream guards for the ported fixes: binary (timestep-norm unification) and triple
# (source prediction). Both np=1 single-rank -- cheap riders on the same allocation.
"$PY" -u -m pytest test/binary test/triple -ra -s -v --tb=short || rc=$?

# (A hernquist diagnostic rider ran here once, 2026-08-27: the detachment fix does NOT move it.
# Baseline still drifts secularly to -8e-2 |dE/E| by t=59 and fails the momentum-drift ceiling
# while RANDOMIZE_GRAVTREE improves that metric 37x -- a separate, correlated-opening-error
# defect. Removed from the acceptance sweep; see project-kokkos-energy-regressions.)

echo "=== summary (results_*.json medians) ==="
"$PY" - <<'EOF'
import json, os
import numpy as np
for v in ("tree", "tree_equaldt", "direct_gravity", "direct_equaldt"):
    p = f"test/fewbody/results_{v}.json"
    if not os.path.exists(p):
        print(f"    {v:<16} -- no results"); continue
    d = json.load(open(p))
    w = np.array([q["worst"] for q in d["problems"] if q.get("worst") is not None])
    print(f"    {v:<16} n={len(w):>2}  median={np.median(w):.5f}  worst={w.max():.4f}  over_10pct={(w>=0.10).sum()}")
EOF
exit $rc
