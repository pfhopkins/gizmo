#!/bin/bash
# Controlled A/B of the Hermite source prediction's cost, on one exclusive Ice Lake node.
#
#   cd /mnt/home/mgrudic/code/gizmo_starforge_dev
#   sbatch run_ab_source_prediction.sbatch
#   sbatch --export=ALL,TEST=binary,RANKS=1 run_ab_source_prediction.sbatch
#   sbatch --export=ALL,TIMEMAX=10.0 run_ab_source_prediction.sbatch
#
# WHY A WHOLE EXCLUSIVE NODE FOR A 2-RUN COMPARISON. This is a timing measurement, and the
# quantity being measured may be a few percent. A shared node means the two variants are timed
# against different background load, which is exactly the size of the effect -- the measurement
# would be noise. Exclusive also stops another job's memory traffic perturbing the tree walk,
# which is the loop under test. Run this alone; do not stack it with the test suite.
#
# The variants build and run SEQUENTIALLY inside one allocation, on the same node, so the
# hardware is identical between them. That matters more than wall-clock efficiency here.
#
#SBATCH --job-name=gizmo_ab_srcpred
#SBATCH --partition=cca
#SBATCH --constraint=icelake
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=4:00:00
#SBATCH --output=ab_srcpred_slurm-%j.out
#SBATCH --error=ab_srcpred_slurm-%j.err
set -uo pipefail

cd "${SLURM_SUBMIT_DIR:-/mnt/home/mgrudic/code/gizmo_starforge_dev}"

TEST=${TEST:-plummer_binaries}
RANKS=${RANKS:-4}
# A tenth of the full run. The metric is normalised per particle-step, so a short run measures
# the same thing as a long one -- there is no reason to pay for the full length.
TIMEMAX=${TIMEMAX:-3.24}

echo "=== $(date) source-prediction A/B on $(hostname) ==="
echo "    branch:  $(git rev-parse --abbrev-ref HEAD)  @ $(git rev-parse --short HEAD)"
echo "    subject: $(git log -1 --format=%s | cut -c1-70)"
echo "    dirty:   $(git status --porcelain | grep -vc '^??') tracked file(s) modified"
echo "    test=$TEST  ranks=$RANKS  TimeMax=$TIMEMAX  cores=$(nproc)"
echo

export SYSTYPE=${SYSTYPE:-RUSTY_ICELAKE}
# Must not be inherited: with concurrent/unpinned mpirun an inherited binding puts every rank on
# the same low-numbered cores, which is silent and costs more than an order of magnitude -- and
# would land differently on the two variants, corrupting exactly the comparison being made.
export OMP_PROC_BIND=false
unset OMP_PLACES
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

./test/ab_source_prediction.sh -t "$TEST" -m "$TIMEMAX" -r "$RANKS"
rc=$?

echo
echo "=== HOW TO READ THIS ==="
echo "  The number that matters is us/particle-step, NOT wall time: the two variants change the"
echo "  forces, so their trajectories diverge and they do not take the same steps. Wall time"
echo "  alone conflates 'slower per unit work' with 'more work'."
echo
echo "  If 'with' is more expensive per particle-step, the cost is the eligible_for_hermite()"
echo "  call in the tree walk's innermost loop (cross-TU, so not inlinable without LTO). The fix"
echo "  would be to precompute eligibility once per pass into an array and test that instead --"
echo "  O(N) rather than O(N x neighbours), and no call in the hot loop."
echo
echo "  If the two are within noise, the prediction is free and the 6x slowdown seen earlier was"
echo "  entirely the wrong IC (100 AU rather than 1000 AU binaries), which is now guarded against"
echo "  in _ensure_ic."
echo "=== $(date) done (exit $rc) ==="
exit $rc
