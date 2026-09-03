#!/bin/bash
# Full test suite across Genoa nodes via disBatch, out of a staged ceph sweep directory.
#
#   ./scripts/sweep_prepare.sh                       # once, prints the SWEEP path
#   GIZMO_SWEEP=<path> sbatch scripts/run_sweep_genoa.sbatch
#
# One test directory per node at a time (-t 1). That is deliberate: run_test() sizes rank
# count from cpu_count()/2 (python_src/gizmo/test.py:103), so on a 96-core Genoa node a
# test runs at 48 ranks. Packing several tests onto a node would either oversubscribe it
# or force the rank count down, and rank count is an axis real defects live on.
#
# 68 test directories. --nodes=8 gives roughly 9 apiece; raise it with `sbatch --nodes=N`
# if the queue is free, since disBatch just hands tasks to whatever engines exist.
#
# Pending on "Nodes required for job are DOWN, DRAINED, ... or reserved" just means the genoa
# rocky9 pool is full. Wait it out. Do NOT reach for the idle rocky8 nodes in the site reservation:
# the module tree is OS-keyed (MODULEPATH is linux-rocky9-x86_64 and lmod/x86_64/rocky9), and Lmod's
# own lua links the system glibc, so on Rocky 8's 2.28 `module load` dies and the job exits 127 in
# one second without running a thing.
#
# Resume a partial sweep (nodes ran out of time, or you fixed something):
#   GIZMO_SWEEP=<path> DB_RESUME=<sweep>/tasks_disBatch_*_status.txt sbatch scripts/run_sweep_genoa.sbatch
#
#SBATCH --job-name=sfdev_sweep
#SBATCH --partition=cca
#SBATCH --nodes=8
#SBATCH --exclusive
#SBATCH --constraint=genoa
#SBATCH --time=12:00:00
#SBATCH --output=sfdev_sweep-%j.out
#SBATCH --error=sfdev_sweep-%j.err
set -uo pipefail

SWEEP="${GIZMO_SWEEP:?set GIZMO_SWEEP to the directory sweep_prepare.sh created}"
[ -f "$SWEEP/tasks.db" ] || { echo "FATAL: no tasks.db under $SWEEP"; exit 1; }

module --force purge
# texlive is required: several tests render labels through matplotlib's usetex path, and
# --force purge strips it.
module load modules/2.4-20250724 gcc openmpi gsl hdf5/mpi-1.12.3 fftw texlive
module load disBatch

export SYSTYPE=RUSTY_GENOA
export GIZMO_TEST_TIMEOUT="${GIZMO_TEST_TIMEOUT:-10800}"
# MHD_MODIFIED_GRADIENT needs hypre, which Rusty has no module for. Build it once with
# scripts/build_hypre_rusty.sh; not built here, because 8 engines racing into one prefix is not
# something an idempotency check saves you from.
export HYPRE_PATH="${HYPRE_PATH:-$HOME/opt/hypre}"
if [ ! -f "$HYPRE_PATH/include/HYPRE.h" ]; then
    echo "FATAL: no hypre at $HYPRE_PATH -- run ./scripts/build_hypre_rusty.sh first"
    exit 1
fi

cd "$SWEEP" || exit 1
echo "=== sweep   $SWEEP"
echo "=== tree    HEAD $(cat "$SWEEP/HEAD_SHA" 2>/dev/null)"
echo "=== nodes   $SLURM_JOB_NUM_NODES x $(nproc) cores, $SLURM_JOB_NODELIST"
echo "=== tasks   $(grep -cv '^#' tasks.db)  timeout ${GIZMO_TEST_TIMEOUT}s each"

RESUME=""
if [ -n "${DB_RESUME:-}" ]; then RESUME="-r ${DB_RESUME} -R"; echo "=== resuming from ${DB_RESUME}"; fi

# -e so the job's exit status reflects a failed test rather than only a disBatch fault.
disBatch -e -t 1 $RESUME tasks.db
rc=$?

cat "$SWEEP"/results/*.rc 2>/dev/null | sort > "$SWEEP/RESULTS.txt"
echo
echo "=== RESULTS (rc=0 is a pass) ==="
cat "$SWEEP/RESULTS.txt"
echo "=== ran $(wc -l < "$SWEEP/RESULTS.txt") of $(wc -l < "$SWEEP/tasks.db") ==="
echo "=== failures ==="
awk '$2 != "rc=0"' "$SWEEP/RESULTS.txt" | grep . || echo "  none"
# A skip exits 0. The harness skips a test whose GIZMO run blows the timeout, so these are
# not passes -- they are tests that did not finish.
echo "=== skipped (did not run to completion) ==="
grep -a "skipped" "$SWEEP/RESULTS.txt" | grep . || echo "  none"
# A test directory with no .rc never finished: node ran out of walltime, or the task died
# outside pytest. Silence here would read as a pass.
echo "=== no result at all ==="
comm -23 <(sort "$SWEEP/testdirs.txt") <(awk '{print $1}' "$SWEEP/RESULTS.txt" | sort) \
    | sed 's/^/  /' | grep . || echo "  none"
exit $rc
