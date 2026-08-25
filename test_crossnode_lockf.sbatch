#!/bin/bash
# Does a file lock actually serialise across NODES on this filesystem?
#
# RESULT for fcntl.flock (BSD): NO. Four tasks on worker6073/6114/6069/6070 all acquired within
# 0.03 s and held it simultaneously -- granted node-locally, no cluster coordination.
#
# This version tests fcntl.lockf (POSIX byte-range), which is a DIFFERENT mechanism: on NFS it
# goes through lockd where flock does not, and GPFS treats the two separately. If it serialises,
# the build lock works cross-node with a one-word change and keeps its best property -- the
# kernel releases it when the holder dies, so a killed job cannot wedge everything behind it.
#
#   sbatch --array=0-3 test_crossnode_flock.sbatch
#
# Four array tasks, spread over different nodes, each take an exclusive flock on one shared file,
# hold it for 20 s, and log when they got it. If flock works cluster-wide the acquire times are
# staggered by ~20 s and only one task is ever inside. If it does not, they all enter at once --
# which is what the validation jobs behaved like, though that was inferred from a corrupted build
# rather than measured.
#
# This matters because the alternative locks are worse: a mkdir/O_EXCL lock works on any POSIX
# filesystem but has no automatic release, so a killed job leaves it held and everything after it
# hangs until someone cleans up by hand. flock is released by the kernel on process death. Worth
# knowing which one is actually needed.
#
#SBATCH --job-name=lockftest
#SBATCH --partition=cca
#SBATCH --constraint=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0:10:00
#SBATCH --output=lockftest-%A_%a.out
set -uo pipefail

LOCK=${LOCK:-/mnt/home/mgrudic/code/gizmo_starforge_dev/.lockftest.lock}
LOG=${LOG:-/mnt/home/mgrudic/code/gizmo_starforge_dev/lockftest_results.txt}

/mnt/home/mgrudic/python_work/bin/python - "$LOCK" "$LOG" "${SLURM_ARRAY_TASK_ID:-0}" <<'PY'
import fcntl, os, socket, sys, time
lock, log, tid = sys.argv[1], sys.argv[2], sys.argv[3]
host = socket.gethostname()
t_start = time.time()
with open(lock, "w") as f:
    fcntl.lockf(f, fcntl.LOCK_EX)          # POSIX byte-range, NOT flock
    t_in = time.time()
    # Append INSIDE the lock. If two tasks overlap here their intervals will overlap in the log,
    # which is the thing being tested -- not whether the append itself is atomic.
    with open(log, "a") as g:
        g.write(f"task {tid} host {host} acquired {t_in:.3f} waited {t_in-t_start:.2f}s\n")
        g.flush()
    time.sleep(20)
    with open(log, "a") as g:
        g.write(f"task {tid} host {host} released {time.time():.3f}\n")
        g.flush()
print(f"  task {tid} on {host}: waited {t_in-t_start:.2f}s for the lock")
PY

echo "=== $(date) task ${SLURM_ARRAY_TASK_ID:-0} on $(hostname) done ==="
echo "  when all four finish:  sort -k5 -n $LOG"
echo "  serialised  -> waits are ~0, 20, 40, 60 s and acquire/release intervals do not overlap"
echo "  NOT working -> all four wait ~0 s and their intervals overlap"
