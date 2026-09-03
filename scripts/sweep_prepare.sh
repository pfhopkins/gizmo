#!/bin/bash
# Stage a full-suite test sweep on ceph: a seed tree, a disBatch task file, and the
# per-task runner. Run this from a workstation or login node (no Slurm needed), then
# submit scripts/run_sweep_genoa.sbatch.
#
#   ./scripts/sweep_prepare.sh [SWEEP_DIR]
#
# The seed is a copy of THIS tree at its current HEAD with build outputs and prior test
# outputs stripped, so downloaded ICs come along and the nodes do not each re-fetch them.
# Every node rsyncs its own working copy off the seed: build_gizmo_for_test rebuilds the
# tree-root sources for each test, so two tests in one tree race on the binary.
set -uo pipefail

# The repo root is the parent of scripts/, not the script's own directory. GIZMO_TREE overrides it
# so a different checkout can be swept -- e.g. upstream, for an A/B against this branch.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TREE="${GIZMO_TREE:-$REPO}"
SWEEP="${1:-/mnt/ceph/users/mgrudic/starforge_dev_sweep/$(date +%Y%m%d_%H%M%S)}"
HEAD_SHA="$(git -C "$TREE" rev-parse --short HEAD)"

if [ -n "$(git -C "$TREE" status --porcelain --untracked-files=no)" ]; then
    echo "WARNING: '$TREE' has uncommitted tracked changes; the sweep will include them."
    git -C "$TREE" status --porcelain --untracked-files=no
fi

mkdir -p "$SWEEP/logs" "$SWEEP/results" || exit 1
echo "=== sweep dir : $SWEEP"
echo "=== source    : $TREE at $HEAD_SHA"

# ---------------------------------------------------------------- seed tree
# Excludes are everything a run regenerates. Keeping them would multiply ~80 GB of stale
# output across every node copy.
echo "=== staging seed (this is the slow part; ~13 GB) ..."
rsync -a --delete \
      --exclude='output' --exclude='output_*' \
      --exclude='.git/' --exclude='__pycache__/' \
      --exclude='*.o' --exclude='GIZMO' --exclude='GIZMO_*' \
      --exclude='*.out' --exclude='*.err' --exclude='*.log' \
      --exclude='slurm-*' \
      "$TREE/" "$SWEEP/seed/" || exit 1
# Makefile:65 does `$(shell cat GIZMO_config.h)` while parsing, so the file must exist before
# any make runs. Sources are at the tree root here -- there is no src/ subdirectory.
touch "$SWEEP/seed/GIZMO_config.h"
echo "$HEAD_SHA" > "$SWEEP/HEAD_SHA"
echo "=== seed staged: $(du -sh "$SWEEP/seed" 2>/dev/null | cut -f1)"

# ---------------------------------------------------------------- task list
# One task per test directory. Not per pytest item: variants inside a directory share the
# directory's ICs and output paths, and splitting them would have two nodes writing the
# same test/<name>/ tree.
#
# SWEEP_TESTS restricts the run to a subset -- a file of directory names, or the names inline.
# Used to re-run only what a fix could plausibly change, rather than paying 8 nodes to watch
# known failures fail again.
( cd "$TREE" && ls test/*/test_*.py ) | sed 's|test/\([^/]*\)/.*|\1|' | sort -u \
    > "$SWEEP/alldirs.txt"

if [ -n "${SWEEP_TESTS:-}" ]; then
    if [ -f "$SWEEP_TESTS" ]; then cat "$SWEEP_TESTS"; else tr ' ' '\n' <<< "$SWEEP_TESTS"; fi \
        | sed '/^$/d' | sort -u > "$SWEEP/testdirs.txt"
    # A name with no test directory would silently vanish from the task list and read as a pass.
    if ! comm -23 "$SWEEP/testdirs.txt" "$SWEEP/alldirs.txt" | grep -q .; then
        echo "=== subset  : $(wc -l < "$SWEEP/testdirs.txt") of $(wc -l < "$SWEEP/alldirs.txt") test dirs"
    else
        echo "FATAL: SWEEP_TESTS names directories that do not exist:"
        comm -23 "$SWEEP/testdirs.txt" "$SWEEP/alldirs.txt" | sed 's/^/  /'
        exit 1
    fi
else
    cp "$SWEEP/alldirs.txt" "$SWEEP/testdirs.txt"
fi
NTASK=$(wc -l < "$SWEEP/testdirs.txt")

# Written out in full rather than via #DISBATCH PREFIX: the prefix is textually prepended
# with no separator, which is one missing trailing space away from silent nonsense.
sed "s|^|bash $SWEEP/run_one.sh |" "$SWEEP/testdirs.txt" > "$SWEEP/tasks.db"

# ---------------------------------------------------------------- per-task runner
cat > "$SWEEP/run_one.sh" <<'RUNONE'
#!/bin/bash
# One test directory, in this engine's private tree. disBatch runs one of these at a
# time per node (-t 1), so each test gets the whole node and its own default rank count.
set -uo pipefail
SWEEP="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TESTDIR="$1"
RANK="${DISBATCH_ENGINE_RANK:-0}"
TREE="$SWEEP/tree_${RANK}"
LOG="$SWEEP/logs/${TESTDIR}.log"

# First task on this node materialises the tree. -t 1 means no concurrent task can race
# us, and the marker distinguishes "finished copying" from "died halfway".
if [ ! -f "$TREE/.seeded" ]; then
    rm -rf "$TREE"
    rsync -a "$SWEEP/seed/" "$TREE/" && touch "$TREE/.seeded"
fi
if [ ! -f "$TREE/.seeded" ]; then
    echo "FATAL: could not stage $TREE" | tee "$LOG"
    exit 1
fi

cd "$TREE" || exit 1
# Detach from Slurm before launching anything. disBatch runs its engines as a one-task-
# per-node srun step, and OpenMPI honours that allocation: mpirun then sees a single slot
# per node and refuses "-np 48" outright ("not enough slots available"). Every test here
# is single-node, so drop the allocation view entirely and let OpenMPI size itself from
# the node's own cores.
unset ${!SLURM_@}
export SYSTYPE=RUSTY_GENOA
export HYPRE_PATH="${HYPRE_PATH:-$HOME/opt/hypre}"
# Without a per-task cpuset an inherited close/cores binding stacks every rank on the
# same cores, which is catastrophic rather than merely slow.
export OMP_PROC_BIND=false
unset OMP_PLACES
export GIZMO_TEST_TIMEOUT="${GIZMO_TEST_TIMEOUT:-10800}"
export MPLBACKEND=Agg

# Preflight, once per node: prove mpirun can actually launch several ranks here. A
# launcher that cannot is not a test failure but it looks exactly like 79 of them, so
# fail the first task loudly rather than burning the whole allocation discovering it.
if [ ! -f "$TREE/.mpiok" ]; then
    if mpirun -np 4 --use-hwthread-cpus true > /dev/null 2>&1; then
        touch "$TREE/.mpiok"
    else
        echo "FATAL: mpirun cannot launch 4 ranks on $(hostname); aborting before any test" \
            | tee "$LOG"
        exit 1
    fi
fi

PY=/mnt/home/mgrudic/python_work/bin/python
t0=$SECONDS
# A SUBSHELL, not a brace group: braces do not fork, so the `exit` below would end
# run_one.sh outright and the result file after it would never be written.
(
    echo "=== $TESTDIR  node $(hostname)  engine $RANK  cores $(nproc)  $(date '+%F %T')"
    "$PY" -u -m pytest "test/$TESTDIR" -ra -s -v --tb=short
    rc=$?
    echo "=== $TESTDIR rc=$rc elapsed=$((SECONDS - t0))s"
    exit $rc
) > "$LOG" 2>&1
rc=$?
# rc alone is not the verdict: run_test() calls pytest.skip when GIZMO exceeds
# GIZMO_TEST_TIMEOUT, and a skipped test exits 0. Carry pytest's own summary line so a
# timed-out run cannot read as a pass.
summary=$(grep -aoE '[0-9]+ (passed|failed|skipped|error)[^=]*' "$LOG" | tail -1 | tr -s ' ')
# One result file per test, not a shared append: O_APPEND is not atomic across hosts on
# Ceph, and a lost line there is a lost failure. The sbatch collates these at the end.
printf '%-32s rc=%-3s %6ss  %-14s %s\n' \
    "$TESTDIR" "$rc" "$((SECONDS - t0))" "$(hostname)" "${summary:-NO SUMMARY}" \
    > "$SWEEP/results/${TESTDIR}.rc"
exit $rc
RUNONE
chmod +x "$SWEEP/run_one.sh"

echo "=== $NTASK tasks written to $SWEEP/tasks.db"
echo
echo "Submit with:"
echo "    cd $TREE && GIZMO_SWEEP=$SWEEP sbatch scripts/run_sweep_genoa.sbatch"
