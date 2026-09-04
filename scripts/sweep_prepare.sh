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
# One task per (test directory, VARIANT). Splitting variants is what keeps the sweep short:
# 67% of node-time sits in multi-variant tests that would otherwise run their variants
# sequentially inside one task, so one engine grinds through hernquist's 8 variants (2h45m)
# while other nodes idle. Per-variant tasks drop the critical path from the longest TEST
# (3.0h) to the longest VARIANT (1.5h).
#
# Safe because each engine rsyncs its own private tree and variant_output_dir() separates
# outputs within it, so two variants on different engines never touch the same files. The
# earlier "variants share the directory's ICs and output paths" concern applied to running
# them concurrently in ONE tree, which never happens here.
#
# SWEEP_TESTS restricts the run to a subset -- a file of directory names, or the names inline.
# Used to re-run only what a fix could plausibly change, rather than paying 8 nodes to watch
# known failures fail again. It selects DIRECTORIES; all of a directory's variants come along.
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
# Enumerate the variants of each selected directory. The pytest id is
# test_X[<variant>-<omp>-<ranks>], and the trailing two components come from cpu_count() on
# whatever machine collects -- 16 ranks here, 48 on a Genoa node -- so only the VARIANT name is
# carried forward. run_one.sh resolves the full node id on the compute node, where it is right.
# A directory whose collection fails (import error, missing optional dep) falls back to a single
# whole-directory task rather than vanishing from the sweep.
# ONE collect for the whole suite, not one per directory: each pytest start re-imports h5py,
# matplotlib and astropy, so 69 of them cost ~7 minutes against 18 seconds for a single pass.
# --continue-on-collection-errors so one unimportable test file does not blank the sweep.
#
# NOSPLIT lists directories that must NOT be split: their variants compare against each other,
# so they need pytest's in-process ordering. Engines work in private trees and disBatch gives no
# ordering barrier, so a split would put a variant on a node where its reference never ran.
NOSPLIT="hernquist_convergence"
PY="${GIZMO_TEST_PYTHON:-/mnt/home/mgrudic/python_work/bin/python}"
echo "=== collecting variants ..."
( cd "$TREE" && "$PY" -m pytest test/ --collect-only -q --continue-on-collection-errors 2>/dev/null ) \
  | "$PY" -c '
import sys, re
want = set(open(sys.argv[1]).read().split())
nosplit = set(sys.argv[2].split())
seen, out = set(), []
for line in sys.stdin:
    m = re.match(r"^test/([^/]+)/.*?(?:\[(.*)\])?$", line.strip())
    if not m or m.group(1) not in want: continue
    d, br = m.group(1), m.group(2)
    if d in nosplit: br = None
    # The id carries the omp/rank counts from cpu_count() on whatever machine collects (16 here,
    # 48 on a Genoa node), so strip them and keep only the variant. Which END they sit on depends
    # on parametrize decorator order -- shu_jets has [2-16-cooling-jets_merge], most tests have
    # [baseline-0-16] -- so strip purely-numeric components from BOTH ends rather than assuming.
    # An all-numeric id means the test is not variant-parametrised at all.
    parts = br.split("-") if br else []
    while parts and parts[0].isdigit(): parts.pop(0)
    while parts and parts[-1].isdigit(): parts.pop()
    v = "-".join(parts)
    key = (d, v)
    if key in seen: continue
    seen.add(key); out.append(f"{d} {v}".rstrip())
for d in sorted(want - {k[0] for k in seen}):   # collection failed: whole-directory fallback
    out.append(d)
print("\n".join(sorted(out)))
' "$SWEEP/testdirs.txt" "$NOSPLIT" > "$SWEEP/tasks.txt"
NTASK=$(wc -l < "$SWEEP/tasks.txt")
NVAR=$(awk 'NF>1' "$SWEEP/tasks.txt" | wc -l)
echo "=== $NTASK tasks from $(wc -l < "$SWEEP/testdirs.txt") directories ($NVAR are per-variant)"

# Written out in full rather than via #DISBATCH PREFIX: the prefix is textually prepended
# with no separator, which is one missing trailing space away from silent nonsense.
sed "s|^|bash $SWEEP/run_one.sh |" "$SWEEP/tasks.txt" > "$SWEEP/tasks.db"

# ---------------------------------------------------------------- per-task runner
cat > "$SWEEP/run_one.sh" <<'RUNONE'
#!/bin/bash
# One (test directory, variant) in this engine's private tree. disBatch runs one of these at
# a time per node (-t 1), so each gets the whole node and its own default rank count.
#
#   run_one.sh <testdir>            whole directory (unparametrised tests)
#   run_one.sh <testdir> <variant>  one variant of it
set -uo pipefail
SWEEP="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TESTDIR="$1"
VARIANT="${2:-}"
RANK="${DISBATCH_ENGINE_RANK:-0}"
TREE="$SWEEP/tree_${RANK}"
TAG="$TESTDIR${VARIANT:+__$VARIANT}"
LOG="$SWEEP/logs/${TAG}.log"

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

# Resolve the pytest node id HERE rather than carrying it from prepare time. The id embeds
# omp/rank counts from cpu_count(), which differ between the machine that staged the sweep and
# this one (16 vs 48), so a pre-computed id would select nothing. Match the variant with the
# numeric components allowed on either side, mirroring how sweep_prepare stripped them; anchoring
# both ends keeps it exact, so subcycle_rt cannot also match subcycle_rt_cooling.
TARGET="test/$TESTDIR"
if [ -n "$VARIANT" ]; then
    VRE=$(printf '%s' "$VARIANT" | sed 's/[][\.*^$+?(){}|\\]/\\&/g')
    TARGET=$("$PY" -m pytest "test/$TESTDIR" --collect-only -q 2>/dev/null \
               | grep -E "\[([0-9]+-)*${VRE}(-[0-9]+)*\]\$" | head -1)
    if [ -z "$TARGET" ]; then
        echo "FATAL: variant '$VARIANT' did not resolve to a pytest id in test/$TESTDIR" | tee "$LOG"
        printf '%-40s rc=%-3s %6ss  %-14s %s\n' "$TAG" 90 0 "$(hostname)" "UNRESOLVED VARIANT" \
            > "$SWEEP/results/${TAG}.rc"
        exit 90
    fi
fi

# A SUBSHELL, not a brace group: braces do not fork, so the `exit` below would end
# run_one.sh outright and the result file after it would never be written.
(
    echo "=== $TAG  node $(hostname)  engine $RANK  cores $(nproc)  $(date '+%F %T')"
    echo "=== target: $TARGET"
    "$PY" -u -m pytest "$TARGET" -ra -s -v --tb=short
    rc=$?
    echo "=== $TAG rc=$rc elapsed=$((SECONDS - t0))s"
    exit $rc
) > "$LOG" 2>&1
rc=$?
# rc alone is not the verdict: run_test() calls pytest.skip when GIZMO exceeds
# GIZMO_TEST_TIMEOUT, and a skipped test exits 0. Carry pytest's own summary line so a
# timed-out run cannot read as a pass.
summary=$(grep -aoE '[0-9]+ (passed|failed|skipped|error)[^=]*' "$LOG" | tail -1 | tr -s ' ')
# One result file per test, not a shared append: O_APPEND is not atomic across hosts on
# Ceph, and a lost line there is a lost failure. The sbatch collates these at the end.
printf '%-40s rc=%-3s %6ss  %-14s %s\n' \
    "$TAG" "$rc" "$((SECONDS - t0))" "$(hostname)" "${summary:-NO SUMMARY}" \
    > "$SWEEP/results/${TAG}.rc"
exit $rc
RUNONE
chmod +x "$SWEEP/run_one.sh"

echo "=== $NTASK tasks written to $SWEEP/tasks.db"
echo
echo "Submit with:"
echo "    cd $TREE && GIZMO_SWEEP=$SWEEP sbatch scripts/run_sweep_genoa.sbatch"
