#!/bin/bash
# Sink / few-body integration accuracy tests, in ascending runtime order.
#
#   ./run_sink_integration_tests.sh [-t SECONDS] [-k EXPR] [-x] [-l]
#
#   -t SECONDS  per-test GIZMO subprocess timeout   (default: 3600)
#   -k EXPR     restrict to tests matching EXPR     (e.g. -k binary)
#   -x          stop at the first failure
#   -l          list the tests and their costs, then exit
#
# These four exercise the Hermite sink integrator from opposite directions:
#
#   binary            2 sinks, e=0.9 q=0.1, 1000 orbits. MAGNITUDE guard on the Hermite
#                     source-prediction fix (b74a8d35): without it the same run leaks 17x over
#                     the drift ceiling. One-bin split -- the most a bound PAIR can produce.
#   triple            3 sinks, 4 AU inner binary + 100 AU tertiary, 50 outer orbits. The
#                     many-timebin case a pair cannot reach (6-bin gap). Guards by drift GROWTH
#                     EXPONENT, not magnitude; calibrated at eta=1.25e-3.
#   plummer_binaries  binaries inside a live cluster potential: the realistic mixed case, where
#                     sinks are also being perturbed by an environment.
#   fewbody           48 few-body problems, 4 variants. Judge it on the direct_gravity variants:
#                     under TREE forces it fails all 48 for an unrelated reason (star-star tree
#                     error; exact forces are ~633x better), so the tree variants say nothing
#                     about the integrator. NOTE the direct_gravity variants need
#                     SINGLE_STAR_DIRECT_GRAVITY, whose implementation
#                     (gravity/star_direct_gravity.cc) is NOT in this tree yet -- it arrives with
#                     origin/starforge_dev commit 5c08e496. Until that is merged, only the two
#                     tree variants can build, and those are the uninformative ones.
#
# NOTE the resting state of binary and triple is 'xfailed', not 'passed': both still record a
# secular residual (the 4th-order block-step impulse asymmetry) as an expected failure. An
# unexpected PASS on those xfail branches is itself worth investigating -- it means the residual
# moved. Real regressions surface as FAILED.
set -uo pipefail

TIMEOUT=3600
KEXPR=
EXITFIRST=
LISTONLY=0
while getopts "t:k:xlh" o; do case $o in
  t) TIMEOUT=$OPTARG ;; k) KEXPR=$OPTARG ;; x) EXITFIRST="-x" ;; l) LISTONLY=1 ;;
  h) sed -n '2,30p' "$0"; exit 0 ;; *) exit 2 ;;
esac; done

REPO=$(cd "$(dirname "$0")" && pwd)
cd "$REPO"

TESTS=(
  "binary|test/binary/test_binary.py|~3 min|1000 orbits, 4002 snapshots"
  "triple|test/triple/test_triple.py|~4 min|50 outer orbits, 6-bin hierarchy"
  "plummer_binaries|test/plummer_binaries/test_plummer_binaries.py|~15-40 min|cluster + binaries"
  "fewbody|test/fewbody/test_fewbody.py|~20-60 min|48 problems x 4 variants"
)

if [ "$LISTONLY" -eq 1 ]; then
    printf "  %-18s %-10s %s\n" "TEST" "COST" "WHAT"
    for e in "${TESTS[@]}"; do IFS='|' read -r n f c w <<< "$e"
        printf "  %-18s %-10s %s\n" "$n" "$c" "$w"; done
    exit 0
fi

FILES=()
for e in "${TESTS[@]}"; do
    IFS='|' read -r n f c w <<< "$e"
    [ -f "$f" ] || { echo "SKIP $n: $f not found"; continue; }
    FILES+=("$f")
done
[ ${#FILES[@]} -gt 0 ] || { echo "FATAL: none of the test files exist under $REPO"; exit 1; }

# texlive: some test plots use matplotlib's usetex path, which shells out to 'latex'. Without it
# a test FAILS after GIZMO has run correctly and every physics assertion has passed -- a failure
# that looks like a physics regression but is a missing font package.
module --force purge >/dev/null 2>&1
module load modules openmpi gsl hdf5/mpi-1.12.3 texlive >/dev/null 2>&1

# The suite sizes MPI ranks from os.cpu_count(); these tests are small and serial-ish, so pin to
# 1 thread and let each test's own parametrization pick its rank count.
export OMP_NUM_THREADS=1
export OMP_PROC_BIND=false
unset OMP_PLACES
export GIZMO_TEST_TIMEOUT=$TIMEOUT

if [ -n "${PYTEST:-}" ]; then PY="$PYTEST"
elif [ -x "$HOME/python_work/bin/pytest" ]; then PY="$HOME/python_work/bin/pytest"
elif command -v pytest >/dev/null 2>&1; then PY=pytest
else echo "FATAL: no pytest found (set \$PYTEST)"; exit 1; fi

SEL=(); [ -n "$KEXPR" ] && SEL+=(-k "$KEXPR")
LOG="$REPO/sink_integration_tests.log"

echo "=== $(date) sink integration tests ==="
echo "    repo:    $REPO ($(git rev-parse --short HEAD 2>/dev/null))"
echo "    pytest:  $PY"
echo "    timeout: ${TIMEOUT}s per GIZMO subprocess"
echo "    tests:   ${FILES[*]}"
echo

"$PY" "${FILES[@]}" -v -s -ra $EXITFIRST "${SEL[@]}" 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}

echo
echo "=== SUMMARY ==="
grep -aE "^(FAILED|ERROR|XFAIL|XPASS|PASSED)" "$LOG" | cut -c1-120 || echo "  (none)"
echo
echo "  full log:  $LOG"
echo "  figures:   test/binary/*.png  test/triple/*.png"
echo
if [ ! -f gravity/star_direct_gravity.cc ]; then
  echo "  WARNING: gravity/star_direct_gravity.cc is absent, so fewbody's direct_gravity and"
  echo "  direct_equaldt variants cannot build. Those are the informative ones -- under tree"
  echo "  forces all 48 problems fail for an unrelated reason (star-star tree error). Merge"
  echo "  origin/starforge_dev (commit 5c08e496) to enable them."
fi
echo "=== $(date) done (exit $rc) ==="
exit $rc
