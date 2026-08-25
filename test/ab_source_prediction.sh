#!/bin/bash
# Controlled A/B: what does the Hermite source prediction COST, per particle-step?
#
#   ./test/ab_source_prediction.sh [-t TEST] [-m TIMEMAX] [-r RANKS]
#
# The question. The prediction is gated on eligible_for_hermite(), which is declared in
# core/proto.h and defined in core/kicks.cc -- a cross-TU call, so not inlinable without LTO --
# and it sits in the innermost loop of the tree walk, executed per (target, source) pair. That is
# the same pattern that cost an order of magnitude earlier in this investigation when opaque
# calls were added to hot loops. Whether it actually costs anything here has never been measured:
# an earlier attempt was confounded by an IC regenerated at the wrong binary separation, which
# changed the problem rather than the code.
#
# Why per PARTICLE-step and not per step or per second. With individual timesteps the two
# variants do not take the same steps -- the prediction changes the forces, so trajectories
# diverge and the step counts differ. Wall time alone therefore conflates "slower per unit work"
# with "more work". GIZMO already counts the work exactly: All.TotNumOfForces, written to
# timings.txt as total-Nf, is the cumulative number of force computations. Cost per particle-step
# = wall / total-Nf is the invariant comparison.
#
# Do NOT read cost per particle-step off cpu.txt: it is written at an interval, not every step,
# and the interval differs between runs, so summing Nactive over its entries counts different
# fractions of each run. That produced an 80x-backwards answer once already.
#
# Both variants: same IC, same params, same node, same build flags apart from the one switch.
set -uo pipefail

TEST=plummer_binaries
TIMEMAX=
RANKS=4
while getopts "t:m:r:h" o; do case $o in
  t) TEST=$OPTARG ;; m) TIMEMAX=$OPTARG ;; r) RANKS=$OPTARG ;;
  h) sed -n '2,30p' "$0"; exit 0 ;; *) exit 2 ;;
esac; done

REPO=$(cd "$(dirname "$0")/.." && pwd); cd "$REPO"
TD="test/$TEST"
[ -f "$TD/$TEST.params" ] || { echo "FATAL: no $TD/$TEST.params"; exit 1; }

module --force purge >/dev/null 2>&1
module load modules openmpi gsl hdf5/mpi-1.12.3 >/dev/null 2>&1
export OMP_NUM_THREADS=1 OMP_PROC_BIND=false; unset OMP_PLACES

echo "=== $(date) source-prediction A/B on $TEST ==="
echo "    repo $REPO @ $(git rev-parse --short HEAD)  ranks=$RANKS"

# The IC must be the one the test intends. A generator run with its own defaults can produce a
# file with the right particle count and a different problem entirely -- that is what invalidated
# the first attempt at this measurement.
/mnt/home/mgrudic/python_work/bin/python - "$TD" "$TEST" <<'PYEOF'
import sys, h5py, numpy as np, os
td, name = sys.argv[1], sys.argv[2]
f = os.path.join(td, f"{name}_ics.hdf5")
if os.path.isfile(f):
    with h5py.File(f) as F:
        x = F["PartType5/Coordinates"][:]
        n = len(x)
    print(f"    IC: {f}  N={n}", end="")
    if name == "plummer_binaries":
        sep = np.median(np.linalg.norm(x[0::2]-x[1::2], axis=1)) * 206264.806
        print(f"  separation={sep:.1f} AU" + ("" if abs(sep-1000.)<1 else "   <-- NOT 1000 AU, WRONG IC"))
    else:
        print()
else:
    print(f"    IC MISSING: {f}")
PYEOF

run_variant () {                     # $1 = label, $2 = extra Config.sh line ("" for none)
    local lab="$1" extra="$2" out="$TD/ab_out_$1"
    cp "$TD/Config.sh" Config.sh
    # OUTPUT_ADDITIONAL_RUNINFO is what writes timings.txt (gravtree.cc:700), i.e. total-Nf.
    # Added to BOTH variants so its own cost cancels.
    echo "OUTPUT_ADDITIONAL_RUNINFO" >> Config.sh
    [ -n "$extra" ] && echo "$extra" >> Config.sh
    make clean >/dev/null 2>&1; make -j8 > "$TD/ab_build_$lab.log" 2>&1
    [ -x GIZMO ] || { echo "    $lab: BUILD FAILED (see $TD/ab_build_$lab.log)"; return 1; }
    mv GIZMO "$TD/GIZMO"

    rm -rf "$out"; mkdir -p "$out"
    sed -e "s|^OutputDir .*|OutputDir  $out|" \
        ${TIMEMAX:+-e "s|^TimeMax .*|TimeMax  $TIMEMAX|"} \
        "$TD/$TEST.params" > "$TD/ab_$lab.params"
    local t0=$(date +%s.%N)
    ( cd "$TD" && mpirun -np "$RANKS" --bind-to none ./GIZMO "ab_$lab.params" 0 > "ab_run_$lab.log" 2>&1 )
    local rc=$? t1=$(date +%s.%N)
    echo "$lab $rc $(echo "$t1 - $t0" | bc)" >> "$TD/ab_times.txt"
    echo "    $lab: exit $rc, $(echo "$t1 - $t0" | bc | cut -c1-7)s"
}

rm -f "$TD/ab_times.txt"
run_variant with    ""
run_variant without "DISABLE_HERMITE_SOURCE_PREDICTION"

echo
echo "=== cost per particle-step ==="
/mnt/home/mgrudic/python_work/bin/python - "$TD" <<'PYEOF'
import re, sys, os
td = sys.argv[1]
times = {}
if os.path.isfile(f"{td}/ab_times.txt"):
    for l in open(f"{td}/ab_times.txt"):
        p = l.split()
        if len(p) == 3: times[p[0]] = (int(p[1]), float(p[2]))

def nf(lab):
    """last cumulative total-Nf from timings.txt; also the final sim time reached"""
    f = f"{td}/ab_out_{lab}/timings.txt"
    tot, t = None, None
    if not os.path.isfile(f): return None, None
    for l in open(f):
        m = re.search(r"total-Nf=\s*(\d+)(\d{9})", l)
        if m: tot = int(m.group(1)) * 10**9 + int(m.group(2))
        m = re.match(r"Step=\s*\d+\s+t=\s*([0-9.eE+-]+)", l)
        if m: t = float(m.group(1))
    return tot, t

rows = []
for lab in ("with", "without"):
    n, t = nf(lab)
    rc, wall = times.get(lab, (None, None))
    rows.append((lab, n, t, wall))
    if n is None:
        print(f"  {lab}: no timings.txt (is OUTPUT_ADDITIONAL_RUNINFO on?)")

print(f"  {'variant':>9} {'particle-steps':>16} {'wall s':>9} {'us/part-step':>13} {'t reached':>10}")
for lab, n, t, wall in rows:
    if n and wall:
        print(f"  {lab:>9} {n:16,d} {wall:9.1f} {1e6*wall/n:13.3f} {t if t else float('nan'):10.4f}")
a = dict((r[0], r) for r in rows)
if a.get("with", (0,None))[1] and a.get("without", (0,None))[1]:
    cw = a["with"][3] / a["with"][1]; co = a["without"][3] / a["without"][1]
    print(f"\n  prediction costs {100*(cw/co-1):+.1f}% per particle-step")
    print(f"  (step counts differ because the two trajectories diverge -- that is why this is")
    print(f"   normalised per particle-step rather than compared as wall time)")
PYEOF
echo "=== $(date) done ==="
